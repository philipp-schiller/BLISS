"""
Cylinder Fitting via Variational EM
=====================================
Generative model:
    s ~ p(s | alpha)       # source density on cylinder surface S_theta
    x | s, beta ~ N(s, sigma^2 * I)   # isotropic Gaussian sensor noise

    theta = (axis direction d, axis point p, radius r)  -- cylinder geometry
    alpha = (uniform surface density, or axial Gaussian bias mu_z, sigma_z)
    beta  = sigma^2  -- noise variance

The marginal likelihood for one point x is:
    p(x | theta, alpha, beta) = integral_{S_theta} N(x; s, sigma^2 I) p(s|alpha) ds

This integral is intractable in closed form (surface integral over cylinder).

Variational EM approximation:
    q(s_i | x_i) = N(s_i; c_i, tau^2 I)   centered at closest point c_i on surface
    -> soft responsibility weights over surface elements

In practice we use a finite discretization of the surface for the E-step,
and update (d, p, r, sigma) in the M-step.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Cylinder geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class Cylinder:
    """Cylinder parametrized by axis point, axis direction, and radius."""
    p: np.ndarray   # point on axis (3,)
    d: np.ndarray   # unit direction of axis (3,)
    r: float        # radius
    h: float = 0.3  # half-height for surface sampling / visualization

    def normalize(self):
        self.d = self.d / (np.linalg.norm(self.d) + 1e-12)
        return self

    def closest_point(self, x: np.ndarray) -> np.ndarray:
        """
        Project x onto the lateral surface of the cylinder.
        x: (N, 3)  ->  closest surface points (N, 3)
        """
        # Axial coordinate
        v = x - self.p[None, :]                     # (N, 3)
        t = v @ self.d                               # (N,) axial parameter
        foot = self.p[None, :] + t[:, None] * self.d[None, :]  # (N, 3) foot on axis
        radial = x - foot                            # (N, 3) radial vector
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
        radial_unit = radial / radial_norm           # (N, 3)
        return foot + self.r * radial_unit           # (N, 3)

    def point_to_surface_dist(self, x: np.ndarray) -> np.ndarray:
        """Signed radial distance from lateral surface. (N,)"""
        v = x - self.p[None, :]
        t = v @ self.d
        foot = self.p[None, :] + t[:, None] * self.d[None, :]
        radial_dist = np.linalg.norm(x - foot, axis=1)
        return radial_dist - self.r

    def sample_surface(self, N: int, alpha_sigma_z: float = None) -> np.ndarray:
        """
        Sample N points from the lateral surface.
        p(s | alpha): uniform in angle, Gaussian or uniform in axial direction.
        """
        angles = np.random.uniform(0, 2 * np.pi, N)

        if alpha_sigma_z is not None:
            # Gaussian source density along axis
            t = np.random.normal(0, alpha_sigma_z, N)
            t = np.clip(t, -self.h, self.h)
        else:
            t = np.random.uniform(-self.h, self.h, N)

        # Build local frame: (e1, e2) perpendicular to d
        e1 = _perp_vector(self.d)
        e2 = np.cross(self.d, e1)
        e1 /= np.linalg.norm(e1); e2 /= np.linalg.norm(e2)

        pts = (self.p[None, :]
               + t[:, None] * self.d[None, :]
               + self.r * np.cos(angles)[:, None] * e1[None, :]
               + self.r * np.sin(angles)[:, None] * e2[None, :])
        return pts


def _perp_vector(d: np.ndarray) -> np.ndarray:
    """Find an arbitrary vector perpendicular to d."""
    d = d / np.linalg.norm(d)
    if abs(d[0]) < 0.9:
        ref = np.array([1., 0., 0.])
    else:
        ref = np.array([0., 1., 0.])
    return np.cross(d, ref)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_cylinder_data(
    cyl: Cylinder,
    N: int = 200,
    sigma_noise: float = 0.05,
    outlier_fraction: float = 0.1,
    alpha_sigma_z: float = None,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    N_inlier = int(N * (1 - outlier_fraction))
    N_outlier = N - N_inlier

    surface_pts = cyl.sample_surface(N_inlier, alpha_sigma_z)
    noise = np.random.randn(*surface_pts.shape) * sigma_noise
    inliers = surface_pts + noise

    # Outliers: random points in bounding box
    bbox = cyl.r * 2.5
    outliers = np.random.uniform(-bbox, bbox, (N_outlier, 3))
    outliers[:, 2] += np.random.uniform(-cyl.h, cyl.h, N_outlier)

    return np.vstack([inliers, outliers])


# ---------------------------------------------------------------------------
# Variational EM
# ---------------------------------------------------------------------------

class VariationalEMCylinderFit:
    """
    Fit a cylinder to a point cloud via Variational EM.

    Variational approximation:
        q(s_i | x_i) is a point mass at the closest surface point c_i(theta).

    This corresponds to hard-assignment EM (ICP-style), but we track the
    ELBO / soft responsibility weights via a temperature-annealed Gaussian
    kernel for robustness.

    Model:
        p(x_i | s_i, beta) = N(x_i; s_i, sigma^2 I)
        p(s_i | alpha)      = uniform on lateral surface  (alpha fixed)
        p(theta)            = flat prior

    Parameters estimated: d (axis dir), p (axis point), r (radius), sigma^2.
    """

    def __init__(
        self,
        sigma_init: float = 0.2,
        temperature_schedule: str = "annealed",
        n_iter: int = 50,
        tol: float = 1e-5,
        outlier_weight: float = 0.1,
        verbose: bool = True
    ):
        self.sigma_init = sigma_init
        self.temperature_schedule = temperature_schedule
        self.n_iter = n_iter
        self.tol = tol
        self.outlier_weight = outlier_weight
        self.verbose = verbose
        self.elbo_history = []

    def _init_cylinder(self, X: np.ndarray) -> Cylinder:
        """PCA-based initialization: axis = dominant direction, radius = median radial dist."""
        centroid = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - centroid)
        d_init = Vt[0]  # principal direction
        cyl = Cylinder(p=centroid, d=d_init, r=1.0)
        cyl.normalize()
        # Estimate radius from radial distances
        v = X - centroid[None, :]
        t = v @ d_init
        foot = centroid[None, :] + t[:, None] * d_init[None, :]
        radial_dists = np.linalg.norm(X - foot, axis=1)
        cyl.r = np.median(radial_dists)
        return cyl

    def _e_step(self, X: np.ndarray, cyl: Cylinder, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Variational E-step:
            c_i = closest point on S_theta to x_i   (hard assignment)
            w_i = N(x_i; c_i, sigma^2 I) / Z        (soft weight, used in ELBO)

        Returns:
            C: (N, 3) closest surface points
            w: (N,)   normalized responsibility weights
        """
        C = cyl.closest_point(X)                          # (N, 3)
        sq_dist = np.sum((X - C) ** 2, axis=1)            # (N,)
        # Inlier log-likelihood under Gaussian sensor model
        log_p_inlier = -0.5 * sq_dist / (sigma ** 2)
        # Background uniform component: constant log-likelihood
        # (models outliers as uniformly distributed)
        log_p_outlier = np.log(self.outlier_weight + 1e-12) * np.ones(len(X))
        # Mix: responsibility = p(inlier | x_i)
        log_p_inlier += np.log(1.0 - self.outlier_weight + 1e-12)
        # Softmax-style normalization
        log_stack = np.stack([log_p_inlier, log_p_outlier], axis=1)  # (N, 2)
        log_stack -= log_stack.max(axis=1, keepdims=True)
        resp = np.exp(log_stack)
        resp /= resp.sum(axis=1, keepdims=True)
        inlier_resp = resp[:, 0]   # (N,) responsibility of inlier component
        # Final weights: inlier responsibilities, normalized
        w = inlier_resp / (inlier_resp.sum() + 1e-12)
        return C, w

    def _m_step(self, X: np.ndarray, C: np.ndarray, w: np.ndarray) -> Tuple[Cylinder, float]:
        """
        Variational M-step: update theta = (d, p, r) and sigma.

        Weighted least-squares on the cylinder parameters.
        We re-estimate:
          1. axis direction d: dominant eigenvector of weighted covariance of closest pts
          2. axis point p:     weighted centroid of closest pts
          3. radius r:         weighted mean radial distance
          4. sigma:            RMS of (x_i - c_i) weighted by w_i
        """
        # 1 & 2: axis direction via weighted PCA of DATA points
        # The axis is the direction that minimizes radial distances,
        # i.e. the direction with least variance in the RADIAL sense.
        # Equivalently: axis = eigenvector of weighted scatter of X with smallest
        # RADIAL contribution -> use the dominant eigenvector of X scatter
        # (points on a cylinder have variance mostly along axis + tangentially)
        mu_x = (w[:, None] * X).sum(axis=0)
        X_centered = X - mu_x[None, :]
        cov = X_centered.T @ (w[:, None] * X_centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        d_new = eigvecs[:, -1]  # dominant eigenvector = axis direction

        p_new = mu_x

        cyl_new = Cylinder(p=p_new, d=d_new, r=1.0)
        cyl_new.normalize()

        # Radius: weighted mean radial distance of X from the new axis
        v = X - p_new[None, :]
        t = v @ d_new
        foot = p_new[None, :] + t[:, None] * d_new[None, :]
        radial = np.linalg.norm(X - foot, axis=1)
        r_new = np.sum(w * radial)
        cyl_new.r = max(r_new, 1e-3)

        # Sigma: weighted RMSE to closest surface points
        C_new = cyl_new.closest_point(X)
        sq_dist = np.sum((X - C_new) ** 2, axis=1)
        sigma_new = np.sqrt(np.sum(w * sq_dist) + 1e-8)
        sigma_new = max(sigma_new, 1e-4)

        return cyl_new, sigma_new

    def _elbo(self, X: np.ndarray, C: np.ndarray, w: np.ndarray, sigma: float) -> float:
        """
        ELBO = E_q[log p(x|s,beta)] - KL[q||p(s|alpha)]
        Under point-mass q and uniform p(s|alpha), KL is a constant,
        so we track only the expected log-likelihood term.
        """
        sq_dist = np.sum((X - C) ** 2, axis=1)
        n = X.shape[0]
        ell = -0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(w * sq_dist) / sigma**2
        return ell

    def fit(self, X: np.ndarray) -> Cylinder:
        cyl = self._init_cylinder(X)
        sigma = self.sigma_init
        self.elbo_history = []

        if self.verbose:
            print(f"{'Iter':>5} | {'ELBO':>12} | {'r':>8} | {'sigma':>8}")
            print("-" * 45)

        for i in range(self.n_iter):
            # Temperature annealing: start with large sigma (soft), decay to estimated sigma
            if self.temperature_schedule == "annealed" and i < self.n_iter // 2:
                sigma_eff = max(sigma, self.sigma_init * (0.95 ** i))
            else:
                sigma_eff = sigma

            # VE step
            C, w = self._e_step(X, cyl, sigma_eff)

            # VM step
            cyl_new, sigma_new = self._m_step(X, C, w)

            elbo = self._elbo(X, C, w, sigma_eff)
            self.elbo_history.append(elbo)

            if self.verbose:
                print(f"{i+1:>5} | {elbo:>12.4f} | {cyl_new.r:>8.4f} | {sigma_new:>8.4f}")

            # Convergence check
            if i > 0 and abs(self.elbo_history[-1] - self.elbo_history[-2]) < self.tol:
                if self.verbose:
                    print(f"\nConverged at iteration {i+1}")
                break

            cyl = cyl_new
            sigma = sigma_new

        self.cyl_ = cyl
        self.sigma_ = sigma
        return cyl


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(X: np.ndarray, cyl_true: Cylinder, cyl_fit: Cylinder, elbo_history):
    fig = plt.figure(figsize=(15, 5))

    # --- 3D point cloud + fitted cylinder ---
    ax1 = fig.add_subplot(131, projection='3d')

    # Surface mesh for visualization
    def cylinder_mesh(cyl, n_theta=30, n_z=10):
        e1 = _perp_vector(cyl.d); e1 /= np.linalg.norm(e1)
        e2 = np.cross(cyl.d, e1); e2 /= np.linalg.norm(e2)
        theta_arr = np.linspace(0, 2*np.pi, n_theta)
        z_arr = np.linspace(-cyl.h, cyl.h, n_z)
        T, Z = np.meshgrid(theta_arr, z_arr)
        Xm = (cyl.p[0] + Z*cyl.d[0] + cyl.r*np.cos(T)*e1[0] + cyl.r*np.sin(T)*e2[0])
        Ym = (cyl.p[1] + Z*cyl.d[1] + cyl.r*np.cos(T)*e1[1] + cyl.r*np.sin(T)*e2[1])
        Zm = (cyl.p[2] + Z*cyl.d[2] + cyl.r*np.cos(T)*e1[2] + cyl.r*np.sin(T)*e2[2])
        return Xm, Ym, Zm

    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], s=5, alpha=0.4, c='steelblue', label='Data')
    Xm, Ym, Zm = cylinder_mesh(cyl_true)
    ax1.plot_surface(Xm, Ym, Zm, alpha=0.15, color='green', label='True')
    Xm, Ym, Zm = cylinder_mesh(cyl_fit)
    ax1.plot_surface(Xm, Ym, Zm, alpha=0.25, color='red', label='Fitted')
    ax1.set_title("Point cloud + cylinders\n(green=true, red=fitted)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_aspect("equal")

    # --- Radial distance histogram ---
    ax2 = fig.add_subplot(132)
    dist_true = cyl_true.point_to_surface_dist(X)
    dist_fit  = cyl_fit.point_to_surface_dist(X)
    bins = np.linspace(-1, 1, 60)
    ax2.hist(dist_true, bins=bins, alpha=0.5, color='green', label=f'True (r={cyl_true.r:.2f})')
    ax2.hist(dist_fit,  bins=bins, alpha=0.5, color='red',   label=f'Fitted (r={cyl_fit.r:.2f})')
    ax2.axvline(0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel("Signed radial distance to surface")
    ax2.set_ylabel("Count")
    ax2.set_title("Radial residuals")
    ax2.legend()

    # --- ELBO history ---
    ax3 = fig.add_subplot(133)
    ax3.plot(elbo_history, marker='o', markersize=3, color='steelblue')
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("ELBO")
    ax3.set_title("Variational ELBO")
    ax3.grid(True, alpha=0.3)
    plt.show()
    plt.tight_layout()
    plt.savefig("cylinder_fit_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot to cylinder_fit_results.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def angle_between(d1, d2):
    cos_a = abs(np.clip(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2)), -1, 1))
    return np.degrees(np.arccos(cos_a))


if __name__ == "__main__":

    # Ground truth cylinder
    """true_cyl = Cylinder(
        p=np.array([0.1, -0.2, 0.0]),
        d=np.array([0.1, 0.2, 1.0]),
        r=1.5,
        h=3.0
    ).normalize()"""

    # for the birch
    true_cyl = Cylinder(
        p = np.array([1.124, 40.3472, -0.9301]),
        d = np.array([-0.0577, -0.0160, 0.9982]),
        r = 0.02,
        h = 0.3
    ).normalize()

    print("=" * 55)
    print("  Cylinder Fitting via Variational EM")
    print("=" * 55)
    print(f"True cylinder: r={true_cyl.r:.3f}, d={true_cyl.d.round(3)}")
    print()

    # Generate noisy data with outliers
    """X = generate_cylinder_data(
        true_cyl,
        N=300,
        sigma_noise=0.05,
        outlier_fraction=0.15,
        alpha_sigma_z=None,   # uniform source density along axis
        seed=0
    )"""

    X = np.loadtxt("C:/Users/vtm263/PycharmProjects/bayes-shape-fitting/tests/fixtures/birch.xyz")

    # Fit
    vem = VariationalEMCylinderFit(
        sigma_init=0.01,
        temperature_schedule="annealed",
        n_iter=60,
        tol=1e-6,
        verbose=True
    )
    fit_cyl = vem.fit(X)

    print()
    print("=" * 55)
    print("  Results")
    print("=" * 55)
    print(f"True   r = {true_cyl.r:.4f}  |  Fitted r = {fit_cyl.r:.4f}")
    print(f"True   d = {true_cyl.d.round(4)}")
    print(f"Fitted d = {fit_cyl.d.round(4)}")
    print(f"Axis angle error : {angle_between(true_cyl.d, fit_cyl.d):.3f} deg")
    print(f"Radius error     : {abs(fit_cyl.r - true_cyl.r):.4f}")
    print(f"Final sigma      : {vem.sigma_:.4f}")

    plot_results(X, true_cyl, fit_cyl, vem.elbo_history)
