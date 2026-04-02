"""
Bayesian Cylinder Fitting with Laser Scanner Shape Prior
=========================================================

Model:
    theta = (cx, cy, cz, ax, ay, az, r)
        cx, cy, cz  : cylinder axis point (a point on the axis)
        ax, ay, az  : axis direction (unit vector, enforced via normalization)
        r           : cylinder radius

Laser Scanner Prior (visibility model):
    A laser scanner only sees the surface facing the scanner. Points are
    concentrated on the "near side" (front arc) of the cylinder. We model this
    as a von Mises-like angular prior: a point at azimuth phi relative to the
    scan direction has higher probability if it is within the visible arc.

    Concretely, for each observed point p_i, we compute its angular position
    around the cylinder axis relative to the scanner origin, and upweight
    configurations where observed points fall within a visible half-circle
    (FOV ~ pi radians facing the scanner).

Likelihood:
    Each point has a distance to the cylinder surface d_i = |dist_to_axis - r|.
    We use a robust Gaussian likelihood:  p(p_i | theta) ~ exp(-d_i^2 / (2*sigma^2))

Posterior (log):
    log p(theta | data) = log likelihood + log prior

Inference:
    MCMC via emcee (ensemble sampler). Falls back to scipy minimize for MAP.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

np.random.seed(42)

# ── optional: emcee for full MCMC ────────────────────────────────────────────
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

# ── optional: matplotlib for visualization ───────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =============================================================================
# Geometry helpers
# =============================================================================

def dist_to_cylinder_axis(points: np.ndarray, axis_pt: np.ndarray,
                           axis_dir: np.ndarray) -> np.ndarray:
    """
    Signed distances from points to an infinite cylinder axis.

    Parameters
    ----------
    points   : (N, 3)
    axis_pt  : (3,)   – a point on the axis
    axis_dir : (3,)   – unit direction of the axis

    Returns
    -------
    radial_dist : (N,)  – distance from each point to the axis line
    """
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
    diff = points - axis_pt[None, :]               # (N, 3)
    proj = diff - (diff @ axis_dir)[:, None] * axis_dir[None, :]  # (N, 3)
    return np.linalg.norm(proj, axis=1)             # (N,)


def azimuth_around_axis(points: np.ndarray, axis_pt: np.ndarray,
                        axis_dir: np.ndarray,
                        scanner_pos: np.ndarray) -> np.ndarray:
    """
    Angular position of each point around the cylinder axis, measured
    relative to the scanner direction (projected onto the axis-normal plane).

    Returns angles in [-pi, pi]; 0 = facing the scanner.
    """
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    # project scanner direction onto plane normal to axis
    to_scanner = scanner_pos - axis_pt
    to_scanner -= (to_scanner @ axis_dir) * axis_dir
    to_scanner_norm = np.linalg.norm(to_scanner)
    if to_scanner_norm < 1e-9:
        # scanner on axis – no preferred direction; return zeros
        return np.zeros(len(points))
    ref = to_scanner / to_scanner_norm              # reference direction

    # build orthonormal frame in the normal plane
    perp = np.cross(axis_dir, ref)
    perp /= np.linalg.norm(perp) + 1e-12

    # project each point's radial vector onto this frame
    diff = points - axis_pt[None, :]
    radial = diff - (diff @ axis_dir)[:, None] * axis_dir[None, :]  # (N,3)
    r_norm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    radial_unit = radial / r_norm

    cos_phi = radial_unit @ ref
    sin_phi = radial_unit @ perp
    return np.arctan2(sin_phi, cos_phi)             # (N,)


# =============================================================================
# Parameter packing / unpacking
# =============================================================================
# theta = [cx, cy, cz, ax, ay, az, r]   (7-D, axis NOT forced unit here;
#          we normalize inside the functions so the sampler is unconstrained)

def unpack(theta):
    axis_pt  = theta[0:3]
    axis_dir = theta[3:6] / (np.linalg.norm(theta[3:6]) + 1e-12)
    r        = theta[6]
    return axis_pt, axis_dir, r


# =============================================================================
# Log-likelihood
# =============================================================================

def log_likelihood(theta, points: np.ndarray, sigma: float = 0.02) -> float:
    """
    Robust Gaussian likelihood on the surface distance residuals.
    """
    if theta[6] <= 0:
        return -np.inf
    axis_pt, axis_dir, r = unpack(theta)
    radial = dist_to_cylinder_axis(points, axis_pt, axis_dir)
    residuals = radial - r                          # signed: + outside, - inside
    return -0.5 * np.sum(residuals**2) / sigma**2


# =============================================================================
# Log-prior
# =============================================================================

def log_prior(theta, points: np.ndarray,
              scanner_pos: np.ndarray,
              r_mean: float, r_std: float,
              visibility_kappa: float = 2.0) -> float:
    """
    Log prior incorporating:
      1. Gaussian prior on radius  r ~ N(r_mean, r_std^2)
      2. Gaussian prior on axis direction norm  (softly encourages unit vector)
      3. Laser-scanner visibility prior:
           points should lie on the "near side" of the cylinder facing the scanner.
           We model visibility as a von Mises distribution centred at phi=0
           (facing scanner) with concentration kappa.

    Parameters
    ----------
    scanner_pos       : (3,) position of the laser scanner in world coordinates
    r_mean, r_std     : hyperparameters for the radius prior
    visibility_kappa  : concentration of the von Mises visibility prior.
                        Higher = tighter constraint that points face the scanner.
                        Typical range: 0.5 (soft) … 5.0 (strict half-cylinder)
    """
    r = theta[6]

    # ── 1. radius prior ───────────────────────────────────────────────────────
    lp_r = -0.5 * ((r - r_mean) / r_std)**2
    if r <= 0:
        return -np.inf

    # ── 2. axis-direction magnitude (weak penalty away from unit length) ──────
    axis_raw_norm = np.linalg.norm(theta[3:6])
    lp_axis = -0.5 * (axis_raw_norm - 1.0)**2 / 0.5**2

    # ── 3. laser-scanner visibility prior ────────────────────────────────────
    #   A real laser scanner sees ~180° of the cylinder facing it.
    #   von Mises: log p(phi) = kappa * cos(phi)  (up to normalisation constant)
    #   phi = 0 means the point is directly facing the scanner → maximum prob.
    axis_pt, axis_dir, _ = unpack(theta)
    phis = azimuth_around_axis(points, axis_pt, axis_dir, scanner_pos)
    lp_visibility = visibility_kappa * np.sum(np.cos(phis))

    return lp_r + lp_axis + lp_visibility


# =============================================================================
# Log-posterior
# =============================================================================

def log_posterior(theta, points, scanner_pos,
                  sigma=0.02, r_mean=0.1, r_std=0.5,
                  visibility_kappa=2.0):
    lp = log_prior(theta, points, scanner_pos,
                   r_mean=r_mean, r_std=r_std,
                   visibility_kappa=visibility_kappa)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, points, sigma=sigma)
    return lp + ll


# =============================================================================
# MAP estimation (fast initialisation or standalone)
# =============================================================================

def fit_map(points: np.ndarray, scanner_pos: np.ndarray,
            theta_init=None,
            sigma=0.02, r_mean=0.1, r_std=0.5,
            visibility_kappa=2.0) -> dict:
    """
    Maximum A Posteriori estimate via L-BFGS-B.

    Returns
    -------
    dict with keys: theta_map, axis_pt, axis_dir, radius, success
    """
    if theta_init is None:
        # sensible default: centre of point cloud, z-axis, median radius
        centre = points.mean(axis=0)
        theta_init = np.array([*centre, 0., 0., 1., r_mean])

    def neg_log_post(theta):
        return -log_posterior(theta, points, scanner_pos,
                              sigma=sigma, r_mean=r_mean, r_std=r_std,
                              visibility_kappa=visibility_kappa)

    # bounds: only radius must be > 0
    bounds = [(-np.inf, np.inf)] * 6 + [(1e-4, np.inf)]
    result = minimize(neg_log_post, theta_init,
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-12})

    theta_map = result.x
    axis_pt, axis_dir, radius = unpack(theta_map)
    return dict(theta_map=theta_map,
                axis_pt=axis_pt, axis_dir=axis_dir, radius=radius,
                success=result.success, message=result.message)


# =============================================================================
# Full Bayesian inference via MCMC (emcee)
# =============================================================================

def fit_mcmc(points: np.ndarray, scanner_pos: np.ndarray,
             theta_init=None,
             sigma=0.02, r_mean=0.1, r_std=0.5,
             visibility_kappa=2.0,
             n_walkers=32, n_steps=2000, n_burn=500) -> dict:
    """
    Full posterior sampling with emcee ensemble sampler.

    Returns
    -------
    dict with keys: sampler, flat_samples, theta_median,
                    theta_lo (16th pct), theta_hi (84th pct),
                    axis_pt, axis_dir, radius
    """
    if not HAS_EMCEE:
        raise ImportError("emcee is required for MCMC. Install with: pip install emcee")

    ndim = 7

    # start walkers near the MAP estimate
    map_result = fit_map(points, scanner_pos, theta_init,
                         sigma, r_mean, r_std, visibility_kappa)
    theta_map = map_result['theta_map']

    rng = np.random.default_rng(42)
    # small Gaussian ball around MAP
    p0 = theta_map[None, :] + rng.normal(0, 1e-3, (n_walkers, ndim))
    # make sure radii are positive
    p0[:, 6] = np.abs(p0[:, 6]) + 1e-4

    def log_prob(theta):
        return log_posterior(theta, points, scanner_pos,
                             sigma=sigma, r_mean=r_mean, r_std=r_std,
                             visibility_kappa=visibility_kappa)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=True)

    flat_samples = sampler.get_chain(discard=n_burn, thin=10, flat=True)

    theta_median = np.median(flat_samples, axis=0)
    theta_lo     = np.percentile(flat_samples, 16, axis=0)
    theta_hi     = np.percentile(flat_samples, 84, axis=0)

    axis_pt, axis_dir, radius = unpack(theta_median)
    return dict(sampler=sampler,
                flat_samples=flat_samples,
                theta_median=theta_median,
                theta_lo=theta_lo,
                theta_hi=theta_hi,
                axis_pt=axis_pt,
                axis_dir=axis_dir,
                radius=radius)


# =============================================================================
# Synthetic data generator (for testing)
# =============================================================================

def generate_laser_scan_points(true_radius=0.15,
                                true_axis_pt=np.array([0., 0., 0.]),
                                true_axis_dir=np.array([0., 0., 1.]),
                                scanner_pos=np.array([-2., 0., 0.5]),
                                n_points=300,
                                noise_std=0.005,
                                fov_half_angle=np.pi * 0.4,
                                seed=0) -> np.ndarray:
    """
    Simulate a laser scanner seeing a partial arc of a cylinder.

    The scanner can only see points within ±fov_half_angle of the scanner
    direction (default ~108°, i.e. slightly more than a hemisphere).
    """
    rng = np.random.default_rng(seed)

    true_axis_dir = true_axis_dir / np.linalg.norm(true_axis_dir)

    # build a 2-D frame in the plane perpendicular to the axis
    ref = scanner_pos - true_axis_pt
    ref -= (ref @ true_axis_dir) * true_axis_dir
    ref /= np.linalg.norm(ref)
    perp = np.cross(true_axis_dir, ref)
    perp /= np.linalg.norm(perp)

    points = []
    attempts = 0
    while len(points) < n_points and attempts < n_points * 20:
        attempts += 1
        phi = rng.uniform(-np.pi, np.pi)
        # accept only if within scanner FOV (von Mises-like rejection sampling)
        if np.abs(phi) > fov_half_angle:
            continue
        z_frac = rng.uniform(-1.0, 1.0)
        z_coord = z_frac * 1.0  # cylinder height ±1 m
        pt = (true_axis_pt
              + z_coord * true_axis_dir
              + true_radius * (np.cos(phi) * ref + np.sin(phi) * perp))
        pt += rng.normal(0, noise_std, 3)
        points.append(pt)

    return np.array(points)


# =============================================================================
# Visualisation
# =============================================================================

def plot_results(points, scanner_pos, map_result, mcmc_result=None):
    if not HAS_MPL:
        print("matplotlib not available – skipping plot.")
        return

    fig = plt.figure(figsize=(14, 5))

    # ── 3-D point cloud + fitted cylinder ────────────────────────────────────
    ax3 = fig.add_subplot(1, 3 if mcmc_result else 2, 1, projection='3d')
    ax3.scatter(*points.T, s=4, c='steelblue', alpha=0.5, label='scan points')
    ax3.scatter(*scanner_pos, s=80, c='red', marker='^', label='scanner')

    # draw MAP cylinder surface
    axis_pt  = map_result['axis_pt']
    axis_dir = map_result['axis_dir']
    r        = map_result['radius']

    ref = np.array([1., 0., 0.]) if abs(axis_dir[0]) < 0.9 else np.array([0., 1., 0.])
    ref -= (ref @ axis_dir) * axis_dir;  ref /= np.linalg.norm(ref)
    perp = np.cross(axis_dir, ref);      perp /= np.linalg.norm(perp)

    phi_grid  = np.linspace(-np.pi, np.pi, 60)
    z_grid    = np.linspace(-1.2, 1.2, 10)
    PHI, Z    = np.meshgrid(phi_grid, z_grid)
    Xc = axis_pt[0] + r*(np.cos(PHI)*ref[0] + np.sin(PHI)*perp[0]) + Z*axis_dir[0]
    Yc = axis_pt[1] + r*(np.cos(PHI)*ref[1] + np.sin(PHI)*perp[1]) + Z*axis_dir[1]
    Zc = axis_pt[2] + r*(np.cos(PHI)*ref[2] + np.sin(PHI)*perp[2]) + Z*axis_dir[2]
    ax3.plot_surface(Xc, Yc, Zc, alpha=0.15, color='orange')
    ax3.set_title('3-D: scan + MAP cylinder')
    ax3.legend(fontsize=7)

    # ── residual histogram ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3 if mcmc_result else 2, 2)
    radial = dist_to_cylinder_axis(points, axis_pt, axis_dir)
    residuals = radial - r
    ax2.hist(residuals * 1e3, bins=30, color='steelblue', edgecolor='white')
    ax2.axvline(0, color='red', lw=1.5, ls='--')
    ax2.set_xlabel('surface residual (mm)')
    ax2.set_ylabel('count')
    ax2.set_title('Fit residuals (MAP)')

    # ── MCMC posterior corner (radius + one axis component) ──────────────────
    if mcmc_result is not None:
        ax4 = fig.add_subplot(1, 3, 3)
        samples = mcmc_result['flat_samples']
        ax4.scatter(samples[:, 0], samples[:, 6], s=1, alpha=0.2, c='purple')
        ax4.set_xlabel('cx (axis x-position)')
        ax4.set_ylabel('radius r')
        ax4.set_title('MCMC posterior: cx vs r')

    plt.tight_layout()
    plt.savefig('cylinder_fit_results.png', dpi=150)
    plt.show()
    print("Figure saved to cylinder_fit_results.png")


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Bayesian Cylinder Fitting – Laser Scanner Prior Demo")
    print("=" * 60)

    # ── ground truth & scanner setup ─────────────────────────────────────────
    TRUE_RADIUS   = 0.15
    TRUE_AXIS_PT  = np.array([0.1, -0.05, 0.0])
    TRUE_AXIS_DIR = np.array([0.05, 0.02, 1.0])   # slightly tilted
    SCANNER_POS   = np.array([-2.0, 0.5, 0.5])

    print(f"\nGround truth: radius={TRUE_RADIUS:.3f}, "
          f"axis_pt={TRUE_AXIS_PT}, axis_dir={TRUE_AXIS_DIR}")
    print(f"Scanner position: {SCANNER_POS}")

    # ── generate synthetic partial-view point cloud ───────────────────────────
    points = generate_laser_scan_points(
        true_radius=TRUE_RADIUS,
        true_axis_pt=TRUE_AXIS_PT,
        true_axis_dir=TRUE_AXIS_DIR,
        scanner_pos=SCANNER_POS,
        n_points=400,
        noise_std=0.005,
        fov_half_angle=0.2 * np.pi,   # ~117° visible arc
    )
    print(f"\nGenerated {len(points)} scan points (partial arc ~117°)")

    # ── hyperparameters ───────────────────────────────────────────────────────
    SIGMA            = 0.008   # measurement noise std [m]
    R_MEAN           = 0.15    # radius prior mean
    R_STD            = 0.10    # radius prior std
    VIS_KAPPA        = 5 #4.5     # visibility concentration (higher = stronger prior)

    # ── MAP estimate ──────────────────────────────────────────────────────────
    print("\n── MAP estimation ──────────────────────────────────────────")
    map_res = fit_map(points, SCANNER_POS,
                      sigma=SIGMA, r_mean=R_MEAN, r_std=R_STD,
                      visibility_kappa=VIS_KAPPA)
    print(f"  radius   : {map_res['radius']:.4f}  (true: {TRUE_RADIUS:.4f})")
    print(f"  axis_pt  : {np.round(map_res['axis_pt'], 4)}")
    print(f"  axis_dir : {np.round(map_res['axis_dir'], 4)}")
    print(f"  success  : {map_res['success']}  ({map_res['message']})")

    # ── MCMC (if emcee is available) ──────────────────────────────────────────
    mcmc_res = None
    if HAS_EMCEE:
        print("\n── MCMC sampling (emcee) ───────────────────────────────────")
        mcmc_res = fit_mcmc(points, SCANNER_POS,
                            sigma=SIGMA, r_mean=R_MEAN, r_std=R_STD,
                            visibility_kappa=VIS_KAPPA,
                            n_walkers=32, n_steps=1500, n_burn=400)
        s = mcmc_res['flat_samples']
        print(f"\n  Posterior (median ± 1σ)  [{s.shape[0]} samples]")
        names = ['cx', 'cy', 'cz', 'ax', 'ay', 'az', 'r']
        for i, n in enumerate(names):
            med = np.median(s[:, i])
            lo  = med - np.percentile(s[:, i], 16)
            hi  = np.percentile(s[:, i], 84) - med
            print(f"    {n:4s}: {med:+.4f}  -{lo:.4f} / +{hi:.4f}")
    else:
        print("\n[emcee not installed – skipping MCMC. Run: pip install emcee]")

    # ── visualise ─────────────────────────────────────────────────────────────
    plot_results(points, SCANNER_POS, map_res, mcmc_res)