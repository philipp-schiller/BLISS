import numpy as np
from scipy.spatial import KDTree
from dataclasses import dataclass
from scipy.special import erf

def compute_normals(
    measurement: np.ndarray,
    k:           int = None,
) -> np.ndarray:
    """
    Computes outward-pointing unit normals via PCA on k-nearest neighbours.

    Parameters
    ----------
    measurement : (N, d)  point cloud, d = 2 or 3
    k           : neighbours for PCA. Defaults to max(10, sqrt(N)).
    """
    N, d = measurement.shape
    assert d in (2, 3), f"Only 2D and 3D supported, got d={d}"

    if k is None:
        k = max(10, int(np.sqrt(N)))
    k = min(k, N - 1)

    tree    = KDTree(measurement)
    normals = np.zeros((N, d))

    _, idx = tree.query(measurement, k=k + 1)    # (N, k+1) — includes self

    for i in range(N):
        neighbours = measurement[idx[i, 1:]]     # (k, d)
        centred    = neighbours - neighbours.mean(axis=0)
        _, _, Vt   = np.linalg.svd(centred, full_matrices=True)
        normals[i] = Vt[-1]

    # ── orient outward from centroid ──────────────────────────────────────
    normals = _orient_normals_outward(measurement, normals)

    # ── normalise (SVD gives unit vectors but be safe) ────────────────────
    norms        = np.linalg.norm(normals, axis=1, keepdims=True)
    normals      = normals / np.maximum(norms, 1e-10)

    return normals


def _orient_normals_outward(
    points: np.ndarray,
    normals: np.ndarray,
    reference_point: np.ndarray | None = None,
) -> np.ndarray:
    """
    Orient normals so they point away from a reference point.

    Parameters
    ----------
    points : (N, d)
    normals : (N, d)
    reference_point : (d,) or None

    Returns
    -------
    normals : (N, d)
        Flipped consistently outward
    """

    if reference_point is None:
        reference_point = points.mean(axis=0)

    oriented = normals.copy()

    # vector from reference → point
    directions = points - reference_point  # (N, d)

    dots = np.sum(oriented * directions, axis=1)

    flip_mask = dots < 0
    oriented[flip_mask] *= -1

    return oriented


@dataclass
class BayesianNormalResult:
    normals:      np.ndarray   # (N, d) — unit normal vectors
    normal_var:   np.ndarray   # (N,)   — scalar variance per normal
    eigenvalues:  np.ndarray   # (N, d) — all local PCA eigenvalues


def compute_normals_bayesian(
    measurement:      np.ndarray,   # (N, d)
    k:                int   = None, # neighbours; defaults to max(10, sqrt(N))
    beta:             float = 0.5,  # noise factor (normalsDistanceFactor in BayesianSSD)
    sigma_points:     float = None, # point position std; if None estimated from data
    reference_point: np.ndarray | None = None
) -> BayesianNormalResult:
    """
    Bayesian PCA normal estimation following the BayesianSSD approach.

    Each point's normal is the eigenvector of the local covariance matrix
    corresponding to the smallest eigenvalue (standard PCA), but the
    uncertainty of that normal is derived from the eigenvalue structure
    under a probabilistic noise model.

    The key idea: given noisy points with isotropic noise σ², the local
    covariance matrix is:

        C = C_true + σ² I

    So the smallest eigenvalue λ_min ≈ σ² (pure noise), and the normal
    uncertainty grows as the gap (λ_2 - λ_min) shrinks — i.e. when the
    surface is nearly flat or the neighbourhood is noisy.

    The normal variance is derived as:

        Var(n) ≈ σ²_eff / (k * (λ_2 - λ_min)²)

    where σ²_eff = beta * λ_min is the effective noise variance estimated
    from the smallest eigenvalue (matching BayesianSSD's normalsDistanceFactor).

    Parameters
    ----------
    measurement   : (N, d) point cloud
    k             : neighbourhood size
    beta          : noise factor — scales how much λ_min contributes to
                    normal uncertainty (normalsDistanceFactor in BayesianSSD)
    sigma_points  : known point position std; if None, estimated per-point
                    from the smallest eigenvalue

    Returns
    -------
    BayesianNormalResult with normals, per-normal variance, and eigenvalues
    """
    N, d = measurement.shape
    assert d in (2, 3), f"Only 2D and 3D supported, got d={d}"

    if k is None:
        k = max(10, int(np.sqrt(N)))
    k = min(k, N - 1)

    tree = KDTree(measurement)
    _, idx = tree.query(measurement, k=k + 1)    # (N, k+1) includes self

    normals     = np.zeros((N, d))
    normal_var  = np.zeros(N)
    eigenvalues = np.zeros((N, d))

    for i in range(N):
        neighbours = measurement[idx[i, 1:]]     # (k, d)
        centred    = neighbours - neighbours.mean(axis=0)

        # ── local covariance matrix ────────────────────────────────────
        C = (centred.T @ centred) / (k - 1)      # (d, d)

        # ── eigendecomposition ─────────────────────────────────────────
        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(C)     # eigvals: (d,), eigvecs: (d, d)

        # normal = eigenvector of smallest eigenvalue
        normals[i]     = eigvecs[:, 0]            # (d,)
        eigenvalues[i] = eigvals                  # (d,) ascending

        # ── Bayesian uncertainty of the normal ─────────────────────────
        lambda_min = eigvals[0]                   # noise floor
        lambda_2   = eigvals[1]                   # next eigenvalue

        # effective noise variance: β * λ_min
        # (if sigma_points provided, use that instead)
        if sigma_points is not None:
            sigma_sq_eff = sigma_points ** 2
        else:
            sigma_sq_eff = beta * lambda_min

        # eigenvalue gap: how well-defined is the normal direction?
        # small gap → ambiguous normal → high variance
        gap = lambda_2 - lambda_min
        gap = max(gap, 1e-10)                     # numerical safety

        # Normal variance from perturbation theory of eigenvectors:
        # Var(n_i) ≈ σ²_eff / (k * gap²)
        # This is the variance of the angular deviation in radians²
        normal_var[i] = sigma_sq_eff / (k * gap ** 2)

    # ── orient outward from centroid ──────────────────────────────────────
    normals = _orient_normals_outward(measurement, normals,reference_point)

    # ── normalise ─────────────────────────────────────────────────────────
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-10)

    return BayesianNormalResult(
        normals     = normals,
        normal_var  = normal_var,
        eigenvalues = eigenvalues,
    )


def _cdf(x: float, mu: float, std: float) -> float:
    return 0.5 * (1.0 + np.abs(erf((x - mu) / (np.sqrt(2.0) * std))))


def compute_normals_bayes(
    measurement: np.ndarray,            # (N, d)
    k: int = None,
    noise_variance: float = 1.0,
    var_by_distance: float = 0.0,
    points_correlation: float = 0.0,
    mean_curvature: float = 0.0,
    return_covariance: bool = True,
    reference_point: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Bayesian normal estimation for point clouds.

    Returns:
        normals: (N, d)
        normals_inv_cov: (N, d, d) or None
    """

    points = measurement
    N, dim = points.shape

    if k is None:
        k = max(10, int(np.sqrt(N)))

    use_covariances = points_correlation > 0.0

    # --- KDTree ---
    tree = KDTree(points)

    # --- pass 1: global mean neighbor scale ---
    mean_neighbor_distance = 0.0

    for i in range(N):
        dists, idx = tree.query(points[i:i+1], k=k)
        neigh = points[idx[0]]

        c = neigh.mean(axis=0)
        mean_dist = np.linalg.norm(neigh - c, axis=1).mean()
        mean_neighbor_distance += mean_dist / N

    mean_sq_dist = mean_neighbor_distance ** 2

    normals = np.zeros_like(points)
    normals_inv_cov = np.zeros((N, dim, dim)) if return_covariance else None

    # --- main loop ---
    for i in range(N):
        dists, idx = tree.query(points[i:i+1], k=k)
        neigh = points[idx[0]]

        c = neigh.mean(axis=0)
        centered = neigh - c

        nCov = np.zeros((dim, dim))
        P = np.zeros((k, dim))
        pCov = np.zeros((k, k))

        mean_dist = np.linalg.norm(centered, axis=1).mean()

        # --- build covariance ---
        for j in range(len(neigh)):
            p = centered[j]
            sq_dist = np.dot(p, p) / mean_sq_dist

            pVar = (1.0 + var_by_distance * (sq_dist - 1.0)) * noise_variance
            pVar += (mean_curvature ** 2) * np.dot(p, p)

            if use_covariances:
                P[j] = p
                pCov[j, j] = pVar
            else:
                inv_var = 1.0 / (pVar + 1e-12)
                nCov += inv_var * np.outer(p, p)

        # --- inter-point covariance (Bayesian part) ---
        if use_covariances:
            sq_mean_pair = (mean_dist ** 2) / k

            for j in range(len(neigh)):
                for l in range(j + 1, len(neigh)):
                    diff = neigh[j] - neigh[l]
                    sq_dist = np.dot(diff, diff)

                    cov = (
                        points_correlation
                        * np.sqrt(pCov[j, j] * pCov[l, l])
                        * np.exp(-sq_dist / (2.0 * sq_mean_pair))
                    )

                    pCov[j, l] = cov
                    pCov[l, j] = cov

            # Solve linear system
            invCovP = np.linalg.solve(
                pCov + 1e-9 * np.eye(len(neigh)), P
            )
            nCov = P.T @ invCovP

        # --- eigen decomposition ---
        eigen_vals, eigen_vecs = np.linalg.eigh(nCov)

        min_idx = np.argmin(np.abs(eigen_vals))
        normal = eigen_vecs[:, min_idx]
        normal /= np.linalg.norm(normal) + 1e-12

        normals[i] = normal

        # --- covariance estimation ---
        if return_covariance:
            ev = eigen_vals.copy()

            sum_var = 0.0
            for d in range(1, dim):
                idx_d = (min_idx + d) % dim
                ev[idx_d] -= ev[min_idx]
                sum_var += 1.0 / (ev[idx_d] + 1e-12)

            sum_std = np.sqrt(sum_var + 1e-12)

            last_cdf = 0.5
            varz = 0.0

            for j in range(15):
                s = (9.0 * j / 15.0) ** 2 * sum_std
                f = (9.0 * (j + 1) / 15.0) ** 2 * sum_std

                x = 0.5 * (s + f)
                vcdf = _cdf(f, 0.0, sum_std)

                v = 1.0 - np.sqrt(max(1 - x * x, 0.0))
                varz += 2.0 * (vcdf - last_cdf) * v * v
                last_cdf = vcdf

            ev[min_idx] = 1.0 / (varz + 1e-12)

            normals_inv_cov[i] = (
                eigen_vecs @ np.diag(ev) @ eigen_vecs.T
            )

        # ── orient outward from centroid ──────────────────────────────────────
    normals = _orient_normals_outward(measurement, normals, reference_point)

    return normals, normals_inv_cov
