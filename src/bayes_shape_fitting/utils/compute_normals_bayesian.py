"""
Bayesian Normal Estimation via Probabilistic PCA
=================================================
Estimates surface normals from point clouds (2D or 3D) using
PPCA, with pluggable uncertainty representations.

Supported uncertainty methods:
    - 'bingham'  : Bingham distribution on S^(d-1)  [default]
    - 'delta'    : Delta-method covariance in R^d (rank d-1)
    - 'planarity': Scalar heuristic (eigenvalue gap, no sigma2 needed)

Usage
-----
    normals, uncertainty = compute_normals_bayes(pts, k=15)
    normals, uncertainty = compute_normals_bayes(pts, k=15, method='delta')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BinghamUncertainty:
    """
    Bingham distribution on S^(d-1).

    Attributes
    ----------
    A : (d, d) concentration matrix.  p(n) ∝ exp(nᵀ A n)
    kappas : (d,) eigenvalues of A, convention: largest = 0, others ≤ 0.
             kappas[0] is most negative  →  strongest repulsion from that axis.
    axes   : (d, d) eigenvectors of A as columns (same as eigenvectors of S).
             axes[:, -1] is the mode (MAP normal).
    """
    A: np.ndarray  # (d, d)
    kappas: np.ndarray  # (d,)
    axes: np.ndarray  # (d, d)

    @property
    def mode(self) -> np.ndarray:
        """MAP normal direction: eigenvector of largest eigenvalue of A."""
        return self.axes[:, -1]

    def sample(self, n_samples: int = 500, seed: int | None = None) -> np.ndarray:
        """
        Draw samples from the Bingham distribution via rejection sampling.
        Returns (n_samples, d) array of unit vectors.
        Efficient when concentrations are not extreme (|kappa| < ~100).
        """
        rng = np.random.default_rng(seed)
        d = self.A.shape[0]
        # Shift A so max eigenvalue = 0  (already by convention, but be safe)
        A_shifted = self.A - self.kappas.max() * np.eye(d)
        samples = []
        while len(samples) < n_samples:
            batch = rng.standard_normal((n_samples * 4, d))
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            batch /= norms
            log_accept = np.einsum('ni,ij,nj->n', batch, A_shifted, batch)
            accept = np.log(rng.random(len(batch))) < log_accept
            samples.extend(batch[accept])
        return np.array(samples[:n_samples])

    def angular_std_deg(self, n_samples: int = 500) -> float:
        """Angular std of posterior samples around the mode, in degrees."""
        samples = self.sample(n_samples)
        cos_a = np.clip(samples @ self.mode, -1.0, 1.0)
        return float(np.degrees(np.std(np.arccos(cos_a))))

    def to_covariance(self, n_samples: int = 300) -> np.ndarray:
        """
        Empirical (d, d) covariance of posterior samples on the sphere.
        Useful for visualisation; not a proper Euclidean covariance.
        """
        s = self.sample(n_samples)
        return np.cov(s.T)  # (d, d)

    def scalar_uncertainty(self, n_samples: int = 300) -> float:
        """Scalar summary: angular std in degrees."""
        return self.angular_std_deg(n_samples)


@dataclass
class DeltaUncertainty:
    """
    Rank-(d-1) covariance of the normal in R^d via eigenvalue perturbation.

    Attributes
    ----------
    cov : (d, d) covariance matrix (rank d-1; zero variance along normal).
    """
    cov: np.ndarray  # (d, d)

    def marginal_std(self) -> np.ndarray:
        """Per-component std of the normal vector."""
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))

    def to_covariance(self) -> np.ndarray:
        """Return the covariance matrix directly."""
        return self.cov

    def scalar_uncertainty(self) -> float:
        """Scalar summary: trace of covariance."""
        return float(np.trace(self.cov))


@dataclass
class PlanarityUncertainty:
    """
    Scalar planarity score ∈ [0, 1].
    1 = perfectly flat (certain normal), 0 = isotropic (no information).
    Does NOT account for n or sigma².
    """
    planarity: float  # scalar in [0, 1]
    eigenvalues: np.ndarray

    def to_covariance(self) -> np.ndarray:
        """
        Pseudo-covariance: isotropic matrix scaled by (1 - planarity).
        Planarity=1 (certain) → zero matrix. Planarity=0 → identity.
        d inferred from eigenvalues length.
        """
        d = len(self.eigenvalues)
        return (1.0 - self.planarity) * np.eye(d)

    def scalar_uncertainty(self) -> float:
        """Scalar summary: 1 - planarity (0=certain, 1=uncertain)."""
        return 1.0 - self.planarity


# ---------------------------------------------------------------------------
# Protocol for uncertainty estimators (extension point)
# ---------------------------------------------------------------------------

@runtime_checkable
class UncertaintyEstimator(Protocol):
    """
    Interface for pluggable uncertainty methods.
    Implement this to add your own estimator.
    """

    def __call__(
            self,
            eigvals: np.ndarray,  # (d,) ascending eigenvalues of S
            eigvecs: np.ndarray,  # (d, d) corresponding eigenvectors
            n_pts: int,  # number of points in neighborhood
            sigma2: float,  # estimated noise variance
    ) -> BinghamUncertainty | DeltaUncertainty | PlanarityUncertainty:
        ...


# ---------------------------------------------------------------------------
# Built-in uncertainty estimators
# ---------------------------------------------------------------------------

def _bingham_estimator(
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        n_pts: int,
        sigma2: float,
) -> BinghamUncertainty:
    """
    Bingham posterior from PPCA.

    Derivation:
        log p(n | X, sigma²) ≈ -n/(2 sigma²) · nᵀ S n  + const
        A = -n/(2 sigma²) · S
        kappa_i = -n/(2 sigma²) · (lambda_i - lambda_{d-1})

    Mode = eigenvector of largest eigenvalue of S = eigvecs[:, -1]
    """
    d = len(eigvals)
    scale = n_pts / (2.0 * sigma2)
    # Shift so largest kappa = 0
    kappas = -scale * (eigvals - eigvals[-1])  # kappas[-1] = 0
    A = eigvecs @ np.diag(kappas) @ eigvecs.T
    return BinghamUncertainty(A=A, kappas=kappas, axes=eigvecs)


def _delta_estimator(
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        n_pts: int,
        sigma2: float,
) -> DeltaUncertainty:
    """
    Delta-method covariance via eigenvalue perturbation theory.

        Cov[n] = sigma²²/n  Σ_{j≠1}  1/(lambda_1 - lambda_j)²  vⱼ vⱼᵀ

    where index 1 = smallest eigenvalue = normal direction.
    Covariance is rank (d-1); zero along the normal itself.
    """
    d = len(eigvals)
    cov = np.zeros((d, d))
    normal_idx = 0  # smallest eigenvalue
    for j in range(1, d):
        vj = eigvecs[:, j]
        gap = eigvals[normal_idx] - eigvals[j]  # negative
        coeff_var = sigma2 ** 2 / (n_pts * gap ** 2)
        cov += coeff_var * np.outer(vj, vj)
    return DeltaUncertainty(cov=cov)


def _planarity_estimator(
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        n_pts: int,
        sigma2: float,
) -> PlanarityUncertainty:
    """
    Scalar planarity heuristic.
    planarity = (lambda_2 - lambda_1) / lambda_{d-1}
    Does not use n_pts or sigma².
    """
    lam1 = eigvals[0]
    lam2 = eigvals[1]
    lamd = eigvals[-1]
    planarity = float((lam2 - lam1) / (lamd + 1e-12))
    planarity = float(np.clip(planarity, 0.0, 1.0))
    return PlanarityUncertainty(planarity=planarity, eigenvalues=eigvals)


_BUILTIN_ESTIMATORS = {
    'bingham': _bingham_estimator,
    'delta': _delta_estimator,
    'planarity': _planarity_estimator,
}


# ---------------------------------------------------------------------------
# Core: fit one local neighborhood
# ---------------------------------------------------------------------------

def _fit_neighborhood(
        pts: np.ndarray,  # (k, d) neighborhood, NOT centered
        sigma2: float | None,
        estimator: UncertaintyEstimator,
) -> tuple[np.ndarray, object]:
    """
    Fit a (d-1)-dimensional plane to a local neighborhood via PPCA.

    Returns
    -------
    normal : (d,) unit normal vector (MAP estimate, sign arbitrary)
    uncertainty : output of the chosen estimator
    """
    center = pts.mean(axis=0)
    X = pts - center
    n, d = X.shape

    S = (X.T @ X) / n
    eigvals, eigvecs = np.linalg.eigh(S)  # ascending: λ₁ ≤ ... ≤ λ_d

    # Noise estimate: smallest eigenvalue = variance orthogonal to surface
    if sigma2 is None:
        sigma2 = float(eigvals[0]) + 1e-12

    # MAP normal = eigenvector of SMALLEST eigenvalue of S
    normal = eigvecs[:, 0]

    uncertainty = estimator(eigvals, eigvecs, n, sigma2)

    return normal, uncertainty


# ---------------------------------------------------------------------------
# Normal orientation
# ---------------------------------------------------------------------------

def _orient_normal(
        normal: np.ndarray,  # (d,)
        point: np.ndarray,  # (d,) the query point
        reference_point: np.ndarray,  # (d,) viewpoint / sensor origin
        outward = True,
) -> np.ndarray:
    """Flip normal to point toward the reference (viewpoint consistency)."""
    direction = reference_point - point
    if np.dot(normal, direction) < 0.0:
        normal = -normal
    if outward:
        normal = -normal
    return normal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_normals_bayes(
        measurement: np.ndarray,
        k: int | None = None,
        reference_point: np.ndarray | None = None,
        method: Literal['bingham', 'delta', 'planarity'] | UncertaintyEstimator = 'bingham',
        sigma2: float | None = None,
) -> tuple[np.ndarray, list]:
    """
    Estimate surface normals with Bayesian uncertainty for a 2D or 3D point cloud.

    Parameters
    ----------
    measurement : (N, d) array, d ∈ {2, 3}
        Input point cloud.
    k : int, optional
        Number of nearest neighbours per point. Defaults to 10 for 3D, 6 for 2D.
    reference_point : (d,) array, optional
        Viewpoint used to orient normals consistently (all normals point toward it).
        If None, normals are unoriented (sign arbitrary).
    method : 'bingham' | 'delta' | 'planarity' | callable
        Uncertainty representation.
        - 'bingham'   : Bingham distribution on S^(d-1)  [default]
        - 'delta'     : Delta-method covariance in R^d
        - 'planarity' : Scalar planarity score
        - callable    : Any object satisfying UncertaintyEstimator protocol
    sigma2 : float, optional
        Fixed noise variance. If None, estimated per-point as the smallest
        eigenvalue of the local covariance.

    Returns
    -------
    normals : (N, d) array of unit normal vectors
    uncertainties : list of N uncertainty objects (type depends on method)

    Examples
    --------
    #>>> pts = np.random.randn(200, 3)
    #>>> normals, unc = compute_normals_bayes(pts, k=15)
    #>>> normals, unc = compute_normals_bayes(pts, k=15, method='delta')
    #>>> # Custom estimator:
    #>>> def my_estimator(eigvals, eigvecs, n_pts, sigma2):
    #...     return {'raw_eigenvalues': eigvals}
    #>>> normals, unc = compute_normals_bayes(pts, method=my_estimator)
    """
    measurement = np.asarray(measurement, dtype=float)
    if measurement.ndim != 2:
        raise ValueError("measurement must be (N, d)")
    N, d = measurement.shape
    if d not in (2, 3):
        raise ValueError(f"Only d=2 or d=3 supported, got d={d}")
    if k is None:
        k = 6 if d == 2 else 10
    if k >= N:
        raise ValueError(f"k={k} must be less than N={N}")

    # Resolve estimator
    if callable(method) and not isinstance(method, str):
        estimator = method
    elif isinstance(method, str):
        if method not in _BUILTIN_ESTIMATORS:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(_BUILTIN_ESTIMATORS)}")
        estimator = _BUILTIN_ESTIMATORS[method]
    else:
        raise TypeError(f"method must be a string or callable, got {type(method)}")

    # Build KD-tree
    tree = KDTree(measurement)
    _, indices = tree.query(measurement, k=k + 1)  # +1: point queries itself
    indices = indices[:, 1:]  # exclude self

    normals = np.zeros((N, d))
    uncertainties = []

    if reference_point is None:
        reference_point = measurement.mean(axis=0)

    for i in range(N):
        neighborhood = measurement[indices[i]]  # (k, d)
        normal, unc = _fit_neighborhood(neighborhood, sigma2, estimator)

        # Orient toward reference if provided

        normal = _orient_normal(normal, measurement[i], reference_point)

        normals[i] = normal
        uncertainties.append(unc)

    return normals, uncertainties