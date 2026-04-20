"""
Fitters/Cylinder_fitters/ransac_cylinder.py

RANSAC-based cylinder fitting.
Each iteration samples 5 points and fits a minimal cylinder, then scores inliers.
"""

import numpy as np
from scipy.optimize import minimize

from scripts.Benchmarking.Fitters.Cylinder_fitters.base_cylinder import BaseCylinderFitter
#from scripts.Benchmarking.Fitters.Cylinder_fitters import (
#    _axis_from_angles, _residuals, _cost,
#)


def _fit_minimal(pts: np.ndarray) -> np.ndarray:
    """Fit a cylinder to a small point set using LS (used inside RANSAC)."""
    center0 = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - center0)
    axis0   = Vt[0]
    polar0  = float(np.arccos(np.clip(axis0[2], -1, 1)))
    azim0   = float(np.arctan2(axis0[1], axis0[0]))

    proj    = (pts - center0) @ axis0
    h0      = max(float(proj.max() - proj.min()), 0.1)
    center0 = center0 + (proj.min() + h0 / 2) * axis0

    axial   = proj[:, None] * axis0
    r0      = max(float(np.mean(np.linalg.norm((pts - center0) - axial, axis=1))), 1e-3)

    x0     = np.array([*center0, r0, h0, polar0, azim0])
    bounds = [(None,None),(None,None),(None,None),(1e-3,None),(1e-3,None),(0,np.pi),(0,2*np.pi)]

    res = minimize(_cost, x0, args=(pts,), method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200})
    return res.x


class RANSACCylinderFitter(BaseCylinderFitter):
    """
    RANSAC cylinder fit.

    Parameters
    ----------
    n_iter : int
        Number of RANSAC iterations.
    inlier_thresh : float
        Maximum |residual| for a point to be an inlier (in the same units as data).
    min_inliers : int
        Minimum inlier count to accept a hypothesis.
    """

    def __init__(self, n_iter: int = 300, inlier_thresh: float = 0.05,
                 min_inliers: int = 10):
        self.n_iter        = n_iter
        self.inlier_thresh = inlier_thresh
        self.min_inliers   = min_inliers

    def fit(self, points: np.ndarray, *args, **kwargs) -> np.ndarray:
        N = len(points)
        if N < 6:
            return np.full(7, np.nan)

        best_params  = None
        best_inliers = -1

        for _ in range(self.n_iter):
            idx    = np.random.choice(N, 6, replace=False)
            sample = points[idx]

            try:
                params = _fit_minimal(sample)
            except Exception:
                continue

            res     = np.abs(_residuals(params, points))
            inliers = np.sum(res < self.inlier_thresh)

            if inliers > best_inliers:
                best_inliers = inliers
                best_params  = params

        if best_params is None or best_inliers < self.min_inliers:
            return np.full(7, np.nan)

        # Refit on all inliers
        res     = np.abs(_residuals(best_params, points))
        inliers = points[res < self.inlier_thresh]

        if len(inliers) < 6:
            return best_params

        bounds = [(None,None),(None,None),(None,None),(1e-3,None),(1e-3,None),(0,np.pi),(0,2*np.pi)]
        result = minimize(_cost, best_params, args=(inliers,), method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": 1000, "ftol": 1e-12})
        return result.x
