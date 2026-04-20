"""
Fitters/Cylinder_fitters/least_squares_cylinder.py

Least-squares cylinder fit via iterative closest-point style optimization.

Cylinder params returned: [x_c, y_c, z_c, r, h, polar, azim]
"""

import numpy as np
from scipy.optimize import minimize

from scripts.Benchmarking.Fitters.Cylinder_fitters.base_cylinder import BaseCylinderFitter


def _axis_from_angles(polar: float, azim: float) -> np.ndarray:
    return np.array([
        np.sin(polar) * np.cos(azim),
        np.sin(polar) * np.sin(azim),
        np.cos(polar),
    ])


def _residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Algebraic distance of each point to the cylinder surface.
    Residual = (distance from axis) - r
    """
    x_c, y_c, z_c, r, h, polar, azim = params
    center = np.array([x_c, y_c, z_c])
    axis   = _axis_from_angles(polar, azim)

    d      = points - center                             # (N, 3)
    axial  = (d @ axis)[:, None] * axis                  # projection onto axis
    radial = d - axial                                   # perpendicular component
    dist   = np.linalg.norm(radial, axis=1)              # distance from axis

    return dist - r                                      # (N,)


def _cost(params: np.ndarray, points: np.ndarray) -> float:
    res = _residuals(params, points)
    return float(np.sum(res ** 2))


class LeastSquaresCylinderFitter(BaseCylinderFitter):
    """
    Non-linear least-squares cylinder fit.
    Minimizes sum of squared (radial_distance - r)².
    Initialization: PCA-based axis estimate + bounding-box center/height.
    """

    def fit(self, points: np.ndarray, **kwargs) -> np.ndarray:
        N = len(points)
        if N < 6:
            return np.full(7, np.nan)

        # ── init: PCA for axis, bounding box for center / height ──────────
        center0 = points.mean(axis=0)
        _, _, Vt = np.linalg.svd(points - center0)
        axis0   = Vt[0]                                  # principal axis

        # polar / azim from axis direction
        polar0  = float(np.arccos(np.clip(axis0[2], -1, 1)))
        azim0   = float(np.arctan2(axis0[1], axis0[0]))

        # project onto axis to get height extent
        proj    = (points - center0) @ axis0
        h0      = float(proj.max() - proj.min())
        center0 = center0 + (proj.min() + h0 / 2) * axis0  # shift to midpoint

        # radial distance → initial radius
        axial   = proj[:, None] * axis0
        radial  = (points - center0) - axial
        r0      = float(np.mean(np.linalg.norm(radial, axis=1)))

        x0 = np.array([*center0, r0, max(h0, 0.1), polar0, azim0])

        bounds = [
            (None, None), (None, None), (None, None),  # center
            (1e-3, None),                               # r > 0
            (1e-3, None),                               # h > 0
            (0.0, np.pi),                               # polar
            (0.0, 2 * np.pi),                           # azim
        ]

        result = minimize(
            _cost, x0, args=(points,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

        if not result.success and result.fun > 1.0:
            return np.full(7, np.nan)

        return result.x
