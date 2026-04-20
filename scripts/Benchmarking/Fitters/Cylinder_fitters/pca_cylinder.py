"""
Fitters/Cylinder_fitters/pca_cylinder.py

PCA-based cylinder fit (closed-form, fast, no optimization).
The principal eigenvector of the point cloud gives the axis direction;
center and radius are then computed analytically.
"""

import numpy as np

from scripts.Benchmarking.Fitters.Cylinder_fitters.base_cylinder import BaseCylinderFitter


class PCACylinderFitter(BaseCylinderFitter):
    """
    Fast closed-form cylinder fitting via PCA.

    Strategy
    --------
    1. PCA on the point cloud → axis = first principal component.
    2. Project points onto the cross-section plane (perpendicular to axis).
    3. Fit a circle to the 2-D projected points (algebraic least squares).
    4. Recover 3-D center and height from the axis projection.

    Returns [x_c, y_c, z_c, r, h, polar, azim].
    """

    def fit(self, points: np.ndarray, *args, **kwargs) -> np.ndarray:
        N = len(points)
        if N < 6:
            return np.full(7, np.nan)

        # ── 1. PCA for axis ───────────────────────────────────────────────
        centroid = points.mean(axis=0)
        _, _, Vt  = np.linalg.svd(points - centroid)
        axis      = Vt[0]                               # unit vector (3,)

        polar = float(np.arccos(np.clip(axis[2], -1.0, 1.0)))
        azim  = float(np.arctan2(axis[1], axis[0]))

        # ── 2. Build orthonormal frame (e1, e2) ⊥ axis ───────────────────
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 \
              else np.array([1.0, 0.0, 0.0])
        e1  = np.cross(ref, axis);   e1 /= np.linalg.norm(e1)
        e2  = np.cross(axis, e1)

        # ── 3. Project onto cross-section plane ───────────────────────────
        d   = points - centroid
        u2d = d @ e1                                    # (N,)
        v2d = d @ e2                                    # (N,)

        # Algebraic circle fit:  u² + v² + Du + Ev + F = 0
        # → least-squares on [u, v, 1] * [D, E, F]ᵀ = -(u² + v²)
        A = np.column_stack([u2d, v2d, np.ones(N)])
        b = -(u2d**2 + v2d**2)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        D, E, F  = coef

        cx2d = -D / 2.0
        cy2d = -E / 2.0
        r    = float(np.sqrt(cx2d**2 + cy2d**2 - F))

        # 3-D circle center projected back
        center_3d = centroid + cx2d * e1 + cy2d * e2

        # ── 4. Axial extent ───────────────────────────────────────────────
        proj  = d @ axis                                 # (N,) signed axial coords
        h_min = float(proj.min())
        h_max = float(proj.max())
        h     = max(h_max - h_min, 1e-3)

        # Shift center to mid-height
        center_3d = center_3d + (h_min + h / 2) * axis

        x_c, y_c, z_c = center_3d

        return np.array([x_c, y_c, z_c, r, h, polar, azim])
