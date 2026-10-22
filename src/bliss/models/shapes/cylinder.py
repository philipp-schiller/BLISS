from dataclasses import dataclass
from typing import override
from scipy.optimize import minimize

import numpy as np

from .base_shape import Shape


@dataclass
class Cylinder(Shape):
    """
    Circular cylinder in 3D: γ(u,v; x_c, y_c, z_c, r, h, polar, azim)

    The cylinder axis is defined by a unit vector û derived from spherical angles:
        polar ∈ [0, π]  — polar angle from z-axis
        azim  ∈ [0, 2π) — azimuthal angle in xy-plane

        û = (sin(polar)cos(azim), sin(polar)sin(azim), cos(polar))

    ê₁, ê₂ are a fixed orthonormal basis of the cross-section plane (no roll).

    Parametrization:
        γ(u, v) = p + h·u·û + r·cos(2πv)·ê₁ + r·sin(2πv)·ê₂

    where p = (x_c, y_c, z_c) is the center of the bottom cap.

    Parameter space: (u,v) ∈ [0,1] × [0,1), v periodic
    Ambient space: R^3
    Shape parameters: [x_c, y_c, z_c, r, h, polar, azim]
    """

    def __init__(self):
        def _build_frame(theta: np.ndarray):
            """Compute û, ê₁, ê₂ from (polar, azim)."""
            polar, azim = theta[5], theta[6]

            u_hat = np.array([
                np.sin(polar) * np.cos(azim),
                np.sin(polar) * np.sin(azim),
                np.cos(polar)
            ])

            # Stable perpendicular vector for Gram-Schmidt
            if np.abs(u_hat[0]) < 0.9:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 1.0, 0.0])

            e1 = perp - np.dot(perp, u_hat) * u_hat
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(u_hat, e1)

            return u_hat, e1, e2

        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """t: (M, 2), theta: [x_c, y_c, z_c, r, h, polar, azim]"""
            t = np.atleast_2d(t)
            u, v = t[:, 0], t[:, 1]
            x_c, y_c, z_c, r, h = theta[0], theta[1], theta[2], theta[3], theta[4]

            u_hat, e1, e2 = _build_frame(theta)
            p = np.array([x_c, y_c, z_c])

            pts = (
                p[None, :]
                + (h * u)[:, None] * u_hat[None, :]
                + (r * np.cos(2 * np.pi * v))[:, None] * e1[None, :]
                + (r * np.sin(2 * np.pi * v))[:, None] * e2[None, :]
            )
            return pts

        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """Returns (M, 3, 2) - derivatives w.r.t (u, v)"""
            t = np.atleast_2d(t)
            u, v = t[:, 0], t[:, 1]
            r, h = theta[3], theta[4]

            u_hat, e1, e2 = _build_frame(theta)

            # ∂γ/∂u = h · û
            dg_du = np.tile(h * u_hat, (len(u), 1))  # (M, 3)

            # ∂γ/∂v = -2πr·sin(2πv)·ê₁ + 2πr·cos(2πv)·ê₂
            dg_dv = (
                (-r * 2 * np.pi * np.sin(2 * np.pi * v))[:, None] * e1[None, :]
                + (r * 2 * np.pi * np.cos(2 * np.pi * v))[:, None] * e2[None, :]
            )  # (M, 3)

            J = np.stack([
                np.stack([dg_du[:, 0], dg_dv[:, 0]], axis=-1),
                np.stack([dg_du[:, 1], dg_dv[:, 1]], axis=-1),
                np.stack([dg_du[:, 2], dg_dv[:, 2]], axis=-1),
            ], axis=1)

            return J

        def initial_fit(measurement: np.ndarray):
            pts = measurement

            def unpack(theta):
                x0, y0, z0, r, h, polar, azim = theta
                ux = np.sin(polar) * np.cos(azim)
                uy = np.sin(polar) * np.sin(azim)
                uz = np.cos(polar)
                axis = np.array([ux, uy, uz])
                bottom = np.array([x0, y0, z0])
                return bottom, axis / np.linalg.norm(axis), r, h

            def objective(theta):
                bottom, axis, r, h = unpack(theta)
                X = pts - bottom
                proj = X @ axis
                proj_clamped = np.clip(proj, 0.0, h)
                closest = bottom + np.outer(proj_clamped, axis)
                d = np.linalg.norm(pts - closest, axis=1)
                return np.sum((d - r) ** 2)

            # init via PCA
            centroid = pts.mean(axis=0)
            X = pts - centroid

            cov = X.T @ X
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, np.argmax(eigvals)]

            proj = X @ axis
            h0 = proj.max() - proj.min()
            bottom0 = centroid + axis * proj.min()

            perp = X - np.outer(proj, axis)
            r0 = np.median(np.linalg.norm(perp, axis=1))

            ux, uy, uz = axis / np.linalg.norm(axis)
            polar0 = np.arccos(uz)
            azim0 = np.arctan2(uy, ux)

            theta0 = np.array([*bottom0, r0, h0, polar0, azim0])

            bounds = [
                (None, None), (None, None), (None, None),
                (1e-3, None),
                (1e-3, None),
                (0, np.pi),
                (0, 2 * np.pi)
            ]

            res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)

            return res.x

        def initial_fit_naive(measurement: np.ndarray):
            xyz = measurement
            centroid = xyz.mean(axis=0)
            X = xyz - centroid
            cov = X.T @ X
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis_dir = eigvecs[:, np.argmax(eigvals)]
            axis_dir /= np.linalg.norm(axis_dir)

            # project onto axis
            proj = X @ axis_dir
            h = proj.max() - proj.min()

            # bottom point
            bottom = centroid + axis_dir * proj.min()

            # radius
            perp = X - np.outer(proj, axis_dir)
            r = np.median(np.linalg.norm(perp, axis=1))

            # spherical angles
            ux, uy, uz = axis_dir
            polar = np.arccos(uz)
            azim = np.arctan2(uy, ux) % (2 * np.pi)

            return np.array([bottom[0], bottom[1], bottom[2], r, h, polar, azim])

        super().__init__(
            parameter_dim=2,
            ambient_dim=3,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=[0.0, 0.0],
            domain_high=[1.0, 1.0],
            periodic=[False, True],
            bounds=[
                (-np.inf, np.inf),  # x_c
                (-np.inf, np.inf),  # y_c
                (-np.inf, np.inf),  # z_c
                (0.01, np.inf),        # r
                (0.01, np.inf),        # h
                (0, np.pi),         # polar
                (0, 2 * np.pi),     # azim
            ],
            initial_fit=initial_fit
        )

    def get_n_params(self) -> int:
        return 7  # x_c, y_c, z_c, r, h, polar, azim

    @override
    def get_closest_point(
            self,
            params: np.ndarray,
            x: np.ndarray,
            return_params: bool = False,
            *args, **kwargs
    ):
        """
        Closed-form closest point on a finite cylinder.

        Strategy: decompose x into axial and radial components relative to the
        cylinder axis, clamp the axial part to [0, h], project radially onto
        the circle of radius r.

        Geometry
        --------
        For query point x:
            1. d = x - p                         (vector from base center)
            2. u_proj = dot(d, û)                (signed axial distance)
            3. u* = clamp(u_proj, 0, h) / h      (parameter u ∈ [0,1])
            4. radial = d - u_proj * û           (component perp. to axis)
            5. project radial onto circle → v*, closest radial point
            6. closest = p + u* h û + r * radial_unit

        Edge cases
        ----------
        - x on the axis: radial == 0 → v* = 0 (arbitrary), point on circle at v=0
        - x beyond caps: u* is clamped, closest is on the end-cap rim

        Parameters
        ----------
        params : np.ndarray, shape (7,)
            [x_c, y_c, z_c, r, h, polar, azim]
        x : np.ndarray, shape (3,) or (M, 3)
        return_params : bool
            If True, also return (u*, v*) with shape (1,) or (M, 2).

        Returns
        -------
        closest : np.ndarray, shape (3,) or (M, 3)
        t_star  : np.ndarray, shape (2,) or (M, 2)  [only if return_params=True]
        """
        x_c, y_c, z_c, r, h, polar, azim = params
        p = np.array([x_c, y_c, z_c])

        # --- Axis unit vector û ---
        u_hat = np.array([
            np.sin(polar) * np.cos(azim),
            np.sin(polar) * np.sin(azim),
            np.cos(polar),
        ])  # (3,)

        # --- Cross-section basis ê₁, ê₂ (same as Shape's construction) ---
        # ê₁: perpendicular to û, no roll — use standard reference vector
        ref = np.array([0.0, 0.0, 1.0]) if abs(u_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(ref, u_hat)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u_hat, e1)  # already unit since û and ê₁ are orthonormal

        x = np.asarray(x, dtype=float)
        scalar_input = x.ndim == 1
        x = np.atleast_2d(x)  # (M, 3)

        # --- Decompose ---
        d = x - p  # (M, 3)
        u_proj = d @ u_hat  # (M,)  axial signed distance
        radial = d - u_proj[:, np.newaxis] * u_hat  # (M, 3) perp. component

        # --- Axial parameter: clamp to [0, 1] ---
        u_star = np.clip(u_proj / h, 0.0, 1.0)  # (M,)

        # --- Radial projection ---
        radial_norm = np.linalg.norm(radial, axis=-1, keepdims=True)  # (M, 1)
        safe_norm = np.where(radial_norm == 0, 1.0, radial_norm)
        radial_unit = radial / safe_norm  # (M, 3)

        # Degenerate: query on the axis → pick ê₁ direction, v* = 0
        on_axis = (radial_norm[:, 0] == 0)
        radial_unit[on_axis] = e1

        # --- Closest point ---
        closest = (
                p
                + (u_star * h)[:, np.newaxis] * u_hat  # axial part
                + r * radial_unit  # radial part
        )  # (M, 3)

        if scalar_input:
            closest = closest[0]

        if not return_params:
            return closest

        # --- v* from angle of radial vector in (ê₁, ê₂) frame ---
        cos_v = radial_unit @ e1  # (M,)
        sin_v = radial_unit @ e2  # (M,)
        v_star = np.arctan2(sin_v, cos_v) / (2 * np.pi) % 1.0  # (M,) ∈ [0,1)
        v_star[on_axis] = 0.0

        # Stack into (M, 2): [u*, v*]
        t_star = np.stack([u_star, v_star], axis=-1)  # (M, 2)

        if scalar_input:
            t_star = t_star[0]  # (2,)

        return closest, t_star