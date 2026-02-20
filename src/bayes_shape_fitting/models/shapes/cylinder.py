from dataclasses import dataclass
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
                (0, np.inf),        # r
                (0, np.inf),        # h
                (0, np.pi),         # polar
                (0, 2 * np.pi),     # azim
            ]
        )

    def get_n_params(self) -> int:
        return 7  # x_c, y_c, z_c, r, h, polar, azim