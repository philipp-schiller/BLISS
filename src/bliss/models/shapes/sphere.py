from dataclasses import dataclass

import numpy as np

from .base_shape import Shape


@dataclass
class Sphere(Shape):
    """
    Sphere in 3D: γ(u,v; x_c, y_c, z_c, r)

    Using spherical coordinates:
    x = x_c + r sin(πu) cos(2πv)
    y = y_c + r sin(πu) sin(2πv)
    z = z_c + r cos(πu)

    Parameter space: (u,v) ∈ [0,1] × [0,1), v periodic
    Ambient space: R^3
    Shape parameters θ = [x_c, y_c, z_c, r]
    """

    def __init__(self):
        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """t: (M, 2), theta: [x_c, y_c, z_c, r]"""
            t = np.atleast_2d(t)
            u, v = t[:, 0], t[:, 1]
            x_c, y_c, z_c, r = theta

            x = x_c + r * np.sin(np.pi * u) * np.cos(2 * np.pi * v)
            y = y_c + r * np.sin(np.pi * u) * np.sin(2 * np.pi * v)
            z = z_c + r * np.cos(np.pi * u)

            return np.stack([x, y, z], axis=-1)

        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """Returns (M, 3, 2) - derivatives w.r.t (u, v)"""
            t = np.atleast_2d(t)
            u, v = t[:, 0], t[:, 1]
            x_c, y_c, z_c, r = theta

            # ∂/∂u
            dx_du = r * np.pi * np.cos(np.pi * u) * np.cos(2 * np.pi * v)
            dy_du = r * np.pi * np.cos(np.pi * u) * np.sin(2 * np.pi * v)
            dz_du = -r * np.pi * np.sin(np.pi * u)

            # ∂/∂v
            dx_dv = -r * 2 * np.pi * np.sin(np.pi * u) * np.sin(2 * np.pi * v)
            dy_dv = r * 2 * np.pi * np.sin(np.pi * u) * np.cos(2 * np.pi * v)
            dz_dv = np.zeros_like(u)

            # Stack as (M, 3, 2)
            J = np.stack([
                np.stack([dx_du, dx_dv], axis=-1),
                np.stack([dy_du, dy_dv], axis=-1),
                np.stack([dz_du, dz_dv], axis=-1)
            ], axis=1)

            return J

        super().__init__(
            parameter_dim=2,
            ambient_dim=3,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=[0.0, 0.0],
            domain_high=[1.0, 1.0],
            periodic=[False, True],  # u not periodic, v periodic
            bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        )

    def get_n_params(self) -> int:
        return 4  # x_c, y_c, z_c, r