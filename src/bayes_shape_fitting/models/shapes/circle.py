from dataclasses import dataclass
import numpy as np
from .base_shape import Shape


@dataclass
class Circle(Shape):
    """
    Circle in 2D: γ(t; x_c, y_c, r) = (x_c + r cos(2πt), y_c + r sin(2πt))

    Parameter space: t ∈ [0, 1), periodic
    Ambient space: R^2
    Shape parameters θ = [x_c, y_c, r]
    """

    def __init__(self):
        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """t: (M, 1) or (M,), theta: [x_c, y_c, r]"""
            t = np.atleast_1d(t).flatten()
            x_c, y_c, r = theta
            x = x_c + r * np.cos(2 * np.pi * t)
            y = y_c + r * np.sin(2 * np.pi * t)
            return np.stack([x, y], axis=-1)

        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """Returns (M, 2, 1)"""
            t = np.atleast_1d(t).flatten()
            x_c, y_c, r = theta
            dx_dt = -r * 2 * np.pi * np.sin(2 * np.pi * t)
            dy_dt = r * 2 * np.pi * np.cos(2 * np.pi * t)
            return np.stack([dx_dt, dy_dt], axis=-1)[:, :, None]

        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=True,
            bounds=[(-10, 10), (-10, 10), (0.1, 10)]  # x_c, y_c, r
        )

    def get_n_params(self) -> int:
        return 3  # x_c, y_c, r