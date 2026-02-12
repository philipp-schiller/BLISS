from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence, Tuple
from .base_shape import Shape


@dataclass
class Ellipse(Shape):
    """
    Ellipse in 2D with center, semi-axes, and rotation.

    Parametrization:
        γ(t; x_c, y_c, a, b, θ) = (x_c, y_c) + R(θ) @ (a*cos(2πt), b*sin(2πt))

    where R(θ) is rotation matrix:
        R(θ) = [[cos(θ), -sin(θ)],
                [sin(θ),  cos(θ)]]

    Parameter space: t ∈ [0, 1), periodic

    Parameters
    ----------
    θ = [x_c, y_c, a, b, rotation]
    - x_c, y_c: center coordinates
    - a: semi-major axis (or first semi-axis)
    - b: semi-minor axis (or second semi-axis)
    - rotation: rotation angle in radians

    Notes
    -----
    The parametrization traces counterclockwise when rotation = 0.
    """

    def __init__(self):
        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            Ellipse parametrization.

            Parameters
            ----------
            t : np.ndarray, shape (M,) or (M, 1)
                Parameter values in [0, 1).
            theta : np.ndarray, shape (5,)
                [x_c, y_c, a, b, rotation]

            Returns
            -------
            points : np.ndarray, shape (M, 2)
                Points on ellipse.
            """
            t = np.atleast_1d(t).flatten()
            x_c, y_c, a, b, angle = theta

            # Parametric ellipse (before rotation)
            x_local = a * np.cos(2 * np.pi * t)
            y_local = b * np.sin(2 * np.pi * t)

            # Rotation matrix
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Rotate and translate
            x = x_c + cos_a * x_local - sin_a * y_local
            y = y_c + sin_a * x_local + cos_a * y_local

            return np.column_stack([x, y])

        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            Jacobian ∂γ/∂t.

            Returns
            -------
            jac : np.ndarray, shape (M, 2, 1)
                Tangent vectors at each point.
            """
            t = np.atleast_1d(t).flatten()
            x_c, y_c, a, b, angle = theta

            # Derivative of parametric ellipse (before rotation)
            # d/dt [a*cos(2πt), b*sin(2πt)] = [-2πa*sin(2πt), 2πb*cos(2πt)]
            dx_dt_local = -2 * np.pi * a * np.sin(2 * np.pi * t)
            dy_dt_local = 2 * np.pi * b * np.cos(2 * np.pi * t)

            # Rotation matrix applied to derivative
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            dx_dt = cos_a * dx_dt_local - sin_a * dy_dt_local
            dy_dt = sin_a * dx_dt_local + cos_a * dy_dt_local

            # Shape (M, 2, 1) for consistency
            return np.stack([dx_dt, dy_dt], axis=1)[:, :, np.newaxis]

        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=True,
            bounds=[
                (-np.inf, np.inf),  # x_c
                (-np.inf, np.inf),  # y_c
                (0, np.inf),  # a (must be positive)
                (0, np.inf),  # b (must be positive)
                (-np.pi, np.pi),  # rotation angle
            ]
        )

    def get_n_params(self) -> int:
        return 5