"""
Implementing a 2D circle
"""

from base_shape import Shape, GammaFn, JacobianFn
import numpy as np

def circle_gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    t = np.atleast_2d(t)
    x_c, y_c, r = theta
    angle = 2 * np.pi * t[:, 0]
    x = x_c + r * np.cos(angle)
    y = y_c + r * np.sin(angle)
    return np.stack([x, y], axis=-1)


def circle_jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    t = np.atleast_2d(t)
    x_c, y_c, r = theta
    angle = 2 * np.pi * t[:, 0]
    dx_dt = -r * 2 * np.pi * np.sin(angle)
    dy_dt = r * 2 * np.pi * np.cos(angle)
    return np.stack([dx_dt, dy_dt], axis=-1)[:, :, None]


class Circle(Shape):
    def __init__(self):
        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=circle_gamma,
            jacobian_fn=circle_jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=True,
            bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        )

    def get_n_params(self) -> int:
        return 3


# Test it
circle = Circle()
t = np.array([0.0, 0.25, 0.5, 0.75])
theta = np.array([0.0, 0.0, 1.0])
points = circle.evaluate(t, theta)
print(points)