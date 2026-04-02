from dataclasses import dataclass
from typing import override

from scipy.optimize import least_squares
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

        def initial_fit(measurement: np.ndarray, method = "geometric"):
            """
            Fit a circle to 2D points using geometric (orthogonal) distances.
            Minimizes sum of (dist_to_center - radius)^2.

            Parameters
            ----------
            measurement
            points : array-like, shape (N, 2)
                Input 2D points.

            Returns
            -------
            xc, yc : float
                Center of the fitted circle.
            r : float
                Radius of the fitted circle.
            result : OptimizeResult
                Full result from scipy least_squares.
            """
            measurement = np.asarray(measurement, dtype=float)
            x, y = measurement[:, 0], measurement[:, 1]

            if method == "geometric":
                def residuals(params):
                    xc, yc, r = params
                    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
                    return distances - r

                # Initial guess: centroid as center, mean distance as radius
                xc0, yc0 = np.mean(x), np.mean(y)
                r0 = np.mean(np.sqrt((x - xc0) ** 2 + (y - yc0) ** 2))

                result = least_squares(residuals, x0=[xc0, yc0, r0])
                xc, yc, r = result.x

            if method == "algebraic":
                # Build the linear system: [2x, 2y, 1] @ [xc, yc, (r²-xc²-yc²)] = x²+y²
                A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
                b = x ** 2 + y ** 2

                result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                xc, yc, c = result
                r = np.sqrt(c + xc ** 2 + yc ** 2)

            return np.array([xc, yc, r])

        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=True,
            bounds=[(-10, 10), (-10, 10), (0.1, 10)], # x_c, y_c, r
            initial_fit = initial_fit
        )

    def get_n_params(self) -> int:
        return 3  # x_c, y_c, r

    @override
    def get_closest_point(self,
            params: np.ndarray,
            x: np.ndarray, *args, **kwargs):

        """
            Closed-form closest point on a circle.

            For a circle centered at (x_c, y_c) with radius r, the closest point
            to any query x is simply the projection onto the circle along the
            radial direction:

                t* = atan2(y - y_c, x - x_c) / (2π)   mod 1
                γ(t*) = (x_c, y_c) + r * (x - center) / ||x - center||

            Degenerate case: if x == center, t* = 0 is returned (arbitrary).

            Parameters
            ----------
            params : np.ndarray, shape (3,)
                Shape parameters [x_c, y_c, r].
            x : np.ndarray, shape (n,) or (M, n)
                Query point(s) in R^2.
            return_params : bool
                If True, also return the parameter values t* in [0, 1).

            Returns
            -------
            closest : np.ndarray, shape (2,) or (M, 2)
            t_star  : np.ndarray, shape (1,) or (M, 1)  [only if return_params=True]
            """
        x_c, y_c, r = params
        center = np.array([x_c, y_c])

        x = np.asarray(x, dtype=float)
        scalar_input = x.ndim == 1
        x = np.atleast_2d(x)  # (M, 2)

        # Vector from center to each query point
        delta = x - center  # (M, 2)

        # Radial distance — guard against x == center
        norms = np.linalg.norm(delta, axis=-1, keepdims=True)  # (M, 1)
        safe_norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        unit = delta / safe_norms  # (M, 2)

        # Closest point: project onto circle
        closest = center + r * unit  # (M, 2)

        # For degenerate case (query == center), return t*=0 → point (x_c + r, y_c)
        degenerate = (norms[:, 0] == 0)
        closest[degenerate] = center + np.array([r, 0.0])

        if scalar_input:
            closest = closest[0]

        # t* = atan2(dy, dx) / (2π), mapped to [0, 1)
        t_star = np.arctan2(delta[:, 1], delta[:, 0]) / (2 * np.pi) % 1.0  # (M,)
        t_star[degenerate] = 0.0
        t_star = t_star[:, np.newaxis]  # (M, 1) to match parameter_dim=1

        if scalar_input:
            t_star = t_star[0]  # (1,)

        return closest, t_star

