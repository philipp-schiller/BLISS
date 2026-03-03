from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence, Tuple
from .base_shape import Shape

def get_ellipse_params(homogenous_coords: np.ndarray) -> np.ndarray:
    """
    Convert general conic Ax² + Bxy + Cy² + Dx + Ey + F = 0
    to unique ellipse parameters (x_c, y_c, a, b, theta)
    with bijective conventions:
        a >= b > 0
        theta in [0, pi)
    """
    A, B, C, D, E, F = homogenous_coords

    # Check ellipse condition
    if B**2 - 4*A*C >= 0:
        raise ValueError("Not an ellipse: discriminant >= 0")

    # Compute center
    denom = B**2 - 4*A*C
    x_c = (2*C*D - B*E) / denom
    y_c = (2*A*E - B*D) / denom

    # Evaluate F at center
    F_bar = A*x_c**2 + B*x_c*y_c + C*y_c**2 + D*x_c + E*y_c + F

    # Normalize so that interior has negative sign
    if F_bar > 0:
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F
        F_bar = -F_bar

    # Orientation angle (full range)
    theta = 0.5 * np.arctan2(B, A - C)

    # Rotation matrix
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotated coefficients
    A_rot = A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2
    C_rot = A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2

    if A_rot <= 0 or C_rot <= 0:
        raise ValueError("Invalid ellipse: negative axis terms")

    # Semi-axes (raw)
    a_raw = np.sqrt(-F_bar / A_rot)
    b_raw = np.sqrt(-F_bar / C_rot)

    # Enforce a >= b by flipping orientation if necessary
    if b_raw > a_raw:
        a, b = b_raw, a_raw
        theta += np.pi/2
    else:
        a, b = a_raw, b_raw

    # Normalize theta into [0, pi)
    theta = theta % np.pi

    return np.array([x_c, y_c, a, b, theta])


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

        def initial_fit(measurement: np.ndarray):
            # computes an initial fit via a self-normalized fit

            x = measurement[:, 0]
            y = measurement[:, 1]

            # Design matrix
            D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T
            S = D.T @ D  # Shape: [N, 6]

            S12 = S[5, :5]
            S22 = S[5, 5]
            Sr = S[:5, :5] - np.outer(S12, S12) / S22

            C = np.array([
                [4 * S12[0], 2 * S12[1], 0, 2 * S12[3], 0],
                [2 * S12[1], S12[0] + S12[2], 2 * S12[1], S12[4], S12[3]],
                [0, 2 * S12[1], 4 * S12[2], 0, 2 * S12[4]],
                [2 * S12[3], S12[4], 0, S22, 0],
                [0, S12[3], 2 * S12[4], 0, S22]
            ])
            L = np.linalg.inv(np.linalg.cholesky(C))
            evals, evects = np.linalg.eigh(np.inner(np.dot(L, Sr), L))
            j = np.argmin(evals)
            G1 = np.dot(evects[:, j], L)
            G2 = -np.dot(G1, S12) / S22

            # to the debias:

            G2 += (G1[0] + G1[2]) * evals[j]
            sol = np.r_[G1, G2]

            return get_ellipse_params(sol)

        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=True,
            bounds=[
                (None, None),  # x_c
                (None, None),  # y_c
                (1e-2, None),  # a (must be positive)
                (1e-2, None),  # b (must be positive)
                (-np.pi, np.pi),  # rotation angle
            ],
            initial_fit = initial_fit
        )

    def get_n_params(self) -> int:
        return 5

