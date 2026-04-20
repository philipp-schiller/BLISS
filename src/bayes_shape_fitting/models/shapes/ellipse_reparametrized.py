from dataclasses import dataclass

import numpy as np

from .base_shape import Shape


def get_ellipse_params(homogenous_coords: np.ndarray) -> np.ndarray:
    A, B, C, D, E, F = homogenous_coords

    if B**2 - 4*A*C >= 0:
        raise ValueError("Not an ellipse: discriminant >= 0")

    # Center
    denom = B**2 - 4*A*C
    x_c = (2*C*D - B*E) / denom
    y_c = (2*A*E - B*D) / denom

    # Value at center
    F_bar = (
        A*x_c**2 + B*x_c*y_c + C*y_c**2 +
        D*x_c + E*y_c + F
    )

    # Ensure interior is negative
    if F_bar > 0:
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F
        F_bar = (
            A*x_c**2 + B*x_c*y_c + C*y_c**2 +
            D*x_c + E*y_c + F
        )

    # Rotation
    theta = 0.5 * np.arctan2(B, A - C)

    # Normalize angle
    theta = ((theta + np.pi/4) % (np.pi/2)) - np.pi/4

    # Rotated coefficients
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A_rot = A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2
    C_rot = A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2

    if A_rot <= 0 or C_rot <= 0 or abs(F_bar) < 1e-12:
        raise ValueError("Degenerate ellipse.")

    # Raw axes
    a_raw = np.sqrt(-F_bar / A_rot)
    b_raw = np.sqrt(-F_bar / C_rot)

    # ✅ Fix orientation & enforce a >= b
    if b_raw > a_raw:
        a_raw, b_raw = b_raw, a_raw
        theta += np.pi/2

    # Normalize θ back to [-π/4, π/4]
    theta = ((theta + np.pi/4) % (np.pi/2)) - np.pi/4

    # Axis ratio r = b/a ∈ [0,1]
    r = b_raw / a_raw

    return np.array([x_c, y_c, a_raw, r, theta])

def get_ellipse_params_(homogenous_coords: np.ndarray) -> np.ndarray:
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
class Ellipse_R(Shape):
    """
    Ellipse in 2D with center, semi-major axis, axis-ratio r = b/a ∈ [0,1],
    and rotation θ. Semi-minor axis is b = a*r.

    Parametrization:
        γ(t; x_c, y_c, a, r, θ) =
            (x_c, y_c) + R(θ) @ (a*cos(2πt), a*r*sin(2πt))
    """

    def __init__(self):
        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            t = np.atleast_1d(t).flatten()
            x_c, y_c, a, r, angle = theta
            b = a * r

            x_local = a * np.cos(2*np.pi*t)
            y_local = b * np.sin(2*np.pi*t)

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x = x_c + cos_a * x_local - sin_a * y_local
            y = y_c + sin_a * x_local + cos_a * y_local
            return np.column_stack([x, y])

        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            t = np.atleast_1d(t).flatten()
            x_c, y_c, a, r, angle = theta
            b = a * r

            dx_dt_local = -2*np.pi*a * np.sin(2*np.pi*t)
            dy_dt_local =  2*np.pi*b * np.cos(2*np.pi*t)

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx_dt = cos_a*dx_dt_local - sin_a*dy_dt_local
            dy_dt = sin_a*dx_dt_local + cos_a*dy_dt_local

            return np.stack([dx_dt, dy_dt], axis=1)[:, :, None]

        def initial_fit(measurement: np.ndarray):
            """
            Compute (x_c, y_c, a, r, θ) from the original ellipse-fit.
            We reuse your existing fit, then convert (a, b) → (a, r=b/a).
            """
            xc, yc, a, b, theta = get_ellipse_params_from_data(measurement)
            r = b / a
            return np.array([xc, yc, a, r, theta])

        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=True,
            bounds=[
                (-5, 5),   # x_c
                (-5, 5),   # y_c
                (0.5, 5),   # a (scale)
                (0.2, 1.0),     # r ∈ [0,1]
                (-np.pi, np.pi) # rotation
            ],
            initial_fit=initial_fit
        )



    def get_n_params(self) -> int:
        return 5

