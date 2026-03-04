from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence, Tuple, override
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

    @override
    def get_closest_point(
            self,
            params: np.ndarray,
            x: np.ndarray,
            *args, **kwargs
    ):
        """
        Closest point on an ellipse via Newton's method in the ellipse frame.

        Strategy
        --------
        Transform x into the canonical ellipse frame (unrotate, uncenter), solve
        the 1D optimality condition there, then transform back.

        In the canonical frame the closest point satisfies:
            F(t) = -diff(t) · γ'(t) = 0
        where diff(t) = γ(t) - x̃  and γ'(t) is the tangent.

        This is a smooth periodic scalar equation → Newton's method converges
        in ~5 iterations from a good initialisation. We seed from a coarse grid
        to handle multi-modal cases (e.g. points near the center).

        Parameters
        ----------
        params : np.ndarray, shape (5,)
            [x_c, y_c, a, b, rotation]
        x : np.ndarray, shape (2,) or (M, 2)
        return_params : bool

        Returns
        -------
        closest : np.ndarray, shape (2,) or (M, 2)
        t_star  : np.ndarray, shape (1,) or (M, 1)  [only if return_params=True]
        """
        x_c, y_c, a, b, rot = params
        center = np.array([x_c, y_c])

        x = np.asarray(x, dtype=float)
        scalar_input = x.ndim == 1
        x = np.atleast_2d(x)  # (M, 2)
        M = x.shape[0]

        # --- Transform to canonical (unrotated, uncentered) frame ---
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        R = np.array([[cos_r, -sin_r],
                      [sin_r, cos_r]])  # (2, 2)
        Rt = R.T  # inverse = transpose for rotation

        x_local = (x - center) @ R  # (M, 2)  [equiv: Rᵀ(x - center)]

        # --- Helpers in canonical frame ---
        two_pi = 2 * np.pi

        def gamma_local(t):
            # t: (M,) → (M, 2)
            return np.stack([a * np.cos(two_pi * t),
                             b * np.sin(two_pi * t)], axis=-1)

        def dgamma_local(t):
            # First derivative γ'(t): (M, 2)
            return np.stack([-a * two_pi * np.sin(two_pi * t),
                             b * two_pi * np.cos(two_pi * t)], axis=-1)

        def d2gamma_local(t):
            # Second derivative γ''(t): (M, 2)
            return np.stack([-a * two_pi ** 2 * np.cos(two_pi * t),
                             -b * two_pi ** 2 * np.sin(two_pi * t)], axis=-1)

        def F(t, xi):
            # Optimality condition: (γ(t) - x) · γ'(t) = 0  →  shape (M,)
            diff = gamma_local(t) - xi  # (M, 2)
            return np.einsum("mi,mi->m", diff, dgamma_local(t))

        def dF(t, xi):
            # F'(t) = ||γ'(t)||² + (γ(t) - x) · γ''(t)
            diff = gamma_local(t) - xi
            g1 = dgamma_local(t)
            g2 = d2gamma_local(t)
            return (np.einsum("mi,mi->m", g1, g1) +
                    np.einsum("mi,mi->m", diff, g2))

        # --- Seed: coarse grid, pick best per point ---
        N_grid = 16
        t_grid = np.linspace(0, 1, N_grid, endpoint=False)  # (G,)
        # Evaluate all grid points for all queries: (G, M, 2)
        gamma_grid = np.stack([gamma_local(np.full(M, tg)) for tg in t_grid])
        # Squared distances: (G, M)
        dist_grid = np.sum((gamma_grid - x_local[np.newaxis]) ** 2, axis=-1)
        best_idx = np.argmin(dist_grid, axis=0)  # (M,)
        t = t_grid[best_idx].copy()  # (M,)

        # --- Newton iterations ---
        for _ in range(12):
            f = F(t, x_local)
            df = dF(t, x_local)
            # Guard against zero derivative (exactly at inflection)
            safe_df = np.where(np.abs(df) < 1e-14, 1.0, df)
            step = f / safe_df
            t = (t - step) % 1.0  # keep periodic

            if np.max(np.abs(step)) < 1e-10:
                break

        # --- Second candidate: for near-center points both ±radial dirs matter ---
        # Run a second pass seeded from t + 0.5 (opposite side), keep the better one
        t2 = (t + 0.5) % 1.0
        for _ in range(12):
            f = F(t2, x_local)
            df = dF(t2, x_local)
            safe_df = np.where(np.abs(df) < 1e-14, 1.0, df)
            step = f / safe_df
            t2 = (t2 - step) % 1.0
            if np.max(np.abs(step)) < 1e-10:
                break

        # Pick whichever candidate is actually closer
        d1 = np.sum((gamma_local(t) - x_local) ** 2, axis=-1)  # (M,)
        d2 = np.sum((gamma_local(t2) - x_local) ** 2, axis=-1)  # (M,)
        t_star = np.where(d1 <= d2, t, t2)  # (M,)

        # --- Map back to world frame ---
        pts_local = gamma_local(t_star)  # (M, 2)
        closest = pts_local @ Rt + center  # (M, 2)  R @ pt + center

        if scalar_input:
            closest = closest[0]

        t_out = t_star[:, np.newaxis]  # (M, 1)
        if scalar_input:
            t_out = t_out[0]  # (1,)

        return closest, t_out