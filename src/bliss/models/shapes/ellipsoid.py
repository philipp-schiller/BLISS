from dataclasses import dataclass
import numpy as np

from .base_shape import Shape


@dataclass
class Ellipsoid_R(Shape):
    """
    Ellipsoid in 3D with center (x_c, y_c, z_c),
    semi-major axis a, axis ratios r1=b/a, r2=c/a,
    and rotation represented by Euler angles (α, β, γ).

    Parametrization:
        u ∈ [0, 1)  → longitude (2πu)
        v ∈ [0, 1)  → latitude   (πv)

        γ(u,v) =
            c + R(α,β,γ) @ (
                a*cos(2πu)*sin(πv),
                a*r1*sin(2πu)*sin(πv),
                a*r2*cos(πv)
            )
    """

    def __init__(self):

        # ────────────────────────────────────────────────────────────
        # Parametrization
        # ────────────────────────────────────────────────────────────
        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            t: (N, 2) in [0,1]²
            theta: (9,) = [x_c, y_c, z_c, a, r1, r2, α, β, γ]
            """
            t = np.atleast_2d(t)
            u = 2 * np.pi * t[:, 0]
            v = np.pi * t[:, 1]

            x_c, y_c, z_c, a, r1, r2, α, β, γ = theta

            # local axes
            X = a * np.cos(u) * np.sin(v)
            Y = a * r1 * np.sin(u) * np.sin(v)
            Z = a * r2 * np.cos(v)

            pts = np.stack([X, Y, Z], axis=1)

            # rotation matrix (Z-Y-X Euler)
            ca, sa = np.cos(α), np.sin(α)
            cb, sb = np.cos(β), np.sin(β)
            cg, sg = np.cos(γ), np.sin(γ)

            Rz = np.array([[ca, -sa, 0],
                           [sa,  ca, 0],
                           [0,    0, 1]])
            Ry = np.array([[ cb, 0, sb],
                           [  0, 1,  0],
                           [-sb, 0, cb]])
            Rx = np.array([[1,  0,   0],
                           [0, cg, -sg],
                           [0, sg,  cg]])

            R = Rz @ Ry @ Rx

            pts = pts @ R.T
            pts += np.array([x_c, y_c, z_c])
            return pts


        # ────────────────────────────────────────────────────────────
        # Jacobian w.r.t. latent parameters (u, v)
        # ────────────────────────────────────────────────────────────
        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            Returns J = dγ/d(u,v)
            Shape: (N, 3, 2)
            """
            t = np.atleast_2d(t)
            u = 2*np.pi * t[:, 0]
            v = np.pi   * t[:, 1]

            a, r1, r2 = theta[3], theta[4], theta[5]
            α, β, γ   = theta[6], theta[7], theta[8]

            dX_du = -2*np.pi * a * np.sin(u) * np.sin(v)
            dY_du =  2*np.pi * a * r1 * np.cos(u) * np.sin(v)
            dZ_du =  np.zeros_like(u)

            dX_dv =  np.pi * a * np.cos(u) * np.cos(v)
            dY_dv =  np.pi * a * r1 * np.sin(u) * np.cos(v)
            dZ_dv = -np.pi * a * r2 * np.sin(v)

            J_local = np.stack([
                np.stack([dX_du, dY_du, dZ_du], axis=1),
                np.stack([dX_dv, dY_dv, dZ_dv], axis=1)
            ], axis=2)  # (N, 3, 2)

            # rotation matrix (same as above)
            ca, sa = np.cos(α), np.sin(α)
            cb, sb = np.cos(β), np.sin(β)
            cg, sg = np.cos(γ), np.sin(γ)

            Rz = np.array([[ca, -sa, 0],
                           [sa,  ca, 0],
                           [0,    0, 1]])
            Ry = np.array([[ cb, 0, sb],
                           [  0, 1,  0],
                           [-sb, 0, cb]])
            Rx = np.array([[1,  0,   0],
                           [0, cg, -sg],
                           [0, sg,  cg]])

            R = Rz @ Ry @ Rx

            return np.einsum('ij,njk->nik', R, J_local)

        """# ────────────────────────────────────────────────────────────
        # Initial fit (placeholder)
        # ────────────────────────────────────────────────────────────
        def initial_fit(measurement: np.ndarray) -> np.ndarray:
           
            center = measurement.mean(axis=0)
            U, S, _ = np.linalg.svd(measurement - center, full_matrices=False)

            a = S[0]
            r1 = S[1] / S[0]
            r2 = S[2] / S[0]

            return np.array([
                center[0], center[1], center[2],
                a, r1, r2,
                0.0, 0.0, 0.0   # no initial rotation
            ])"""


        # ────────────────────────────────────────────────────────────
        # Initial fit (fast PCA + robust extent + 1-step LS refinement)
        # ────────────────────────────────────────────────────────────
        def initial_fit(measurement: np.ndarray) -> np.ndarray:
            """
                Fast ellipsoid initializer.

                Parameters
                ----------
                measurement : (N,3) array
                    Point cloud sampled from the ellipsoid surface (possibly noisy).

                Returns
                -------
                theta0 : (9,) array
                    [x_c, y_c, z_c, a, r1, r2, alpha, beta, gamma]

                Notes
                -----
                Strategy:
                  1) center by mean
                  2) estimate orientation from covariance eigenvectors
                  3) in the PCA frame, estimate center/axes from robust bounds
                  4) refine semi-axes via one linear least-squares solve
                  5) convert rotation matrix to Z-Y-X Euler angles

                Assumes the observed points cover a reasonably large portion
                of the ellipsoid. This is much better than plain SVD singular values,
                but still extremely fast.
                """
            X = np.asarray(measurement, dtype=float)
            if X.ndim != 2 or X.shape[1] != 3:
                raise ValueError("measurement must have shape (N, 3)")
            if X.shape[0] < 9:
                raise ValueError("Need at least 9 points for a stable initialization")

            # --------------------------------------------------------
            # Helper: rotation matrix -> Z-Y-X Euler angles
            # R = Rz(alpha) @ Ry(beta) @ Rx(gamma)
            # --------------------------------------------------------
            def rotmat_to_euler_zyx(R: np.ndarray) -> np.ndarray:
                """
                    Returns angles [alpha, beta, gamma] such that
                    R = Rz(alpha) @ Ry(beta) @ Rx(gamma)
                    with beta in [-pi/2, pi/2].
                    """
                # Clamp for numerical safety
                r20 = np.clip(R[2, 0], -1.0, 1.0)

                # Generic case
                if abs(r20) < 1.0 - 1e-8:
                    beta = -np.arcsin(r20)
                    cb = np.cos(beta)
                    alpha = np.arctan2(R[1, 0] / cb, R[0, 0] / cb)
                    gamma = np.arctan2(R[2, 1] / cb, R[2, 2] / cb)
                else:
                    # Gimbal lock: beta ≈ ±pi/2
                    beta = np.pi / 2 if r20 <= -1.0 + 1e-8 else -np.pi / 2
                    alpha = np.arctan2(-R[0, 1], R[1, 1])
                    gamma = 0.0

                return np.array([alpha, beta, gamma], dtype=float)

            # --------------------------------------------------------
            # 1) Rough center
            # --------------------------------------------------------
            center0 = X.mean(axis=0)

            # --------------------------------------------------------
            # 2) PCA orientation via covariance eigenvectors
            #    eigh is cheaper / more appropriate than SVD here
            # --------------------------------------------------------
            Y = X - center0
            C = (Y.T @ Y) / max(len(X) - 1, 1)

            evals, evecs = np.linalg.eigh(C)  # ascending order
            order = np.argsort(evals)[::-1]  # descending
            evecs = evecs[:, order]

            # Ensure a proper right-handed rotation
            if np.linalg.det(evecs) < 0:
                evecs[:, 2] *= -1.0

            R = evecs  # columns are principal directions

            # --------------------------------------------------------
            # 3) Robust center/extent estimate in PCA frame
            # --------------------------------------------------------
            Q0 = (X - center0) @ R  # coordinates in PCA frame

            # Robust low/high quantiles instead of pure min/max
            q_lo = np.quantile(Q0, 0.02, axis=0)
            q_hi = np.quantile(Q0, 0.98, axis=0)

            center_local = 0.5 * (q_hi + q_lo)
            axes_box = 0.5 * (q_hi - q_lo)

            # Avoid degeneracies
            axes_box = np.maximum(axes_box, 1e-6)

            # Update center in world coordinates
            center = center0 + R @ center_local

            # Recompute aligned coordinates around updated center
            Q = (X - center) @ R

            # --------------------------------------------------------
            # 4) One cheap LS refinement of semi-axes in aligned frame
            #
            # Ellipsoid in aligned frame:
            #   x^2/a^2 + y^2/b^2 + z^2/c^2 ≈ 1
            #
            # Solve for s = [1/a^2, 1/b^2, 1/c^2] in least squares:
            #   [x_i^2, y_i^2, z_i^2] s ≈ 1
            # --------------------------------------------------------
            A = Q ** 2
            rhs = np.ones(Q.shape[0])

            try:
                s, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                s = np.maximum(s, 1e-12)
                axes_ls = 1.0 / np.sqrt(s)

                # Guard against pathological values; blend with box estimate if needed
                if np.all(np.isfinite(axes_ls)) and np.all(axes_ls > 0):
                    axes = axes_ls
                else:
                    axes = axes_box
            except np.linalg.LinAlgError:
                axes = axes_box

            # Sort axes descending to match your parameterization:
            # a = major axis, b = a*r1, c = a*r2
            perm = np.argsort(axes)[::-1]
            axes = axes[perm]
            R = R[:, perm]

            # Re-enforce right-handedness after permutation
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1.0

            a, b, c = axes

            # --------------------------------------------------------
            # 5) Convert rotation matrix to Euler angles
            # --------------------------------------------------------
            alpha, beta, gamma = rotmat_to_euler_zyx(R)

            # Axis ratios
            r1 = np.clip(b / max(a, 1e-12), 1e-6, 1.0)
            r2 = np.clip(c / max(a, 1e-12), 1e-6, 1.0)

            return np.array([
                center[0], center[1], center[2],
                a, r1, r2,
                alpha, beta, gamma
            ], dtype=float)


        super().__init__(
            parameter_dim=2,
            ambient_dim=3,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=np.array([0.0, 0.0]),
            domain_high=np.array([1.0, 1.0]),
            periodic=[True, False],   # u periodic, v non-periodic
            bounds=[
                (-5, 5),   # x_c
                (-5, 5),   # y_c
                (-5, 5),   # z_c
                (0.1, 2),  # a
                (0.2, 1.0),# r1
                (0.2, 1.0),# r2
                (-np.pi/2, np.pi/2),  # α
                (-np.pi/2, np.pi/2),  # β
                (-np.pi/2, np.pi/2)   # γ
            ],
            initial_fit=initial_fit
        )

    def get_n_params(self) -> int:
        return 9