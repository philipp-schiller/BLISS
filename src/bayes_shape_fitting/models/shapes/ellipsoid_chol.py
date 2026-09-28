from dataclasses import dataclass
import numpy as np

from .base_shape import Shape


@dataclass
class Ellipsoid_chol(Shape):
    """
    Ellipsoid in 3D, parametrized by center c and the Cholesky factor L
    of its SPD shape matrix A = L @ L.T, where the ellipsoid is

        { x : (x - c)^T A (x - c) = 1 }.

    Why this parametrization:
    --------------------------------------------------------------
    The classic (center, a, r1, r2, alpha, beta, gamma) parametrization
    has several non-identifiabilities that hurt optimization:
      - a discrete D2 (Klein four-group) symmetry: 180-degree rotations
        about any principal axis give the same ellipsoid (4 equivalent
        rotation matrices per shape)
      - a label-swap symmetry whenever r1 ~ r2 (near-degenerate, not just
        exactly equal -> ill-conditioning, not just an exact singularity)
      - a full continuous O(2) symmetry (rotation about the symmetry
        axis) when r1 == r2 (prolate/oblate spheroid) -> the Jacobian
        w.r.t. rotation becomes rank-deficient, a flat valley rather
        than a point minimum
      - full SO(3) symmetry when r1 == r2 == 1 (sphere) -> rotation is
        completely unidentifiable
      - Euler angles additionally have gimbal lock at beta = +-pi/2,
        independent of the ellipsoid's own symmetry

    Cholesky decomposition of an SPD matrix with positive diagonal
    entries is *unique*. So (center, L) is a genuine bijection between
    parameters and ellipsoids: no discrete flips, no label ambiguity,
    no flat valleys at the spheroid/sphere limit, no gimbal lock. All
    of those degeneracies are pushed into a single closed-form
    eigendecomposition done only when you want interpretable output
    (see `canonical_parameters`), where they are harmless.

    Parametrization:
        u ∈ [0, 1)  → longitude (2πu)
        v ∈ [0, 1)  → latitude   (πv)

        s(u, v) = (cos(2πu) sin(πv), sin(2πu) sin(πv), cos(πv))   # unit sphere

        γ(u, v) = c + L^{-1 , T-applied-as} ... concretely:
                = c + s(u, v) @ M,     M := L^{-1}   (so that (x-c)^T A (x-c) = 1)

    Derivation: let z = L^T (x - c). Then (x-c)^T A (x-c) = (x-c)^T L L^T (x-c)
    = z^T z, so z lies on the unit sphere iff x is on the ellipsoid. Hence
    x - c = L^{-T} z, i.e. x = c + (L^{-T} z). Writing this as a row-vector
    convention (matching the rest of this codebase, x_row = s_row @ M),
    M = L^{-1} and x = c + s @ M.

    theta: (9,) = [x_c, y_c, z_c, l11, l21, l22, l31, l32, l33]
        L = [[l11,   0,   0],
             [l21, l22,   0],
             [l31, l32, l33]]
        l11, l22, l33 > 0 (enforced via bounds) for uniqueness.

    Note: the (u, v) surface parametrization itself still has the usual
    lat/long coordinate singularities at the poles (v = 0, pi). That is
    a property of this surface chart, not of the ellipsoid's shape
    parameters, and is unrelated to (and not fixed by) the symmetries
    discussed above.
    """

    def __init__(self):

        # ────────────────────────────────────────────────────────────
        # Helpers
        # ────────────────────────────────────────────────────────────
        def _L_from_theta(theta: np.ndarray) -> np.ndarray:
            _, _, _, l11, l21, l22, l31, l32, l33 = theta
            return np.array([
                [l11, 0.0, 0.0],
                [l21, l22, 0.0],
                [l31, l32, l33],
            ])

        # ────────────────────────────────────────────────────────────
        # Parametrization
        # ────────────────────────────────────────────────────────────
        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            t: (N, 2) in [0,1]^2
            theta: (9,) = [x_c, y_c, z_c, l11, l21, l22, l31, l32, l33]
            """
            t = np.atleast_2d(t)
            u = 2 * np.pi * t[:, 0]
            v = np.pi * t[:, 1]

            center = theta[:3]
            L = _L_from_theta(theta)
            M = np.linalg.inv(L)  # M = L^{-1}

            S = np.stack([
                np.cos(u) * np.sin(v),
                np.sin(u) * np.sin(v),
                np.cos(v),
            ], axis=1)  # (N, 3), points on the unit sphere

            pts = S @ M
            pts += center
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
            u = 2 * np.pi * t[:, 0]
            v = np.pi * t[:, 1]

            L = _L_from_theta(theta)
            M = np.linalg.inv(L)

            dS_du = 2 * np.pi * np.stack([
                -np.sin(u) * np.sin(v),
                np.cos(u) * np.sin(v),
                np.zeros_like(u),
            ], axis=1)

            dS_dv = np.pi * np.stack([
                np.cos(u) * np.cos(v),
                np.sin(u) * np.cos(v),
                -np.sin(v),
            ], axis=1)

            J_local = np.stack([dS_du, dS_dv], axis=2)  # (N, 3, 2)

            return J_local.transpose(0, 2, 1) @ M
            # -> transpose to (N,2,3), apply M, giving (N,2,3); transpose back
            # (kept as one line above; see note below if you prefer explicit steps)

        # NOTE on the jacobian line above: J_local is (N,3,2). We want, for
        # each of the 2 latent directions, the row-vector d(s)/dt @ M (a
        # (3,) vector), stacked back into (N,3,2). Doing this via a single
        # einsum is clearer than the transpose trick used in the one-liner
        # above; both are equivalent. Replacing jacobian's return with an
        # explicit einsum for clarity and to avoid transpose bugs:
        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:  # noqa: F811
            t = np.atleast_2d(t)
            u = 2 * np.pi * t[:, 0]
            v = np.pi * t[:, 1]

            L = _L_from_theta(theta)
            M = np.linalg.inv(L)

            dS_du = 2 * np.pi * np.stack([
                -np.sin(u) * np.sin(v),
                np.cos(u) * np.sin(v),
                np.zeros_like(u),
            ], axis=1)  # (N, 3)

            dS_dv = np.pi * np.stack([
                np.cos(u) * np.cos(v),
                np.sin(u) * np.cos(v),
                -np.sin(v),
            ], axis=1)  # (N, 3)

            dX_du = dS_du @ M  # (N, 3)
            dX_dv = dS_dv @ M  # (N, 3)

            return np.stack([dX_du, dX_dv], axis=2)  # (N, 3, 2)

        # ────────────────────────────────────────────────────────────
        # Initial fit (PCA orientation + robust extent + LS refinement,
        # converted into the Cholesky parametrization at the end)
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
                [x_c, y_c, z_c, l11, l21, l22, l31, l32, l33]

            Strategy: identical geometric approach to before (PCA orientation,
            robust quantile extent, one linear LS refinement of semi-axes),
            but the result is assembled into a shape matrix A and Cholesky-
            factorized rather than stored as (a, r1, r2, alpha, beta, gamma).
            This means the discrete axis-order / rotation-sign choices made
            below only affect *which* representative of the true orbit we
            start from -- not the optimum itself, since all representatives
            map to the same A.
            """
            X = np.asarray(measurement, dtype=float)
            if X.ndim != 2 or X.shape[1] != 3:
                raise ValueError("measurement must have shape (N, 3)")
            if X.shape[0] < 9:
                raise ValueError("Need at least 9 points for a stable initialization")

            # 1) Rough center
            center0 = X.mean(axis=0)

            # 2) PCA orientation via covariance eigenvectors
            Y = X - center0
            C = (Y.T @ Y) / max(len(X) - 1, 1)
            evals, evecs = np.linalg.eigh(C)  # ascending order
            order = np.argsort(evals)[::-1]  # descending
            evecs = evecs[:, order]
            if np.linalg.det(evecs) < 0:
                evecs[:, 2] *= -1.0
            R = evecs

            # 3) Robust center/extent estimate in PCA frame
            Q0 = (X - center0) @ R
            q_lo = np.quantile(Q0, 0.02, axis=0)
            q_hi = np.quantile(Q0, 0.98, axis=0)
            center_local = 0.5 * (q_hi + q_lo)
            axes_box = np.maximum(0.5 * (q_hi - q_lo), 1e-6)

            center = center0 + R @ center_local
            Q = (X - center) @ R

            # 4) One cheap LS refinement of semi-axes in aligned frame
            A_mat = Q ** 2
            rhs = np.ones(Q.shape[0])
            try:
                s, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                s = np.maximum(s, 1e-12)
                axes_ls = 1.0 / np.sqrt(s)
                axes = axes_ls if (np.all(np.isfinite(axes_ls)) and np.all(axes_ls > 0)) else axes_box
            except np.linalg.LinAlgError:
                axes = axes_box

            # Sort descending purely for a sane starting point (harmless:
            # any ordering/sign choice here lands on the same shape matrix A)
            perm = np.argsort(axes)[::-1]
            axes = axes[perm]
            R = R[:, perm]
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1.0

            # 5) Assemble shape matrix A = R diag(1/a_i^2) R^T and Cholesky-factorize
            inv_sq = 1.0 / np.maximum(axes, 1e-8) ** 2
            A_shape = (R * inv_sq) @ R.T
            # symmetrize defensively against floating-point drift
            A_shape = 0.5 * (A_shape + A_shape.T)

            try:
                L = np.linalg.cholesky(A_shape)
            except np.linalg.LinAlgError:
                # nudge onto the SPD cone if numerically borderline
                jitter = 1e-8 * np.eye(3)
                L = np.linalg.cholesky(A_shape + jitter)

            return np.array([
                center[0], center[1], center[2],
                L[0, 0], L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[2, 2],
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
                (-5, 5),      # x_c
                (-5, 5),      # y_c
                (-5, 5),      # z_c
                (0.05, 10),   # l11 (> 0 for Cholesky uniqueness)
                (-10, 10),    # l21
                (0.05, 10),   # l22 (> 0)
                (-10, 10),    # l31
                (-10, 10),    # l32
                (0.05, 10),   # l33 (> 0)
            ],
            initial_fit=initial_fit,
        )

    def get_n_params(self) -> int:
        return 9

    # ────────────────────────────────────────────────────────────
    # Reporting helper: convert (center, L) -> interpretable
    # (center, a, r1, r2, R, euler angles). This is the ONLY place
    # the classic symmetries (axis order, sign, gimbal lock) show up,
    # and here they're harmless since it's a closed-form, one-shot
    # canonicalization rather than something the optimizer has to
    # fight through.
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def canonical_parameters(theta: np.ndarray) -> dict:
        """
        theta: (9,) = [x_c, y_c, z_c, l11, l21, l22, l31, l32, l33]

        Returns a dict with:
            center : (3,)
            a, r1, r2 : floats, a >= a*r1 >= a*r2 > 0
            R : (3,3) rotation matrix, det(R) = +1
            euler_zyx : (3,) [alpha, beta, gamma] with R = Rz(a) Ry(b) Rx(g)
        """
        center = theta[:3]
        l11, l21, l22, l31, l32, l33 = theta[3:]
        L = np.array([
            [l11, 0.0, 0.0],
            [l21, l22, 0.0],
            [l31, l32, l33],
        ])
        A_shape = L @ L.T  # SPD shape matrix

        evals, evecs = np.linalg.eigh(A_shape)  # ascending eigenvalues
        # semi-axis lengths: A = R diag(1/axes^2) R^T  =>  axes_i = 1/sqrt(eval_i)
        axes = 1.0 / np.sqrt(np.maximum(evals, 1e-16))
        order = np.argsort(axes)[::-1]  # descending: a >= b >= c
        axes = axes[order]
        R = evecs[:, order]

        # canonicalize handedness
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0

        # canonicalize sign per column (first nonzero entry positive)
        for j in range(3):
            col = R[:, j]
            idx = np.argmax(np.abs(col) > 1e-12)
            if col[idx] < 0:
                R[:, j] *= -1.0
        if np.linalg.det(R) < 0:
            # if the sign fix broke handedness, flip the last axis back
            R[:, 2] *= -1.0

        a, b, c = axes
        r1 = float(np.clip(b / max(a, 1e-12), 0.0, 1.0))
        r2 = float(np.clip(c / max(a, 1e-12), 0.0, 1.0))

        # R -> Z-Y-X Euler angles (same convention as the old class)
        r20 = np.clip(R[2, 0], -1.0, 1.0)
        if abs(r20) < 1.0 - 1e-8:
            beta = -np.arcsin(r20)
            cb = np.cos(beta)
            alpha = np.arctan2(R[1, 0] / cb, R[0, 0] / cb)
            gamma_ang = np.arctan2(R[2, 1] / cb, R[2, 2] / cb)
        else:
            beta = np.pi / 2 if r20 <= -1.0 + 1e-8 else -np.pi / 2
            alpha = np.arctan2(-R[0, 1], R[1, 1])
            gamma_ang = 0.0

        return {
            "center": center,
            "a": float(a),
            "r1": r1,
            "r2": r2,
            "R": R,
            "euler_zyx": np.array([alpha, beta, gamma_ang]),
        }

    @staticmethod
    def from_classic_parameters(
        center: np.ndarray,
        a: float,
        r1: float,
        r2: float,
        alpha: float,
        beta: float,
        gamma_ang: float,
    ) -> np.ndarray:
        """
        Inverse of `canonical_parameters`: convert the classic
        (center, a, r1, r2, alpha, beta, gamma) representation into
        theta = [x_c, y_c, z_c, l11, l21, l22, l31, l32, l33].

        Uses the same conventions as the old parametrization:
          - semi-axes (a, a*r1, a*r2) along the local x, y, z directions
            (matching gamma(u,v) = c + R @ (a cos u sin v, a r1 sin u sin v,
            a r2 cos v))
          - R = Rz(alpha) @ Ry(beta) @ Rx(gamma)  (Z-Y-X Euler angles)

        This is a many-to-one map (the classic parametrization's
        symmetries all collapse onto the same theta), which is exactly
        the point: feel free to pass in any representative -- e.g. one
        obtained from a prior, a manual guess, or another tool's output
        -- and it will land on the correct, canonical (center, L).
        """
        center = np.asarray(center, dtype=float)
        a = float(a)
        r1 = float(r1)
        r2 = float(r2)
        if a <= 0 or r1 <= 0 or r2 <= 0:
            raise ValueError("a, r1, r2 must be strictly positive")

        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma_ang), np.sin(gamma_ang)

        Rz = np.array([[ca, -sa, 0],
                       [sa,  ca, 0],
                       [0,    0, 1]])
        Ry = np.array([[cb, 0, sb],
                       [0,  1,  0],
                       [-sb, 0, cb]])
        Rx = np.array([[1,  0,   0],
                       [0, cg, -sg],
                       [0, sg,  cg]])
        R = Rz @ Ry @ Rx

        axes = np.array([a, a * r1, a * r2])
        A_shape = (R * (1.0 / axes ** 2)) @ R.T
        A_shape = 0.5 * (A_shape + A_shape.T)  # guard against fp asymmetry

        try:
            L = np.linalg.cholesky(A_shape)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(A_shape + 1e-10 * np.eye(3))

        return np.array([
            center[0], center[1], center[2],
            L[0, 0], L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[2, 2],
        ], dtype=float)