from dataclasses import dataclass
from typing import Optional
import numpy as np

from .base_shape import Shape


@dataclass(init=False)
class BSplineCurve(Shape):
    """
    B-spline curve (open or closed / periodic).

    Parametrization:
        γ(t) = Σ_i N_{i,p}(t) P_i

    - Open splines: clamped uniform knots
    - Closed splines: periodic via control-point extension
    """

    def __init__(
        self,
        degree: int = 3,
        n_control: int = 6,
        closed: bool = False,
    ):
        if degree < 1:
            raise ValueError("Degree must be ≥ 1.")
        if n_control < degree + 1:
            raise ValueError("Need at least degree+1 control points.")

        self.degree = degree
        self.n_control = n_control
        self.closed = closed

        # Number of control points used for evaluation
        self.n_eval_ctrl = (
            n_control + degree if closed else n_control
        )

        # Knot vector
        self.knot_vector = (
            self._periodic_knots()
            if closed
            else self._open_uniform_knots()
        )

        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            t = np.atleast_1d(t).flatten()
            ctrl = theta.reshape(self.n_control, 2)

            if self.closed:
                ctrl = np.vstack([ctrl, ctrl[:self.degree]])

            N = self._bspline_basis(t)
            return N @ ctrl

        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            t = np.atleast_1d(t).flatten()
            ctrl = theta.reshape(self.n_control, 2)

            if self.closed:
                ctrl = np.vstack([ctrl, ctrl[:self.degree]])

            dN = self._bspline_basis_derivative(t)
            tangent = dN @ ctrl
            return tangent[:, :, None]

        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=closed,
            bounds=[(-10.0, 10.0)] * (2 * n_control),
        )

    # ─────────────────────────────────────────────────────────────
    # Knot vectors
    # ─────────────────────────────────────────────────────────────

    def _open_uniform_knots(self) -> np.ndarray:
        p = self.degree
        n = self.n_control
        m = n + p + 1

        U = np.zeros(m)
        U[p : m - p] = np.linspace(0, 1, m - 2 * p)
        U[m - p :] = 1.0
        return U

    def _periodic_knots(self) -> np.ndarray:
        """
        Uniform, non-clamped knot vector for periodic B-splines.
        """
        p = self.degree
        n = self.n_control + p
        m = n + p + 1
        return np.arange(m, dtype=float)

    # ─────────────────────────────────────────────────────────────
    # Cox–de Boor basis
    # ─────────────────────────────────────────────────────────────

    def _bspline_basis(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        M = len(t)
        p = self.degree
        U = self.knot_vector
        n = self.n_eval_ctrl

        # Map [0,1] → valid knot span
        t0 = U[p]
        t1 = U[n]
        t = t0 + t * (t1 - t0)

        N = np.zeros((M, n))

        # Degree 0 basis
        for i in range(n):
            N[:, i] = ((U[i] <= t) & (t < U[i + 1])).astype(float)

        # Right endpoint fix
        N[t == U[n], n - 1] = 1.0

        # Cox–de Boor recursion
        for k in range(1, p + 1):
            Nk = np.zeros_like(N)
            for i in range(n):
                # First term
                if i + k < len(U) and U[i + k] != U[i]:
                    Nk[:, i] += (
                                        (t - U[i]) / (U[i + k] - U[i])
                                ) * N[:, i]

                # Second term (guarded!)
                if (
                        i + 1 < n
                        and i + k + 1 < len(U)
                        and U[i + k + 1] != U[i + 1]
                ):
                    Nk[:, i] += (
                                        (U[i + k + 1] - t)
                                        / (U[i + k + 1] - U[i + 1])
                                ) * N[:, i + 1]

            N = Nk

        return N

    # ─────────────────────────────────────────────────────────────
    # Basis derivative
    # ─────────────────────────────────────────────────────────────

    def _bspline_basis_derivative(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        p = self.degree
        U = self.knot_vector
        n = self.n_eval_ctrl

        if p == 0:
            return np.zeros((len(t), n))

        N_low = self._bspline_basis_degree(t, p - 1)
        dN = np.zeros((len(t), n))

        for i in range(n):
            if U[i + p] != U[i]:
                dN[:, i] += p / (U[i + p] - U[i]) * N_low[:, i]

            if (
                    i + 1 < n
                    and U[i + p + 1] != U[i + 1]
            ):
                dN[:, i] -= p / (U[i + p + 1] - U[i + 1]) * N_low[:, i + 1]

        # Chain rule due to parameter remapping
        dN *= (U[n] - U[p])

        return dN

    def _bspline_basis_degree(self, t: np.ndarray, degree: int) -> np.ndarray:
        saved = self.degree
        self.degree = degree
        N = self._bspline_basis(t)
        self.degree = saved
        return N

    # ─────────────────────────────────────────────────────────────

    def get_n_params(self) -> int:
        return 2 * self.n_control