from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence, Tuple
from .base_shape import Shape


@dataclass(init=False)
class BezierCurve(Shape):
    """
    Bezier curve of degree N (with N+1 control points).

    Parametrization:
        γ(t; P_0, ..., P_N) = Σ_{i=0}^N B_i^N(t) P_i

    where B_i^N(t) are Bernstein basis polynomials:
        B_i^N(t) = C(N, i) * t^i * (1-t)^(N-i)

    Parameter space: t ∈ [0, 1], NOT periodic (open curve)

    Parameters
    ----------
    degree : int
        Degree of Bezier curve (number of control points = degree + 1).

    Shape parameters:
        θ = [x_0, y_0, x_1, y_1, ..., x_N, y_N]
        Total: 2(N+1) parameters where N is the degree

    Notes
    -----
    - degree=1: Line segment
    - degree=2: Quadratic Bezier (3 control points)
    - degree=3: Cubic Bezier (4 control points)
    - Higher degrees: More flexible curves

    Properties:
    - Curve starts at P_0 and ends at P_N
    - Curve lies in convex hull of control points
    - Tangent at start is parallel to (P_1 - P_0)
    - Tangent at end is parallel to (P_N - P_{N-1})
    """

    degree: int  # Degree N (number of control points = N + 1)

    def __init__(self, degree: int = 4):
        self.degree = degree
        if self.degree < 1:
            raise ValueError(f"Degree must be at least 1, got {self.degree}")

        # Precompute binomial coefficients C(N, i)
        self.binomial_coeffs = self._compute_binomial_coefficients()

        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            Bezier curve evaluation using de Casteljau's algorithm.

            Parameters
            ----------
            t : np.ndarray, shape (M,) or (M, 1)
                Parameter values in [0, 1].
            theta : np.ndarray, shape (2(N+1),)
                Flattened control points [x_0, y_0, ..., x_N, y_N]

            Returns
            -------
            points : np.ndarray, shape (M, 2)
                Points on Bezier curve.
            """
            t = np.atleast_1d(t).flatten()
            M = len(t)

            # Reshape control points to (N+1, 2)
            n_control = self.degree + 1
            control_points = theta.reshape(n_control, 2)

            # Compute Bernstein basis polynomials
            basis = self._bernstein_basis(t)  # Shape (M, N+1)

            # Linear combination: γ(t) = Σ B_i(t) P_i
            # points[j] = Σ_i basis[j, i] * control_points[i]
            points = basis @ control_points  # (M, N+1) @ (N+1, 2) = (M, 2)

            return points

        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            Jacobian ∂γ/∂t using derivative of Bernstein polynomials.

            Derivative: γ'(t) = N * Σ_{i=0}^{N-1} B_i^{N-1}(t) (P_{i+1} - P_i)

            Returns
            -------
            jac : np.ndarray, shape (M, 2, 1)
                Tangent vectors.
            """
            t = np.atleast_1d(t).flatten()
            M = len(t)

            n_control = self.degree + 1
            control_points = theta.reshape(n_control, 2)

            if self.degree == 1:
                # Linear: derivative is constant (P_1 - P_0)
                tangent = control_points[1] - control_points[0]
                tangent_vectors = np.tile(tangent, (M, 1))
            else:
                # Derivative of Bezier curve of degree N is degree N-1
                # γ'(t) = N * Σ B_i^{N-1}(t) ΔP_i where ΔP_i = P_{i+1} - P_i

                # Compute Bernstein basis of degree N-1
                basis_deriv = self._bernstein_basis(t, degree=self.degree - 1)

                # Differences between consecutive control points
                delta_P = np.diff(control_points, axis=0)  # Shape (N, 2)

                # γ'(t) = N * basis_deriv @ delta_P
                tangent_vectors = self.degree * (basis_deriv @ delta_P)

            # Shape (M, 2, 1)
            return tangent_vectors[:, :, np.newaxis]

        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=False,  # Bezier curves are NOT periodic
            bounds=[(-10.0, 10.0)] * (2 * (self.degree + 1))
        )

    def get_n_params(self) -> int:
        return 2 * (self.degree + 1)

    def _compute_binomial_coefficients(self) -> np.ndarray:
        """
        Compute binomial coefficients C(N, i) for i = 0, ..., N.

        Returns
        -------
        coeffs : np.ndarray, shape (N+1,)
            Binomial coefficients.
        """
        from math import comb
        N = self.degree
        return np.array([comb(N, i) for i in range(N + 1)])

    def _bernstein_basis(self, t: np.ndarray, degree: Optional[int] = None) -> np.ndarray:
        """
        Evaluate Bernstein basis polynomials.

        B_i^N(t) = C(N, i) * t^i * (1-t)^(N-i)

        Parameters
        ----------
        t : np.ndarray, shape (M,)
            Parameter values.
        degree : int, optional
            Degree of Bernstein polynomials. If None, uses self.degree.

        Returns
        -------
        basis : np.ndarray, shape (M, N+1)
            Bernstein basis values.
        """
        if degree is None:
            degree = self.degree
            binomial = self.binomial_coeffs
        else:
            from math import comb
            binomial = np.array([comb(degree, i) for i in range(degree + 1)])

        M = len(t)
        N = degree

        # Prepare t and (1-t) powers
        # t^i for i=0,...,N
        t_powers = np.power.outer(t, np.arange(N + 1))  # Shape (M, N+1)

        # (1-t)^(N-i) for i=0,...,N
        one_minus_t = 1 - t
        one_minus_t_powers = np.power.outer(one_minus_t, np.arange(N, -1, -1))

        # B_i^N(t) = C(N,i) * t^i * (1-t)^(N-i)
        basis = binomial * t_powers * one_minus_t_powers

        return basis