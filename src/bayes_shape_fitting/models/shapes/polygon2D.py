from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence, Tuple
from .base_shape import Shape


@dataclass(init=False)
class Polygon2D(Shape):
    """
    Free-form polygon with N vertices.

    Unlike regular polygon, vertices can be at arbitrary positions.
    The curve is piecewise linear interpolation between vertices.

    Parametrization:
        γ(t; x_1, y_1, ..., x_N, y_N) = piecewise linear through vertices

    Parameter space: t ∈ [0, 1), periodic

    Parameters
    ----------
    N : int
        Number of vertices.

    Shape parameters:
        θ = [x_1, y_1, x_2, y_2, ..., x_N, y_N]
        Total: 2N parameters

    Notes
    -----
    - The polygon is closed (last vertex connects to first)
    - Vertices are visited in order as t increases from 0 to 1
    - Not necessarily convex
    - Jacobian is undefined at vertices (piecewise constant between)

    """

    def __init__(self, n: int):
        self. N = n
        if self.N < 3:
            raise ValueError(f"Polygon must have at least 3 vertices, got {self.N}")

        def gamma(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            Piecewise linear interpolation through vertices.

            Parameters
            ----------
            t : np.ndarray, shape (M,) or (M, 1)
                Parameter values in [0, 1).
            theta : np.ndarray, shape (2N,)
                Flattened vertices [x_1, y_1, x_2, y_2, ..., x_N, y_N]

            Returns
            -------
            points : np.ndarray, shape (M, 2)
                Points on polygon.
            """
            t = np.atleast_1d(t).flatten()

            # Reshape theta to (N, 2) vertices
            vertices = theta.reshape(self.N, 2)

            # Map t ∈ [0, 1) to fractional edge index
            idx = t * self.N

            # Integer part: which edge (0 to N-1)
            i0 = np.floor(idx).astype(int) % self.N
            i1 = (i0 + 1) % self.N

            # Fractional part: position along edge
            w = idx - np.floor(idx)

            # Linear interpolation
            return vertices[i0] * (1 - w)[:, np.newaxis] + vertices[i1] * w[:, np.newaxis]

        def jacobian(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
            """
            Jacobian ∂γ/∂t (edge vectors, piecewise constant).

            Returns
            -------
            jac : np.ndarray, shape (M, 2, 1)
                Tangent vectors (constant along each edge).
            """
            t = np.atleast_1d(t).flatten()
            vertices = theta.reshape(self.N, 2)

            # Determine which edge each t is on
            idx = t * self.N
            i0 = np.floor(idx).astype(int) % self.N
            i1 = (i0 + 1) % self.N

            # Edge vectors (scaled by N because dt corresponds to 1/N of perimeter)
            edge_vectors = (vertices[i1] - vertices[i0]) * self.N

            # Shape (M, 2, 1)
            return edge_vectors[:, :, np.newaxis]

        super().__init__(
            parameter_dim=1,
            ambient_dim=2,
            parametrization=gamma,
            jacobian_fn=jacobian,
            domain_low=0.0,
            domain_high=1.0,
            periodic=True,
            bounds=[(-10, 10)] * (2 * self.N)  # All vertices in [-10, 10]²
        )

    def get_n_params(self) -> int:
        return 2 * self.N