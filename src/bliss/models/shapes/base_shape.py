"""
Base shape class for k-dimensional parameter spaces and n-dimensional ambient spaces.

This module provides the foundational Shape class that all geometric objects inherit from.
Shapes can represent curves (k=1), surfaces (k=2), or higher-dimensional manifolds,
embedded in ambient spaces of dimension n (typically 2 or 3).
"""

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Union, Sequence, Tuple

from scipy.special import erf
from sklearn.neighbors import KDTree


import numpy as np
from scipy.integrate import dblquad, quad
from scipy.optimize import minimize

# Type aliases for clarity and documentation
GammaFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""
Parametrization function signature: (t, θ) -> x

Parameters
----------
t : np.ndarray
    Parameter values, shape (M, k) where k is parameter_dim
θ : np.ndarray  
    Shape parameters, shape (n_params,)

Returns
-------
x : np.ndarray
    Points in ambient space, shape (M, n) where n is ambient_dim
"""

JacobianFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""
Jacobian function signature: (t, θ) -> J

Parameters
----------
t : np.ndarray
    Parameter values, shape (M, k)
θ : np.ndarray
    Shape parameters, shape (n_params,)

Returns
-------
J : np.ndarray
    Jacobian matrices ∂γ/∂t, shape (M, n, k)
"""


@dataclass
class Shape(ABC):
    """
    Abstract base class for geometric shapes with parametric representations.

    A Shape represents a k-dimensional manifold embedded in n-dimensional space,
    given by a parametrization γ: D ⊂ ℝ^k → ℝ^n that depends on shape parameters θ.

    Examples
    --------
    - Circle in 2D: k=1, n=2, γ(t; x_c, y_c, r) = (x_c + r cos(2πt), y_c + r sin(2πt))
    - Sphere in 3D: k=2, n=3, γ(u,v; x_c, y_c, z_c, r) = center + r * direction(u,v)

    Parameters
    ----------
    parameter_dim : int
        Dimension k of the parameter space (1 for curves, 2 for surfaces, etc.)
    ambient_dim : int
        Dimension n of the ambient space (2 for plane, 3 for 3D space, etc.)
    parametrization : GammaFn
        Function mapping (t, θ) to points in ambient space.
        Must handle vectorized input: t of shape (M, k) -> output shape (M, n)
    jacobian_fn : Optional[JacobianFn], default=None
        Function computing ∂γ/∂t. If provided, enables metric computations.
        Input: t of shape (M, k) -> output shape (M, n, k)
    domain_low : Union[float, Sequence[float]], default=0.0
        Lower bounds of parameter domain. Scalar broadcasts to all dimensions.
    domain_high : Union[float, Sequence[float]], default=1.0
        Upper bounds of parameter domain. Scalar broadcasts to all dimensions.
    periodic : Union[bool, Sequence[bool]], default=False
        Whether each parameter dimension is periodic (wraps at boundaries).
        Scalar broadcasts to all dimensions.
    bounds : Optional[Sequence[Tuple[float, float]]], default=None
        Box constraints on shape parameters θ for optimization.
        Length should equal get_n_params().

    Attributes
    ----------
    domain_low : np.ndarray
        Lower bounds of parameter domain, shape (k,)
    domain_high : np.ndarray
        Upper bounds of parameter domain, shape (k,)
    periodic : np.ndarray
        Periodicity flags for each dimension, shape (k,)

    Notes
    -----
    The parameter domain is D = [domain_low[0], domain_high[0]] × ... × [domain_low[k-1], domain_high[k-1]].
    For periodic dimensions, the parameter space forms a circle (wraps around).

    All concrete subclasses must implement get_n_params() to specify the number
    of shape parameters θ required by their parametrization.
    """

    parameter_dim: int
    ambient_dim: int
    parametrization: GammaFn
    jacobian_fn: Optional[JacobianFn] = None
    domain_low: Union[float, Sequence[float]] = 0.0
    domain_high: Union[float, Sequence[float]] = 1.0
    periodic: Union[bool, Sequence[bool]] = False
    bounds: Optional[Sequence[Tuple[float, float]]] = None
    initial_fit: Optional[Callable] = None

    def __post_init__(self) -> None:
        """
        Validate and normalize inputs after initialization.

        Converts scalar domain bounds and periodicity flags to arrays,
        and validates that dimensions are consistent.

        Raises
        ------
        AssertionError
            If domain bounds or periodic flags have incorrect length
        ValueError
            If parameter_dim or ambient_dim are not positive
        """
        # Validate dimensions
        if self.parameter_dim <= 0:
            raise ValueError(f"parameter_dim must be positive, got {self.parameter_dim}")
        if self.ambient_dim <= 0:
            raise ValueError(f"ambient_dim must be positive, got {self.ambient_dim}")

        # Convert domain_low to array
        if isinstance(self.domain_low, (int, float)):
            object.__setattr__(
                self, 'domain_low',
                np.full(self.parameter_dim, float(self.domain_low))
            )
        else:
            object.__setattr__(self, 'domain_low', np.array(self.domain_low, dtype=float))

        # Convert domain_high to array
        if isinstance(self.domain_high, (int, float)):
            object.__setattr__(
                self, 'domain_high',
                np.full(self.parameter_dim, float(self.domain_high))
            )
        else:
            object.__setattr__(self, 'domain_high', np.array(self.domain_high, dtype=float))

        # Convert periodic to array
        if isinstance(self.periodic, bool):
            object.__setattr__(
                self, 'periodic',
                np.full(self.parameter_dim, self.periodic)
            )
        else:
            object.__setattr__(self, 'periodic', np.array(self.periodic, dtype=bool))

        # Validate array lengths
        assert len(self.domain_low) == self.parameter_dim, \
            f"domain_low length {len(self.domain_low)} != parameter_dim {self.parameter_dim}"
        assert len(self.domain_high) == self.parameter_dim, \
            f"domain_high length {len(self.domain_high)} != parameter_dim {self.parameter_dim}"
        assert len(self.periodic) == self.parameter_dim, \
            f"periodic length {len(self.periodic)} != parameter_dim {self.parameter_dim}"

        # Validate domain bounds
        if not np.all(self.domain_low < self.domain_high):
            raise ValueError("domain_low must be strictly less than domain_high")

        # Validate bounds if provided
        if self.bounds is not None:
            n_params = self.get_n_params()
            if len(self.bounds) != n_params:
                raise ValueError(
                    f"bounds length {len(self.bounds)} != n_params {n_params}"
                )

            for i, (low, high) in enumerate(self.bounds):

                # Case 1: (None, None) — always valid
                if low is None and high is None:
                    continue

                # Case 2: (None, high) — valid only if high is numeric
                if low is None:
                    if high is None:
                        continue
                    if not isinstance(high, (float, int)):
                        raise ValueError(f"Invalid upper bound at index {i}: upper={high}")
                    continue

                # Case 3: (low, None) — valid only if low is numeric
                if high is None:
                    if not isinstance(low, (float, int)):
                        raise ValueError(f"Invalid lower bound at index {i}: lower={low}")
                    continue

                # Case 4: (low, high) — both numbers → require low < high
                if not (isinstance(low, (float, int)) and isinstance(high, (float, int))):
                    raise ValueError(f"Bounds must be numeric or None at index {i}: ({low}, {high})")

                if low >= high:
                    raise ValueError(
                        f"Invalid bounds at index {i}: lower={low} >= upper={high}"
                    )

    # ------------------------------------------------------------------
    # Default probabilistic models
    # ------------------------------------------------------------------

    def default_source_density(self):
        from bliss.inference.source_densitites import (
            UniformSourceDensity,
        )

        centers = (self.domain_low + self.domain_high) / 2
        widths = (self.domain_high - self.domain_low) / 2

        fixed_params = np.empty(2 * self.parameter_dim)
        fixed_params[0::2] = centers
        fixed_params[1::2] = widths

        return UniformSourceDensity(
            n_dim=self.parameter_dim,
            domain_low=self.domain_low,
            domain_high=self.domain_high,
            periodic=self.periodic.tolist(),
            #fixed_params=fixed_params,
        )

    def default_noise_function(self):
        from bliss.inference.noise_functions import (
            IsotropicNoise,
        )

        return IsotropicNoise(
            dim_param=self.parameter_dim,
            dim_ambient=self.ambient_dim,
        )

    def evaluate(self, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Evaluate the shape at given parameter values.

        Parameters
        ----------
        t : np.ndarray
            Parameter values. Can be:
            - Shape (M, k): M points in k-dimensional parameter space
            - Shape (k,): Single point (will be reshaped to (1, k))
            - Shape (M,): For k=1, M points (will be reshaped to (M, 1))
        theta : np.ndarray
            Shape parameters, shape (n_params,)

        Returns
        -------
        points : np.ndarray
            Points in ambient space, shape (M, n)

        """
        # Normalize input shape
        t = self._normalize_parameter_input(t)

        # Call parametrization
        return self.parametrization(t, theta)

    def jacobian(self, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix ∂γ/∂t at given parameter values.

        The Jacobian represents the differential of the parametrization,
        encoding how the shape locally changes with respect to parameters.

        Parameters
        ----------
        t : np.ndarray
            Parameter values, shape (M, k) or (k,)
        theta : np.ndarray
            Shape parameters, shape (n_params,)

        Returns
        -------
        J : np.ndarray
            Jacobian matrices, shape (M, n, k)
            J[i, j, l] = ∂γ_j/∂t_l at point i

        Raises
        ------
        NotImplementedError
            If jacobian_fn was not provided during initialization
        """
        if self.jacobian_fn is None:
            raise NotImplementedError(
                f"Jacobian not provided for {self.__class__.__name__}. "
                "Provide jacobian_fn during initialization or use finite differences."
            )

        # Normalize input shape
        t = self._normalize_parameter_input(t)

        return self.jacobian_fn(t, theta)

    def evaluate_normals(
            self,
            t: np.ndarray,
            theta: np.ndarray,
            return_covariance: bool = True,
            noise_variance: float = 1.0,
            reference_point: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute consistent (outward) normals with covariance.

        Parameters
        ----------
        reference_point : Optional[np.ndarray]
            Point inside the shape used to orient normals outward.
            If None, estimated automatically.
        """

        t = self._normalize_parameter_input(t)
        X = self.evaluate(t, theta)  # (M, n)
        J = self.jacobian(t, theta)

        M, n = X.shape
        k = self.parameter_dim

        normals = np.zeros((M, n))
        covariances = np.zeros((M, n, n)) if return_covariance else None

        # --- Estimate interior reference point if not provided ---
        if reference_point is None:
            reference_point = X.mean(axis=0)  # robust default

        for i in range(M):
            J_i = J[i]

            # --- SVD for tangent / normal space ---
            U, S, Vt = np.linalg.svd(J_i, full_matrices=True)
            rank = np.sum(S > 1e-10)

            U_perp = U[:, rank:]  # normal space

            if U_perp.shape[1] == 0:
                normals[i] = 0
                if return_covariance:
                    covariances[i] = np.eye(n) * 1e6
                continue

            # Representative normal
            n_vec = U_perp[:, 0]
            n_vec = n_vec / np.linalg.norm(n_vec)

            # ---  ORIENTATION FIX ---
            direction = X[i] - reference_point

            if np.dot(n_vec, direction) < 0:
                n_vec = -n_vec

            normals[i] = n_vec

            # --- Covariance ---
            if return_covariance:
                G = J_i.T @ J_i
                G += np.eye(k) * 1e-10

                try:
                    Sigma_tangent = np.linalg.inv(G / noise_variance)
                except np.linalg.LinAlgError:
                    Sigma_tangent = np.eye(k) * 1e6

                P_perp = U_perp @ U_perp.T
                scale = np.trace(Sigma_tangent)

                covariances[i] = scale * P_perp

        return normals, covariances


    def sample_parameter_space_uniform(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample uniformly from the parameter domain.

        For non-periodic dimensions, samples uniformly from [domain_low, domain_high].
        For periodic dimensions, the distribution is the same but conceptually wraps.

        Parameters
        ----------
        size : int
            Number of samples M
        rng : Optional[np.random.Generator], default=None
            Random number generator. If None, uses np.random default.

        Returns
        -------
        samples : np.ndarray
            Parameter samples, shape (M, k)
        """
        if rng is None:
            rng = np.random.default_rng()

        samples = rng.uniform(
            low=self.domain_low,
            high=self.domain_high,
            size=(size, self.parameter_dim)
        )
        return samples

    def sample_parameter_space_grid(self, n_points_per_dim: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Create a regular grid of parameter values.

        Useful for visualization and structured quadrature.

        Parameters
        ----------
        n_points_per_dim : Union[int, Sequence[int]]
            Number of grid points along each dimension.
            If int, uses same number for all dimensions.

        Returns
        -------
        grid : np.ndarray
            Grid points, shape (N, k) where N = prod(n_points_per_dim)
        """
        if isinstance(n_points_per_dim, int):
            n_points_per_dim = [n_points_per_dim] * self.parameter_dim
        else:
            n_points_per_dim = list(n_points_per_dim)

        assert len(n_points_per_dim) == self.parameter_dim

        # Create 1D grids for each dimension
        grids_1d = []
        for i in range(self.parameter_dim):
            endpoint = not self.periodic[i]  # Exclude endpoint for periodic dims
            grids_1d.append(
                np.linspace(
                    self.domain_low[i],
                    self.domain_high[i],
                    n_points_per_dim[i],
                    endpoint=endpoint
                )
            )

        # Create meshgrid
        mesh = np.meshgrid(*grids_1d, indexing='ij')

        # Flatten and stack
        grid = np.stack([m.flatten() for m in mesh], axis=-1)

        return grid

    def metric_density(self, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the metric density √det(g) where g = J^T J.

        The metric density represents the local "volume element" of the manifold,
        useful for integration and geometric computations.

        For curves (k=1): Returns ||∂γ/∂t|| (arc length element)
        For surfaces (k=2): Returns √det(J^T J) (area element)

        Parameters
        ----------
        t : np.ndarray
            Parameter values, shape (M, k)
        theta : np.ndarray
            Shape parameters

        Returns
        -------
        density : np.ndarray
            Metric density at each point, shape (M,)

        Raises
        ------
        NotImplementedError
            If jacobian_fn was not provided

        Examples
        """
        J = self.jacobian(t, theta)  # (M, n, k)

        if self.parameter_dim == 1:
            # For curves: metric density is ||dγ/dt||
            return np.linalg.norm(J.squeeze(-1), axis=-1)
        else:
            # For surfaces/higher-dim: metric density is √det(J^T J)
            # g = J^T J has shape (M, k, k)
            g = np.einsum('mni,mnj->mij', J, J)
            det_g = np.linalg.det(g)

            # Handle numerical issues (det should be non-negative)
            det_g = np.maximum(det_g, 0.0)

            return np.sqrt(det_g)

    def area(self, theta: np.array):
        """
        Compute the area of the shape with parameter theta
        Resembles the length in case of a curve
        """

        if self.jacobian is None:
            raise ValueError("Shape has no jacobian_fn defined.")

         # we need to integrate the speed over the parameter space

        speed = functools.partial(self.metric_density, theta = theta)

        def speed_scalar(t):
            return float(np.asarray(speed(t)).ravel()[0])

        if self.parameter_dim == 1:

            # here it's just a 1D integral
            length, _ = quad(speed_scalar, self.domain_low[0], self.domain_high[0])
            return float(length)

        if self.parameter_dim == 2:

            speed_uv = lambda u,v : speed([[u,v]])[0]

            u0, v0 = float(self.domain_low[0]), float(self.domain_low[1])
            u1, v1 = float(self.domain_high[0]), float(self.domain_high[1])

            area, _ = dblquad(speed_uv, u0, u1, lambda u: v0, lambda u: v1)
            return float(area)
        else:
            raise ValueError(f"Unsupported parameter_dim={self.parameter_dim}. Only 1 or 2 are recognized.")

    def _normalize_parameter_input(self, t: np.ndarray) -> np.ndarray:
        """
        Normalize parameter input to shape (M, k).

        Parameters
        ----------
        t : np.ndarray
            Input array of various possible shapes

        Returns
        -------
        t_normalized : np.ndarray
            Array of shape (M, k)
        """
        t = np.atleast_1d(t)

        if t.ndim == 1:
            # Shape (M,) -> (M, 1) for k=1, or (1, k) for k>1
            if self.parameter_dim == 1:
                t = t.reshape(-1, 1)
            else:
                t = t.reshape(1, -1)

        # Ensure 2D
        if t.ndim != 2:
            raise ValueError(
                f"Expected parameter input with 1 or 2 dimensions, got {t.ndim}"
            )

        # Validate shape
        if t.shape[1] != self.parameter_dim:
            raise ValueError(
                f"Parameter dimension mismatch: expected {self.parameter_dim}, "
                f"got {t.shape[1]}"
            )

        return t

    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        """
        Get box constraints for shape parameters θ (for optimization).

        Returns
        -------
        bounds : Optional[Sequence[Tuple[float, float]]]
            List of (lower, upper) bounds for each parameter, or None if unbounded
        """
        return self.bounds

    def get_domain_volume(self) -> float:
        """
        Compute the volume of the parameter domain.

        Returns
        -------
        volume : float
            Product of interval lengths
        """
        return np.prod(self.domain_high - self.domain_low)

    @abstractmethod
    def get_n_params(self) -> int:
        """
        Get the number of shape parameters θ.

        Returns
        -------
        n_params : int
            Number of parameters required by the parametrization
        """
        pass

    def __repr__(self) -> str:
        """String representation of the shape."""
        return (
            f"{self.__class__.__name__}("
            f"parameter_dim={self.parameter_dim}, "
            f"ambient_dim={self.ambient_dim}, "
            f"n_params={self.get_n_params()})"
        )

    def get_closest_point(
            self,
            params: np.ndarray,
            x: np.ndarray,
            n_restarts: int = 8,
            tol: float = 1e-7,
            return_params: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Find the closest point(s) on the shape to query point(s) x.

        Solves: t* = argmin_{t ∈ D} ||γ(t; θ) - x||²

        Uses L-BFGS-B with multiple random restarts seeded from a coarse grid
        to avoid local minima on non-convex shapes.

        Parameters
        ----------
        params: shape params
        x : np.ndarray, shape (n,) or (M, n)
            One or more query points in ambient space.
        n_restarts : int, default=8
            Number of optimizer restarts per query point. More restarts = more
            robust against local minima, but slower. For smooth convex shapes,
            n_restarts=1 may suffice.
        tol : float, default=1e-7
            Convergence tolerance passed to the optimizer.
        return_params : bool, default=False
            If True, also return the optimal parameter values t*.

        Returns
        -------
        closest : np.ndarray, shape (n,) or (M, n)
            Closest point(s) on the shape. Matches input shape convention.
        params : np.ndarray, shape (k,) or (M, k)  [only if return_params=True]
            Optimal parameter values t* achieving the minimum distance.

        Notes
        -----
        - Complexity: O(M * n_restarts * iters_per_opt * cost(γ))
        - For repeated calls with slowly-moving x, pass warm_start_params to seed
          restarts near the previous solution (not shown here, but trivial to add).
        - If the shape has a jacobian_fn, gradient info is used automatically via
          finite differences inside scipy (or you can extend this to pass jac=True).
        """
        x = np.asarray(x, dtype=float)
        scalar_input = x.ndim == 1
        if scalar_input:
            x = x[np.newaxis, :]  # (1, n)

        M = x.shape[0]
        k = self.parameter_dim

        low = self.domain_low  # (k,)
        high = self.domain_high  # (k,)
        periodic = self.periodic  # (k,) bool

        bounds_opt = [
            (None, None) if periodic[i] else (low[i], high[i])
            for i in range(k)
        ]

        # --- Coarse grid for seeding restarts ---
        # Sample a grid of n_restarts^k points (capped to avoid explosion)
        pts_per_dim = max(1, int(round(n_restarts ** (1.0 / k))))
        grid_1d = [np.linspace(low[i], high[i], pts_per_dim + 2)[1:-1] for i in range(k)]
        grid = np.stack(np.meshgrid(*grid_1d, indexing="ij"), axis=-1).reshape(-1, k)
        # Pad with random restarts if grid is small
        if len(grid) < n_restarts:
            extra = np.random.uniform(low, high, size=(n_restarts - len(grid), k))
            grid = np.vstack([grid, extra])

        # --- Per-point optimization ---
        best_t = np.empty((M, k))
        best_pts = np.empty((M, x.shape[1]))

        def objective(t_flat, xi):
            t = t_flat.reshape(1, k)
            # Wrap periodic dims into [low, high]
            for i in range(k):
                if periodic[i]:
                    span = high[i] - low[i]
                    t[0, i] = low[i] + (t[0, i] - low[i]) % span
            pt = self.parametrization(t, params)  # (1, n)
            diff = pt[0] - xi
            return float(np.dot(diff, diff))

        for m in range(M):
            xi = x[m]
            best_val = np.inf
            best_t_m = grid[0].copy()

            for t0 in grid[:n_restarts]:
                result = minimize(
                    objective,
                    t0.copy(),
                    args=(xi,),
                    method="L-BFGS-B",
                    bounds=bounds_opt,
                    options={"ftol": tol ** 2, "gtol": tol, "maxiter": 200},
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_t_m = result.x

            # Final eval at best t
            t_star = best_t_m.reshape(1, k)
            for i in range(k):
                if periodic[i]:
                    span = high[i] - low[i]
                    t_star[0, i] = low[i] + (t_star[0, i] - low[i]) % span
            best_t[m] = t_star[0]
            best_pts[m] = self.parametrization(t_star, params)[0]

        if scalar_input:
            return (best_pts[0], best_t[0]) if return_params else best_pts[0]
        return (best_pts, best_t) if return_params else best_pts

    def geometric_distance(self, params: np.ndarray, measurement: np.ndarray):
        """
        Computes the geometric distance (orthogonal) to a set of measured points
        Parameters
        ----------
        params
        measurement

        Returns
        -------

        """

        # compute the closest points
        closest_points = self.get_closest_point(params, measurement, return_params=True)[0]

        # now compute the distances
        dists = np.linalg.norm(measurement - closest_points, axis=1)

        return np.sum(dists)

    def transformed(self, shift=None, scale=None):
        shift = np.zeros(self.ambient_dim) if shift is None else shift
        scale = 1.0 if scale is None else scale

        base_gamma = self.parametrization
        base_jac = self.jacobian_fn

        def gamma_new(t, theta):
            return scale * base_gamma(t, theta) + shift[None, :]

        def jac_new(t, theta):
            if base_jac is None:
                return None
            return scale * base_jac(t, theta)

        # --- clone SAME CLASS ---
        new = self.__class__()  # ⚠ requires default constructor!!

        # overwrite behavior
        new.parametrization = gamma_new
        new.jacobian_fn = jac_new

        return new

    def geometric_residuals(
            self,
            params: np.ndarray,
            measurement: np.ndarray,
    ):
        closest_points = self.get_closest_point(
            params,
            measurement,
            return_params=True,
        )[0]

        return np.linalg.norm(
            measurement - closest_points,
            axis=1,
        )
