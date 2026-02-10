"""
Base shape class for k-dimensional parameter spaces and n-dimensional ambient spaces.

This module provides the foundational Shape class that all geometric objects inherit from.
Shapes can represent curves (k=1), surfaces (k=2), or higher-dimensional manifolds,
embedded in ambient spaces of dimension n (typically 2 or 3).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Sequence, Tuple
import numpy as np

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
                if low >= high:
                    raise ValueError(
                        f"Invalid bounds at index {i}: lower={low} >= upper={high}"
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

    def arc_length(self, t: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute arc length for curves (parameter_dim=1).

        Integrates the metric density over the parameter domain.

        Parameters
        ----------
        t : np.ndarray
            Parameter values where metric is evaluated, shape (M, 1) or (M,)
        theta : np.ndarray
            Shape parameters

        Returns
        -------
        length : float
            Total arc length

        Raises
        ------
        ValueError
            If parameter_dim != 1

        Examples
        """
        if self.parameter_dim != 1:
            raise ValueError(
                f"arc_length only defined for curves (parameter_dim=1), "
                f"got parameter_dim={self.parameter_dim}"
            )

        t = self._normalize_parameter_input(t)
        density = self.metric_density(t, theta)

        # Integrate using trapezoidal rule
        from scipy.integrate import trapezoid
        return trapezoid(density, t.flatten())

    def surface_area(self, t: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute surface area for surfaces (parameter_dim=2).

        Integrates the metric density over the parameter domain.

        Parameters
        ----------
        t : np.ndarray
            Parameter values, shape (M, 2)
        theta : np.ndarray
            Shape parameters

        Returns
        -------
        area : float
            Total surface area

        Raises
        ------
        ValueError
            If parameter_dim != 2

        Notes
        -----
        For accurate results, t should be a dense grid covering the domain.
        """
        if self.parameter_dim != 2:
            raise ValueError(
                f"surface_area only defined for surfaces (parameter_dim=2), "
                f"got parameter_dim={self.parameter_dim}"
            )

        density = self.metric_density(t, theta)

        # Approximate integral using mean value
        domain_volume = np.prod(self.domain_high - self.domain_low)
        return np.mean(density) * domain_volume

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