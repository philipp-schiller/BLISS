import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from scipy.stats import beta as beta_dist, norm
from src.bayes_shape_fitting.inference.priors import Prior


class FunctionType(Enum):
    FIXED = "fixed"  # Function is known, no parameters
    PARAMETRIC = "parametric"  # Function has learnable parameters


class SourceDensity(ABC):
    """
    Abstract base class for parametric or fixed probability densities.
    Supports evaluation and sampling.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the domain."""
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate density p(x). x must be shape (N, dim)."""
        pass

    @abstractmethod
    def sample(self, n: int, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw n samples. Returns array of shape (n, dim)."""
        pass

    @abstractmethod
    def get_function_type(self) -> FunctionType:
        pass

    @abstractmethod
    def get_n_params(self) -> int:
        pass

    @abstractmethod
    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        pass

    @abstractmethod
    def get_prior(self) -> Optional[Prior]:
        pass

    def _ensure_shape(self, x: np.ndarray) -> np.ndarray:
        """Convert 1D arrays into shape (N, dim)."""
        x = np.asarray(x)
        if x.ndim == 1:
            if self.dim == 1:
                x = x.reshape(-1, 1)
            else:
                raise ValueError(f"x must have shape (N,{self.dim}), got {x.shape}")
        return x


@dataclass(frozen=True)
class FixedSourceDensity(SourceDensity):
    density_function: Callable[[np.ndarray], np.ndarray]
    dim: int

    def dim(self) -> int:
        return self.dim

    def evaluate(self, x, params=None):
        x = self._ensure_shape(x)
        return self.density_function(x)

    def sample(self, n, params=None):
        raise NotImplementedError("Sampling not supported for arbitrary fixed densities.")

    def get_function_type(self):
        return FunctionType.FIXED

    def get_n_params(self):
        return 0

    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        return None

    def get_prior(self):
        return None

@dataclass(frozen=True)
class UniformSourceDensity(SourceDensity):
    dim: int
    prior: Optional[Prior] = None
    fixed_params: Optional[np.ndarray] = None #[a_0,b_0,a_1,b_1,...a_dim,b_dim]

    # required by interface
    def dim(self) -> int:
        return self.dim

    def _unpack_params(self, params):
        """
        Returns parameters, using fixed_params if params is None.
        """
        if self.fixed_params is not None:
            return self.fixed_params

        if params is None:
            raise ValueError("UniformSourceDensity: params must be provided unless fixed_params is set.")

        params = np.asarray(params)
        if params.shape[0] != 2 * self.dim:
            raise ValueError(
                f"UniformSourceDensity expected {2 * self.dim} parameters (a_i,b_i), got {params.size}"
            )
        return params

    def evaluate(self, x, params=None):
        """
        Evaluate the uniform density at points x.
        x must be shape (N, dim). Returns shape (N,).
        """
        params = self._unpack_params(params)
        x = np.atleast_2d(x)  # ensure shape (N, dim)

        if x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (N, {self.dim}), got {x.shape}")

        # extract per-dim bounds
        a = params[0::2]  # (dim,)
        b = params[1::2]  # (dim,)

        # compute per-dimension support mask: (N, dim)
        inside = (x >= a) & (x <= b)

        # point is inside support only if inside all dims
        mask = np.all(inside, axis=1)  # (N,)

        # compute volume
        widths = b - a
        if np.any(widths <= 0):
            raise ValueError("UniformSourceDensity: each interval must satisfy b > a.")

        volume = np.prod(widths)
        density_value = 1.0 / volume

        out = np.zeros(x.shape[0])
        out[mask] = density_value
        return out

    # ---------------------------------------------------------
    #                      SAMPLING
    # ---------------------------------------------------------
    def sample(self, n, params=None):
        """
        Draw n samples from the uniform density.
        Returns shape (n, dim)
        """
        params = self._unpack_params(params)
        a = params[0::2]
        b = params[1::2]

        if np.any(b <= a):
            raise ValueError("UniformSourceDensity: each interval must satisfy b > a.")

        # uniform per dimension
        return a + (b - a) * np.random.rand(n, self.dim)


    def get_function_type(self):
        return FunctionType.FIXED if self.fixed_params is not None else FunctionType.PARAMETRIC

    def get_n_params(self):
        return 0 if self.fixed_params is not None else self.dim * 2

    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        return None

    def get_prior(self):
        return None

@dataclass(frozen=True)
class BetaSourceDensity(SourceDensity):
    prior: Optional[Prior] = None
    fixed_params: Optional[np.ndarray] = None  # [alpha, beta]

    @property
    def dim(self):
        return 1

    def _params(self, params):
        if params is None:
            if self.fixed_params is None:
                raise ValueError("Must provide params or define fixed_params.")
            return self.fixed_params
        return params

    def evaluate(self, x, params=None):
        x = self._ensure_shape(x)
        alpha, beta = self._params(params)
        return beta_dist.pdf(x[:, 0], alpha, beta)

    def sample(self, n, params=None):
        alpha, beta = self._params(params)
        from scipy.stats import beta as beta_dist
        s = beta_dist.rvs(alpha, beta, size=n)
        return s.reshape(-1, 1)

    def get_function_type(self):
        return FunctionType.FIXED if self.fixed_params is not None else FunctionType.PARAMETRIC

    def get_n_params(self):
        return 0 if self.fixed_params is not None else 2

    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        return None if self.fixed_params else [(0,np.inf),(0,np.inf)]

    def get_prior(self):
        return None if self.fixed_params else self.prior


@dataclass(frozen=True)
class SplineSourceDensity(SourceDensity):
    n_knots: int
    prior: Optional[Prior] = None
    knots: Optional[np.ndarray] = None

    @property
    def dim(self):
        return 1

    def __post_init__(self):
        if self.knots is None:
            object.__setattr__(self, "knots", np.linspace(0, 1, self.n_knots))

    def evaluate(self, x, params=None):
        if params is None:
            raise ValueError("SplineSourceDensity requires params")

        x = self._ensure_shape(x)
        weights = np.exp(params)

        norm_const = np.trapezoid(weights, self.knots)
        density = weights / norm_const

        interpolator = interp1d(
            self.knots, density, kind="cubic",
            bounds_error=False, fill_value=0.0
        )

        return interpolator(x[:, 0])

    def sample(self, n, params=None):
        """Inverse transform sampling via CDF."""
        if params is None:
            raise ValueError("Spline source requires params")

        weights = np.exp(params)
        norm_const = np.trapezoid(weights, self.knots)
        density = weights / norm_const

        cdf = np.cumsum(density)
        cdf = cdf / cdf[-1]

        us = np.random.rand(n)
        idx = np.searchsorted(cdf, us)
        samples = self.knots[idx]
        return samples.reshape(-1, 1)

    def get_function_type(self):
        return FunctionType.PARAMETRIC

    def get_n_params(self):
        return self.n_knots

    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        return None

    def get_prior(self):
        return self.prior


@dataclass(frozen=True)
class MixtureSourceDensity(SourceDensity):
    n_components: int
    fixed_params: Optional[np.ndarray] = None
    prior: Optional[Prior] = None

    @property
    def dim(self):
        return 1

    def _params(self, params):
        if params is None:
            if self.fixed_params is None:
                raise ValueError("Must provide params or define fixed_params.")
            return self.fixed_params
        return params

    def evaluate(self, x, params=None):
        params = self._params(params)
        x = self._ensure_shape(x)
        params = params.reshape(self.n_components, 3)
        mus, sigmas, logits = params[:, 0], params[:, 1], params[:, 2]

        weights = np.exp(logits)
        weights /= weights.sum()

        out = np.zeros(len(x))
        for mu, sigma, w in zip(mus, np.abs(sigmas), weights):
            out += w * norm.pdf(x[:, 0], mu, sigma)

        return out

    def sample(self, n, params=None):
        params = self._params(params)

        params = params.reshape(self.n_components, 3)
        mus, sigmas, logits = params[:, 0], params[:, 1], params[:, 2]

        weights = np.exp(logits)
        weights /= weights.sum()

        comps = np.random.choice(self.n_components, size=n, p=weights)
        samples = np.array([
            norm.rvs(loc=mus[c], scale=np.abs(sigmas[c]))
            for c in comps
        ])
        return samples.reshape(-1, 1)

    def get_function_type(self):
        return FunctionType.FIXED if self.fixed_params is not None else FunctionType.PARAMETRIC

    def get_n_params(self):
        return 3 * self.n_components

    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        return None

    def get_prior(self):
        return self.prior