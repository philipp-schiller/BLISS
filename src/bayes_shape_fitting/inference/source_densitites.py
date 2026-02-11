import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
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

    def get_prior(self):
        return self.prior


@dataclass(frozen=True)
class MixtureSourceDensity(SourceDensity):
    n_components: int
    prior: Optional[Prior] = None

    @property
    def dim(self):
        return 1

    def evaluate(self, x, params=None):
        if params is None:
            raise ValueError("MixtureSourceDensity requires params")

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
        if params is None:
            raise ValueError("MixtureSourceDensity requires params")

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
        return FunctionType.PARAMETRIC

    def get_n_params(self):
        return 3 * self.n_components

    def get_prior(self):
        return self.prior