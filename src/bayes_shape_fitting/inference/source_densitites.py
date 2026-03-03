import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, List, Dict
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

    def _wrap_to_domain(self, x: np.ndarray) -> np.ndarray:
        """
        Wrap/clip points to the domain [domain_low, domain_high].

        - Periodic dimensions are wrapped using modular arithmetic (circle/cylinder/torus).
        - Non-periodic dimensions are clipped to [low, high].

        Parameters
        ----------
        x : np.ndarray, shape (N, dim)

        Returns
        -------
        x_wrapped : np.ndarray, shape (N, dim)
        """
        x = np.array(x, dtype=float)  # copy so we don't mutate input

        low = self.domain_low  # (dim,)
        high = self.domain_high  # (dim,)
        period = high - low  # (dim,)

        periodic = self.periodic if self.periodic is not None else [False] * self.dim

        for i, is_periodic in enumerate(periodic):
            if is_periodic:
                # map x into [low, high) via modular arithmetic
                x[:, i] = (x[:, i] - low[i]) % period[i] + low[i]
            else:
                # hard clip to [low, high]
                x[:, i] = np.clip(x[:, i], low[i], high[i])

        return x


# we introduce a wrapper class to fix parts of the source density

class PartiallyFixedDensity(SourceDensity):
    """
    Wraps a SourceDensity and fixes a subset of its parameters.

    Example:
        # Original has 3 params: [mean, std, skew]
        # Fix param index 1 (std=2.0), leaving [mean, skew] free
        wrapped = PartiallyFixedDensity(base, fixed_params={1: 2.0})
        wrapped.evaluate(x, params=np.array([mean_val, skew_val]))
    """

    def __init__(self, base: SourceDensity, fixed_params: Dict[int, float]):
        """
        Args:
            base: The underlying density with N total parameters.
            fixed_params: Mapping of {param_index: fixed_value} in the
                          original parameter space.
        """
        total = base.get_n_params()
        if any(i < 0 or i >= total for i in fixed_params):
            raise ValueError(f"Fixed param indices must be in [0, {total - 1}]")

        self._base = base
        self._fixed_params = fixed_params  # {original_index: value}

        # Indices of the free parameters in the original space
        self._free_indices = [i for i in range(total) if i not in fixed_params]

    def _reconstruct_params(self, free_params: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Reconstruct the full parameter vector by inserting fixed values."""
        if self._base.get_n_params() == 0:
            return None

        full = np.empty(self._base.get_n_params())

        for orig_idx, val in self._fixed_params.items():
            full[orig_idx] = val

        if free_params is not None:
            if len(free_params) != len(self._free_indices):
                raise ValueError(
                    f"Expected {len(self._free_indices)} free params, got {len(free_params)}"
                )
            for free_idx, orig_idx in enumerate(self._free_indices):
                full[orig_idx] = free_params[free_idx]

        return full

    # --- SourceDensity interface ---

    @property
    def dim(self) -> int:
        return self._base.dim

    def evaluate(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        return self._base.evaluate(x, self._reconstruct_params(params))

    def sample(self, n: int, params: Optional[np.ndarray] = None) -> np.ndarray:
        return self._base.sample(n, self._reconstruct_params(params))

    def get_function_type(self) -> FunctionType:
        return FunctionType.FIXED if not self._free_indices else FunctionType.PARAMETRIC

    def get_n_params(self) -> int:
        return len(self._free_indices)

    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        bounds = self._base.get_parameter_bounds()
        if bounds is None:
            return None
        return [bounds[i] for i in self._free_indices]

    def get_prior(self) -> Optional["Prior"]:
        prior = self._base.get_prior()
        if prior is None or not self._free_indices:
            return None
        return prior


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

@dataclass(frozen=False)
class UniformSourceDensity(SourceDensity):
    """
    Uniform density over an n-dimensional box, parametrized by center and half-width per dimension.

    Parameters: [c_0, w_0, c_1, w_1, ..., c_{dim-1}, w_{dim-1}]
        c_i : center in dimension i
        w_i : half-width in dimension i  (full width = 2*w_i)

    Optionally wraps around the domain in periodic dimensions, e.g. for a cylinder
    where v ∈ [0, 1) is periodic.
    """
    n_dim: int
    domain_low: np.ndarray                        # (dim,) lower bound of parameter domain
    domain_high: np.ndarray                       # (dim,) upper bound of parameter domain
    periodic: Optional[List[bool]] = None         # (dim,) True if dimension is periodic
    prior: Optional[Prior] = None
    fixed_params: Optional[np.ndarray] = None     # [c_0, w_0, c_1, w_1, ...]

    def dim(self):
        return self.n_dim

    def _unpack_params(self, params):
        if self.fixed_params is not None:
            return self.fixed_params
        if params is None:
            raise ValueError("UniformSourceDensity: params must be provided unless fixed_params is set.")
        params = np.asarray(params)
        if params.shape[0] != 2 * self.n_dim:
            raise ValueError(
                f"UniformSourceDensity expected {2 * self.n_dim} parameters (c_i, w_i), got {params.shape[0]}"
            )
        return params

    def _get_centers_and_halfwidths(self, params):
        centers     = params[0::2]   # (dim,)
        halfwidths  = params[1::2]   # (dim,)
        if np.any(halfwidths <= 0):
            raise ValueError("UniformSourceDensity: all half-widths must be > 0.")
        return centers, halfwidths

    def _periodic_distance(self, x_i: np.ndarray, c: float, w: float, period: float) -> np.ndarray:
        """
        Signed distance from center c in a periodic dimension with given period.
        Returns |x - c| wrapped to [-period/2, period/2].
        """
        diff = (x_i - c) % period          # in [0, period)
        diff = np.where(diff > period / 2, diff - period, diff)
        return np.abs(diff)

    def evaluate(self, x, params=None) -> np.ndarray:
        """
        Evaluate uniform density at points x.

        Parameters
        ----------
        x : (N, dim)
        params : [c_0, w_0, ..., c_{dim-1}, w_{dim-1}]

        Returns
        -------
        (N,) density values
        """
        params = self._unpack_params(params)
        x = np.atleast_2d(x)

        if x.shape[1] != self.n_dim:
            raise ValueError(f"x must have shape (N, {self.n_dim}), got {x.shape}")

        centers, halfwidths = self._get_centers_and_halfwidths(params)

        domain_low  = np.asarray(self.domain_low)
        domain_high = np.asarray(self.domain_high)
        periods     = domain_high - domain_low
        periodic    = self.periodic if self.periodic is not None else [False] * self.n_dim

        # Build support mask (N, dim)
        inside = np.ones((x.shape[0], self.n_dim), dtype=bool)
        for i in range(self.n_dim):
            if periodic[i]:
                dist = self._periodic_distance(x[:, i], centers[i], halfwidths[i], periods[i])
                inside[:, i] = dist <= halfwidths[i]
            else:
                inside[:, i] = np.abs(x[:, i] - centers[i]) <= halfwidths[i]

        mask = np.all(inside, axis=1)   # (N,)

        # Volume: for periodic dims the effective width is clipped to the period
        effective_widths = np.where(
            periodic,
            np.minimum(2 * halfwidths, periods),
            2 * halfwidths
        )
        volume = np.prod(effective_widths)

        out = np.zeros(x.shape[0])
        out[mask] = 1.0 / volume
        return out

    def sample(self, n, params=None) -> np.ndarray:
        """
        Draw n samples from the uniform box.

        Returns
        -------
        (n, dim)
        """
        params = self._unpack_params(params)
        centers, halfwidths = self._get_centers_and_halfwidths(params)

        domain_low  = np.asarray(self.domain_low)
        domain_high = np.asarray(self.domain_high)
        periods     = domain_high - domain_low
        periodic    = self.periodic if self.periodic is not None else [False] * self.n_dim

        # Sample uniformly in [-w_i, w_i] around each center, then wrap if periodic
        offsets = (2 * np.random.rand(n, self.n_dim) - 1) * halfwidths[None, :]  # (n, dim)
        samples = centers[None, :] + offsets                                       # (n, dim)

        for i in range(self.n_dim):
            if periodic[i]:
                samples[:, i] = (samples[:, i] - domain_low[i]) % periods[i] + domain_low[i]
            else:
                samples[:, i] = np.clip(samples[:, i], domain_low[i], domain_high[i])

        return samples

    def get_function_type(self):
        return FunctionType.FIXED if self.fixed_params is not None else FunctionType.PARAMETRIC

    def get_n_params(self):
        return 0 if self.fixed_params is not None else self.n_dim * 2

    def get_parameter_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        domain_low  = np.asarray(self.domain_low)
        domain_high = np.asarray(self.domain_high)
        periods     = domain_high - domain_low
        bounds = []
        for i in range(self.n_dim):
            bounds.append((domain_low[i], domain_high[i]))          # center
            bounds.append((0, periods[i] / 2))                      # half-width
        return bounds

    def get_prior(self):
        return self.prior


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
        return None if (self.fixed_params is not None) else self.prior


@dataclass(frozen=True)
class SplineSourceDensity(SourceDensity):

    #TODO: this needs some fix regarding positivity (density values can be negative)

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