import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
from src.bayes_shape_fitting.inference.priors import Prior
from enum import Enum
from abc import ABC, abstractmethod

class FunctionType(Enum):
    FIXED = "fixed"  # Function is known, no parameters
    PARAMETRIC = "parametric"  # Function has learnable parameters

class NoiseFunction(ABC):
    """
    Abstract base class for parametric or fixed probability densities.
    Supports evaluation and sampling.
    """

    @property
    @abstractmethod
    def dim_param(self) -> int:
        """Dimensionality k of the parameter domain."""
        pass

    @property
    @abstractmethod
    def dim_ambient(self) -> int:
        """Dimensionality n of the ambient domain"""

    @abstractmethod
    def evaluate(self, t: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns covariance matrices (M, n, n), where M = len(t)"""
        pass

    @abstractmethod
    def get_function_type(self) -> FunctionType:
        pass

    @abstractmethod
    def get_n_params(self) -> int:
        """
        Number of (learnable) parameters for this model *per location* (or a base count for
        parameter-independent models). If there are no learnable params, return 0.
        """
        pass

    @abstractmethod
    def get_prior(self) -> Optional[Prior]:
        """Return the prior over parameters, or None if not used."""
        pass

    # ----- helpers -----

    def _ensure_t_shape(self, t: np.ndarray) -> np.ndarray:
        """
        Ensure t has shape (M, dim_param).
        Accepts (M,) for dim_param == 1.
        """
        t = np.asarray(t)
        if t.ndim == 1:
            if self.dim_param != 1:
                raise ValueError(
                    f"t must have shape (M, {self.dim_param}) but got 1D array with shape {t.shape}."
                )
            return t.reshape(-1, 1)
        if t.ndim != 2 or t.shape[1] != self.dim_param:
            raise ValueError(
                f"t must have shape (M, {self.dim_param}); got {t.shape}."
            )
        return t

    def _eye_stack(self, M: int) -> np.ndarray:
        """Return an array of M copies of I_n with shape (M, n, n)."""
        n = self.dim_ambient
        I = np.eye(n)
        return np.broadcast_to(I, (M, n, n)).copy()
        # copy() to avoid accidental writes to a broadcasted view


# ------------------------------------------------------------
# Fixed noise function (user-provided function)
# ------------------------------------------------------------

@dataclass(frozen=True)
class FixedNoiseFunction(NoiseFunction):
    """
    Known noise covariance function (no learnable parameters).

    Parameters
    ----------
    noise_function : Callable[[np.ndarray], np.ndarray]
        A function that takes t with shape (M, k) (or (M,) if k=1) and returns (M, n, n).
    dim_param : int
        k, dimension of parameter domain.
    dim_ambient : int
        n, dimension of ambient space (covariance matrices are n×n).
    """
    noise_function: Callable[[np.ndarray], np.ndarray]
    dim_param: int
    dim_ambient: int

    def dim_param(self) -> int:
        return self.dim_param

    def dim_ambient(self) -> int:
        return self.dim_ambient

    def evaluate(self, t: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        T = self._ensure_t_shape(t)
        cov = np.asarray(self.noise_function(T))
        expected = (T.shape[0], self.dim_ambient, self.dim_ambient)
        if cov.shape != expected:
            raise ValueError(f"noise_function must return shape {expected}, got {cov.shape}.")
        return cov

    def get_function_type(self) -> FunctionType:
        return FunctionType.FIXED

    def get_n_params(self) -> int:
        return 0

    def get_prior(self) -> Optional[Prior]:
        return None


# ------------------------------------------------------------
# Isotropic noise: Σ(t) = σ^2 I
# Parameterization: params = [log(σ^2)]
# ------------------------------------------------------------

@dataclass(frozen=False)
class IsotropicNoise(NoiseFunction):
    """
    Constant isotropic noise: Σ(t) = σ^2 I_n

    Parameters
    ----------
    dim_param : int
        Parameter domain dimension (k).
    dim_ambient : int
        Ambient dimension (n).
    prior : Optional[Prior]
        Prior over log(σ^2). None means no prior is used.
    fixed_sigma : Optional[float]
        If provided, σ is fixed and no learnable parameter is needed.
    """
    dim_param: int
    dim_ambient: int
    prior: Optional[Prior] = None
    fixed_sigma: Optional[float] = None

    def dim_param(self) -> int:
        return self.dim_param

    def dim_ambient(self) -> int:
        return self.dim_ambient

    def evaluate(self, t: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        T = self._ensure_t_shape(t)
        M = T.shape[0]

        if params is None:
            if self.fixed_sigma is None:
                raise ValueError("IsotropicNoise: provide params=sigma or set fixed_sigma.")
            sigma_sq = float(self.fixed_sigma) ** 2
        else:
            params = np.asarray(params, dtype=float)
            if params.ndim == 0:
                sigma = float(params)
            else:
                if params.shape[0] < 1:
                    raise ValueError("IsotropicNoise: params must have at least one element [log(sigma^2)].")
                sigma = float(params[0])
            sigma_sq = sigma**2

        # Build (M, n, n) = sigma^2 * I
        cov = self._eye_stack(M)
        cov *= sigma_sq
        return cov

    def get_function_type(self) -> FunctionType:
        return FunctionType.FIXED if self.fixed_sigma is not None else FunctionType.PARAMETRIC

    def get_n_params(self) -> int:
        return 0 if self.fixed_sigma is not None else 1

    def get_prior(self) -> Optional[Prior]:
        return None if self.fixed_sigma is not None else self.prior


# ------------------------------------------------------------
# Parameter-dependent isotropic: σ(t) = exp(a + b t), Σ(t) = σ(t)^2 I
# params = [a, b]
# ------------------------------------------------------------

@dataclass(frozen=True)
class ParameterDependentNoise(NoiseFunction):
    """
    Noise varying with the parameter t (assumed 1D): σ(t) = exp(a + b·t), Σ(t) = σ(t)^2 I_n

    Parameters
    ----------
    dim_ambient : int
        Ambient dimension n.
    prior : Optional[Prior]
        Prior over [a, b]. None means no prior used.
    """
    dim_ambient: int
    prior: Optional[Prior] = None

    @property
    def dim_param(self) -> int:
        return 1

    def dim_ambient(self) -> int:
        return self.dim_ambient

    def evaluate(self, t: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        if params is None:
            raise ValueError("ParameterDependentNoise requires params = [a, b].")

        T = self._ensure_t_shape(t)  # (M, 1)
        M = T.shape[0]
        params = np.asarray(params, dtype=float)
        if params.shape[0] < 2:
            raise ValueError("ParameterDependentNoise: params must be [a, b].")

        a, b = float(params[0]), float(params[1])
        sigmas = np.exp(a + b * T[:, 0])           # (M,)
        sigma_sq = sigmas ** 2                     # (M,)

        cov = self._eye_stack(M)                   # (M, n, n)
        cov *= sigma_sq[:, None, None]             # broadcast across matrices
        return cov

    def get_function_type(self) -> FunctionType:
        return FunctionType.PARAMETRIC

    def get_n_params(self) -> int:
        return 2

    def get_prior(self) -> Optional[Prior]:
        return self.prior


# ------------------------------------------------------------
# Anisotropic full covariance via Cholesky factor L (lower-triangular)
# params encodes L with diagonal in log-space: for i>=j, in row-major lower-tri order
# parameter_dependent=True allows per-t parameter vectors of length n_chol each.
# ------------------------------------------------------------

@dataclass(frozen=True)
class AnisotropicNoise(NoiseFunction):
    """
    Full covariance noise Σ(t; L) with Cholesky factor L (lower triangular).
    Diagonal entries parameterized as exp(·) to ensure positive definiteness.

    Parameters
    ----------
    dim_param : int
        Parameter domain dimension k (usually 1).
    dim_ambient : int
        Ambient dimension n (size of covariance matrices).
    prior : Optional[Prior]
        Prior over the Cholesky parameters. None means no prior used.
    parameter_dependent : bool
        If True, expect params shaped (M, n_chol) to vary with t.
        If False, expect a single vector (n_chol,) applied to all t.
    """
    dim_param: int
    dim_ambient: int
    prior: Optional[Prior] = None
    parameter_dependent: bool = False

    def dim_param(self) -> int:
        return self.dim_param

    def dim_ambient(self) -> int:
        return self.dim_ambient

    def get_n_params(self) -> int:
        d = self.dim_ambient
        return d * (d + 1) // 2  # number of elements in lower triangular

    def _build_L_from_params(self, p: np.ndarray) -> np.ndarray:
        """Convert vector p (n_chol,) to lower-triangular L (n, n), with diag = exp(·)."""
        d = self.dim_ambient
        L = np.zeros((d, d), dtype=float)
        idx = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    L[i, j] = np.exp(p[idx])  # positive diagonal
                else:
                    L[i, j] = p[idx]
                idx += 1
        return L

    def evaluate(self, t: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        if params is None:
            raise ValueError("AnisotropicNoise requires params (n_chol,) or (M, n_chol) if parameter_dependent=True.")

        T = self._ensure_t_shape(t)
        M = T.shape[0]
        n_chol = self.get_n_params()

        P = np.asarray(params, dtype=float)

        if self.parameter_dependent:
            # Accept (n_chol,) (broadcast) or (M, n_chol)
            if P.ndim == 1 and P.shape[0] == n_chol:
                P = np.broadcast_to(P, (M, n_chol))
            elif P.ndim == 2 and P.shape == (M, n_chol):
                pass
            else:
                raise ValueError(
                    f"AnisotropicNoise (parameter_dependent=True): params must be (n_chol,) or (M, n_chol), got {P.shape}."
                )

            cov = np.empty((M, self.dim_ambient, self.dim_ambient), dtype=float)
            for i in range(M):
                L = self._build_L_from_params(P[i])
                cov[i] = L @ L.T
            return cov

        # parameter_dependent == False: expect single vector (n_chol,)
        if P.ndim != 1 or P.shape[0] != n_chol:
            raise ValueError(
                f"AnisotropicNoise: params must be a vector of length {n_chol}, got shape {P.shape}."
            )
        L = self._build_L_from_params(P)
        cov_single = L @ L.T
        cov = np.broadcast_to(cov_single, (M, self.dim_ambient, self.dim_ambient)).copy()
        return cov

    def get_function_type(self) -> FunctionType:
        return FunctionType.PARAMETRIC

    def get_prior(self) -> Optional[Prior]:
        return self.prior

