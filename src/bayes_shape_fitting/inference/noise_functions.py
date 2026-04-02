import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
from src.bayes_shape_fitting.inference.priors import Prior
from typing import Callable, Optional, Sequence, Tuple, List, Dict
from enum import Enum
from abc import ABC, abstractmethod

from src.bayes_shape_fitting.inference.source_densitites import FunctionType
from src.bayes_shape_fitting.models.shapes.base_shape import Shape


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
# Parameterization: params = σ^2
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
    fixed_params : Optional[float]
        If provided, σ is fixed and no learnable parameter is needed.
    """
    dim_param: int
    dim_ambient: int
    prior: Optional[Prior] = None
    fixed_params: Optional[float] = None

    def dim_param(self) -> int:
        return self.dim_param

    def dim_ambient(self) -> int:
        return self.dim_ambient

    def evaluate(self, t: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        T = self._ensure_t_shape(t)
        M = T.shape[0]

        if params is None:
            if self.fixed_params is None:
                raise ValueError("IsotropicNoise: provide params=sigma or set fixed_sigma.")
            sigma_sq = float(self.fixed_params)# ** 2
        else:
            params = np.asarray(params, dtype=float)
            if params.ndim == 0:
                sigma = float(params)
            else:
                if params.shape[0] < 1:
                    raise ValueError("IsotropicNoise: params must have at least one element [log(sigma^2)].")
                sigma = float(params[0])
            sigma_sq = sigma#**2

        # Build (M, n, n) = sigma^2 * I
        cov = self._eye_stack(M)
        cov *= sigma_sq
        return cov

    def get_function_type(self) -> FunctionType:
        return FunctionType.FIXED if self.fixed_params is not None else FunctionType.PARAMETRIC

    def get_n_params(self) -> int:
        return 0 if self.fixed_params is not None else 1

    def get_prior(self) -> Optional[Prior]:
        return None if self.fixed_params is not None else self.prior


@dataclass(frozen=True)
class LaserDependentNoise(NoiseFunction):
    """
    Noise dependent on a Laser-scanner s

    Parameters are \sigma_0 and gamma
    """
    dim_param: int = 1
    dim_ambient: int = 2
    prior: Optional[Prior] = None
    fixed_params: Optional[np.array] = None
    shape: Shape = None
    scanner: Optional[np.array] = None

    def evaluate(self, t: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:

        t = self._ensure_t_shape(t)

        if params is None:
            if self.fixed_params is None:
                raise ValueError("provide params or set fixed_sigma.")


        # --- resolve parameters -------------------------------------------------

        if params is None:
            if self.fixed_params is None:
                raise ValueError("LaserDependentNoise: provide params or set fixed_params.")
            params = self.fixed_params

        # ellipse parameters + sigma0 + gamma
        # assume structure: [a, b, x0, y0, phi, sigma_0, gamma]
        ellipse_params = params[:self.shape.get_n_params()]
        sigma_0, gamma = params[-2:]

        t = np.atleast_1d(t)
        M = len(t)

        # --- compute geometric components --------------------------------------

        # points on ellipse
        points = self.shape.evaluate(t, ellipse_params)

        # vectors from laser
        laser = np.asarray(self.scanner)
        vectors = points - laser
        norms = np.linalg.norm(vectors, axis=1)

        # tangent vectors
        tangents = self.shape.jacobian(t, ellipse_params)[:,:,0]

        # incidence angle (beam vs tangent)
        dot = np.einsum('ij,ij->i', vectors, tangents)
        v_norms = norms
        t_norms = np.linalg.norm(tangents, axis=1)

        cos_angle = dot / (v_norms * t_norms + 1e-12)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angle = np.where(angle > np.pi / 2, np.pi - angle, angle)
        incidence = np.pi / 2 - angle  # final incidence angle

        # rotation of covariance: angle of vector from laser
        rotation_angle = np.arctan2(vectors[:, 1], vectors[:, 0])

        # --- compute laser sigmas ----------------------------------------------

        # radial uncertainty
        sigma_rad = 0.5 * norms * np.tan(gamma)

        # propagation uncertainty
        sigma_prop = sigma_0 + sigma_rad * np.clip(np.tan(incidence), None, 3)

        # --- construct covariance matrices -------------------------------------

        cos_r = np.cos(rotation_angle)
        sin_r = np.sin(rotation_angle)

        # rotation matrices
        R = np.zeros((M, 2, 2))
        R[:, 0, 0] = cos_r
        R[:, 0, 1] = -sin_r
        R[:, 1, 0] = sin_r
        R[:, 1, 1] = cos_r

        # diagonal sigma matrices
        D = np.zeros((M, 2, 2))
        D[:, 0, 0] = sigma_prop
        D[:, 1, 1] = sigma_rad

        # Σ = R D Rᵀ
        cov = np.einsum('mik,mkl,mjl->mij', R, D, R)

        return cov

    def get_function_type(self) -> FunctionType:
        return FunctionType.FIXED if self.fixed_params is not None else FunctionType.PARAMETRIC

    def get_n_params(self) -> int:
        return 0 if self.fixed_params is not None else 2 + self.shape.get_n_params()

    def get_prior(self) -> Optional[Prior]:
        return None if self.fixed_params is not None else self.prior


class PartiallyFixedNoise(NoiseFunction):
    """
    Wraps a NoiseFunction and fixes a subset of its parameters.

    Example:
        # Base noise has parameters: [sigma, alpha, beta]
        # Fix index 1: alpha = 0.2
        wrapped = PartiallyFixedNoise(base, fixed_params={1: 0.2})

        # Now the wrapper expects only [sigma, beta] as params.
        covs = wrapped.evaluate(t, params=np.array([sigma_val, beta_val]))
    """

    def __init__(self, base: NoiseFunction, fixed_params: Dict[int, float]):
        """
        Args:
            base: Underlying NoiseFunction with N parameters.
            fixed_params: Mapping {index: fixed_value}.
        """
        total = base.get_n_params()

        # Validate fixed indices:
        if any(i < 0 or i >= total for i in fixed_params):
            raise ValueError(
                f"Fixed param indices must be between 0 and {total - 1}"
            )

        self._base = base
        self._fixed_params = dict(fixed_params)
        #self.fixed_params = np.array(list(self._fixed_params.values()))
        self.fixed_params = None

        # Indices of parameters that remain free
        self._free_indices = [i for i in range(total) if i not in fixed_params]

    # ------------------------------------------------------------------
    # Required NoiseFunction properties
    # ------------------------------------------------------------------

    @property
    def dim_param(self) -> int:
        """Number of free parameters this wrapped model expects."""
        return len(self._free_indices)

    @property
    def dim_ambient(self) -> int:
        """Ambient dimension (delegated to base)."""
        return self._base.dim_ambient

    # ------------------------------------------------------------------
    # Parameter reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_params(self, params: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Reconstruct full parameter vector by inserting fixed and free params.

        Supports BOTH:
          - free params only  (len = len(self._free_indices))
          - full params       (len = base.get_n_params())
        """

        total = self._base.get_n_params()
        if total == 0:
            return None

        if params is None:
            return None

        params = np.asarray(params)

        # ✅ Case 1: User passed FULL parameter vector → auto-reduce
        if len(params) == total:
            # filter to only free params
            params = params[self._free_indices]
        # ✅ Case 2: User passed FREE parameter vector (expected)
        elif len(params) != len(self._free_indices):
            raise ValueError(
                f"Wrong number of params: expected {len(self._free_indices)} "
                f"free params or {total} full params, got {len(params)}"
            )

        # ✅ Now rebuild full parameter vector
        full = np.empty(total)

        # Insert fixed params
        for idx, val in self._fixed_params.items():
            full[idx] = val

        # Insert free params
        for k, orig_idx in enumerate(self._free_indices):
            full[orig_idx] = params[k]

        return full

    # ------------------------------------------------------------------
    # Core interface delegation
    # ------------------------------------------------------------------

    def evaluate(
        self, t: np.ndarray, params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Evaluate covariance matrices using reconstructed full parameter vector."""
        return self._base.evaluate(t, self._reconstruct_params(params))

    def get_function_type(self) -> FunctionType:
        """If no free params remain, this wrapper is a FIXED-function."""
        if len(self._free_indices) == 0:
            return FunctionType.FIXED
        return FunctionType.PARAMETRIC

    def get_n_params(self) -> int:
        """Return number of free parameters."""
        return len(self._free_indices)

    def get_prior(self) -> Optional[Prior]:
        """Return prior restricted to free parameters."""
        prior = self._base.get_prior()
        if prior is None or len(self._free_indices) == 0:
            return None
        return prior  # Optionally you can remap prior—depends on your design







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

