from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from .noise_functions import NoiseFunction
from .source_densitites import SourceDensity, FunctionType
from ..models.shapes.base_shape import Shape

# define the abstract Likelihood class
class Likelihood(ABC):
    """
    Abstract base class for likelihood models
        L(X | theta)
    Accepts optionally also source_density, noise_function
    """

    shape: Shape

    @abstractmethod
    def log_likelihood(
        self,
        measurement: np.ndarray,
        shape_params: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        pass


@dataclass()
class MarginalLikelihood(Likelihood):
    """
    Computes the marginal likelihood based on a source density and noise_function
    """
    shape: Shape
    source_density: SourceDensity
    noise_function: NoiseFunction
    num_samples: int = 500 # number of samples on the shape to compute the surface integral


    def __post_init__(self) -> None:
        """
        We check which of the source density and noise function are fixed and store the number of their parameters
        """
        self.num_shape_params = self.shape.get_n_params()
        self.num_source_params = self.source_density.get_n_params()
        self.num_noise_params = self.noise_function.get_n_params()

    def log_likelihood(
            self,
            measurement: np.ndarray,
            shape_params: np.ndarray,
            source_params: np.ndarray = None,
            noise_params: np.ndarray = None,
            *args,
    ) -> np.ndarray:

        # first assert, if the source-density and noise function
        if source_params is None:
            assert self.source_density.get_function_type() == FunctionType.FIXED, "Source density needs parameters!"
        if noise_params is None:
            assert self.noise_function.get_function_type() == FunctionType.FIXED, "Noise functions needs parameters!"

        noise_function = self.noise_function
        source_density = self.source_density

        param_dim = self.shape.parameter_dim  # 1 for curves, 2 for surfaces
        domain_low = np.atleast_1d(self.shape.domain_low)
        domain_high = np.atleast_1d(self.shape.domain_high)
        n_per_dim = self.num_samples

        # we only consider t_vals where the source density is bigger than epsilon for efficiency
        t_vals_full = self.shape.sample_parameter_space_grid(n_points_per_dim=self.num_samples)
        pi_vals_full = source_density.evaluate(t_vals_full, params=source_params)

        eps = 0.0001
        mask = pi_vals_full >= eps
        t_vals = t_vals_full[mask]
        pi_vals = pi_vals_full[mask]

        if param_dim == 1:
            # --- 1D case ---

            covariances = noise_function.evaluate(t_vals, params=noise_params)
            inv_covs = np.linalg.inv(covariances)
            det_cov = np.linalg.det(covariances)
            sigmasq = 2 * np.pi * np.sqrt(det_cov)

            # compute the source points
            sampled_points = self.shape.evaluate(t_vals, shape_params)

            # compute the mahalanobis distances to the source points
            diff = measurement[:, None, :] - sampled_points[None, :, :]
            dist_sq = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)
            integrand = np.exp(-0.5 * dist_sq) / sigmasq[None, :] * pi_vals[None, :]

            # Put back onto full grid and integrate
            integrand_full = np.zeros((measurement.shape[0], n_per_dim))
            integrand_full[:, mask] = integrand
            values = simpson(integrand_full, x=t_vals_full[:, 0], axis=-1)

        elif param_dim == 2:
            # --- 2D case ---

            u_vals = np.linspace(domain_low[0], domain_high[0], n_per_dim)
            v_vals = np.linspace(domain_low[1], domain_high[1], n_per_dim)

            covariances = noise_function.evaluate(t_vals, params=noise_params)
            inv_covs = np.linalg.inv(covariances)
            det_cov = np.linalg.det(covariances)

            d = measurement.shape[1]  # dimension = 3
            sigmasq = (2 * np.pi) ** (d / 2) * np.sqrt(det_cov)
            sampled_points = self.shape.evaluate(t_vals, shape_params)

            diff = measurement[:, None, :] - sampled_points[None, :, :]
            dist_sq = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)
            integrand = np.exp(-0.5 * dist_sq) / sigmasq[None, :] * pi_vals[None, :]  # (M, N')

            # Put back onto full (n_v, n_u) grid and integrate via nested Simpson
            integrand_full = np.zeros((measurement.shape[0], n_per_dim * n_per_dim))
            integrand_full[:, mask] = integrand
            integrand_grid = integrand_full.reshape(measurement.shape[0], n_per_dim, n_per_dim)
            # integrand_grid is (M, n_v, n_u) — integrate u first, then v
            inner = simpson(integrand_grid, x=u_vals, axis=-1)  # (M, n_v)
            values = simpson(inner, x=v_vals, axis=-1)  # (M,)

        else:
            raise NotImplementedError(f"log_likelihood not implemented for parameter_dim={param_dim}")
        return np.log(values + 1e-200)

#-------------------------------------------------------
# define likelihood with respect to the closest point
#-------------------------------------------------------


@dataclass()
class ClosestPointLikelihood(Likelihood):
    shape: Shape
    source_density: SourceDensity
    noise_function: NoiseFunction

    def __post_init__(self) -> None:
        """
        We check which of the source density and noise function are fixed and store the number of their parameters
        """
        self.num_shape_params = self.shape.get_n_params()
        self.num_source_params = self.source_density.get_n_params()
        self.num_noise_params = self.noise_function.get_n_params()

    def log_likelihood(
            self,
            measurement: np.ndarray,
            shape_params: np.ndarray,
            source_params: np.ndarray = None,
            noise_params: np.ndarray = None,
            *args,
            **kwargs
    ) -> np.ndarray:
        """
        Compute the log-likelihood of measurements given shape parameters theta.

        For each measurement x_i, we find the closest point γ(t*) on the shape,
        then evaluate:

            log p(x_i | θ) = -½ (x_i - γ(t*))ᵀ Σ(t*)⁻¹ (x_i - γ(t*))
                             - ½ log|2π Σ(t*)|
                             + log p(t*)          ← source density term

        Parameters
        ----------
        noise_params
        source_params
        shape_params : np.ndarray
            Shape parameters θ passed to the parametrization.
        measurement : np.ndarray, shape (M, n) or (n,)
            One or more observed points in ambient space.

        Returns
        -------
        log_liks : np.ndarray, shape (M,)
            Per-measurement log-likelihoods.
        """
        noise_function = self.noise_function
        source_density = self.source_density

        # noise_params: pulled from kwargs, else fallback to self attribute
        noise_params = kwargs.get("noise_params", getattr(self, "noise_params", None))

        measurement = np.atleast_2d(measurement)  # (M, n)
        M = measurement.shape[0]
        n = measurement.shape[1]

        # --- Closest points on shape ---
        # closest_points: (M, n),  t_vals: (M, k)
        closest_points, t_vals = self.shape.get_closest_point(params=shape_params, x=measurement, return_params=True, tol=0.3, n_restarts=1
                                                              )

        # --- Covariances at closest parameter values ---
        # covariances: (M, n, n)
        covariances = noise_function.evaluate(t_vals, params=noise_params)
        inv_covs = np.linalg.inv(covariances)  # (M, n, n)
        sign, log_det_cov = np.linalg.slogdet(covariances)  # (M,) — numerically stable

        # --- Mahalanobis distance squared ---
        # diff: (M, n)
        diff = measurement - closest_points

        # dist_sq[i] = diff[i]ᵀ Σ⁻¹[i] diff[i]
        # einsum: 'mi, mij, mj -> m'
        dist_sq = np.einsum("mi, mij, mj -> m", diff, inv_covs, diff)  # (M,)

        # --- Gaussian log-likelihood ---
        # log p(x | t*) = -½ (dist_sq + n*log(2π) + log|Σ|)
        log_liks = -0.5 * (dist_sq + n * np.log(2 * np.pi) + log_det_cov)  # (M,)

        return log_liks  # shape (M,)