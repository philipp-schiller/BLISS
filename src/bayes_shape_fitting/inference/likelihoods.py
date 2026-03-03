import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar

from src.bayes_shape_fitting.inference.noise_functions import NoiseFunction
from src.bayes_shape_fitting.inference.source_densitites import SourceDensity
from src.bayes_shape_fitting.models.shapes.base_shape import Shape
from src.bayes_shape_fitting.utils.integration import Integrator


# define the abstract Likelihood class
class Likelihood(ABC):
    """
    Abstract base class for likelihood models
        L(X | theta)
    Accepts optionally also source_density, noise_function
    """

    @abstractmethod
    def log_likelihood(
        self,
        shape: Shape,
        theta: np.ndarray,
        measurement: np.ndarray,
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
            theta: np.ndarray,
            measurement: np.ndarray,
            *args,
    ) -> np.ndarray:

        source_fixed = True if self.num_source_params == 0 else False
        noise_fixed = True if self.num_noise_params == 0 else False

        assert len(theta) == self.num_shape_params + self.num_source_params + self.num_noise_params

        shape_params = theta[0:self.num_shape_params]
        source_params = theta[
                        self.num_shape_params: self.num_shape_params + self.num_source_params] if not source_fixed else None
        noise_params = theta[
                       self.num_shape_params + self.num_source_params: self.num_shape_params + self.num_source_params + self.num_noise_params] if not noise_fixed else None

        noise_function = self.noise_function
        source_density = self.source_density

        param_dim = self.shape.parameter_dim  # 1 for curves, 2 for surfaces
        domain_low = np.atleast_1d(self.shape.domain_low)
        domain_high = np.atleast_1d(self.shape.domain_high)
        n_per_dim = self.num_samples

        # we only consider t_vals where the source density is bigger than epsilon for efficiency
        t_vals_full = self.shape.sample_parameter_space_grid(n_points_per_dim=self.num_samples)
        pi_vals_full = source_density.evaluate(t_vals_full, params=source_params)

        eps = 0
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
            sigmasq = 2 * np.pi * np.sqrt(det_cov)

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
