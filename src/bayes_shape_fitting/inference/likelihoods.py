import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.integrate import simpson

from src.bayes_shape_fitting.inference.noise_functions import NoiseFunction
from src.bayes_shape_fitting.inference.source_densitites import SourceDensity
from src.bayes_shape_fitting.models.shapes.base_shape import Shape



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

@dataclass(frozen=True)
class MarginalLikelihoodFixed(Likelihood):
    """
    Computes the marginal likelihood over source_parameters for a fixed SourceDensity and NoiseFunction
    """
    source_density: SourceDensity
    noise_function: NoiseFunction
    num_samples: int = 500 #number of samples on the shape to compute the surface integral

    def __post_init__(self) -> None:
        """
        We validate that source_density and noise_function are FIXED
        """
        assert self.source_density.get_function_type().value == "fixed", "Source density needs to be fixed!"
        assert self.noise_function.get_function_type().value == "fixed", "Noise function needs to be fixed!"

    def log_likelihood(
        self,
        shape: Shape,
        theta: np.ndarray,
        measurement: np.ndarray,
        *args,
    ) -> np.ndarray:

        # compute the marginal integral over all sources
        M = self.num_samples
        N = len(measurement)

        param_samples_full = shape.sample_parameter_space_uniform(size=M)

        # based on this, we compute the corresponding pi_vals with the source density
        pi_vals_full = self.source_density.evaluate(param_samples_full)

        # we only consider param_samples where the source density is bigger than epsilon for efficiency
        eps = 1e-8
        param_samples = param_samples_full[pi_vals_full > eps]
        pi_vals = pi_vals_full[pi_vals_full > eps]

        # --- Noise covariance ---
        cov = self.noise_function.evaluate(param_samples)  # shape (M,k,k)
        det_cov = np.linalg.det(cov)  # shape (M,)
        # Normalization constant of multivariate Gaussian
        norm = (2 * np.pi) ** (cov.shape[-1] / 2) * np.sqrt(det_cov)

        # Instead of inverse, use solve for Mahalanobis distance
        # shape (N, M, k)
        sampled = shape.parametrization(param_samples, theta)
        diff = measurement[:, None, :] - sampled[None, :, :]
        # Solve cov @ x = diff
        # reshape to (M, k, N)
        diff_T = diff.transpose(1, 2, 0)
        solved = np.linalg.solve(cov, diff_T)  # (M, k, N)
        solved = solved.transpose(2, 0, 1)  # (N, M, k)

        # Mahalanobis: Σ_i (diff * solved)
        dist_sq = np.sum(diff * solved, axis=-1)  # (N, M)

        # Integrand: shape (M, N)
        integrand = np.exp(-0.5 * dist_sq) / norm * pi_vals[None, :]

        # now we need to integrate

        if shape.parameter_dim == 1:
            # here it's just a 1D integral
            values = simpson(integrand, param_samples[:, 0])
            return values

        if shape.parameter_dim == 2:
            # for now, we use a simple Riemann sum to integrate over the 2D grid

            # we reshape the integral
            integrand = integrand.reshape(N, M, M)

            # take the periodicity into account:

            def compute_spacing(low, high, M, periodic):
                if periodic:
                    return (high - low) / M  # endpoint=False
                else:
                    return (high - low) / (M - 1)  # endpoint=True

            u0, v0 = shape.domain_low
            u1, v1 = shape.domain_high

            du = compute_spacing(u0, u1, M, periodic=shape.periodic[0])
            dv = compute_spacing(v0, v1, M, periodic=shape.periodic[1])

            values = np.sum(integrand, axis=(1, 2)) * du * dv
            return values

        else:
            raise ValueError(f"Unsupported parameter_dim={shape.parameter_dim}. Only 1 or 2 are recognized.")



