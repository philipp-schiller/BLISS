import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.integrate import simpson

from src.bayes_shape_fitting.inference.likelihoods import Likelihood, MarginalLikelihood
from src.bayes_shape_fitting.inference.noise_functions import NoiseFunction
from src.bayes_shape_fitting.inference.priors import Prior
from src.bayes_shape_fitting.inference.source_densitites import SourceDensity
from src.bayes_shape_fitting.models.shapes.base_shape import Shape


# define the abstract Posterior class
class Posterior(ABC):
    """
    Abstract base class for Posterior models
        p(theta | X)
    Accepts optionally also source_density, noise_function
    """
    shape: Shape
    measurement: np.ndarray
    likelihood: Likelihood
    prior: Prior = None


    @abstractmethod
    def log_posterior(
            self,
            theta: np.ndarray,
            *args,
            **kwargs
    ) -> np.ndarray:
        pass


@dataclass(frozen = True)
class MarginalPosterior:
    shape: Shape
    measurement: np.ndarray
    likelihood: MarginalLikelihood
    prior: Prior = None

    def log_posterior(self, theta: np.ndarray) -> float:
        """
        Compute log p(θ| X)

        Accepts either ModelParameters or flat array
        """

        # note that theta = (shape_params, source_density_params, noise_function_params)
        # so first, we check if source density and noise function are fixed or not

        source_fixed = True if self.likelihood.num_source_params == 0 else False
        noise_fixed = True if self.likelihood.num_noise_params == 0 else False

        n_shape = self.likelihood.num_shape_params # number of shape params
        n_source = self.likelihood.num_source_params # number of source params
        n_noise = self.likelihood.num_noise_params # number of noise params

        assert len(
            theta) == n_shape + n_source + n_noise, f"Parameter size does not fit the likelihood! {theta},  {n_shape}, {n_source}, {n_noise}"

        shape_params = theta[0:n_shape]
        if not source_fixed:
            source_params = theta[n_shape: n_shape + n_source]
        else:
            source_params = None
        if not noise_fixed:
            noise_params = theta[n_shape + n_source: n_shape + n_source + n_noise]
        else:
            noise_params = None

        # Prior
        log_prior = 0
        if self.prior is not None:
            # this is the shape prior
            log_prior = self.prior.log_prob(shape_params)# * self.measurement.shape[0]
            if not np.isfinite(log_prior):
                return -1e200

        log_source_prior = 0
        # get the source-density prior
        source_prior = self.likelihood.source_density.get_prior()
        if source_prior is not None and source_params is not None:
            log_source_prior = source_prior.log_prob(source_params)
            if not np.isfinite(log_source_prior):
                return -1e200

        # now the noise prior
        log_noise_prior = 0
        noise_prior = self.likelihood.noise_function.get_prior()
        if noise_prior is not None and noise_params is not None:
            log_noise_prior = noise_prior.log_prob(noise_params)
            if not np.isfinite(log_source_prior):
                return -1e200

        # Likelihood
        log_like = np.sum(self.likelihood.log_likelihood(
            theta=theta,
            measurement=self.measurement,
        ))

        if not np.isfinite(log_like):
            print("Warning - Likelihood not finite!")
            return -1e200

        return log_like + log_prior + log_source_prior + log_noise_prior

