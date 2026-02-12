import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.integrate import simpson

from src.bayes_shape_fitting.inference.likelihoods import Likelihood, MarginalLikelihoodFixed
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

    @abstractmethod
    def log_posterior(
            self,
            shape: Shape,
            measurement: np.ndarray,
            likelihood: Likelihood,
            prior: Prior = None,
            *args,
            **kwargs
    ) -> np.ndarray:
        pass

@dataclass(frozen=True)
class PosteriorFixed:
    shape: Shape
    measurement: np.ndarray
    likelihood: MarginalLikelihoodFixed
    prior: Prior = None

    def log_posterior(self, theta: np.ndarray) -> float:
        """
        Compute log p(θ| X)

        Accepts either ModelParameters or flat array
        """
        # Prior
        log_prior = 0
        if self.prior is not None:
            log_prior = self.prior.log_prob(params)
            if not np.isfinite(log_prior):
                return -np.inf

        # Likelihood
        log_like = np.sum(
            self.likelihood.log_likelihood(
                theta=theta,
                measurement=self.measurement,
            )
        )

        return log_prior + log_like
