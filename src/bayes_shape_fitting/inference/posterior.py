from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..inference.likelihoods import Likelihood
from ..inference.priors import Prior
from ..models.shapes.base_shape import Shape


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
    normals: np.ndarray = None
    prior: Prior = None


    @abstractmethod
    def log_posterior(
            self,
            theta: np.ndarray,
            *args,
            **kwargs
    ) -> np.ndarray:
        pass

@dataclass(frozen=True)
class MarginalPosterior:
    shape: Shape
    measurement: np.ndarray
    likelihood: Likelihood
    normals: np.ndarray = None
    prior: Prior = None

    def log_posterior(self, shape_params: np.ndarray, source_params: np.array = None, noise_params: np.array = None) -> float:
        """
        Compute log p(θ| X)

        Accepts either ModelParameters or flat array
        """

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
            if not np.isfinite(log_noise_prior):
                return -1e200

        # Likelihood
        log_like = np.sum(self.likelihood.log_likelihood(
            shape_params=shape_params,
            source_params=source_params,
            noise_params=noise_params,
            measurement=self.measurement,
            normals = self.normals
        ))

        if not np.isfinite(log_like):
            print("Warning - Likelihood not finite!")
            return -1e200

        return log_like + log_prior + log_source_prior + log_noise_prior

    """def normalized(self):
        X = self.measurement

        mu = X.mean(axis=0)
        scale = np.std(X) + 1e-12

        X_scaled = (X - mu) / scale

        shape = self.shape

        def gamma_scaled(t, theta):
            return (shape.evaluate(t, theta) - mu) / scale

        def jacobian_scaled(t, theta):
            J = shape.jacobian(t, theta)
            return J / scale

        # scale normals if present (VERY IMPORTANT!)
        normals_scaled = None
        if self.normals is not None:
            normals_scaled = self.normals / (np.linalg.norm(self.normals, axis=1, keepdims=True) + 1e-12)

        # scale likelihood (we adjust noise inside it)
        likelihood_scaled = self.likelihood.scaled(scale)

        return (
            MarginalPosterior(
                shape=self.shape,
                measurement=X_scaled,
                likelihood=likelihood_scaled,
                normals=normals_scaled,
                prior=self.prior
            ),
            mu,
            scale
        )
    """