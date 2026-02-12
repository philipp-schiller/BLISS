"""
Test the functionalities of posterior functions
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.bayes_shape_fitting.inference.likelihoods import MarginalLikelihoodFixed
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.posterior import PosteriorFixed
from src.bayes_shape_fitting.inference.source_densitites import BetaSourceDensity
from src.bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from tests.fixtures.sample_data import simulate_measurement


def _case_id(c):
    return c["name"]

@pytest.fixture(params=shape_cases, ids=_case_id)
def case(request):
    return request.param

@pytest.fixture
def shape(case):
    return case["shape"]

@pytest.fixture
def theta(case):
    return case["theta"]

@pytest.fixture
def source_density():
    return BetaSourceDensity(fixed_params=np.array([1,1]))

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_sigma=0.1)

@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)

# ---------------------------------------------------------------------
# Fixed Marginal likelihood
# ---------------------------------------------------------------------

class TestFixedPosterior:

    def test_log_posterior(self, shape, theta, measurement, source_density, noise_function):
        """
        Test the basic log_likelihood function
        """
        likelihood = MarginalLikelihoodFixed(shape=shape,source_density=source_density, noise_function=noise_function)
        posterior = PosteriorFixed(shape=shape, measurement=measurement, likelihood = likelihood)
        print(posterior.log_posterior(theta))

