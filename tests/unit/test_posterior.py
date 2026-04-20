"""
Test the functionalities of posterior functions
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from tests.fixtures.sample_data import simulate_measurement

np.random.seed(43)

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
def source_density(shape):
    dim = shape.parameter_dim
    return UniformSourceDensity(n_dim=dim, fixed_params=np.array(dim * [0.3, 0.1]), domain_low=shape.domain_low,
                                domain_high=shape.domain_high, periodic=shape.periodic)

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)

@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)

# ---------------------------------------------------------------------
# Marginal likelihood
# ---------------------------------------------------------------------

class TestMarginalPosterior:

    def test_log_posterior(self, shape, theta, measurement, source_density, noise_function):
        """
        Test the basic log_likelihood function
        """
        likelihood = MarginalLikelihood(shape=shape,source_density=source_density, noise_function=noise_function)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood = likelihood)
        print(posterior.log_posterior(theta))
