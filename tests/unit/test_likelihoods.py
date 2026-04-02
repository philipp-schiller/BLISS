"""
Test the functionalities of likelihood functions
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
import time

from src.bayes_shape_fitting.inference.likelihoods import MarginalLikelihood, ClosestPointLikelihood
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.source_densitites import BetaSourceDensity, UniformSourceDensity
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
def source_density(shape):
    dim = shape.parameter_dim
    return UniformSourceDensity(n_dim=dim, fixed_params=np.array(dim*[0.3,0.1]), domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)

@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)

# ---------------------------------------------------------------------
# Marginal likelihood
# ---------------------------------------------------------------------

class TestMarginalLikelihood:

    def test_log_likelihood(self, shape, theta, measurement, source_density, noise_function):
        """
        Test the basic log_likelihood function
        """
        likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)
        likelihood_values = likelihood.log_likelihood(measurement = measurement, shape_params=theta)
        assert len(likelihood_values) == len(measurement)

    def test_log_likelihood_variable(self, shape, theta, measurement):
        """
        Test the basic log_likelihood function with variable source-and noise parameters
        """

        dim = shape.parameter_dim
        source_density_var = UniformSourceDensity(n_dim=dim, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)
        source_params = np.array(dim*[0.3,0.1])

        noise_function_var = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim)
        noise_params = np.array([0.1])

        likelihood = MarginalLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function_var)
        likelihood_values = likelihood.log_likelihood(shape_params=theta, source_params=source_params, noise_params=noise_params, measurement=measurement)
        assert len(likelihood_values) == len(measurement)


class TestClosestPointLikelihood:

    def test_log_likelihood_fixed(self, shape, theta, measurement, source_density, noise_function):
        """
        Test the basic log_likelihood function with fixed source_density and noise function
        """
        likelihood = ClosestPointLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)

        start = time.perf_counter()
        likelihood_values = likelihood.log_likelihood(shape_params=theta, measurement=measurement)
        end = time.perf_counter()

        elapsed = end - start
        print(f"log_likelihood computation took {elapsed:.6f} seconds")

        assert len(likelihood_values) == len(measurement)