"""
Test the fitting of 3D surfaces
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement

np.random.seed(23)

def _case_id(c):
    return c["name"]

# filter the curves
curve_cases = [
    c for c in shape_cases if "surface" in c["markers"]
]

@pytest.fixture(params=curve_cases, ids=_case_id)
def case(request):
    return request.param

@pytest.fixture
def shape(case):
    return case["shape"]

@pytest.fixture
def theta(case):
    return case["theta"]

@pytest.fixture
def init_theta(case):
    return case["init_params"]

@pytest.fixture
def source_density(shape):
    return UniformSourceDensity(n_dim = 2, domain_low=shape.domain_low,
                                domain_high=shape.domain_high, fixed_params=np.array([0.2,0.5, 0.1, 0.6])) # params are [a_0,b_0,a_1,b_1]

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)

@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)

class TestMAP:

    def test_map(self, shape, measurement, theta, init_theta, source_density, noise_function):
        likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=50)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)

        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
        result = optimizer.optimize(posterior, init_theta)
        print(f"MAP estimate: {result.shape_params}")

        print(posterior.log_posterior(shape_params = init_theta))
        print(posterior.log_posterior(shape_params = result.shape_params))

        if True:
            plotter = ShapePlotter3D()
            fig, ax = plotter.plot_shape(shape, theta, label = "Ground Truth")
            plotter.plot_shape(shape, result.shape_params, ax = ax, label = "Fitted Curve", color = "green")
            plotter.plot_shape(shape, init_theta, ax = ax, label = "Initial Fit", color = "orange")
            plotter.plot_observations(observations=measurement, ax=ax)
            plt.legend()
            plt.show()