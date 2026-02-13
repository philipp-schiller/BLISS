"""
Test the fitting of 2D curves
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.bayes_shape_fitting.inference.likelihoods import MarginalLikelihoodFixed
from src.bayes_shape_fitting.inference.posterior import PosteriorFixed
from src.bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from src.bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.source_densitites import BetaSourceDensity
from src.bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from tests.fixtures.sample_data import simulate_measurement

np.random.seed(23)

def _case_id(c):
    return c["name"]

# filter the curves
curve_cases = [
    c for c in shape_cases if "curve" in c["markers"]
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
def source_density():
    return BetaSourceDensity(fixed_params=np.array([3,2]))

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_sigma=0.1)

@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)

class TestMAP:

    def test_map_fixed(self, shape, measurement, theta, init_theta, source_density, noise_function):
        likelihood = MarginalLikelihoodFixed(shape=shape, source_density=source_density, noise_function=noise_function)
        posterior = PosteriorFixed(shape=shape, measurement=measurement, likelihood=likelihood)

        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
        result = optimizer.optimize(posterior, init_theta)
        print(f"MAP estimate: {result.theta_map}")
        print(f"Log posterior: {result.log_posterior}")

        print(posterior.log_posterior(theta = init_theta))
        print(posterior.log_posterior(theta = result.theta_map))

        if True:
            plotter = ShapePlotter2D()
            fig, ax, _ = plotter.plot_shape(shape, theta, label = "Ground Truth")
            plotter.plot_shape(shape, result.theta_map, ax = ax, label = "Fitted Curve", color = "green")
            plotter.plot_shape(shape, init_theta, ax = ax, label = "Initial Fit", color = "orange")
            plotter.plot_observations(observations=measurement, ax=ax)
            plt.legend()

            # plot the likelihood of the original and the solution
            plt.figure()
            plotter.plot_likelihood(likelihood = likelihood, theta = theta, ax = plt.gca(), x_min=-4,x_max=4,y_min=-4,y_max=4, plot_measurement=True, measurement=measurement)

            plt.figure()
            plotter.plot_likelihood(likelihood=likelihood, theta=result.theta_map, ax=plt.gca(), x_min=-4, x_max=4, y_min=-4, y_max=4, plot_measurement=True, measurement=measurement)
            plt.show()