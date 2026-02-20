"""
Test the functionalities of the plot 2d functionalities
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.bayes_shape_fitting.inference.likelihoods import MarginalLikelihoodFixed, MarginalLikelihood
from src.bayes_shape_fitting.inference.posterior import PosteriorFixed, MarginalPosterior
from src.bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from src.bayes_shape_fitting.visualization.animation_2d import ShapePlotterAnimations
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
    return BetaSourceDensity(fixed_params=np.array([2,7]))

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_sigma=0.1)

@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)

class TestPlots:

    def test_shape_plot(self, shape, theta):
        plotter = ShapePlotter2D()
        plotter.plot_shape(shape, theta, n_points=500)
        plt.show()

    def test_observation_plot(self, shape, theta, measurement):
        plotter = ShapePlotter2D()
        plotter.plot_shape(shape, theta)
        plotter.plot_observations(measurement)
        plt.show()

    def test_plot_source_density(self, shape, source_density, theta):
        plotter = ShapePlotter2D()
        fig, ax = plotter.plot_shape(shape, theta)
        plotter.plot_source_density(shape, source_density, source_density.fixed_params, ax=ax, plot_on_shape=True, shape_params= theta)
        plotter.plot_source_density(shape, source_density, source_density.fixed_params)
        plt.show()

    def test_plot_likelihood(self, shape, theta, source_density, noise_function, measurement):
        plotter = ShapePlotter2D()
        likelihood = MarginalLikelihoodFixed(shape, source_density, noise_function)
        plotter.plot_shape(shape, theta)
        plotter.plot_likelihood(likelihood = likelihood, theta = theta, ax = plt.gca(), x_min=-4,x_max=4,y_min=-4,y_max=4, plot_measurement=True, measurement=measurement)
        plt.show()

    def test_animation(self, shape, theta, init_theta, source_density, noise_function, measurement):
        # here we test the basic animation, for this we need to do a fit and get the param history
        likelihood = MarginalLikelihoodFixed(shape=shape, source_density=source_density, noise_function=noise_function)
        posterior = PosteriorFixed(shape=shape, measurement=measurement, likelihood=likelihood)
        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
        result = optimizer.optimize(posterior, init_theta)
        param_history = result.param_history

        # now we plot the basic setup
        plotter = ShapePlotter2D()
        fig, ax, _ = plotter.plot_shape(shape, theta, label="Ground Truth")
        plotter.plot_shape(shape, result.theta_map, ax=ax, label="Fitted Curve", color="green")
        plotter.plot_shape(shape, init_theta, ax=ax, label="Initial Fit", color="orange")
        plotter.plot_observations(observations=measurement, ax=ax)

        animator = ShapePlotterAnimations()
        animator.animate_param_history(shape, param_history, ax = ax)
        #plt.show()

    def test_source_density_animation(self, shape, theta, init_theta, source_density, noise_function, measurement):
        # here we test the basic animation, for this we need to do a fit and get the param history
        # we define the source_density to be variable
        source_density_var = BetaSourceDensity()

        likelihood = MarginalLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)
        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
        bounds = shape.bounds + [(0.01, np.inf), (0.01, np.inf)]
        result = optimizer.optimize(posterior, np.concatenate([init_theta,source_density.fixed_params]), bounds=bounds)
        param_history = result.param_history

        shape_param_hist = [r[:shape.get_n_params()] for r in param_history]
        source_param_hist = [r[shape.get_n_params():] for r in param_history]

        # now we plot the basic setup
        plotter = ShapePlotter2D()
        fig, ax, _ = plotter.plot_shape(shape, theta, label="Ground Truth")
        plotter.plot_shape(shape, result.theta_map[:shape.get_n_params()], ax=ax, label="Fitted Curve", color="green")
        plotter.plot_shape(shape, init_theta, ax=ax, label="Initial Fit", color="orange")
        plotter.plot_observations(observations=measurement, ax=ax)

        animator = ShapePlotterAnimations()
        animator.animate_param_history(shape, shape_param_hist, ax=ax, show = True)

        animator.animate_source_density(shape, source_density_var, source_param_hist, show = True)