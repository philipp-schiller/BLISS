"""
Test the functionalities of the plot 2d functionalities
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from src.bayes_shape_fitting.inference.posterior import MarginalPosterior
from src.bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from src.bayes_shape_fitting.visualization.animation_3d import ShapePlotterAnimations3D
from src.bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.source_densitites import BetaSourceDensity, UniformSourceDensity
from src.bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from tests.fixtures.sample_data import simulate_measurement

np.random.seed(23)

def _case_id(c):
    return c["name"]

# filter the surfaces
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
    return UniformSourceDensity(n_dim = 2, fixed_params=np.array([0.4,0.3,0.3,0.2]), domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic) # params are [a_0,b_0,a_1,b_1]

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_sigma=0.1)

@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)

class TestPlots:

    def test_shape_plot(self, shape, theta):
        plotter = ShapePlotter3D()
        plotter.plot_shape(shape, theta, n_points=500)
        plt.show()

    def test_observation_plot(self, shape, theta, measurement):
        plotter = ShapePlotter3D()
        plotter.plot_shape(shape, theta)
        plotter.plot_observations(measurement)
        plt.show()

    def test_plot_source_density(self, shape, source_density, theta):
        plotter = ShapePlotter3D()
        fig, ax = plotter.plot_shape(shape, theta)
        plotter.plot_source_density(shape, source_density, source_density.fixed_params, ax=ax, plot_on_shape=True, shape_params= theta)
        plotter.plot_source_density(shape, source_density, source_density.fixed_params)
        plt.show()

    def test_plot_likelihood(self, shape, theta, source_density, noise_function, measurement):
        plotter = ShapePlotter3D()
        likelihood = MarginalLikelihood(shape, source_density, noise_function)
        plotter.plot_shape(shape, theta)
        plotter.plot_likelihood(likelihood = likelihood, theta = theta, plot_measurement=True, measurement=measurement)
        plt.show()

    def test_animation(self, shape, theta, init_theta, source_density, noise_function, measurement):
        # here we test the basic animation, for this we need to do a fit and get the param history
        likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=50)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)
        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
        result = optimizer.optimize(posterior, init_theta)
        param_history = result.param_history

        # now we plot the basic setup
        plotter = ShapePlotter3D()
        fig, ax = plotter.plot_shape(shape, theta, label="Ground Truth")
        plotter.plot_shape(shape, result.theta_map, ax=ax, label="Fitted Surface", color="green")
        plotter.plot_shape(shape, init_theta, ax=ax, label="Initial Fit", color="orange")
        plotter.plot_observations(observations=measurement, ax=ax)

        animator = ShapePlotterAnimations3D()
        animator.animate_param_history(shape, param_history, ax = ax, show=True)
        plt.show()

    def test_source_density_animation(self, shape, theta, init_theta, source_density, noise_function, measurement):
        # here we test the basic animation, for this we need to do a fit and get the param history
        # we define the source_density to be variable
        source_density_var = UniformSourceDensity(n_dim = 2, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)

        likelihood = MarginalLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function, num_samples=50)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)
        optimizer = MAPOptimizer(method='Powell', verbose=True, maxiter=10)
        bounds = shape.bounds + 4*[(0.1,0.5)]
        result = optimizer.optimize(posterior, np.concatenate([init_theta,source_density.fixed_params]), bounds=bounds)
        param_history = result.param_history

        shape_param_hist = [r[:shape.get_n_params()] for r in param_history]
        source_param_hist = [r[shape.get_n_params():] for r in param_history]

        # now we plot the basic setup
        plotter = ShapePlotter3D()
        fig, ax = plotter.plot_shape(shape, theta, label="Ground Truth")
        plotter.plot_shape(shape, result.theta_map[:shape.get_n_params()], ax=ax, label="Fitted Curve", color="green")
        plotter.plot_shape(shape, init_theta, ax=ax, label="Initial Fit", color="orange")
        plotter.plot_observations(observations=measurement, ax=ax)

        animator = ShapePlotterAnimations3D()
        animator.animate_param_history(shape, shape_param_hist, ax=ax, show = True)

        animator.animate_source_density(shape, source_density_var, source_param_hist, show = True)