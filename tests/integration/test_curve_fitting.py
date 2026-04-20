"""
Test the fitting of 2D curves
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.inference.samplers.ecm_estimation import ECMOptimizer
from bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import BetaSourceDensity, UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
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
def source_density(shape):
    #return BetaSourceDensity(fixed_params=np.array([3,2]))
    return UniformSourceDensity(n_dim=1, fixed_params=np.array([0.3, 0.2]), domain_low=shape.domain_low,
                                domain_high=shape.domain_high, periodic=shape.periodic)

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)


@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)

class TestMAP:

    def test_map_fixed(self, shape, measurement, theta, init_theta, source_density, noise_function):
        likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)

        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
        result = optimizer.optimize(posterior, init_theta)
        print(f"MAP estimate: {result.theta_map}")
        print(f"Log posterior: {result.log_posterior}")

        print(posterior.log_posterior(shape_params = init_theta))
        print(posterior.log_posterior(shape_params = result.theta_map))

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

class TestECM:

    def test_map_fixed(self, shape, measurement, theta, init_theta, source_density, noise_function):
        likelihood =MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)

        optimizer = ECMOptimizer(method='L-BFGS-B', verbose=True)
        result = optimizer.optimize(posterior, shape_init = init_theta)
        print(f"MAP estimate: {result.shape_params}")
        print(f"Log posterior: {result.log_posterior_final}")

        if True:
            plotter = ShapePlotter2D()
            fig, ax, _ = plotter.plot_shape(shape, theta, label = "Ground Truth")
            plotter.plot_shape(shape, result.shape_params, ax = ax, label = "Fitted Curve", color = "green")
            plotter.plot_shape(shape, init_theta, ax = ax, label = "Initial Fit", color = "orange")
            plotter.plot_observations(observations=measurement, ax=ax)
            plt.legend()
            plt.show()

    def test_map_var(self, shape, measurement, theta, init_theta, source_density, noise_function):

        dim = shape.parameter_dim
        source_density_var = UniformSourceDensity(n_dim=1, domain_low=shape.domain_low,
                                domain_high=shape.domain_high, periodic=shape.periodic)

        noise_function_var = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim)

        likelihood =MarginalLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function_var)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)

        optimizer = ECMOptimizer(method='L-BFGS-B', verbose=True)

        #init_theta = shape.initial_fit(measurement)

        result = optimizer.optimize(posterior, shape_init = init_theta, source_init = np.array([0.3,0.2]), noise_init = np.array([0.1]), source_bounds=[(0,1), (0.1,0.5)])
        print(f"MAP estimate: {result.shape_params}")
        print(f"Log posterior: {result.log_posterior_final}")

        print(f"MAP estimate source: {result.source_params}")
        print(f"MAP estimate noise: {result.noise_params}")

        if True:
            plotter = ShapePlotter2D()
            fig, ax, _ = plotter.plot_shape(shape, theta, label = "Ground Truth")
            plotter.plot_shape(shape, result.shape_params, ax = ax, label = "Fitted Curve", color = "green")
            plotter.plot_shape(shape, init_theta, ax = ax, label = "Initial Fit", color = "orange")
            plotter.plot_observations(observations=measurement, ax=ax)
            plt.legend()
            plt.show()