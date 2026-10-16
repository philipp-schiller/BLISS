"""
Test the functionalities of the plot 2d functionalities
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.likelihoods import  MarginalLikelihood, ClosestPointLikelihood, MarginalLikelihoodSum
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.visualization.animation_2d import ShapePlotterAnimations
from bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from bayes_shape_fitting.inference.noise_functions import LaserDependentEllipseNoise, IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import BetaSourceDensity, SmoothUniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement
from bayes_shape_fitting.utils.compute_normals import compute_normals, compute_normals_bayesian
from bayes_shape_fitting.utils.compute_normals_bayesian import compute_normals_bayes

np.random.seed(25)

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
    #return BetaSourceDensity(fixed_params=np.array([2,3]))
    return SmoothUniformSourceDensity(fixed_params=np.array([0.5,0.2]), n_dim=1, domain_low=np.array([0,]), domain_high=np.array([1,]), periodic=[True], sharpness=100)

@pytest.fixture
def noise_function(shape, theta):
    params = np.hstack((theta,np.array([0.01,0.03])))
    #return LaserDependentEllipseNoise(scanner = np.array([-10,10]), fixed_params=params,shape=shape)
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.0005)

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

    def test_covariances(self, shape, theta, source_density, noise_function, measurement):
        plotter = ShapePlotter2D()
        plotter.plot_shape(shape, theta)
        plotter.plot_observations(measurement)
        plotter.plot_covariances(shape, theta, source_density, noise_function)
        plt.show()

    def test_closest_points(self, shape, theta, measurement):
        plotter = ShapePlotter2D()
        plotter.plot_shape(shape, theta)
        plotter.plot_observations(measurement)

        # compute the closest points
        closest_points, t = shape.get_closest_point(params=theta, x=measurement, return_params=True)
        plotter.plot_observations(closest_points, title="closest points")

        plt.show()

    def test_normals(self, shape, theta, measurement):
        normals = compute_normals(measurement=measurement)
        assert len(normals) == len(measurement)

        # now plot
        plotter = ShapePlotter2D()
        fig, ax, _ = plotter.plot_shape(shape, theta)
        plotter.plot_observations(measurement, ax=ax)
        plotter.plot_normals(measurement=measurement, normals=normals, ax=ax)
        plt.show()

    def test_normals_bayesian(self, shape, theta, measurement):
        result = compute_normals_bayesian(measurement=measurement, k=10, beta=0.5)
        # now plot
        plotter = ShapePlotter2D()
        fig, ax, _ = plotter.plot_shape(shape, theta)
        plotter.plot_observations(measurement, ax=ax)
        plotter.plot_normals(measurement=measurement, normals=result.normals, normal_var=result.normal_var, ax=ax)
        plt.show()

    def test_normals_bayesian_two(self, shape, theta, measurement):
        result, covs = compute_normals_bayes(measurement=measurement, k=10, method="bingham")
        # now plot
        plotter = ShapePlotter2D()
        fig, ax, _ = plotter.plot_shape(shape, theta)
        plotter.plot_observations(measurement, ax=ax)
        plotter.plot_normals(measurement=measurement, normals=result, uncertainties=covs, ax=ax)
        plt.show()

    def test_plot_source_density(self, shape, source_density, theta):
        plotter = ShapePlotter2D()
        fig, ax, _ = plotter.plot_shape(shape, theta)
        plotter.plot_source_density(shape, source_density, source_density.fixed_params, ax=ax, plot_on_shape=True, shape_params= theta)
        plotter.plot_source_density(shape, source_density, source_density.fixed_params)
        plt.show()

    def test_plot_likelihood(self, shape, theta, source_density, noise_function, measurement):
        plotter = ShapePlotter2D()
        likelihood = MarginalLikelihood(shape, source_density, noise_function)
        plotter.plot_shape(shape, theta)
        plotter.plot_likelihood(likelihood = likelihood, theta = theta, ax = plt.gca(), x_min=-4,x_max=4,y_min=-4,y_max=4, plot_measurement=True, measurement=measurement)

        #plt.figure()
        #likelihood = ClosestPointLikelihood(shape, source_density, noise_function)
        likelihood = MarginalLikelihoodSum(shape, source_density, noise_function, num_samples=2000)
        plotter.plot_shape(shape, theta)
        plotter.plot_likelihood(likelihood = likelihood, theta = theta, ax = plt.gca(), x_min=-4,x_max=4,y_min=-4,y_max=4, plot_measurement=True, measurement=measurement)
        plt.title("Closest Point")
        plt.show()

    def test_animation(self, shape, theta, init_theta, source_density, noise_function, measurement):
        # here we test the basic animation, for this we need to do a fit and get the param history
        likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)
        #likelihood = ClosestPointLikelihood(shape, source_density, noise_function)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)
        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
        result = optimizer.optimize(posterior, init_theta)
        param_history = result.shape_history

        # now we plot the basic setup
        plotter = ShapePlotter2D()
        fig, ax, _ = plotter.plot_shape(shape, theta, label="Ground Truth")
        plotter.plot_shape(shape, result.shape_params, ax=ax, label="Fitted Curve", color="green")
        plotter.plot_shape(shape, init_theta, ax=ax, label="Initial Fit", color="orange")
        plotter.plot_observations(observations=measurement, ax=ax)

        animator = ShapePlotterAnimations()
        #animator.animate_param_history(shape, param_history, ax = ax, show = True)
        animator.animate_optimization_result(result, shape, source_density, noise_function, ax=ax, show = True)


    def test_source_density_animation(self, shape, theta, init_theta, source_density, noise_function, measurement):
        # here we test the basic animation, for this we need to do a fit and get the param history
        # we define the source_density to be variable
        source_density_var = SmoothUniformSourceDensity(n_dim=1, domain_low=np.array([0,]), domain_high=np.array([1,]), periodic=[True], sharpness=100)
        #source_density_var = BetaSourceDensity()

        likelihood = MarginalLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function)
        posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)
        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
        result = optimizer.optimize(posterior, shape_init=init_theta, source_init=source_density.fixed_params, shape_bounds = shape.bounds, source_bounds=[(0.01, np.inf), (0.01, np.inf)])

        # now we plot the basic setup
        plotter = ShapePlotter2D()
        fig, ax, _ = plotter.plot_shape(shape, theta, label="Ground Truth")
        plotter.plot_shape(shape, result.shape_params, ax=ax, label="Fitted Curve", color="green")
        plotter.plot_shape(shape, init_theta, ax=ax, label="Initial Fit", color="orange")
        plotter.plot_observations(observations=measurement, ax=ax)

        animator = ShapePlotterAnimations()
        #animator.animate_optimization_result(result, shape, source_density_var, ax, show=True)
        animator.animate_param_history(shape, result.shape_history, ax=ax, show = True)

        animator.animate_source_density(shape, source_density_var, result.source_history, show = True)