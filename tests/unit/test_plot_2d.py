"""
Test the functionalities of the plot 2d functionalities
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.source_densitites import BetaSourceDensity
from src.bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from tests.fixtures.sample_data import simulate_measurement


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
def source_density():
    return BetaSourceDensity(fixed_params=np.array([3,2]))

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