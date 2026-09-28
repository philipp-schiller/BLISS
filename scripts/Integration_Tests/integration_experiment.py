"""
Test the fitting of a 3D cylinder
"""

import numpy as np
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood, MarginalLikelihood_Standard
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement

np.random.seed(28)

# first we define the shape case
#case = shape_cases[5] # cylinder
case = shape_cases[-1] # Ellipsoid

shape = case["shape"]
theta = case["theta"]
init_theta = case["init_params"]

source_density_true = UniformSourceDensity(n_dim = 2, domain_low=shape.domain_low, domain_high=shape.domain_high, fixed_params=np.array([0.5,0.5, 0.6, 0.2]), periodic=shape.periodic)

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)
measurement = simulate_measurement(shape, theta, source_density_true, noise_function, num_samples=100)

likelihood = MarginalLikelihood_Standard(shape=shape, source_density=source_density_true, noise_function=noise_function, num_samples=50)

posterior_fixed = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)

optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
result = optimizer.optimize(posterior_fixed, init_theta)

# we plot the result of the fit
from bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D

plotter = ShapePlotter3D()

fig, ax = plotter.plot_shape(shape, theta, label="Ground Truth")
plotter.plot_shape(shape, result.theta_map, ax=ax, label="Fitted Curve", color="green")
plotter.plot_shape(shape, init_theta, ax=ax, label="Initial Fit", color="orange")
plotter.plot_observations(observations=measurement, ax=ax)
plt.show()