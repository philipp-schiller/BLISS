"""
Test the fitting of 2D curves
"""

import numpy as np
import matplotlib.pyplot as plt

from bliss.inference.likelihoods import MarginalLikelihood
from bliss.inference.posterior import MarginalPosterior
from bliss.inference.samplers.map_estimation import MAPOptimizer
from bliss.visualization.plotting_2d import ShapePlotter2D
from bliss.inference.noise_functions import IsotropicNoise
from bliss.inference.source_densitites import BetaSourceDensity, UniformSourceDensity
from bliss.models.shapes.shape_examples import shape_cases
from bliss.utils.sample_data import simulate_measurement

np.random.seed(25)

# first we define the shape case
case = shape_cases[0]

shape = case["shape"]
theta = case["theta"]
init_theta = case["init_params"]

source_density_true = UniformSourceDensity(n_dim=1, fixed_params=np.array([0.4, 0.2]), domain_low=shape.domain_low,
                                           domain_high=shape.domain_high, periodic=shape.periodic)

source_density = BetaSourceDensity()
noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)
measurement = simulate_measurement(shape, theta, source_density_true, noise_function, num_samples=100)

# define a variable and a fixed likelihood
likelihood_variable = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)
likelihood_fixed = MarginalLikelihood(shape=shape, source_density=source_density_true, noise_function=noise_function)
posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)
#posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)
posterior_fixed = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_fixed)

optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)

# we define an initial guess and the bounds
source_init = np.array([2,7])
source_bounds = [(1, 13), (1, 13)]  # bounds for the Beta

result_var = optimizer.optimize(posterior_var, shape_init = init_theta, source_init=source_init, source_bounds=source_bounds)
result_fixed = optimizer.optimize(posterior_fixed, init_theta)

init = init_theta[0:3]

print(f"Result fixed:  {result_fixed.shape_params}")
print(f"Result variable:  {result_var.shape_params}")

plotter = ShapePlotter2D()
fig, ax, _ = plotter.plot_shape(shape, theta, label="Ground Truth")
plotter.plot_shape(shape, result_var.shape_params, ax=ax, label="Fitted Curve Variable", color="green")
plotter.plot_shape(shape, result_fixed.shape_params, ax=ax, label="Fitted Curve Fixed", color="red")
plotter.plot_shape(shape, init, ax=ax, label="Initial Fit", color="orange")
plotter.plot_observations(observations=measurement, ax=ax)
plt.legend()
plt.show()