"""
Test the fitting of a 3D cylinder
"""

import numpy as np
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D
from bayes_shape_fitting.visualization.animation_3d import ShapePlotterAnimations3D
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement
from bayes_shape_fitting.models.shapes.cylinder import Cylinder

np.random.seed(27)

# first we define the shape case
case = shape_cases[5]

shape = case["shape"]
theta = case["theta"]
init_theta = case["init_params"]

source_density_true = UniformSourceDensity(n_dim = 2, domain_low=shape.domain_low, domain_high=shape.domain_high, fixed_params=np.array([0.5,0.5, 0.6, 0.2]), periodic=shape.periodic)

plotter = ShapePlotter3D()
#plotter.plot_source_density(shape, source_density_true, params=source_density_true.fixed_params)
#plt.show()

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)
measurement = simulate_measurement(shape, theta, source_density_true, noise_function, num_samples=100)

# we set a Gaussian Prior on the source_density

# define a variable and a fixed likelihood

likelihood_fixed = MarginalLikelihood(shape=shape, source_density=source_density_true, noise_function=noise_function, num_samples=50)

#posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)
posterior_fixed = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_fixed)

optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)

# we define an initial guess and the bounds
#init_theta_var = np.hstack([init_theta, [2, 7]])
init_theta_var = np.hstack([init_theta, np.array([0.5,0.2, 0.6, 0.3])])

source_init = np.array([0.5,0.2, 0.6, 0.3])
noise_init = np.array([0.1])

source_prior= GaussianPrior(mean=np.array([0.5, 0.2, 0.6, 0.3]), cov=np.diag([1, 0.0001, 1, 0.0001]))
source_density = UniformSourceDensity(n_dim=2, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic, prior=source_prior)

source_bounds = [(0, 1), (0.175, 0.225),(0, 1), (0.275, 0.325)]
noise_bounds = [(0.01,0.2)]

likelihood_variable = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=50)
posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)

result_var = optimizer.optimize(posterior_var, shape_init= init_theta, source_init=source_init, noise_init=noise_init, source_bounds=source_bounds, noise_bounds=noise_bounds)
result_fixed = optimizer.optimize(posterior_fixed, init_theta)

#theta_map = result_var.theta_map[:shape.get_n_params()]
#source_map = result_var.theta_map[shape.get_n_params():]
init = init_theta[:shape.get_n_params()]

#print(f"Log posterior variable: {result_var.log_posterior}")
print(f"Log posterior fixed: {result_fixed.log_posterior}")

print(f"Result fixed:  {result_fixed.theta_map}")
#print(f"Result variable:  {result_var.theta_map}")


fig, ax = plotter.plot_shape(shape, theta, label="Ground Truth")
#plotter.plot_shape(shape, theta_map, ax=ax, label="Fitted Curve Variable", color="green")
plotter.plot_shape(shape, result_fixed.theta_map, ax=ax, label="Fitted Curve Fixed", color="red")
plotter.plot_shape(shape, init, ax=ax, label="Initial Fit", color="orange")
plotter.plot_observations(observations=measurement, ax=ax)
plt.legend()
plt.show()

animator = ShapePlotterAnimations3D()
animator.animate_param_history(shape, result_fixed.param_history, show=True)
