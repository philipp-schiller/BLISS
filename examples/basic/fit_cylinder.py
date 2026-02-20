"""
Test the fitting of a 3D cylinder
"""

import numpy as np
import matplotlib.pyplot as plt

from src.bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from src.bayes_shape_fitting.inference.posterior import PosteriorFixed, MarginalPosterior
from src.bayes_shape_fitting.inference.priors import Prior, GaussianPrior, UniformPrior
from src.bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from src.bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D
from src.bayes_shape_fitting.visualization.animation_3d import ShapePlotterAnimations3D
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.source_densitites import BetaSourceDensity, UniformSourceDensity, SplineSourceDensity
from src.bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from tests.fixtures.sample_data import simulate_measurement

np.random.seed(25)

# first we define the shape case
case = shape_cases[5]

shape = case["shape"]
theta = case["theta"]
init_theta = case["init_params"]

source_density_true = UniformSourceDensity(n_dim = 2, domain_low=shape.domain_low, domain_high=shape.domain_high, fixed_params=np.array([0.5,0.2, 0.6, 0.3]), periodic=shape.periodic)
source_density = UniformSourceDensity(n_dim=2, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)

plotter = ShapePlotter3D()
#plotter.plot_source_density(shape, source_density_true, params=source_density_true.fixed_params)
#plt.show()

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_sigma=0.1)
measurement = simulate_measurement(shape, theta, source_density_true, noise_function, num_samples=100)

# we set a Gaussian Prior on the source_density

# define a variable and a fixed likelihood
likelihood_variable = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=50)
likelihood_fixed = MarginalLikelihood(shape=shape, source_density=source_density_true, noise_function=noise_function, num_samples=50)
posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)
#posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)
posterior_fixed = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_fixed)

optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)

# we define an initial guess and the bounds
#init_theta_var = np.hstack([init_theta, [2, 7]])
init_theta_var = np.hstack([init_theta, np.array([0.5,0.2, 0.6, 0.3])])
bounds_var = shape.bounds + 4*[(0.01,1)]

#result_var = optimizer.optimize(posterior_var, init_theta_var, bounds=bounds_var)
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

# plot the source density
#plotter.plot_source_density(shape, source_density, params=source_map)
#plotter.plot_source_density(shape, source_density, params=[2,7])
#plotter.plot_source_density(shape, source_density_true, params = [0.2,0.4])
#plt.show()

# plot the likelihood
#plotter.plot_likelihood(likelihood=likelihood_variable, theta = np.array([-1, 0, 2, 2, 7]), plot_measurement=True, measurement=measurement)
#plotter.plot_likelihood(likelihood=likelihood_fixed, theta = result_fixed.theta_map, plot_measurement=True, measurement=measurement)
#plotter.plot_likelihood(likelihood=likelihood_variable, theta= result_var.theta_map, plot_measurement=True, measurement=measurement)
#plt.show()
