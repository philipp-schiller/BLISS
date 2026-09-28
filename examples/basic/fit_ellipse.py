"""
Test the fitting of 2D curves
"""

import numpy as np
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity, \
    PartiallyFixedDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement

np.random.seed(63)

# first we define the shape case of the ellipse
case = shape_cases[1]

shape = case["shape"]
theta = case["theta"]
init_theta = case["init_params"]

# number of shape-parameters
n = shape.get_n_params()

source_density_true = UniformSourceDensity(n_dim = 1, fixed_params=np.array([0.3,0.2]), domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)
source_density = UniformSourceDensity(n_dim = 1, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)
source_density_partial = PartiallyFixedDensity(source_density, {0: 0.3})

noise_function_true = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.05)
noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim)
measurement = simulate_measurement(shape, theta, source_density_true, noise_function_true, num_samples=150)

# we set a Gaussian Prior on the source_density
mean = np.array([0.3,0.2])
cov =  np.diag([1,0.00001])
prior = GaussianPrior(mean=mean, cov=cov)
source_density.prior = prior

# we also set a prior on the partial source density
source_density_partial.prior = GaussianPrior(mean=np.array([0.2]), cov = np.array([0.01]))

mean = np.array([0.1])
cov =  np.diag([0.0001])
noise_prior = GaussianPrior(mean=mean, cov=cov)
noise_function.prior = noise_prior

# define a variable and a fixed likelihood
likelihood_variable = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=300)
likelihood_partial = MarginalLikelihood(shape=shape, source_density=source_density_partial, noise_function=noise_function_true, num_samples=300)
likelihood_fixed = MarginalLikelihood(shape=shape, source_density=source_density_true, noise_function=noise_function_true, num_samples=300)

# now define the posterior
posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)
posterior_partial = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_partial)
posterior_fixed = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_fixed)

#optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True, maxiter=1000)
optimizer = MAPOptimizer(method='Powell', verbose=True,  ftol = 1e-9, gtol = 1e-4, maxiter=1000)

# perform the MAP estimation
init_theta = shape.initial_fit(measurement)

init_source_density = np.array([0.4, 0.1]) # interval_center, half-width
init_source_density_partial = np.array([0.3]) # resembles the interval_center
init_noise = np.array([0.1])
init_theta_partial = np.hstack([init_theta,init_source_density_partial])
init_theta_var = np.hstack([init_theta,init_source_density,init_noise]) # for variable source-density

# we set the bounds reasonable
shape_bounds = 2*[(-5,5)] + 2*[(0,10)] + [(-np.pi,np.pi)]
source_bounds = [(0,1),(0.05,0.4)]
source_bounds_partial = [(0.1,0.4)]
noise_bounds = [(0.05,0.15)]
bounds = shape_bounds + source_bounds + noise_bounds # for variable source-density
bounds_partial = shape_bounds + source_bounds_partial # for variable source-density


result_var = optimizer.optimize(posterior_var, shape_init=init_theta, source_init = init_source_density, noise_init=init_noise, shape_bounds=shape_bounds, source_bounds = source_bounds, noise_bounds=noise_bounds)
result_partial = optimizer.optimize(posterior_partial,  shape_init=init_theta, source_init = init_source_density_partial, shape_bounds = shape_bounds, source_bounds=source_bounds_partial)
result_fixed = optimizer.optimize(posterior_fixed, shape_init = init_theta, shape_bounds=shape_bounds)

theta_map = result_var.theta_map[:n]
theta_map_partial = result_partial.theta_map[:n]
source_map = result_var.theta_map[n:n+2]
init = init_theta[:n]

print(f"Log posterior variable: {result_var.log_posterior}")
print(f"Log posterior partial: {result_partial.log_posterior}")
print(f"Log posterior fixed: {result_fixed.log_posterior}")

print(f"Result fixed:  {result_fixed.theta_map}")
print(f"Result partial:  {result_partial.theta_map}")
print(f"Result variable:  {result_var.theta_map}")


plotter = ShapePlotter2D()
fig, ax, _ = plotter.plot_shape(shape, theta, label="Ground Truth")
plotter.plot_shape(shape, theta_map, ax=ax, label="Fitted Curve Variable", color="green")
plotter.plot_shape(shape, result_fixed.theta_map, ax=ax, label="Fitted Curve Fixed", color="red")
plotter.plot_shape(shape, theta_map_partial, ax=ax, label="Fitted Curve Partial", color="Purple")
plotter.plot_shape(shape, init, ax=ax, label="Initial Fit", color="orange")
plotter.plot_observations(observations=measurement, ax=ax)
plt.legend()
plt.show()

