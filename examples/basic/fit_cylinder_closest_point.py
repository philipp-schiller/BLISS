"""
Test the fitting of a 3D cylinder
"""

import numpy as np
import matplotlib.pyplot as plt

from src.bayes_shape_fitting.inference.likelihoods import MarginalLikelihood, ClosestPointLikelihood
from src.bayes_shape_fitting.inference.posterior import MarginalPosterior
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

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)
measurement = simulate_measurement(shape, theta, source_density_true, noise_function, num_samples=100)

# define a variable and a fixed likelihood
likelihood_closest = ClosestPointLikelihood(shape, source_density, noise_function)
likelihood_marginal = MarginalLikelihood(shape=shape, source_density=source_density_true, noise_function=noise_function, num_samples=50)
posterior_closest =MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_closest)
posterior_marginal = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_marginal)

optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)

# we define an initial guess and the bounds
init_source = np.array([0.5,0.2, 0.6, 0.3])
bounds_var = 2*[(0,1),(0.1,0.5)]

result_closest = optimizer.optimize(posterior_closest, shape_init=init_theta, source_init=init_source, source_bounds=bounds_var)
result_marginal = optimizer.optimize(posterior_marginal, init_theta)

init = init_theta[:shape.get_n_params()]

print(f"Log posterior closest: {result_closest.log_posterior}")
print(f"Log posterior fixed: {result_marginal.log_posterior}")

print(f"Result closest:  {result_closest.theta_map}")
print(f"Result variable:  {result_marginal.theta_map}")

shape_params_closest = result_closest.theta_map[:7]
shape_params_marginal = result_marginal.theta_map[:7]

fig, ax = plotter.plot_shape(shape, theta, label="Ground Truth")
plotter.plot_shape(shape, shape_params_closest, ax=ax, label="Fitted Curve closest", color="green")
plotter.plot_shape(shape, shape_params_marginal, ax=ax, label="Fitted Curve marginal", color="red")
plotter.plot_shape(shape, init, ax=ax, label="Initial Fit", color="orange")
plotter.plot_observations(observations=measurement, ax=ax)
plt.legend()
plt.show()

animator = ShapePlotterAnimations3D()
animator.animate_param_history(shape, result_closest.param_history, show=True)