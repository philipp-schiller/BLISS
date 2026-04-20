"""
Test the fitting of a 3D cylinder
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import time
from src.bayes_shape_fitting.inference.likelihoods import ClosestPointLikelihood
from src.bayes_shape_fitting.inference.posterior import MarginalPosterior
from src.bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from src.bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from src.bayes_shape_fitting.models.shapes.shape_examples import shape_cases

np.random.seed(25)

# first we define the shape case
case = shape_cases[5]

shape = case["shape"]

# get the data
ROOT = Path(__file__).resolve().parents[2]   # adjust level if needed
DATA_PATH = ROOT / "tests" / "fixtures" / "spruce.xyz"

measurement = np.loadtxt(DATA_PATH)

# get the data
measurement_moved = np.loadtxt(DATA_PATH)
# move it to the center
measurement = measurement_moved - measurement_moved.mean(axis=0)

init_theta = shape.initial_fit(measurement)
#init_theta = initial_cylinder_parameters(measurement)

source_density = UniformSourceDensity(n_dim=2, domain_low=shape.domain_low, fixed_params=np.array([0.5,0.5, 0.5, 0.25]), domain_high=shape.domain_high, periodic=shape.periodic)
source_density_var = UniformSourceDensity(n_dim=2, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)

plotter = ShapePlotter3D()

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.5)

# we set a Gaussian Prior on the source_density

# define a variable and a fixed likelihood
#likelihood_variable = MarginalLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function, num_samples=50)
likelihood_variable = ClosestPointLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function)
posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)

optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
#optimizer = MAPOptimizer(method='Powell', verbose=True)

# we do the least squares fit as comparison
start = time.time()
result_ls = shape.initial_fit(measurement)
end = time.time()
runtime = end - start
print(f"Runtime: {runtime:.3f}s")

# for the variable result we need to give an init_value
init_source = np.array([0.2,0.48, 0.4, 0.48])
result_var = optimizer.optimize(posterior_var, shape_init=result_ls, source_init = init_source, shape_bounds=shape.bounds, source_bounds = 2*[(0,1),(0.1,0.49)])

print(result_var.theta_map)
print(result_ls)

theta_map = result_var.theta_map[:shape.get_n_params()]
source_map = result_var.theta_map[shape.get_n_params():]
init = init_theta[:shape.get_n_params()]

print(f"Log posterior variable: {result_var.log_posterior}")
print(f"Result variable:  {result_var.theta_map}")

fig, ax = plotter.plot_shape(shape, theta_map, label="Fitted Curve Variable", color="green")

plotter.plot_shape(shape, result_ls, ax=ax, label="Least squares", color = "purple")
plotter.plot_shape(shape, init, ax=ax, label="Initial Fit", color="orange")
plotter.plot_observations(observations=measurement, ax=ax)
plt.legend()
plt.show()

