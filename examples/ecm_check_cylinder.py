"""
Here we check the ECM optimization
"""
from bayes_shape_fitting.inference.samplers.ecm_estimation import ECMOptimizer

"""
Test the fitting of 2D curves
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
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity, \
    PartiallyFixedDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement
from bayes_shape_fitting.visualization.animation_2d import ShapePlotterAnimations
from pathlib import Path

np.random.seed(25)

# first we define the shape case
case = shape_cases[5]

shape = case["shape"]

ground_truth = np.array([1.124, 40.3472,-0.9301, 0.02, 0.3, 0, 0])

# get the data
ROOT = Path(__file__).resolve().parents[1]   # adjust level if needed
DATA_PATH = ROOT / "tests" / "fixtures" / "spruce.xyz"

DATA_PATH = ROOT / "examples" / "advanced" / "Cylinder_data" / "birch_pc_3.txt"

# get the data
measurement_moved = np.loadtxt(DATA_PATH)
# move it to the center
measurement = measurement_moved - measurement_moved.mean(axis=0)


init_theta = shape.initial_fit(measurement)
#init_theta = initial_cylinder_parameters(measurement)

source_density_var = UniformSourceDensity(n_dim=2, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)
source_density = PartiallyFixedDensity(source_density_var, {0: 0.5,1 : 0.5})

plotter = ShapePlotter3D()

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim)

# we set a Gaussian Prior on the source_density
mean = np.array([0.5,0.15])
cov =  np.diag([100,0.0001])
prior = GaussianPrior(mean=mean, cov=cov)
source_density.prior = prior

# define a variable and a fixed likelihood
likelihood_partial = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=100)

# now define the posterior
posterior_partial = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_partial)


optimizer = ECMOptimizer(verbose=True, max_iter=10)

# perform the MAP estimation
init_theta = shape.initial_fit(measurement)

init_source_density = np.array([0.4, 0.1]) # interval_center, half-width # resembles the interval_center
init_noise = np.array([0.001])
 # for variable source-density

# we set the bounds reasonable
source_bounds = [(0.01,1),(0.05,0.4)]
noise_bounds = [(0.0005,0.0015)]
shape_bounds = bounds = [
                (None, None), (None, None), (None, None),
                (1e-3, None),
                (1e-2, None),
                (0, np.pi),
                (0, 2 * np.pi)
            ]

result_partial = optimizer.optimize(posterior_partial,  shape_init=init_theta, source_init = init_source_density, noise_init = init_noise, shape_bounds = shape_bounds, source_bounds=source_bounds, noise_bounds = noise_bounds)

theta_map_partial = result_partial.shape_params
source_map = result_partial.shape_params


plotter = ShapePlotter3D()

fig, ax = plotter.plot_shape(shape, theta_map_partial,label="Fitted Curve Partial", color="Purple")
plotter.plot_shape(shape, init_theta, ax=ax, label="Initial Fit", color="orange")
plotter.plot_observations(observations=measurement, ax=ax)
plt.legend()

animator = ShapePlotterAnimations3D()
animator.animate_param_history(shape, result_partial.shape_history, show=True, ax = ax)


