"""
Test the fitting of a 3D cylinder
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import time

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases

np.random.seed(25)

# first we define the shape case
case = shape_cases[5]

shape = case["shape"]

ground_truth = np.array([2,0,0,0.25,1,0,5.3])

# get the data
ROOT = Path(__file__).resolve().parents[2]   # adjust level if needed
DATA_PATH = ROOT / "tests" / "fixtures" / "measurements/measured_points_ac_0.01.npy"

measurement = np.load(DATA_PATH)
# move it to the center
#measurement = measurement_moved - measurement_moved.mean(axis=0)

def initial_cylinder_parameters(xyz):
    """
    Given an (N,3) point cloud xyz, return an initial cylinder estimate
    in the desired format:
        [x_c, y_c, z_c, r, h, polar, azim]

    Geometry conventions:
        - p = (x_c, y_c, z_c) is the center of the *bottom* cap
        - axis direction is encoded using spherical coords (polar, azim)
        - h = total height (extent) of the point cloud along the axis
    """

    # --- 1) Center data ---
    centroid = xyz.mean(axis=0)
    X = xyz - centroid

    # --- 2) PCA for axis direction (largest variance) ---
    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_dir = eigvecs[:, np.argmax(eigvals)]
    axis_dir /= np.linalg.norm(axis_dir)

    # --- 3) Project onto axis to find height extent ---
    proj = X @ axis_dir
    h = proj.max() - proj.min()

    # --- 4) Pick bottom-cap point (minimum projection) ---
    bottom_point = centroid + axis_dir * proj.min()

    # --- 5) Compute radius estimate ---
    # perpendicular distances to axis
    perp = X - np.outer(proj, axis_dir)
    radial_distances = np.linalg.norm(perp, axis=1)
    r = np.median(radial_distances)

    # --- 6) Compute spherical angles (polar, azim) for axis_dir ---
    # axis_dir = (ux, uy, uz)
    ux, uy, uz = axis_dir
    polar = np.arccos(uz)                   # angle from +z
    azim = np.arctan2(uy, ux) % (2*np.pi)   # azimuth in [0,2π)

    # Enforce consistent orientation, e.g., prefer uz >= 0
    if axis_dir[2] < 0:
        axis_dir = -axis_dir

    # --- 7) Return cylinder parameters ---
    x_c, y_c, z_c = bottom_point
    return np.array([x_c, y_c, z_c, r, h, polar, azim])

init_theta = initial_cylinder_parameters(measurement)

source_density = UniformSourceDensity(n_dim=2, domain_low=shape.domain_low, fixed_params=np.array([0.5,0.5, 0.5, 0.25]), domain_high=shape.domain_high, periodic=shape.periodic)
source_density_var = UniformSourceDensity(n_dim=2, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)

plotter = ShapePlotter3D()
#fig, ax = plotter.plot_shape(shape, params=init_theta)
#plotter.plot_observations(measurement, ax = ax)
#plotter.plot_source_density(shape, source_density, params=source_density.fixed_params, plot_on_shape=True, shape_params = init_theta)

#ax.set_aspect("equal")
#plt.show()

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.01)

# we set a Gaussian Prior on the source_density

# define a variable and a fixed likelihood
likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=10)
likelihood_variable = MarginalLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function, num_samples=10)
posterior = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood)
posterior_var = MarginalPosterior(shape=shape, measurement=measurement, likelihood=likelihood_variable)

optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)

# we do the least squares fit as comparison
start = time.time()
result_ls = shape.initial_fit(measurement)
end = time.time()
runtime = end - start
print(f"Runtime: {runtime:.3f}s")

result_fixed = optimizer.optimize(posterior, init_theta, shape_bounds=shape.bounds)
# for the variable result we need to give an init_value
init_source = np.array([0.4,0.4, 0.4, 0.25])
result_var = optimizer.optimize(posterior_var, shape_init=init_theta, source_init=init_source, shape_bounds=shape.bounds, source_bounds = 2*[(0.01,1),(0.01,0.49)])


#print(result_var.theta_map)
#print(result_ls)

theta_map = result_var.theta_map[:shape.get_n_params()]
source_map = result_var.theta_map[shape.get_n_params():]
init = init_theta[:shape.get_n_params()]

#print(f"Log posterior variable: {result_var.log_posterior}")
#print(f"Log posterior fixed: {result_fixed.log_posterior}")

np.set_printoptions(precision=4, suppress=True)

print("Result LS:", result_ls)
print("Result fixed:", result_fixed.theta_map)
print("Result variable:", result_var.theta_map)


fig, ax = plotter.plot_shape(shape, theta_map, label="Fitted Curve Variable", color="green")

#plotter.plot_shape(shape, ground_truth, ax=ax, label="Ground Truth", color = "black")
plotter.plot_shape(shape, result_ls, ax=ax, label="Least squares", color = "purple")
plotter.plot_shape(shape, result_fixed.theta_map, ax=ax, label="Fitted Curve Fixed", color="red")
plotter.plot_shape(shape, init, ax=ax, label="Initial Fit", color="orange")
plotter.plot_observations(observations=measurement, ax=ax)
plt.legend()
plt.show()

#animator = ShapePlotterAnimations3D()
#animator.animate_param_history(shape, result_var.param_history, show=True)

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
