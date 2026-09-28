import numpy as np
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.compute_normals import compute_normals_bayesian
from bayes_shape_fitting.utils.sample_data import simulate_measurement

from benchmark_integrators import run_benchmark
from bayes_shape_fitting.utils.samplers import GridSampler, TopKSampler, ClosestPointWindowSampler, AdaptiveGridSampler, ClosestPointSampler
from bayes_shape_fitting.utils.integration import SimpsonIntegrator, TrapezoidIntegrator, MonteCarloIntegrator, \
    SupportOnlyIntegrator

from bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D

np.random.seed(26)

# first we define the shape case
case = shape_cases[1] # ellipse
#case = shape_cases[-1] # ellipsoid

shape = case["shape"]
theta = case["theta"]
init_theta = case["init_params"]

source_density = UniformSourceDensity(n_dim = 1, domain_low=shape.domain_low, domain_high=shape.domain_high, fixed_params=np.array([0.7,0.1]), periodic=shape.periodic)

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.05)
measurement = simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)
normals = compute_normals_bayesian(measurement, k = 10, reference_point=np.array([-2,0]))

# let's also show the true normals
t_vals = np.linspace(0,1,50)
sources = shape.evaluate(t_vals, theta)
normals_true, covariances = shape.evaluate_normals(t_vals, theta)

likelihood = MarginalLikelihood(shape=shape, source_density=source_density,
        noise_function=noise_function, num_samples=200)
#print(likelihood.log_likelihood(measurement=measurement, shape_params=theta))
#print(likelihood.log_likelihood(measurement=measurement, shape_params=theta, normals=normals.normals))

# now fit
posterior = MarginalPosterior(shape = shape, measurement = measurement, likelihood = likelihood)
posterior_normal = MarginalPosterior(shape = shape, measurement = measurement, likelihood = likelihood, normals=normals.normals)
optimizer = MAPOptimizer(verbose=True)

result = optimizer.optimize(posterior, init_theta)
result_normals = optimizer.optimize(posterior_normal, init_theta)

plotter = ShapePlotter2D()
fig, ax, _ = plotter.plot_shape(shape, theta, label = "GT")
plotter.plot_shape(shape, result.theta_map, color = "green", label="Without", ax=ax)
plotter.plot_shape(shape, result_normals.theta_map, color = "red", label="With", ax=ax)
plotter.plot_observations(measurement, ax=ax)
plotter.plot_normals(measurement, normals.normals, normal_var=normals.normal_var, ax=ax)
#plotter.plot_normals(sources, normals_true, ax=ax)

#plotter.plot_likelihood(likelihood=likelihood, ax=ax, theta=theta, plot_measurement=True, measurement=measurement)

plt.show()

