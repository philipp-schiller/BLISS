from benchmark_fitting import run_fitting_benchmark
from bayes_shape_fitting.utils.samplers import GridSampler, TopKSampler, ClosestPointWindowSampler, AdaptiveGridSampler
from bayes_shape_fitting.utils.integration import SimpsonIntegrator, TrapezoidIntegrator, SupportOnlyIntegrator
from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood, MarginalLikelihood_Standard
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement

import numpy as np

# case = shape_cases[5] # cylinder
case = shape_cases[-1] # ellipsoid

shape = case["shape"]
theta = case["theta"]
init_theta = case["init_params"]

source_density_true = UniformSourceDensity(n_dim = 2, domain_low=shape.domain_low, domain_high=shape.domain_high, fixed_params=np.array([0.5,0.5, 0.5, 0.5]), periodic=shape.periodic)

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.01)
measurement = simulate_measurement(shape, theta, source_density_true, noise_function, num_samples=50)


results = run_fitting_benchmark(
    shape            = shape,
    measurement      = measurement,
    source_density   = source_density_true,
    noise_function   = noise_function,
    init_theta       = init_theta,
    theta_true       = theta,
    samplers         = [GridSampler(), AdaptiveGridSampler()],
    integrators      = [TrapezoidIntegrator(), SupportOnlyIntegrator()],
    num_samples_list = [25, 50],
    save_fig         = "fitting_benchmark.png",
)