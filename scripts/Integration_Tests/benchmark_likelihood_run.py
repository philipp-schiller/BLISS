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


np.random.seed(27)

# first we define the shape case
case = shape_cases[5] # cylinder
#case = shape_cases[-1] # ellipsoid

shape = case["shape"]
theta = case["theta"]
init_theta = case["init_params"]

source_density = UniformSourceDensity(n_dim = 2, domain_low=shape.domain_low, domain_high=shape.domain_high, fixed_params=np.array([0.5,0.5, 0.6, 0.2]), periodic=shape.periodic)

noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)
measurement = simulate_measurement(shape, theta, source_density, noise_function, num_samples=100)


def make_ml(sampler, integrator):
    return MarginalLikelihood(
        shape=shape, source_density=source_density,
        noise_function=noise_function, num_samples=200,
        sampler=sampler, integrator=integrator,
    )

results = run_benchmark(
    ml_factory    = make_ml,
    measurement   = measurement,
    shape_params  = theta,
    samplers      = [GridSampler(), ClosestPointWindowSampler(n_samples=500), AdaptiveGridSampler(n_coarse=10, n_fine=200)],
    integrators   = [SimpsonIntegrator(), TrapezoidIntegrator(), SupportOnlyIntegrator()],
    n_repeats     = 5,
    save_fig      = "benchmark_likelihood.png",
)
