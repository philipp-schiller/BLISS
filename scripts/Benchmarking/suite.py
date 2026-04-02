# suite.py
import numpy as np
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from scripts.Benchmarking.Fitters.Ellipse_fitters.base import BaseEllipseFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.bayes_fitter import BayesFitter
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.priors import Prior, GaussianPrior
from src.bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from scripts.Benchmarking.ground_truth_config import GroundTruthConfig
from scripts.Benchmarking.axes import SHAPE_CASES, FITTERS_STATIC, LABELS, \
    NOISE_FUNCTION_CASES, SOURCE_DENSITY_CASES


@dataclass
class BenchmarkCase:
    name: str
    shape_name: str
    source_name: str
    noise_name: str
    ground_truth_config: GroundTruthConfig
    fitters: list[BaseEllipseFitter]
    labels: list[str] = field(default_factory=list)
    num_it: int = 50
    num_points: int = 100

    def __post_init__(self):
        if not self.labels:
            self.labels = [f.__class__.__name__ for f in self.fitters]

    def make_bayes_correct(self) -> BaseEllipseFitter | None:
        """Build a Bayes fitter mirroring the current ground truth params."""
        if self.bayes_correct_factory is None:
            return None
        return self.bayes_correct_factory(self.ground_truth_config)

    def build_bayes_fitters(self) -> list[BayesFitter]:
        """Builds the Bayesian fitters based on the Ground Truth"""

        source_params = self.ground_truth_config.source_params
        noise_params = self.ground_truth_config.noise_params
        # let's convert noise_params to a float if its an array
        if isinstance(noise_params, np.ndarray):
            if noise_params.size == 1:
                noise_params = float(noise_params.item())

        shape=self.ground_truth_config.shape

        # let's define some priors
        mean = np.array(
            [0.5, 0.20])  # the first parameter resembles the interval center c, and the second the half-width w
        strong_cov = np.diag([10, 0.0001])  # we set a prior independently on those
        strong_source_prior = GaussianPrior(mean=mean, cov=strong_cov)

        weak_cov = np.diag([10, 0.01])
        weak_source_prior = GaussianPrior(mean=mean, cov=weak_cov)

        mean_noise = np.array(noise_params)
        weak_cov_noise = np.array([0.1])
        strong_cov_noise = np.array([0.001])
        strong_noise_prior = GaussianPrior(mean=mean_noise, cov=strong_cov_noise)
        weak_noise_prior = GaussianPrior(mean=mean_noise, cov=weak_cov_noise)

        # weak prior
        Bayes_weak = BayesFitter(
            name="Bayes_weak_prior",

            source_density=UniformSourceDensity(n_dim=1, domain_low=shape.domain_low,
                                                domain_high=shape.domain_high, periodic=shape.periodic,
                                                prior=weak_source_prior),
            noise_function=IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim,
                                          prior=weak_noise_prior),

            source_bounds=[(0, 1), (0.05, 0.4)],
            source_init=np.array([0.5, 0.2]) + 0.05 * np.random.randn(2),

            noise_bounds=[(0.001, 0.015)],
            noise_init=np.array(noise_params)
        )

        # strong prior
        Bayes_strong = BayesFitter(
            name="Bayes_strong_prior",
            source_density=UniformSourceDensity(n_dim=1, domain_low=shape.domain_low,
                                                domain_high=shape.domain_high, periodic=shape.periodic,
                                                prior=strong_source_prior),
            noise_function=IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim,
                                          prior=strong_noise_prior),

            source_bounds=[(0, 1), (0.05, 0.4)],
            source_init=np.array([0.5, 0.2]) + 0.05 * np.random.randn(2),

            noise_bounds=[(0.001, 0.015)],
            noise_init=np.array(noise_params)
        )

        # correct Fitter
        Bayes_correct = BayesFitter(
            name = "Bayes_correct",
            source_density=UniformSourceDensity(n_dim=1, domain_low=shape.domain_low,
                                                domain_high=shape.domain_high, periodic=shape.periodic,
                                                fixed_params = source_params),
            noise_function=IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim,
                                          fixed_params= noise_params)
        )

        return [Bayes_weak, Bayes_strong, Bayes_correct]



def build_suite() -> list[BenchmarkCase]:
    cases = []

    for shape_case, noise, density in product(SHAPE_CASES, NOISE_FUNCTION_CASES, SOURCE_DENSITY_CASES):
        gt_config = GroundTruthConfig(
            shape=shape_case["shape"],
            shape_params=shape_case["shape_params"],
            noise_function=noise["noise_function"],
            source_density=density["source_density"],
            # no shape/source/noise params → all random
        )

        cases.append(BenchmarkCase(
            name=f"{shape_case['name']}__{noise["name"]}__{density["name"]}",
            shape_name=f"{shape_case['name']}",
            source_name=f"{density['name']}",
            noise_name=f"{noise['name']}",
            ground_truth_config=gt_config,
            fitters=FITTERS_STATIC,          # static fitters only here
            labels=LABELS,
            ),
        )

    return cases

