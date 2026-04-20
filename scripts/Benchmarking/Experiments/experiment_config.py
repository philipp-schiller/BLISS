from dataclasses import dataclass, field
import numpy as np
import time
from tqdm import tqdm
import copy

from scripts.Benchmarking.Fitters.Cylinder_fitters.bayes_fitter_cylinder import BayesCylinderFitter
from scripts.Benchmarking.results import SweepResult
from bayes_shape_fitting.inference.noise_functions import LaserDependentNoise, PartiallyFixedNoise
from bayes_shape_fitting.inference.priors import Prior
from scripts.Benchmarking.Fitters.Ellipse_fitters.bayes_fitter import BayesFitter

from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound="ExperimentConfig")


@dataclass
class ExperimentConfig:
    # --- basic metadata ---
    name: str = None
    iterations: int = 100
    num_points: int = 100

    # --- geometry / shape ---
    shape: object = None

    # --- ground‑truth model ---
    source_density: object = None
    noise_function: object = None

    # --- ground-truth parameters
    shape_params: np.ndarray = field(default=None)
    source_params: np.ndarray = field(default=None)
    noise_params: np.ndarray = field(default=None)


    # --- priors for source ---
    source_prior_weak: Prior = None
    source_prior_strong: Prior = None
    source_bounds_weak: list = None
    source_bounds_strong: list = None

    # --- priors for noise ---
    noise_prior_weak: object = None
    noise_prior_strong: object = None
    noise_bounds_weak: list = None
    noise_bounds_strong: list = None

    # --- initialization values ---
    source_init: np.ndarray = None
    noise_init: np.ndarray = None

    # --- fitters & labels ---
    fitters_static: list = field(default_factory=list)
    labels: list = field(default_factory=list)


    def __post_init__(self):
        # remember which were unset (random) before we fill them in

        # first, check if the source density and noise function have fixed params
        #self.source_params = self.source_density.fixed_params
        #self.noise_params = self.noise_function.fixed_params

        # if they are still none, we proceed to generate random ones
        self._shape_is_random = self.shape_params is None
        self._source_is_random = self.source_params is None
        self._noise_is_random = self.noise_params is None

        if self._shape_is_random:
            self.shape_params = self._random_shape_params()
        if self._source_is_random:
            self.source_params = self._random_source_params()
        if self._noise_is_random:
            self.noise_params = self._random_noise_params()

    # ── samplers ──────────────────────────────────────────────────────────────

    def _random_shape_params(self) -> np.ndarray:
        cx    = np.random.uniform(-1, 1)
        cy    = np.random.uniform(-1, 1)
        a     = np.random.uniform(2, 2.5)
        b     = np.random.uniform(1.5, 2)
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        return np.array([cx, cy, a, b, angle])

    def _random_source_params(self) -> np.ndarray:
        c = np.random.uniform(0, 1)
        w = np.random.uniform(0.175,0.225)
        return np.array([c, w])

    def _random_noise_params(self) -> np.ndarray:
        #sigma = np.random.uniform(0.0225, 0.0225) # for laser experiment
        sigma = np.random.uniform(0.1**2,0.2**2)
        return np.array([sigma])

    # ── per-iteration resamplers ───────────────────────────────────────────────

    def resample(self):
        """Resample all params that were originally None (i.e. random)."""
        if self._shape_is_random:
            self.shape_params = self._random_shape_params()
        if self._source_is_random:
            self.source_params = self._random_source_params()
        if self._noise_is_random:
            self.noise_params = self._random_noise_params()

    @classmethod
    def get_from_dict(cls: Type[T], cfg_dict: Dict[str, Any]) -> T:
        """
        Build an ExperimentConfig instance from a dictionary.

        Extra keys in cfg_dict are ignored (but warned about).
        Missing keys use class defaults.
        """
        valid_fields = set(cls.__dataclass_fields__.keys())

        # keep only valid keys
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}

        # warn about unused keys
        unused = set(cfg_dict.keys()) - valid_fields
        if unused:
            print(f"[ExperimentConfig] Warning: ignoring unknown fields: {unused}")

        return cls(**filtered)

    # performing one run

    def get_measurement(self, n_points:int=100):
        # we simulate a measurement

        # first sample the parameters t
        parameter_samples = self.source_density.sample(n=n_points, params=self.source_params)

        # based on this, compute the respective covariances:
        covariances = self.noise_function.evaluate(parameter_samples, params=self.noise_params)

        # compute the respective source points
        source_points = self.shape.parametrization(parameter_samples, self.shape_params)

        # now sample from the covariances
        errors = np.array([np.random.multivariate_normal(np.zeros(self.shape.ambient_dim), cov) for cov in covariances])

        # we add the errors to the source
        measured_points = source_points + errors

        return measured_points, parameter_samples

    # --- build the bayesian fitters

    def build_bayes_fitters(self) -> list[BayesFitter]:
        """Builds the Bayesian fitters based on the Ground Truth"""

        source_params = self.source_params
        noise_params = self.noise_params
        # let's convert noise_params to a float if its an array
        if isinstance(noise_params, np.ndarray):
            if noise_params.size == 1:
                noise_params = float(noise_params.item())

        # copy the source_density and noise_function and set fixed-parameters and priors

        # we start with the weak prior
        source_weak = copy.copy(self.source_density)
        source_weak.prior = self.source_prior_weak

        noise_weak = copy.copy(self.noise_function)
        noise_weak.prior = self.noise_prior_weak

        # weak prior
        Bayes_weak = BayesFitter(
            name="Bayes_weak_prior",

            source_density=source_weak,
            noise_function=noise_weak,

            source_bounds=self.source_bounds_weak,
            source_init=self.source_init,

            noise_bounds=self.noise_bounds_strong,
            noise_init=self.noise_init
        )

        # then the strong prior
        source_strong = copy.copy(self.source_density)
        source_strong.prior = self.source_prior_strong

        noise_strong = copy.copy(self.noise_function)
        noise_strong.prior = self.noise_prior_strong

        Bayes_strong = BayesFitter(
            name="Bayes_weak_prior",

            source_density=source_strong,
            noise_function=noise_strong,

            source_bounds=self.source_bounds_strong,
            source_init=self.source_init,

            noise_bounds=self.noise_bounds_strong,
            noise_init=self.noise_init
        )

        # correct Fitter
        source_fixed = copy.copy(self.source_density)
        source_fixed.fixed_params = source_params

        #noise_fixed = copy.copy(self.noise_function)
        #noise_fixed.fixed_params = noise_params

        # for the laser case we do it dirty:
        base_noise = LaserDependentNoise(dim_param=1, dim_ambient=2, shape=self.shape, scanner=np.array([-8,1]))
        fixed_params = {0: -1, 1: 0, 2: 3, 3: 2, 4: np.pi / 4, 5: noise_params, 6: 0.08}
        noise_fixed = PartiallyFixedNoise(base_noise, fixed_params=fixed_params)

        Bayes_correct = BayesFitter(
            name="Bayes_correct",
            source_density=source_fixed,
            noise_function=noise_fixed
        )

        return [Bayes_weak, Bayes_strong, Bayes_correct]

    def build_bayes_fitter_cylinder(self):
        """Builds the Bayesian fitters based on the Ground Truth"""

        source_params = self.source_params
        noise_params = self.noise_params
        # let's convert noise_params to a float if its an array
        if isinstance(noise_params, np.ndarray):
            if noise_params.size == 1:
                noise_params = float(noise_params.item())

        # copy the source_density and noise_function and set fixed-parameters and priors

        # we start with the weak prior
        source_weak = copy.copy(self.source_density)
        source_weak.prior = self.source_prior_weak

        noise_weak = copy.copy(self.noise_function)
        noise_weak.prior = self.noise_prior_weak

        # weak prior
        Bayes_weak = BayesCylinderFitter(
            name="Bayes_weak_prior",

            source_density=source_weak,
            noise_function=noise_weak,

            source_bounds=self.source_bounds_weak,
            source_init=self.source_init,

            noise_bounds=self.noise_bounds_strong,
            noise_init=self.noise_init
        )

        # then the strong prior
        source_strong = copy.copy(self.source_density)
        source_strong.prior = self.source_prior_strong

        noise_strong = copy.copy(self.noise_function)
        noise_strong.prior = self.noise_prior_strong

        Bayes_strong = BayesCylinderFitter(
            name="Bayes_weak_prior",

            source_density=source_strong,
            noise_function=noise_strong,

            source_bounds=self.source_bounds_strong,
            source_init=self.source_init,

            noise_bounds=self.noise_bounds_strong,
            noise_init=self.noise_init
        )

        # correct Fitter
        source_fixed = copy.copy(self.source_density)
        source_fixed.fixed_params = source_params
        noise_fixed = copy.copy(self.noise_function)
        noise_fixed.fixed_params = noise_params

        Bayes_correct = BayesCylinderFitter(
            name="Bayes_correct",
            source_density=source_fixed,
            noise_function=noise_fixed
        )

        return [Bayes_weak, Bayes_strong, Bayes_correct]


    # --- run the experiment

    def run(self):
        n_fitters = len(self.fitters_static) + 3  # number of Static fitters + Bayesian Fitters
        fitting_params = np.zeros((self.iterations, n_fitters, self.shape.get_n_params()))

        m = 1 if isinstance(self.noise_params, float) else len(self.noise_params)

        # store per-iteration ground truth
        shape_gt_all = np.zeros((self.iterations, self.shape.get_n_params()))
        source_gt_all = np.zeros((self.iterations, len(self.source_params)))
        noise_gt_all = np.zeros((self.iterations, m))

        # store the fitting times as well
        fit_times = np.zeros((self.iterations, n_fitters))

        for it in tqdm(range(self.iterations), desc=self.name):

            self.resample()  # no-op if nothing is random
            shape_gt_all[it] = self.shape_params
            source_gt_all[it] = self.source_params
            noise_gt_all[it] = self.noise_params

            # sample points from the shape
            points, parameter_samples = self.get_measurement(self.num_points)

            # build the per-iteration fitter list
            fitters = self.fitters_static + self.build_bayes_fitters()
            #fitters = self.fitters_static + self.build_bayes_fitter_cylinder()

            for i, fitter in enumerate(fitters):
                try:
                    start = time.perf_counter()

                    params_fit = fitter.fit(
                        points,
                        uncertainties=self.noise_function.evaluate(
                            parameter_samples, params=self.noise_params
                        ),
                        shape_init=None,
                    )

                    end = time.perf_counter()
                    fit_times[it, i] = end - start

                    fitting_params[it, i, :] = params_fit
                except Exception as e:
                    fitting_params[it, i, :] = np.nan
                    fit_times[it, i] = np.nan
                    print(f"  {fitter.__class__.__name__} failed: {e}")

        return SweepResult(
            experiment_config=self,
            fitting_params=fitting_params,
            fitting_times=fit_times,
            shape_gt=shape_gt_all,
            source_gt=source_gt_all,
            noise_gt=noise_gt_all
        )
