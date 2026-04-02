import numpy as np
from dataclasses import dataclass, field
from src.bayes_shape_fitting.inference.noise_functions import NoiseFunction
from src.bayes_shape_fitting.inference.source_densitites import SourceDensity
from src.bayes_shape_fitting.models.shapes.base_shape import Shape
from tests.fixtures.sample_data import simulate_measurement


@dataclass
class GroundTruthConfig:

    shape: Shape

    # Density function over T
    source_density: SourceDensity

    # Noise function over T
    noise_function: NoiseFunction

    # Ground truth parameters
    shape_params: np.ndarray = field(default=None)
    source_params: np.ndarray = field(default=None)
    noise_params: np.ndarray = field(default=None)

    # for logging
    name: str = ""

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

        return measured_points


    def __post_init__(self):
        # remember which were unset (random) before we fill them in

        # first, check if the source density and noise function have fixed params
        self.source_params = self.source_density.fixed_params
        self.noise_params = self.noise_function.fixed_params

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
        angle = np.random.uniform(-np.pi/2, np.pi/2)
        return np.array([cx, cy, a, b, angle])

    def _random_source_params(self) -> np.ndarray:
        c = np.random.uniform(0, 1)
        w = np.random.uniform(0.15,0.25)
        return np.array([c, w])

    def _random_noise_params(self) -> np.ndarray:
        #sigma = np.random.uniform(0.01, 0.01) # for laser experiment
        sigma = np.random.uniform(0.001, 0.01)
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