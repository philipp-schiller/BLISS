import numpy as np
from typing import Optional

from bayes_shape_fitting.inference.samplers.ecm_estimation import ECMOptimizer
from bayes_shape_fitting.models.shapes.ellipsoid import Ellipsoid_R

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.noise_functions   import NoiseFunction
from bayes_shape_fitting.inference.posterior         import MarginalPosterior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer

from bayes_shape_fitting.inference.source_densitites import SourceDensity
from scripts.Benchmarking.Fitters.Ellipsoid_fitters.base_ellipsoid import BaseEllipsoidFitter

from bayes_shape_fitting.inference.fit import fit as standard_fit



class BayesEllipsoidFitter(BaseEllipsoidFitter):
    """
    MAP ellipsoid fitter.

    Parameters
    ----------
    source_density : SourceDensity
        Prior / density over the (u, v) parameter space.
    noise_function : NoiseFunction
        Noise model (e.g. IsotropicNoise).
    shape_init : array (7,), optional
        Initial cylinder params.  If None, uses LeastSquaresCylinderFitter.
    source_init / noise_init : arrays, optional
        Initial source / noise params for joint optimization.
    source_bounds / noise_bounds : list of (lo, hi), optional
    use_ecm : bool
        If True, use ECM (alternating); otherwise CMA-ES.
    """

    def __init__(
        self,
        name: str = "BayesEllipsoidFitter",
        source_density: Optional[SourceDensity] = None,
        noise_function:  Optional[NoiseFunction]  = None,
        shape_init:  Optional[np.ndarray] = None,
        source_init: Optional[np.ndarray] = None,
        noise_init:  Optional[np.ndarray] = None,
        source_bounds=None,
        noise_bounds=None,
        num_samples: int = 50,          # integration grid per dimension
    ):
        self.name          = name
        self.source_density = source_density
        self.noise_function = noise_function
        self.shape_init    = shape_init
        self.source_init   = source_init
        self.noise_init    = noise_init
        self.source_bounds = source_bounds
        self.noise_bounds  = noise_bounds
        self.num_samples   = num_samples

    # ──────────────────────────────────────────────────────────────────────

    def fit(self, points: np.ndarray, source_density: SourceDensity = None, noise_function: NoiseFunction = None,
            shape_init: np.ndarray = None, source_init: np.ndarray = None, noise_init: np.ndarray = None,
            source_bounds=None, noise_bounds=None, minimize_kwargs=None, *args, **kwargs) -> np.ndarray:

        # use instance defaults, but allow call-time override
        source_density = source_density or self.source_density
        noise_function = noise_function or self.noise_function

        shape_init = shape_init if shape_init is not None else self.shape_init
        source_init = source_init if source_init is not None else self.source_init
        noise_init = noise_init if noise_init is not None else self.noise_init
        source_bounds = source_bounds if source_bounds is not None else self.source_bounds
        noise_bounds = noise_bounds if noise_bounds is not None else self.noise_bounds

        optimizer = ECMOptimizer(verbose=True, scale_factor=1, max_iter=10, inner_max_iter=10, methods={
            'shape': 'L-BFGS-B',
            'source': 'L-BFGS-B',
            'noise': 'L-BFGS-B'
        })

        shape = Ellipsoid_R()

        likelihood = MarginalLikelihood(
            shape=shape,
            source_density=self.source_density,
            noise_function=self.noise_function,
            num_samples=100,
        )

        fit_result = standard_fit(points=points, shape=shape, likelihood=likelihood, shape_init=shape_init,
                                  source_init=source_init, noise_init=noise_init, shape_bounds=shape.bounds,
                                  source_bounds=source_bounds, noise_bounds=noise_bounds,
                                  optimizer=optimizer)

        return fit_result.shape_params