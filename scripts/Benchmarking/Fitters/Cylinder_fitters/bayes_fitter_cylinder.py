"""
Fitters/Cylinder_fitters/bayes_fitter_cylinder.py

Bayesian MAP cylinder fitter using MarginalLikelihood + ECM / CMA-ES.
Mirrors the structure of BayesFitter for ellipses but targets the Cylinder shape.
"""

import numpy as np
from typing import Optional

from scripts.Benchmarking.Fitters.Cylinder_fitters.base_cylinder import BaseCylinderFitter
from scripts.Benchmarking.Fitters.Cylinder_fitters.least_squares_cylinder import LeastSquaresCylinderFitter
from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.noise_functions   import NoiseFunction
from bayes_shape_fitting.inference.posterior         import MarginalPosterior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer

from bayes_shape_fitting.inference.source_densitites import SourceDensity
from bayes_shape_fitting.models.shapes.cylinder      import Cylinder


class BayesCylinderFitter(BaseCylinderFitter):
    """
    MAP cylinder fitter.

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
        name: str = "BayesCylinderFitter",
        source_density: Optional[SourceDensity] = None,
        noise_function:  Optional[NoiseFunction]  = None,
        shape_init:  Optional[np.ndarray] = None,
        source_init: Optional[np.ndarray] = None,
        noise_init:  Optional[np.ndarray] = None,
        source_bounds=None,
        noise_bounds=None,
        use_ecm: bool = True,
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
        self.use_ecm       = use_ecm
        self.num_samples   = num_samples

    # ──────────────────────────────────────────────────────────────────────

    def fit(self, points: np.ndarray, *args, **kwargs) -> np.ndarray:

        shape = Cylinder()

        likelihood = MarginalLikelihood(
            shape=shape,
            source_density=self.source_density,
            noise_function=self.noise_function,
            num_samples=50,
        )

        """likelihood = ClosestPointLikelihood(
            shape=shape,
            source_density=self.source_density,
            noise_function=self.noise_function,
        )"""


        """ # ── initial shape estimate ─────────────────────────────────────────
        shape_init = self.shape_init
        if shape_init is None:
            try:
                shape_init = LeastSquaresCylinderFitter().fit(points)
            except Exception:
                print("We go to baseline")
                shape_init = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 0.3, 0.5])

        if np.any(np.isnan(shape_init)):
            shape_init = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 0.3, 0.5])"""

        shape_init = np.array([0.2, 1.2, -0.3, 2.0, 8.5, 0.6, 0.1])

        # ── posterior ─────────────────────────────────────────────────────
        posterior = MarginalPosterior(
            shape=shape,
            measurement=points,
            likelihood=likelihood,
        )

        # ── shape parameter bounds ─────────────────────────────────────────
        shape_bounds = shape.bounds  # list of 7 (lo, hi) pairs

        shape_bounds = [
            (-5, 5),  # x_c
            (-5, 5),  # y_c
            (-5, 5),  # z_c
            (0.1, 5.0),  # r
            (1, 20.0),  # h
            (0.0, np.pi),  # polar
            (0.0, 2 * np.pi)  # azim
        ]

        # CMA-ES with per-parameter scales
        CMA_stds = [0.3, 0.3, 0.3,   # center xyz
                    0.1,              # r
                    0.2,              # h
                    0.05, 0.1]        # polar, azim
        if self.source_init is not None:
            dim_src = len(self.source_init)
            CMA_stds += [0.3, 0.1] * (dim_src // 2)
        if self.noise_init is not None:
            CMA_stds += [0.005] * len(np.atleast_1d(self.noise_init))

        #optimizer = CMAESOptimizer(sigma0=0.1, maxiter=20, verbose=True)

        optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True, ftol=1e-5, gtol=1e-4, maxiter=100)

        result = optimizer.optimize(
            posterior,
            shape_init=shape_init,
            source_init=self.source_init,
            noise_init=self.noise_init,
            shape_bounds=shape_bounds,
            source_bounds=self.source_bounds,
            noise_bounds=self.noise_bounds,
            #CMA_stds=CMA_stds
        )

        # Return only the 7 shape params
        return np.array(result.theta_map[:7])
