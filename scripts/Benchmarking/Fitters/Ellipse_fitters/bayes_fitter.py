import numpy as np

from scripts.Benchmarking.Fitters.Ellipse_fitters.base import BaseEllipseFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.direct_ellipse_fit import DirectEllipseFit
from scripts.Benchmarking.Fitters.Ellipse_fitters.ellipse_utils import convert_ellipse_params
from scripts.Benchmarking.Fitters.Ellipse_fitters.least_squares import LeastSquaresFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.self_normalized import SelfNormalizedFitter
from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.noise_functions import NoiseFunction
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.samplers.map_estimation_global import CMAESOptimizer
from bayes_shape_fitting.inference.source_densitites import SourceDensity
from bayes_shape_fitting.models.shapes.ellipse_reparametrized import Ellipse_R


class BayesFitter(BaseEllipseFitter):

    def __init__(
            self,
            name: str = "BayesFitter",
            source_density: SourceDensity = None,
            noise_function: NoiseFunction = None,
            shape_init: np.ndarray = None,
            source_init: np.ndarray = None,
            noise_init: np.ndarray = None,
            source_bounds=None,
            noise_bounds=None,
    ):
        self.name = name
        self.source_density = source_density
        self.noise_function = noise_function
        self.shape_init = shape_init
        self.source_init = source_init
        self.noise_init = noise_init
        self.source_bounds = source_bounds
        self.noise_bounds = noise_bounds

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
        shape = Ellipse_R()
        # define the likelihood
        likelihood = MarginalLikelihood(shape=shape, source_density=source_density,
                                                 noise_function=noise_function, num_samples=300)

        # --- 1. Initial fit (may fail) ---
        if shape_init is None:
            ls = SelfNormalizedFitter()
            try:
                shape_init = ls.fit(points)
            except Exception as e:
                print(f"Least-squares init failed: {e}")
                shape_init = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        # 2. Bounds
        bounds = [
            (-5, 5),  # x_c
            (-5, 5),  # y_c
            (0.1, 5),  # a
            (0.1, 5), # b
            (-np.pi, np.pi)  # rotation
        ]

        # 3. Check each parameter against bounds
        in_bounds = all(
            lower <= val <= upper
            for val, (lower, upper) in zip(shape_init, bounds)
        )

        # 4. If invalid → fallback to unit circle
        if not in_bounds or np.any(np.isnan(shape_init)):
            dls = DirectEllipseFit()
            shape_init = dls.fit(points)

        # we convert the initial_fit to the other parametrization

        shape_init_converted = convert_ellipse_params(shape_init)

        # now define the posterior
        posterior = MarginalPosterior(shape=shape, measurement=points, likelihood=likelihood)

        #optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True, ftol=1e-7, gtol=1e-4, maxiter=1000)
        #optimizer = MAPOptimizer(method='Powell', verbose=False, ftol=1e-7, gtol=1e-4, maxiter=1000)

        CMA_stds = [
            0.5,  # cx
            0.5,  # cy
            0.2,  # a
            0.2,  # b
            0.1  # angle
        ]

        if source_init is not None:
            # std for source function
            CMA_stds += [
                1,  # c
                0.1,  # w
            ]
        if noise_init is not None:
            CMA_stds += [
                0.01, # sigma
            ]

        optimizer = CMAESOptimizer(sigma0=0.01,maxiter=200,verbose=True)

        #result = optimizer.optimize(posterior, shape_init=shape_init_converted,source_init= source_init,noise_init=noise_init, shape_bounds=shape.bounds, source_bounds=source_bounds, noise_bounds=noise_bounds, CMA_stds=CMA_stds)
        result = optimizer.optimize(posterior, shape_init=shape_init_converted,source_init= source_init,noise_init=noise_init, shape_bounds=shape.bounds, source_bounds=source_bounds, noise_bounds=noise_bounds)

        print("Full solution: " + str(result.theta_map))

        theta = result.theta_map[:5] # only the first 5 entries resemble the ellipse
        # convert the minor axis back to standard form
        theta[3] = theta[3]*theta[2]

        return theta
