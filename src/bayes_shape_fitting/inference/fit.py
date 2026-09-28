"""
This file contains a fitting function
"""
from bayes_shape_fitting.inference.likelihoods import Likelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.models.shapes.base_shape import Shape
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.inference.samplers.ecm_estimation import ECMOptimizer
import numpy as np

def fit(points: np.ndarray, shape: Shape, likelihood: Likelihood, shape_init = None, optimizer = None, **kwargs):

    # ── posterior ─────────────────────────────────────────────────────
    posterior = MarginalPosterior(
        shape=shape,
        measurement=points,
        likelihood=likelihood,
    )

    if shape_init is None:
        shape_init = shape.initial_fit(points)

    # ── optional inputs (None if not provided) ───────────────────
    source_init = kwargs.get("source_init", None)
    noise_init = kwargs.get("noise_init", None)

    shape_bounds = kwargs.get("shape_bounds", None)

    if shape_bounds is None:
        shape_bounds = shape.bounds

    source_bounds = kwargs.get("source_bounds", None)
    noise_bounds = kwargs.get("noise_bounds", None)

    # check if there are noise or source bounds in args / kwargs

    #optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True, ftol=1e-5, gtol=1e-4, maxiter=100)

    if optimizer is None:
        optimizer = ECMOptimizer(verbose=False, scale_factor=1, max_iter=10, inner_max_iter=100, methods= {
                'shape': 'L-BFGS-B',
                'source': 'Powell',
                'noise': 'Powell'
            })

    result = optimizer.optimize(
        posterior,
        shape_init=shape_init,
        source_init=source_init,
        noise_init=noise_init,
        shape_bounds=shape_bounds,
        source_bounds=source_bounds,
        noise_bounds=noise_bounds,
    )

    return result