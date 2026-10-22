"""
This file contains a fitting function
"""
from bliss.inference.source_densitites import SourceDensity
from bliss.inference.noise_functions import NoiseFunction
from bliss.inference.likelihoods import Likelihood, MarginalLikelihood
from bliss.inference.posterior import MarginalPosterior
from bliss.models.shapes.base_shape import Shape
from bliss.inference.samplers.map_estimation import MAPOptimizer
from bliss.inference.samplers.ecm_estimation import ECMOptimizer
import numpy as np

def fit(
    points: np.ndarray,
    shape: Shape,
    *,
    source_density: SourceDensity | None = None,
    noise_function: NoiseFunction | None = None,
    likelihood_kwargs: dict | None = None,
    optimizer=None,
    shape_init=None,
    source_init=None,
    noise_init=None,
    shape_bounds=None,
    source_bounds=None,
    noise_bounds=None,
):

    # -------------------------------------------------
    # Default models
    # -------------------------------------------------

    if source_density is None:
        source_density = shape.default_source_density()

    if noise_function is None:
        noise_function = shape.default_noise_function()

    # -------------------------------------------------
    # Initial values
    # -------------------------------------------------

    if shape_init is None:

        if shape.initial_fit is None:
            raise ValueError(
                f"{shape.__class__.__name__} does not provide "
                "an initial_fit routine. Please specify shape_init."
            )

        shape_init = shape.initial_fit(points)

    if shape_bounds is None:
        shape_bounds = shape.bounds

    if getattr(noise_function, "fixed_params", None) is not None:
        noise_init = None
        noise_bounds = []
    else:

        if noise_bounds is None:
            noise_bounds = noise_function.get_bounds()

        if noise_init is None:

            residuals = shape.geometric_residuals(
                shape_init,
                points,
            )

            noise_init = np.array([
                np.median(residuals ** 2)
            ])

    if getattr(source_density, "fixed_params", None) is not None:
        source_init = None
        source_bounds = []
    else:

        if source_bounds is None:
            source_bounds = source_density.get_parameter_bounds()

        if source_init is None:
            source_init = np.array([
                (low + high) / 2
                for low, high in source_bounds
            ])

    # -------------------------------------------------
    # Construct likelihood
    # -------------------------------------------------

    if likelihood_kwargs is None:
        likelihood_kwargs = {}

    # sensible defaults depending on manifold dimension
    if "num_samples" not in likelihood_kwargs:
        likelihood_kwargs["num_samples"] = (
            60 if shape.parameter_dim >= 2 else 300
        )

    likelihood = MarginalLikelihood(
        shape=shape,
        source_density=source_density,
        noise_function=noise_function,
        **likelihood_kwargs,
    )

    posterior = MarginalPosterior(
        shape=shape,
        measurement=points,
        likelihood=likelihood,
    )

    # -------------------------------------------------
    # Default optimizer
    # -------------------------------------------------

    if optimizer is None:
        optimizer = ECMOptimizer(
            verbose=False,
            scale_factor=1,
            max_iter=20,
            inner_max_iter=100,
            tol_lp=1e-4,
            methods={
                "shape": "L-BFGS-B",
                "source": "Powell",
                "noise": "Powell",
            },
        )

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

def _fitting(points: np.ndarray, shape: Shape, likelihood: Likelihood, shape_init = None, optimizer = None, **kwargs):

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

    if optimizer is None:
        optimizer = ECMOptimizer(verbose=False, scale_factor=1, max_iter=20, inner_max_iter=100, methods= {
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