"""
Here we define functions to simulate a measurement
"""
import numpy as np

from bayes_shape_fitting.inference.noise_functions import NoiseFunction
from bayes_shape_fitting.inference.source_densitites import SourceDensity
from bayes_shape_fitting.models.shapes.base_shape import Shape


def simulate_measurement(shape: Shape, theta: np.ndarray, source_density: SourceDensity, noise_function: NoiseFunction, num_samples = 100) -> np.ndarray:
    """
    Simulates a measurement based on a measurement model

    Parameters
    ----------
    shape: underlying shape class
    theta: shape parameter
    source_density: source_density of the model, needs to be fixed!
    noise_function: noise_function of the model, needs to be fixed!
    num_samples: number of sampled points

    Returns
    -------
    (num_samples, shape.ambient_dim) np.array of observed points
    """

    assert source_density.get_function_type().value == "fixed"
    assert noise_function.get_function_type().value == "fixed"
    assert hasattr(source_density, "sample")

    # first sample the parameters t
    parameter_samples = source_density.sample(n = num_samples)

    # based on this, compute the respective covariances:
    covariances = noise_function.evaluate(parameter_samples) # size (num_samples, n,n)

    # compute the respective source points
    source_points = shape.parametrization(parameter_samples, theta)

    # now sample from the covariances
    errors = np.array([np.random.multivariate_normal(np.zeros(shape.ambient_dim), cov) for cov in covariances])

    # we add the errors to the source
    measured_points = source_points + errors

    return measured_points


