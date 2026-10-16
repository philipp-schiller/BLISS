"""
Tests for the public fit() interface.
"""

import numpy as np
import pytest

from bayes_shape_fitting.inference.fit import fit
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement


def _case_id(c):
    return c["name"]


@pytest.fixture(params=shape_cases, ids=_case_id)
def case(request):
    return request.param


@pytest.fixture
def shape(case):
    return case["shape"]


@pytest.fixture
def theta(case):
    return case["theta"]

@pytest.fixture
def init_theta(case):
    return case["init_params"]

@pytest.fixture
def source_density(shape):
    dim = shape.parameter_dim

    return UniformSourceDensity(
        n_dim=dim,
        fixed_params=np.array(dim * [0.5, 0.3]),
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
    )


@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(
        dim_param=shape.parameter_dim,
        dim_ambient=shape.ambient_dim,
        fixed_params=0.1,
    )


@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(
        shape,
        theta,
        source_density,
        noise_function,
        num_samples=65,
    )

class TestFitDefaults:

    def test_fit_defaults(
        self,
        shape,
        measurement,
        init_theta
    ):
        result = fit(
            measurement,
            shape,
            shape_init = init_theta
        )

        assert result is not None
        assert result.shape_params is not None
        assert np.all(np.isfinite(result.shape_params))

class TestFitSourceDensity:

    def test_fit_custom_source_density(
        self,
        shape,
        measurement,
        source_density,
        init_theta,
    ):
        result = fit(
            measurement,
            shape,
            source_density=source_density,
            shape_init=init_theta,
        )

        assert result is not None
        assert np.all(np.isfinite(result.shape_params))

class TestFitNoiseFunction:

    def test_fit_custom_noise_function(
        self,
        shape,
        measurement,
        noise_function,
        init_theta,
    ):
        result = fit(
            measurement,
            shape,
            noise_function=noise_function,
            shape_init=init_theta,
        )

        assert result is not None
        assert np.all(np.isfinite(result.shape_params))

class TestLikelihoodKwargs:

    def test_fit_num_samples(
        self,
        shape,
        measurement,
        init_theta,
    ):
        result = fit(
            measurement,
            shape,
            shape_init=init_theta,
            likelihood_kwargs={
                "num_samples": 50,
            },
        )

        assert result is not None

class TestNoiseInitialization:

    def test_fit_noise_init(
        self,
        shape,
        measurement,
        init_theta,
    ):
        result = fit(
            measurement,
            shape,
            shape_init=init_theta,
            noise_init=np.array([0.05]),
        )

        assert result is not None
        assert result.noise_params is not None

class TestSourceInitialization:

    def test_fit_source_init(
        self,
        shape,
        measurement,
        init_theta,
    ):
        dim = shape.parameter_dim

        source_density = UniformSourceDensity(
            n_dim=dim,
            domain_low=shape.domain_low,
            domain_high=shape.domain_high,
            periodic=shape.periodic,
        )

        source_init = np.array(dim * [0.5, 0.25])

        result = fit(
            measurement,
            shape,
            shape_init=init_theta,
            source_density=source_density,
            source_init=source_init,
        )

        assert result is not None

class TestBounds:

    def test_fit_shape_bounds(
        self,
        shape,
        measurement,
        init_theta,
    ):
        result = fit(
            measurement,
            shape,
            shape_init=init_theta,
            shape_bounds=shape.bounds,
        )

        assert result is not None

    def test_fit_source_bounds(
        self,
        shape,
        measurement,
        init_theta,
    ):
        dim = shape.parameter_dim

        source_density = UniformSourceDensity(
            n_dim=dim,
            domain_low=shape.domain_low,
            domain_high=shape.domain_high,
            periodic=shape.periodic,
        )

        result = fit(
            measurement,
            shape,
            shape_init=init_theta,
            source_density=source_density,
            source_bounds=source_density.get_parameter_bounds(),
        )

        assert result is not None

class TestOptimizationResult:

    def test_result_fields(
        self,
        shape,
        measurement,
        init_theta,
    ):
        result = fit(
            measurement,
            shape,
            shape_init=init_theta,
        )

        assert hasattr(result, "shape_params")
        assert hasattr(result, "source_params")
        assert hasattr(result, "noise_params")

        assert hasattr(result, "log_posterior_final")
        assert hasattr(result, "runtime")
        assert hasattr(result, "converged")

        assert np.isfinite(result.log_posterior_final)