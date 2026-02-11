import numpy as np
import pytest

from src.bayes_shape_fitting.inference.source_densitites import (
    FixedSourceDensity,
    BetaSourceDensity,
    SplineSourceDensity,
    MixtureSourceDensity,
)

from src.bayes_shape_fitting.inference.priors import GaussianPrior, UniformPrior


# ---------------------------------------------------------------------
# FIXED DENSITY
# ---------------------------------------------------------------------

def test_fixed_density_evaluate_correctly():
    # simple known density: p(x) = x in [0,1]
    f = FixedSourceDensity(lambda x: x[:, 0], dim=1)

    x = np.array([0.2, 0.5, 0.8])
    out = f.evaluate(x)

    assert np.allclose(out, x)


def test_fixed_density_sample_not_implemented():
    f = FixedSourceDensity(lambda x: x, dim=1)
    with pytest.raises(NotImplementedError):
        f.sample(10)


# ---------------------------------------------------------------------
# BETA DENSITY
# ---------------------------------------------------------------------

def test_beta_density_evaluate():
    beta = BetaSourceDensity(prior=None, fixed_params=np.array([2.0, 3.0]))

    x = np.array([0.2, 0.4, 0.8])
    out = beta.evaluate(x)

    assert out.shape == (3,)
    assert np.all(out >= 0)


def test_beta_density_sampling():
    beta = BetaSourceDensity(prior=None, fixed_params=np.array([2.0, 5.0]))

    samples = beta.sample(1000)
    assert samples.shape == (1000, 1)
    assert np.all(samples >= 0) and np.all(samples <= 1)


def test_beta_missing_params_raises():
    beta = BetaSourceDensity(prior=None, fixed_params=None)

    with pytest.raises(ValueError):
        beta.evaluate(np.array([0.2]))


# ---------------------------------------------------------------------
# SPLINE DENSITY
# ---------------------------------------------------------------------

def test_spline_density_evaluate_shape():
    prior = GaussianPrior(np.zeros(5), np.eye(5))
    spl = SplineSourceDensity(n_knots=5, prior=prior)

    params = np.zeros(5)
    x = np.linspace(0, 1, 20)

    out = spl.evaluate(x, params=params)
    assert out.shape == (20,)
    assert np.all(out >= 0)


def test_spline_density_samples_in_unit_interval():
    prior = GaussianPrior(np.zeros(5), np.eye(5))
    spl = SplineSourceDensity(n_knots=5, prior=prior)

    params = np.random.randn(5)
    samples = spl.sample(1000, params=params)

    assert samples.shape == (1000, 1)
    assert np.all(samples >= 0) and np.all(samples <= 1)


def test_spline_missing_params_raises():
    prior = GaussianPrior(np.zeros(5), np.eye(5))
    spl = SplineSourceDensity(n_knots=5, prior=prior)

    with pytest.raises(ValueError):
        spl.evaluate(np.array([0.5]))


# ---------------------------------------------------------------------
# MIXTURE DENSITY
# ---------------------------------------------------------------------

def test_mixture_density_evaluate():
    prior = GaussianPrior(np.zeros(6), np.eye(6))
    mixture = MixtureSourceDensity(n_components=2, prior=prior)

    params = np.array([0.0, 0.5, 0.0,     # mu1, sigma1, logit1
                       1.0, 0.5, 0.0])    # mu2, sigma2, logit2

    x = np.linspace(-3, 3, 50)
    out = mixture.evaluate(x, params=params)

    assert out.shape == (50,)
    assert np.all(out > 0)


def test_mixture_density_sampling():
    prior = GaussianPrior(np.zeros(6), np.eye(6))
    mixture = MixtureSourceDensity(n_components=2, prior=prior)

    params = np.array([0.0, 0.3, 1.0,
                       1.5, 0.4, -1.0])

    samples = mixture.sample(2000, params=params)
    assert samples.shape == (2000, 1)
    # mixture support is real line, but check for finite values
    assert np.all(np.isfinite(samples))


def test_mixture_missing_params_raises():
    prior = GaussianPrior(np.zeros(3), np.eye(3))
    mixture = MixtureSourceDensity(n_components=1, prior=prior)

    with pytest.raises(ValueError):
        mixture.evaluate(np.array([0.0]))