import numpy as np
import pytest

from bayes_shape_fitting.inference.source_densitites import (
    FixedSourceDensity,
    BetaSourceDensity,
    UniformSourceDensity
)

from bayes_shape_fitting.inference.priors import GaussianPrior, UniformPrior


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
# Uniform DENSITY
# ---------------------------------------------------------------------

def test_uniform_density_evaluate():
    uni = UniformSourceDensity(prior=None, n_dim = 2, domain_low=[0,0], domain_high=[1,1], fixed_params=np.array([0,1,0,2*np.pi]))

    x = np.array([0.2, 0.4]).reshape(1,-1)
    out = uni.evaluate(x)

    assert out.shape == (1,)
    assert np.all(out >= 0)


def test_unform_density_sampling():
    uni = UniformSourceDensity(prior=None, n_dim = 2, domain_low=[0,0], domain_high=[1,1], fixed_params=np.array([0,1,0,2*np.pi]))

    samples = uni.sample(1000)
    assert samples.shape == (1000, uni.n_dim)
