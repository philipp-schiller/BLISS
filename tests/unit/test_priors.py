import numpy as np
import pytest
from bayes_shape_fitting.inference.priors import GaussianPrior, UniformPrior   # adjust import to your file name


# -----------------------------
# GaussianPrior Tests
# -----------------------------
def test_gaussian_log_prob_shape_match():
    mean = np.zeros(3)
    cov = np.eye(3)
    prior = GaussianPrior(mean, cov)

    value = np.array([0.1, -0.2, 0.3])
    lp = prior.log_prob(value)

    assert isinstance(lp, float)
    assert np.isfinite(lp)


def test_gaussian_sampling_shape():
    mean = np.zeros(2)
    cov = np.eye(2)
    prior = GaussianPrior(mean, cov)

    samples = prior.sample(size=5)
    assert samples.shape == (5, 2)
    assert np.all(np.isfinite(samples))


def test_gaussian_sampling_mean_variance():
    mean = np.array([1.0, -2.0])
    cov = np.eye(2)
    prior = GaussianPrior(mean, cov)

    samples = prior.sample(size=50000)
    assert np.allclose(samples.mean(axis=0), mean, atol=0.1)
    assert np.allclose(np.cov(samples.T), cov, atol=0.1)


# -----------------------------
# UniformPrior Tests
# -----------------------------
def test_uniform_log_prob_inside():
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0, 2.0])
    prior = UniformPrior(lower, upper)

    value = np.array([0.5, 1.5])
    lp = prior.log_prob(value)

    expected = -np.sum(np.log(upper - lower))
    assert np.isclose(lp, expected)


def test_uniform_log_prob_outside():
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0, 1.0])
    prior = UniformPrior(lower, upper)

    value = np.array([1.5, 0.2])  # outside bounds
    lp = prior.log_prob(value)

    assert lp == -np.inf


def test_uniform_sampling_shape():
    lower = np.array([0.0, 1.0])
    upper = np.array([1.0, 3.0])
    prior = UniformPrior(lower, upper)

    samples = prior.sample(size=10)
    assert samples.shape == (10, 2)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)


def test_uniform_sampling_distribution():
    lower = np.array([0.0])
    upper = np.array([1.0])
    prior = UniformPrior(lower, upper)

    samples = prior.sample(size=50000)

    # mean of U(0,1) should be 0.5
    assert np.allclose(samples.mean(), 0.5, atol=0.02)