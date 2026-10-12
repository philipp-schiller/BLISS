import numpy as np
import pytest

from bayes_shape_fitting.inference.noise_functions import (
    FunctionType,
    FixedNoiseFunction,
    IsotropicNoise,
    ParameterDependentNoise,
    AnisotropicNoise,
    LaserDependentEllipseNoise
)


# -----------------------------
# Utilities
# -----------------------------

def make_fixed_noise_identity_scaling(dim_param, dim_ambient=2):
    def fn(t):
        # t shaped (M, dim_param). Use a simple scalar weight w_i = 1 + sum(t_i)^2
        t = np.asarray(t)
        M = t.shape[0]
        w = 1.0 + (t ** 2).sum(axis=1)   # (M,)
        cov = np.zeros((M, dim_ambient, dim_ambient))
        for i in range(M):
            cov[i] = w[i] * np.eye(dim_ambient)
        return cov
    return fn


# -----------------------------
# FixedNoiseFunction
# -----------------------------

def test_fixed_noise_evaluate_shape_and_values():
    dim_param, dim_ambient = 1, 3
    f = FixedNoiseFunction(
        noise_function=make_fixed_noise_identity_scaling(dim_param, dim_ambient),
        dim_param=dim_param,
        dim_ambient=dim_ambient,
    )
    t = np.linspace(0.0, 1.0, 5)  # (M,)
    cov = f.evaluate(t)
    assert cov.shape == (5, dim_ambient, dim_ambient)
    # Check diagonal increases with t^2
    d0 = np.diag(cov[0])
    d4 = np.diag(cov[-1])
    assert np.all(d4 > d0)
    assert f.get_function_type() == FunctionType.FIXED
    assert f.get_n_params() == 0
    assert f.get_prior() is None


def test_fixed_noise_bad_return_shape_raises():
    def bad_fn(t):
        return np.zeros((3, 2, 2))  # wrong M
    f = FixedNoiseFunction(bad_fn, dim_param=1, dim_ambient=2)
    with pytest.raises(ValueError):
        f.evaluate(np.array([0.1, 0.2]))


# -----------------------------
# IsotropicNoise
# -----------------------------

def test_isotropic_noise_fixed_sigma():
    n = 3
    iso = IsotropicNoise(dim_param=1, dim_ambient=n, fixed_params=0.5)
    t = np.linspace(0.0, 1.0, 7)
    cov = iso.evaluate(t)
    assert cov.shape == (7, n, n)
    assert np.allclose(cov[0], 0.5 * np.eye(n))
    assert iso.get_function_type() == FunctionType.FIXED
    assert iso.get_n_params() == 0
    assert iso.get_prior() is None


def test_isotropic_noise_missing_params_raises():
    iso = IsotropicNoise(dim_param=1, dim_ambient=2, fixed_params=None)
    with pytest.raises(ValueError):
        iso.evaluate(np.array([0.0, 0.1]))


# -----------------------------
# ParameterDependentNoise
# -----------------------------

def test_parameter_dependent_noise_shapes_and_monotonicity():
    n = 2
    pdn = ParameterDependentNoise(dim_ambient=n, prior=None)
    t = np.linspace(0.0, 1.0, 5)
    a, b = np.log(0.5), 0.7  # sigma(t) = exp(a + b t) increases with t
    cov = pdn.evaluate(t, params=np.array([a, b]))
    assert cov.shape == (5, n, n)
    d0 = np.diag(cov[0])
    d4 = np.diag(cov[-1])
    assert np.all(d4 > d0)
    assert pdn.get_function_type() == FunctionType.PARAMETRIC
    assert pdn.get_n_params() == 2
    assert pdn.get_prior() is None


def test_parameter_dependent_missing_params_raises():
    pdn = ParameterDependentNoise(dim_ambient=2, prior=None)
    with pytest.raises(ValueError):
        pdn.evaluate(np.array([0.0, 0.5]))


# -----------------------------
# AnisotropicNoise
# -----------------------------

def test_anisotropic_noise_global_params_2d():
    ani = AnisotropicNoise(dim_param=1, dim_ambient=2, prior=None, parameter_dependent=False)
    # params for L (lower-tri): [l11_log, l21, l22_log]
    params = np.array([np.log(2.0), 0.1, np.log(0.5)])
    t = np.linspace(0, 1, 4)
    cov = ani.evaluate(t, params=params)
    assert cov.shape == (4, 2, 2)
    # All covariances identical across M when parameter_dependent == False
    assert np.allclose(cov[0], cov[-1])
    assert ani.get_function_type() == FunctionType.PARAMETRIC
    assert ani.get_n_params() == 3
    assert ani.get_prior() is None


def test_anisotropic_noise_parameter_dependent():
    ani = AnisotropicNoise(dim_param=1, dim_ambient=2, parameter_dependent=True)
    t = np.linspace(0, 1, 3)  # M=3
    # two different Ls; broadcast to all 3 should fail unless exact shape or broadcasted vector
    p0 = np.array([np.log(1.0), 0.0, np.log(1.0)])  # identity
    p1 = np.array([np.log(2.0), 0.2, np.log(0.5)])
    params = np.vstack([p0, p1, p0])  # (M, n_chol)
    cov = ani.evaluate(t, params=params)
    assert cov.shape == (3, 2, 2)
    # different covariances for row 0 and 1
    assert not np.allclose(cov[0], cov[1])

    # test broadcast from single vector (n_chol,)
    cov_b = ani.evaluate(t, params=p0)
    assert np.allclose(cov_b[0], cov_b[1])