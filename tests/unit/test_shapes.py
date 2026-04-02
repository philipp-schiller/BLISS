"""
Test the functionalities of the basic shape class
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.bayes_shape_fitting.models.shapes.shape_examples import shape_cases

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

def test_sample_parameter_space_grid(shape):
    # get the number of sampled points per dimension
    N = 100
    k = shape.parameter_dim
    sample = shape.sample_parameter_space_grid(n_points_per_dim=N)
    assert sample.shape == (N**k, k)


def test_sample_parameter_space_uniform(shape):
    # Number of samples
    N = 100
    k = shape.parameter_dim
    sample = shape.sample_parameter_space_uniform(size = N)
    assert sample.shape == (N,k)


def test_jacobian(shape, theta):
    # test the jacobian
    dim = shape.parameter_dim

    # we chose different t_values, based on the dimension
    if dim == 1:
        t = np.array([0.2, 0.4, 0.65, 0.9])
    if dim == 2:
        t = np.array([[0.2, 0.4, 0.65, 0.9], [0.1, 0.4, 0.5, 0.5]]).T
    dens = shape.metric_density(t, theta)
    assert np.all(dens > 0)
    assert len(dens) == 4
    print(shape.metric_density(t, theta))


def test_area(shape, theta):
    # test the area function
    area = shape.area(theta)
    assert area > 0
    print(area)


def test_get_domain_volume(shape):
    domain_volume = shape.get_domain_volume()
    print(domain_volume)
    assert domain_volume > 0


def test_get_n_params(shape):
    n = shape.get_n_params()
    assert n > 0
    print(n)
