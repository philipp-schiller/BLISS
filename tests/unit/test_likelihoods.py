"""
Test the functionalities of likelihood functions
"""

import time

import numpy as np
import pytest

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood, ClosestPointLikelihood, \
    MarginalLikelihoodEfficient, MarginalLikelihoodSum
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.compute_normals_bayesian import compute_normals_bayes
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
def source_density(shape):
    dim = shape.parameter_dim
    return UniformSourceDensity(n_dim=dim, fixed_params=np.array(dim*[0.3,0.1]), domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)

@pytest.fixture
def noise_function(shape):
    return IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)

@pytest.fixture
def measurement(shape, theta, source_density, noise_function):
    return simulate_measurement(shape, theta, source_density, noise_function, num_samples=65)

# ---------------------------------------------------------------------
# Marginal likelihood
# ---------------------------------------------------------------------

class TestMarginalLikelihood:

    def test_log_likelihood(self, shape, theta, measurement, source_density, noise_function):
        """
        Test the basic log_likelihood function
        """
        likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)
        likelihood_values = likelihood.log_likelihood(measurement = measurement, shape_params=theta)
        assert len(likelihood_values) == len(measurement)

    def test_log_likelihood_normals(self, shape, theta, measurement, source_density, noise_function):
        """
        Test the basic log_likelihood function with normals
        """

        normals, normals_uncertainty = compute_normals_bayes(measurement)
        normals_cov = [normals_uncertainty[i].to_covariance() for i in range(len(normals))]

        likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)
        likelihood_values = likelihood.log_likelihood(measurement = measurement, shape_params=theta, normals=normals, normals_cov=normals_cov)
        assert len(likelihood_values) == len(measurement)

    def test_log_likelihood_variable(self, shape, theta, measurement):
        """
        Test the basic log_likelihood function with variable source-and noise parameters
        """

        dim = shape.parameter_dim
        source_density_var = UniformSourceDensity(n_dim=dim, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic)
        source_params = np.array(dim*[0.3,0.1])

        noise_function_var = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim)
        noise_params = np.array([0.1])

        likelihood = MarginalLikelihood(shape=shape, source_density=source_density_var, noise_function=noise_function_var)
        likelihood_values = likelihood.log_likelihood(shape_params=theta, source_params=source_params, noise_params=noise_params, measurement=measurement)
        assert len(likelihood_values) == len(measurement)


class TestClosestPointLikelihood:

    def test_log_likelihood_fixed(self, shape, theta, measurement, source_density, noise_function):
        """
        Test the basic log_likelihood function with fixed source_density and noise function
        """
        likelihood = ClosestPointLikelihood(shape=shape, source_density=source_density, noise_function=noise_function)

        start = time.perf_counter()
        likelihood_values = likelihood.log_likelihood(shape_params=theta, measurement=measurement)
        end = time.perf_counter()

        elapsed = end - start
        print(f"log_likelihood computation took {elapsed:.6f} seconds")

        assert len(likelihood_values) == len(measurement)

class TestEfficient:

    def test_log_likelihood_efficient(self, shape, theta, measurement, source_density, noise_function):
        """
        Test the basic log_likelihood function
        """
        likelihood = MarginalLikelihoodEfficient(shape=shape, source_density=source_density, noise_function=noise_function)
        likelihood_values = likelihood.log_likelihood(measurement=measurement, shape_params=theta)
        assert len(likelihood_values) == len(measurement)

class TestComparison:

    def test_comparison(self, shape, theta, measurement, source_density, noise_function):
        """
        Compares the efficient and the standard one, and measures runtime.
        """
        likelihood_eff = MarginalLikelihoodEfficient(
            shape=shape,
            source_density=source_density,
            noise_function=noise_function,
            num_samples=1000
        )
        likelihood = MarginalLikelihoodSum(
            shape=shape,
            source_density=source_density,
            noise_function=noise_function,
            num_samples=1000
        )

        likelihood_sum = MarginalLikelihoodSum(
            shape=shape,
            source_density=source_density,
            noise_function=noise_function,
            num_samples=5000
        )

        # --- standard ---
        t0 = time.perf_counter()
        likelihood_values = likelihood.log_likelihood(
            measurement=measurement,
            shape_params=theta,
        )
        t1 = time.perf_counter()

        # --- efficient ---
        t2 = time.perf_counter()
        likelihood_values_eff = likelihood_eff.log_likelihood(
            measurement=measurement,
            shape_params=theta,
        )
        t3 = time.perf_counter()

        # --- sum ---
        t4 = time.perf_counter()
        likelihood_values_sum = likelihood_sum.log_likelihood(
            measurement=measurement,
            shape_params=theta,
        )
        t5 = time.perf_counter()

        #assert np.allclose(likelihood_values, likelihood_values_eff)

        standard_time = t1 - t0
        efficient_time = t3 - t2
        sum_time = t5 - t4

        print(f"Standard time:  {standard_time:.6f} s")
        print(f"Efficient time: {efficient_time:.6f} s")
        print(f"Speedup efficient:        {standard_time / efficient_time:.2f}×")

        print(f"Sum time: {sum_time:.6f} s")
        print(f"Speedup sum:        {standard_time / sum_time:.2f}×")

        # --- differences (vector-wise) ---
        diff_eff = likelihood_values_eff - likelihood_values
        diff_sum = likelihood_values_sum - likelihood_values

        abs_diff_eff = np.abs(diff_eff)
        abs_diff_sum = np.abs(diff_sum)

        rel_diff_eff = abs_diff_eff / (np.abs(likelihood_values) + 1e-12)
        rel_diff_sum = abs_diff_sum / (np.abs(likelihood_values) + 1e-12)

        print("\n--- Mean log-likelihood ---")
        print(f"Standard:   {likelihood_values.mean():.6f}")
        print(f"Efficient:  {likelihood_values_eff.mean():.6f}")
        print(f"Sum:        {likelihood_values_sum.mean():.6f}")

        # --- absolute error stats ---
        print("\n--- Absolute error statistics ---")
        print(f"Eff vs Std: mean={abs_diff_eff.mean():.6e}, "
              f"median={np.median(abs_diff_eff):.6e}, "
              f"max={abs_diff_eff.max():.6e}")

        print(f"Sum vs Std: mean={abs_diff_sum.mean():.6e}, "
              f"median={np.median(abs_diff_sum):.6e}, "
              f"max={abs_diff_sum.max():.6e}")

        # --- relative error stats ---
        print("\n--- Relative error statistics ---")
        print(f"Eff vs Std: mean={rel_diff_eff.mean():.6e}, "
              f"median={np.median(rel_diff_eff):.6e}, "
              f"max={rel_diff_eff.max():.6e}")

        print(f"Sum vs Std: mean={rel_diff_sum.mean():.6e}, "
              f"median={np.median(rel_diff_sum):.6e}, "
              f"max={rel_diff_sum.max():.6e}")

        # --- bias (important!)
        print("\n--- Signed bias ---")
        print(f"Eff vs Std: mean diff={diff_eff.mean():.6e}")
        print(f"Sum vs Std: mean diff={diff_sum.mean():.6e}")

        # --- quantiles (very informative)
        print("\n--- Error quantiles (Eff vs Std) ---")
        print("q05, q25, q50, q75, q95:",
              np.quantile(diff_eff, [0.05, 0.25, 0.5, 0.75, 0.95]))

        print("\n--- Error quantiles (Sum vs Std) ---")
        print("q05, q25, q50, q75, q95:",
              np.quantile(diff_sum, [0.05, 0.25, 0.5, 0.75, 0.95]))


