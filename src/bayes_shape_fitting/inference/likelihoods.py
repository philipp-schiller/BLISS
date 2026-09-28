import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import simpson
from scipy.special import logsumexp

from .noise_functions import NoiseFunction
from .source_densitites import SourceDensity, FunctionType
from ..models.shapes.base_shape import Shape
from ..utils.integration import SimpsonIntegrator, Integrator
from ..utils.samplers import GridSampler, Sampler


# define the abstract Likelihood class
class Likelihood(ABC):
    """
    Abstract base class for likelihood models
        L(X | theta)
    Accepts optionally also source_density, noise_function
    """

    shape: Shape

    @abstractmethod
    def log_likelihood(
        self,
        measurement: np.ndarray,
        shape_params: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        pass

@dataclass()
class MarginalLikelihood(Likelihood):
    shape:          Shape
    source_density: SourceDensity
    noise_function: NoiseFunction
    num_samples:    int        = 500
    sampler:        Sampler    = field(default_factory=GridSampler)
    integrator:     Integrator = field(default_factory=SimpsonIntegrator)

    def log_likelihood(self, measurement, shape_params,
                       source_params=None, noise_params=None, normals=None, normals_cov = None, *args):

        use_normals = normals is not None

        param_dim   = self.shape.parameter_dim
        domain_low  = np.atleast_1d(self.shape.domain_low)
        domain_high = np.atleast_1d(self.shape.domain_high)
        n_per_dim   = self.num_samples

        t_vals_full  = self.shape.sample_parameter_space_grid(n_points_per_dim=n_per_dim)
        pi_vals_full = self.source_density.evaluate(t_vals_full, params=source_params)

        # ── 1. Sample ─────────────────────────────────────────────────
        t_vals, pi_vals, mask = self.sampler.sample(
            shape=self.shape, source_density=self.source_density,
            noise_function=self.noise_function, measurement=measurement,
            shape_params=shape_params, source_params=source_params,
            noise_params=noise_params, t_vals_full=t_vals_full,
            pi_vals_full=pi_vals_full, domain_low=domain_low,
            domain_high=domain_high, param_dim=param_dim,
        )

        # ── 2. Build the integral ─────────────────────────────────────────────────
        pts = self.shape.evaluate(t_vals, shape_params)
        pts_covs = self.noise_function.evaluate(t_vals, params=noise_params)

        # now evaluate normals
        shape_normals = None
        if use_normals:
            shape_normals, _ = self.shape.evaluate_normals(t_vals, shape_params, return_covariance=False)

        integrand = self.integrator.build_integrand(
            measurement, pts, pts_covs, normals, shape_normals, normals_cov, pi_vals
        )

        values = self.integrator.integrate(
            integrand, t_vals, mask,
            domain_low, domain_high, n_per_dim, param_dim,
        )

        return np.log(values + 1e-200)

    def scaled(self, scale):
        new = copy.deepcopy(self)
        new.noise_function = self.noise_function.scaled(scale)
        return new

@dataclass
class MarginalLikelihoodSum(Likelihood):
    shape: Shape
    source_density: SourceDensity
    noise_function: NoiseFunction
    num_samples: int = 2000

    def log_likelihood(self, measurement, shape_params,
                       source_params=None, noise_params=None, normals = None, *args):

        # --- sample parameter space ---
        t_vals = self.shape.sample_parameter_space_uniform(self.num_samples)

        # --- evaluate covariances ---
        covariances = self.noise_function.evaluate(t_vals, params=noise_params)  # (M, d, d)

        inv_covs = np.linalg.inv(covariances)  # (M, d, d)
        log_det_cov = np.log(np.linalg.det(covariances))  # (M,)

        d = measurement.shape[1]

        # --- correct normalization ---
        log_norm = 0.5 * (d * np.log(2 * np.pi) + log_det_cov)  # (M,)

        # --- evaluate surface points ---
        Y = self.shape.evaluate(t_vals, shape_params)  # (M, d)

        # --- compute differences ---
        diff = measurement[:, None, :] - Y[None, :, :]  # (N, M, d)

        # --- Mahalanobis distances ---
        dist_sq = np.einsum('nmd,mdk,nmk->nm', diff, inv_covs, diff)  # (N, M)

        # --- log Gaussian ---
        log_gauss = -0.5 * dist_sq - log_norm[None, :]  # (N, M)

        """ # --- include source density ---
        pi_vals = self.source_density.evaluate(t_vals, params=source_params)
        pi_vals = pi_vals / (np.sum(pi_vals) + 1e-12)
        log_pi = np.log(pi_vals + 1e-12)

        log_weighted = log_gauss + log_pi[None, :]  # (N, M)

        # --- marginal likelihood ---
        log_p = logsumexp(log_weighted, axis=1)  # (N,)

        return log_p"""
        # --- density ---
        pi_vals = self.source_density.evaluate(t_vals, params=source_params)

        # --- metric (IMPORTANT if surface sampling)
        metric = self.shape.metric_density(t_vals, shape_params)

        weights = pi_vals * metric

        log_weights = np.log(weights + 1e-12)

        log_p = logsumexp(log_gauss + log_weights[None, :], axis=1) \
                - np.log(self.num_samples)

        return log_p

    def scaled(self, scale):
        new = copy.deepcopy(self)
        new.noise_function = self.noise_function.scaled(scale)
        return new

@dataclass()
class MarginalLikelihoodComplex(Likelihood):
    shape:          Shape
    source_density: SourceDensity
    noise_function: NoiseFunction
    num_samples:    int        = 500
    sampler:        Sampler    = field(default_factory=GridSampler)
    integrator:     Integrator = field(default_factory=SimpsonIntegrator)

    # ── Normal likelihood helper ───────────────────────────────────────
    @staticmethod
    def _vmf_log_likelihood(observed_normals: np.ndarray,
                            predicted_normals: np.ndarray,
                            kappa: float) -> np.ndarray:
        """
        Von Mises-Fisher log-likelihood for unit normals.

        Parameters
        ----------
        observed_normals  : (M, d) unit vectors from measurement
        predicted_normals : (S, d) or (M, S, d) unit vectors from shape
        kappa             : concentration parameter (scalar); larger = tighter

        Returns
        -------
        log_vmf : (M, S) array — one score per (measurement, sample) pair
        """
        # Normalise defensively
        obs  = observed_normals  / (np.linalg.norm(observed_normals,  axis=-1, keepdims=True) + 1e-300)
        pred = predicted_normals / (np.linalg.norm(predicted_normals, axis=-1, keepdims=True) + 1e-300)

        if pred.ndim == 2:          # (S, d) → broadcast over M
            cos_sim = obs[:, None, :] @ pred[None, :, :].swapaxes(-1, -2)  # (M, S, S)
            cos_sim = np.einsum('md,sd->ms', obs, pred)                    # (M, S)
        else:                       # (M, S, d) — per-measurement samples
            cos_sim = np.einsum('md,msd->ms', obs, pred)                   # (M, S)

        # log C(kappa, d) is a constant w.r.t. t → omit for relative comparisons,
        # but include for correctly scaled absolute likelihoods.
        d = obs.shape[-1]
        log_norm = (
            (d / 2 - 1) * np.log(kappa)
            - (d / 2) * np.log(2 * np.pi)
            - np.log(np.i0(kappa))          # scipy.special.i0 for d=3; generalise if needed
        )
        return log_norm + kappa * cos_sim   # (M, S)

    # ── Main entry point ──────────────────────────────────────────────
    def log_likelihood(self, measurement, shape_params,
                       source_params=None, noise_params=None, normals=None, *args):

        use_normals = normals is not None
        kappa = (noise_params.get("kappa", 10.0)
                 if isinstance(noise_params, dict) else 10.0)

        param_dim   = self.shape.parameter_dim
        domain_low  = np.atleast_1d(self.shape.domain_low)
        domain_high = np.atleast_1d(self.shape.domain_high)
        n_per_dim   = self.num_samples

        t_vals_full  = self.shape.sample_parameter_space_grid(n_points_per_dim=n_per_dim)
        pi_vals_full = self.source_density.evaluate(t_vals_full, params=source_params)

        # ── 1. Sample ─────────────────────────────────────────────────
        t_vals, pi_vals, mask = self.sampler.sample(
            shape=self.shape, source_density=self.source_density,
            noise_function=self.noise_function, measurement=measurement,
            shape_params=shape_params, source_params=source_params,
            noise_params=noise_params, t_vals_full=t_vals_full,
            pi_vals_full=pi_vals_full, domain_low=domain_low,
            domain_high=domain_high, param_dim=param_dim,
        )

        # ── 2. Evaluate shape + noise + (optionally) normals ──────────
        if self.sampler.per_measurement:
            M      = measurement.shape[0]
            values = np.zeros(M)

            for m in range(M):
                pts  = self.shape.evaluate(t_vals[m], shape_params)
                covs = self.noise_function.evaluate(t_vals[m], params=noise_params)
                integrand = self.integrator.build_integrand(
                    measurement[[m]], pi_vals[m], pts, covs
                )                                                   # (1, S)

                # Multiply in vMF normal term if normals provided
                if use_normals:
                    pred_n, _  = self.shape.evaluate_normals(t_vals[m], shape_params, return_covariance=False)  # (S, d)
                    log_vmf  = self._vmf_log_likelihood(normals[[m]], pred_n, kappa) # (1, S)
                    integrand = integrand * np.exp(log_vmf)         # (1, S)

                log_f = np.log(integrand[0, 0] + 1e-300)
                if self.sampler.trivial_integral:
                    Sigma = covs[0]
                    J = self.shape.jacobian(t_vals[m][0], shape_params)[0]
                    inv_Sigma = np.linalg.inv(Sigma)
                    H_noise = J.T @ inv_Sigma @ J
                    eigvals, eigvecs = np.linalg.eigh(H_noise)
                    eigvals = np.maximum(eigvals, 1e-8)
                    H_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    sign, log_det_H = np.linalg.slogdet(H_pd)
                    if sign <= 0:
                        values[m] = np.exp(log_f)
                    else:
                        log_volume = 0.5 * param_dim * np.log(2 * np.pi) - 0.5 * log_det_H
                        values[m] = np.exp(log_f + log_volume)
                else:
                    values[m] = self.integrator.integrate(
                        integrand, t_vals[m], mask=None,
                        domain_low=domain_low, domain_high=domain_high,
                        n_per_dim=n_per_dim, param_dim=param_dim,
                    )[0]

        else:
            # Shared samples across all measurements
            pts  = self.shape.evaluate(t_vals, shape_params)
            covs = self.noise_function.evaluate(t_vals, params=noise_params)
            integrand = self.integrator.build_integrand(
                measurement, pi_vals, pts, covs
            )                                                       # (M, N')

            # Multiply in vMF normal term if normals provided
            if use_normals:
                pred_n, _   = self.shape.evaluate_normals(t_vals, shape_params)    # (N', d)
                log_vmf  = self._vmf_log_likelihood(normals, pred_n, kappa)     # (M, N')
                integrand = integrand * np.exp(log_vmf)            # (M, N')

            values = self.integrator.integrate(
                integrand, t_vals, mask,
                domain_low, domain_high, n_per_dim, param_dim,
            )

        return np.log(values + 1e-200)


@dataclass()
class MarginalLikelihood_OG(Likelihood):
    shape:          Shape
    source_density: SourceDensity
    noise_function: NoiseFunction
    num_samples:    int        = 500
    sampler:        Sampler    = field(default_factory=GridSampler)
    integrator:     Integrator = field(default_factory=SimpsonIntegrator)

    def log_likelihood(self, measurement, shape_params,
                       source_params=None, noise_params=None, *args):

        param_dim   = self.shape.parameter_dim
        domain_low  = np.atleast_1d(self.shape.domain_low)
        domain_high = np.atleast_1d(self.shape.domain_high)
        n_per_dim   = self.num_samples

        t_vals_full  = self.shape.sample_parameter_space_grid(n_points_per_dim=n_per_dim)
        pi_vals_full = self.source_density.evaluate(t_vals_full, params=source_params)

        # ── 1. Sample: decide WHERE to evaluate ───────────────────────
        t_vals, pi_vals, mask = self.sampler.sample(
            shape          = self.shape,
            source_density = self.source_density,
            noise_function = self.noise_function,
            measurement    = measurement,
            shape_params   = shape_params,
            source_params  = source_params,
            noise_params   = noise_params,
            t_vals_full    = t_vals_full,
            pi_vals_full   = pi_vals_full,
            domain_low     = domain_low,
            domain_high    = domain_high,
            param_dim      = param_dim,
        )

        # ── 2. Evaluate shape + noise at sampled points ────────────────

        if self.sampler.per_measurement:
            # t_vals: (M, S, param_dim), pi_vals: (M, S)
            M      = measurement.shape[0]
            values = np.zeros(M)

            for m in range(M):
                pts  = self.shape.evaluate(t_vals[m], shape_params)
                covs = self.noise_function.evaluate(t_vals[m], params=noise_params)
                integrand = self.integrator.build_integrand(
                    measurement[[m]], pi_vals[m], pts, covs
                )                                               # (1, S)

                log_f = np.log(integrand[0, 0] + 1e-300)
                # check for the trivial case (if the integral depends on only one point)
                if self.sampler.trivial_integral:
                    # Laplace correction
                    Sigma = covs[0]  # (d, d)
                    J = self.shape.jacobian(t_vals[m][0], shape_params)[0]  # (d, param_dim)

                    # Full Hessian of -log integrand at t*
                    inv_Sigma = np.linalg.inv(Sigma)
                    H_noise = J.T @ inv_Sigma @ J  # (param_dim, param_dim)

                    # Optional: add pi curvature via finite differences if pi is not flat
                    # For now assume pi flat near t* → H ≈ H_noise

                    # Ensure positive definite
                    eigvals, eigvecs = np.linalg.eigh(H_noise)
                    eigvals = np.maximum(eigvals, 1e-8)  # clip negative eigenvalues
                    H_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

                    sign, log_det_H = np.linalg.slogdet(H_pd)
                    if sign <= 0:
                        # Fallback: no volume correction
                        values[m] = np.exp(log_f)
                    else:
                        log_volume = 0.5 * param_dim * np.log(2 * np.pi) - 0.5 * log_det_H
                        values[m] = np.exp(log_f + log_volume)

                else:
                    values[m] = self.integrator.integrate(
                        integrand, t_vals[m], mask=None,
                        domain_low=domain_low, domain_high=domain_high,
                        n_per_dim=n_per_dim, param_dim=param_dim,
                    )[0]
        else:
            # t_vals: (N', param_dim) — shared across all measurements
            pts  = self.shape.evaluate(t_vals, shape_params)
            covs = self.noise_function.evaluate(t_vals, params=noise_params)
            integrand = self.integrator.build_integrand(
                measurement, pi_vals, pts, covs
            )                                                   # (M, N')
            values = self.integrator.integrate(
                integrand, t_vals, mask,
                domain_low, domain_high, n_per_dim, param_dim,
            )

        return np.log(values + 1e-200)


@dataclass()
class MarginalLikelihood_Standard(Likelihood):
    """
    Computes the marginal likelihood based on a source density and noise_function
    """
    shape: Shape
    source_density: SourceDensity
    noise_function: NoiseFunction
    num_samples: int = 500 # number of samples on the shape to compute the surface integral


    def __post_init__(self) -> None:
        """
        We check which of the source density and noise function are fixed and store the number of their parameters
        """
        self.num_shape_params = self.shape.get_n_params()
        self.num_source_params = self.source_density.get_n_params()
        self.num_noise_params = self.noise_function.get_n_params()

    def log_likelihood(
            self,
            measurement: np.ndarray,
            shape_params: np.ndarray,
            source_params: np.ndarray = None,
            noise_params: np.ndarray = None,
            *args, **kwargs
    ) -> np.ndarray:

        # first assert, if the source-density and noise function
        if source_params is None:
            assert self.source_density.get_function_type() == FunctionType.FIXED, "Source density needs parameters!"
        if noise_params is None:
            assert self.noise_function.get_function_type() == FunctionType.FIXED, "Noise functions needs parameters!"

        noise_function = self.noise_function
        source_density = self.source_density

        param_dim = self.shape.parameter_dim  # 1 for curves, 2 for surfaces
        domain_low = np.atleast_1d(self.shape.domain_low)
        domain_high = np.atleast_1d(self.shape.domain_high)
        n_per_dim = self.num_samples

        # we only consider t_vals where the source density is bigger than epsilon for efficiency
        t_vals_full = self.shape.sample_parameter_space_grid(n_points_per_dim=self.num_samples)
        pi_vals_full = source_density.evaluate(t_vals_full, params=source_params)

        eps = 0.0001
        mask = pi_vals_full >= eps
        t_vals = t_vals_full[mask]
        pi_vals = pi_vals_full[mask]

        if param_dim == 1:
            # --- 1D case ---

            covariances = noise_function.evaluate(t_vals, params=noise_params)
            inv_covs = np.linalg.inv(covariances)
            det_cov = np.linalg.det(covariances)
            sigmasq = 2 * np.pi * np.sqrt(det_cov)

            # compute the source points
            sampled_points = self.shape.evaluate(t_vals, shape_params)

            # compute the mahalanobis distances to the source points
            diff = measurement[:, None, :] - sampled_points[None, :, :]
            dist_sq = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)
            integrand = np.exp(-0.5 * dist_sq) / sigmasq[None, :] * pi_vals[None, :]

            #values = np.trapezoid(integrand, x=t_vals[:, 0], axis=-1)

            # Put back onto full grid and integrate
            integrand_full = np.zeros((measurement.shape[0], n_per_dim))
            integrand_full[:, mask] = integrand
            values = simpson(integrand_full, x=t_vals_full[:, 0], axis=-1)

        elif param_dim == 2:
            # --- 2D case ---

            u_vals = np.linspace(domain_low[0], domain_high[0], n_per_dim)
            v_vals = np.linspace(domain_low[1], domain_high[1], n_per_dim)

            covariances = noise_function.evaluate(t_vals, params=noise_params)
            inv_covs = np.linalg.inv(covariances)
            det_cov = np.linalg.det(covariances)

            d = measurement.shape[1]  # dimension = 3
            sigmasq = (2 * np.pi) ** (d / 2) * np.sqrt(det_cov)
            sampled_points = self.shape.evaluate(t_vals, shape_params)

            diff = measurement[:, None, :] - sampled_points[None, :, :]
            dist_sq = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)
            integrand = np.exp(-0.5 * dist_sq) / sigmasq[None, :] * pi_vals[None, :]  # (M, N')

            # Put back onto full (n_v, n_u) grid and integrate via nested Simpson
            integrand_full = np.zeros((measurement.shape[0], n_per_dim * n_per_dim))
            integrand_full[:, mask] = integrand
            integrand_grid = integrand_full.reshape(measurement.shape[0], n_per_dim, n_per_dim)
            # integrand_grid is (M, n_v, n_u) — integrate u first, then v
            inner = simpson(integrand_grid, x=u_vals, axis=-1)  # (M, n_v)
            values = simpson(inner, x=v_vals, axis=-1)  # (M,)

        else:
            raise NotImplementedError(f"log_likelihood not implemented for parameter_dim={param_dim}")
        return np.log(values + 1e-200)

#-------------------------------------------------------
# define likelihood with respect to the closest point
#-------------------------------------------------------


@dataclass()
class ClosestPointLikelihood(Likelihood):
    shape: Shape
    source_density: SourceDensity
    noise_function: NoiseFunction

    def __post_init__(self) -> None:
        """
        We check which of the source density and noise function are fixed and store the number of their parameters
        """
        self.num_shape_params = self.shape.get_n_params()
        self.num_source_params = self.source_density.get_n_params()
        self.num_noise_params = self.noise_function.get_n_params()

    def log_likelihood(
            self,
            measurement: np.ndarray,
            shape_params: np.ndarray,
            source_params: np.ndarray = None,
            noise_params: np.ndarray = None,
            *args,
            **kwargs
    ) -> np.ndarray:
        """
        Compute the log-likelihood of measurements given shape parameters theta.

        For each measurement x_i, we find the closest point γ(t*) on the shape,
        then evaluate:

            log p(x_i | θ) = -½ (x_i - γ(t*))ᵀ Σ(t*)⁻¹ (x_i - γ(t*))
                             - ½ log|2π Σ(t*)|
                             + log p(t*)          ← source density term

        Parameters
        ----------
        noise_params
        source_params
        shape_params : np.ndarray
            Shape parameters θ passed to the parametrization.
        measurement : np.ndarray, shape (M, n) or (n,)
            One or more observed points in ambient space.

        Returns
        -------
        log_liks : np.ndarray, shape (M,)
            Per-measurement log-likelihoods.
        """
        noise_function = self.noise_function
        source_density = self.source_density

        # noise_params: pulled from kwargs, else fallback to self attribute
        noise_params = kwargs.get("noise_params", getattr(self, "noise_params", None))

        measurement = np.atleast_2d(measurement)  # (M, n)
        M = measurement.shape[0]
        n = measurement.shape[1]

        # --- Closest points on shape ---
        # closest_points: (M, n),  t_vals: (M, k)
        closest_points, t_vals = self.shape.get_closest_point(params=shape_params, x=measurement, return_params=True, tol=0.3, n_restarts=1
                                                              )

        # --- Covariances at closest parameter values ---
        # covariances: (M, n, n)
        covariances = noise_function.evaluate(t_vals, params=noise_params)
        inv_covs = np.linalg.inv(covariances)  # (M, n, n)
        sign, log_det_cov = np.linalg.slogdet(covariances)  # (M,) — numerically stable

        # --- Mahalanobis distance squared ---
        # diff: (M, n)
        diff = measurement - closest_points

        # dist_sq[i] = diff[i]ᵀ Σ⁻¹[i] diff[i]
        # einsum: 'mi, mij, mj -> m'
        dist_sq = np.einsum("mi, mij, mj -> m", diff, inv_covs, diff)  # (M,)

        # --- Gaussian log-likelihood ---
        # log p(x | t*) = -½ (dist_sq + n*log(2π) + log|Σ|)
        log_liks = -0.5 * (dist_sq + n * np.log(2 * np.pi) + log_det_cov)  # (M,)

        return log_liks  # shape (M,)

@dataclass()
class MarginalLikelihoodEfficient(Likelihood):
    """
    Computes the marginal likelihood based on a source density and noise_function
    """
    shape: Shape
    source_density: SourceDensity
    noise_function: NoiseFunction
    num_samples: int = 500 # number of samples on the shape to compute the surface integral


    def __post_init__(self) -> None:
        """
        We check which of the source density and noise function are fixed and store the number of their parameters
        """
        self.num_shape_params = self.shape.get_n_params()
        self.num_source_params = self.source_density.get_n_params()
        self.num_noise_params = self.noise_function.get_n_params()

        # Precompute covariance constants if noise function is fixed
        if self.noise_function.get_function_type() == FunctionType.FIXED:
            _test_t = self.shape.sample_parameter_space_uniform(10)
            _test_covs = self.noise_function.evaluate(_test_t)  # (10, d, d)
            self._constant_cov = np.allclose(_test_covs, _test_covs[0])
            if self._constant_cov:
                d = _test_covs.shape[-1]
                self._inv_cov = np.linalg.inv(_test_covs[0])  # (d, d)
                _, logdet = np.linalg.slogdet(_test_covs[0])
                self._log_norm_const = (d / 2) * np.log(2 * np.pi) + 0.5 * logdet  # scalar
        else:
            self._constant_cov = False

    def log_likelihood(
            self,
            measurement: np.ndarray,
            shape_params: np.ndarray,
            source_params: np.ndarray = None,
            noise_params: np.ndarray = None,
            *args,
    ) -> np.ndarray:

        if source_params is None:
            assert self.source_density.get_function_type() == FunctionType.FIXED, "Source density needs parameters!"
        if noise_params is None:
            assert self.noise_function.get_function_type() == FunctionType.FIXED, "Noise functions needs parameters!"

        noise_function = self.noise_function
        source_density = self.source_density

        param_dim = self.shape.parameter_dim
        domain_low = np.atleast_1d(self.shape.domain_low)
        domain_high = np.atleast_1d(self.shape.domain_high)
        Q = self.num_samples
        d = measurement.shape[1]

        if param_dim == 1:
            # Uniform grid
            t_vals = np.linspace(domain_low[0], domain_high[0], Q)  # (Q,)

            # Mask near-zero source density
            pi_vals = source_density.evaluate(t_vals[:, None], params=source_params)  # (Q,)
            active = pi_vals >= 1e-4
            t_vals_a = t_vals[active]  # (N',)
            pi_vals_a = pi_vals[active]  # (N',)

            covariances = noise_function.evaluate(t_vals_a[:, None], params=noise_params)  # (N', d, d)

            if self._constant_cov:
                inv_cov = self._inv_cov  # (d, d)
                log_norm = self._log_norm_const  # scalar
                sampled_points = self.shape.evaluate(t_vals_a[:, None], shape_params)  # (N', d)
                diff = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
                dist_sq = np.einsum('mni,ij,mnj->mn', diff, inv_cov, diff)  # (M, N')
            else:
                inv_covs = np.linalg.inv(covariances)  # (N', d, d)
                _, logdet = np.linalg.slogdet(covariances)  # (N',)
                log_norm = (d / 2) * np.log(2 * np.pi) + 0.5 * logdet  # (N',)
                sampled_points = self.shape.evaluate(t_vals_a[:, None], shape_params)  # (N', d)
                diff = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
                dist_sq = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)  # (M, N')

            log_gauss = -0.5 * dist_sq - log_norm  # (M, N')
            integrand = np.where(log_gauss > -40,
                                 np.exp(log_gauss) * pi_vals_a[None, :],
                                 0.0)  # (M, N')

            # Integrate directly on masked grid — no zero-padding
            values = np.trapezoid(integrand, x=t_vals_a, axis=-1)  # (M,)

        elif param_dim == 2:
            u_vals = np.linspace(domain_low[0], domain_high[0], Q)  # (Q,)
            v_vals = np.linspace(domain_low[1], domain_high[1], Q)  # (Q,)

            UU, VV = np.meshgrid(u_vals, v_vals, indexing='ij')  # (Q, Q)
            t_vals = np.stack([UU.ravel(), VV.ravel()], axis=-1)  # (Q², 2)

            pi_vals = source_density.evaluate(t_vals, params=source_params)  # (Q²,)
            active = pi_vals >= 1e-4
            t_vals_a = t_vals[active]  # (N', 2)
            pi_vals_a = pi_vals[active]  # (N',)

            covariances = noise_function.evaluate(t_vals_a, params=noise_params)  # (N', d, d)

            if np.allclose(covariances, covariances[0]):
                inv_cov = self._inv_cov
                log_norm = self._log_norm_const
                sampled_points = self.shape.evaluate(t_vals_a, shape_params)  # (N', d)
                diff = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
                dist_sq = np.einsum('mni,ij,mnj->mn', diff, inv_cov, diff)  # (M, N')
            else:
                inv_covs = np.linalg.inv(covariances)
                _, logdet = np.linalg.slogdet(covariances)
                log_norm = (d / 2) * np.log(2 * np.pi) + 0.5 * logdet  # (N',)
                sampled_points = self.shape.evaluate(t_vals_a, shape_params)  # (N', d)
                diff = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
                dist_sq = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)  # (M, N')

            log_gauss = -0.5 * dist_sq - log_norm  # (M, N')
            integrand = np.where(log_gauss > -40,
                                 np.exp(log_gauss) * pi_vals_a[None, :],
                                 0.0)  # (M, N')

            # Reconstruct 2D grid for nested trapz
            integrand_full = np.zeros((measurement.shape[0], Q * Q))  # (M, Q²)
            integrand_full[:, active] = integrand
            integrand_grid = integrand_full.reshape(measurement.shape[0], Q, Q)  # (M, Q, Q)
            inner = np.trapezoid(integrand_grid, x=u_vals, axis=-2)  # (M, Q)
            values = np.trapezoid(inner, x=v_vals, axis=-1)  # (M,)

        else:
            raise NotImplementedError(f"log_likelihood not implemented for parameter_dim={param_dim}")

        return np.log(values + 1e-200)


    def log_likelihood_more_accurate(
            self,
            measurement: np.ndarray,
            shape_params: np.ndarray,
            source_params: np.ndarray = None,
            noise_params: np.ndarray = None,
            *args,
    ) -> np.ndarray:

        if source_params is None:
            assert self.source_density.get_function_type() == FunctionType.FIXED, "Source density needs parameters!"
        if noise_params is None:
            assert self.noise_function.get_function_type() == FunctionType.FIXED, "Noise functions needs parameters!"

        noise_function = self.noise_function
        source_density = self.source_density

        param_dim = self.shape.parameter_dim
        domain_low = np.atleast_1d(self.shape.domain_low)  # (param_dim,)
        domain_high = np.atleast_1d(self.shape.domain_high)  # (param_dim,)
        Q = self.num_samples  # quadrature points per dimension

        # --- Build Gauss-Legendre nodes and weights on [-1, 1] ---
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(Q)  # (Q,), (Q,)

        def gl_to_domain(low, high):
            """Map GL nodes from [-1, 1] -> [low, high], return nodes and scaled weights."""
            scale = (high - low) / 2.0
            shift = (high + low) / 2.0
            nodes = scale * gl_nodes + shift  # (Q,)
            weights = scale * gl_weights  # (Q,) — absorbs the Jacobian
            return nodes, weights

        d = measurement.shape[1]  # ambient dimension (2 for curves, 3 for surfaces)

        if param_dim == 1:
            # ------------------------------------------------------------------ 1D
            t_nodes, t_weights = gl_to_domain(domain_low[0], domain_high[0])
            t_vals = t_nodes[:, None]  # (Q, 1)

            # Source density — mask near-zero regions for efficiency
            pi_vals = source_density.evaluate(t_vals, params=source_params)  # (Q,)
            active = pi_vals >= 1e-4
            t_vals_a = t_vals[active]  # (N', 1)
            pi_vals_a = pi_vals[active]  # (N',)
            t_weights_a = t_weights[active]  # (N',)

            print("Active nodes: " )
            print(len(t_vals_a))

            covariances = noise_function.evaluate(t_vals_a, params=noise_params)  # (N', d, d)

            # Optimise for spatially-constant covariance
            if np.allclose(covariances, covariances[0]):
                inv_cov = np.linalg.inv(covariances[0])  # (d, d)
                log_norm = (d / 2) * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(covariances[0]))
                sampled_points = self.shape.evaluate(t_vals_a, shape_params)  # (N', d)
                diff = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
                dist_sq = np.einsum('mni,ij,mnj->mn', diff, inv_cov, diff)  # (M, N')
            else:
                inv_covs = np.linalg.inv(covariances)  # (N', d, d)
                sign, logdet = np.linalg.slogdet(covariances)  # (N',)
                log_norm = (d / 2) * np.log(2 * np.pi) + 0.5 * logdet  # (N',)
                sampled_points = self.shape.evaluate(t_vals_a, shape_params)  # (N', d)
                diff = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
                dist_sq = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)  # (M, N')

            # log-space Gaussian, gated to avoid exp on negligible values
            log_gauss = -0.5 * dist_sq - log_norm  # (M, N')
            integrand = np.where(log_gauss > -40,
                                 np.exp(log_gauss) * pi_vals_a[None, :],
                                 0.0)  # (M, N')

            # GL quadrature: sum over weighted nodes
            values = integrand @ t_weights_a  # (M,)

        elif param_dim == 2:
            # ------------------------------------------------------------------ 2D
            u_nodes, u_weights = gl_to_domain(domain_low[0], domain_high[0])  # (Q,)
            v_nodes, v_weights = gl_to_domain(domain_low[1], domain_high[1])  # (Q,)

            # Tensor-product grid
            UU, VV = np.meshgrid(u_nodes, v_nodes, indexing='ij')  # (Q, Q)
            WW = np.outer(u_weights, v_weights)  # (Q, Q) — 2D GL weights

            t_vals = np.stack([UU.ravel(), VV.ravel()], axis=-1)  # (Q², 2)
            quad_weights = WW.ravel()  # (Q²,)

            # Source density mask
            pi_vals = source_density.evaluate(t_vals, params=source_params)  # (Q²,)
            active = pi_vals >= 1e-4
            t_vals_a = t_vals[active]  # (N', 2)
            pi_vals_a = pi_vals[active]  # (N',)
            quad_weights_a = quad_weights[active]  # (N',)

            covariances = noise_function.evaluate(t_vals_a, params=noise_params)  # (N', d, d)

            if np.allclose(covariances, covariances[0]):
                inv_cov = np.linalg.inv(covariances[0])
                log_norm = (d / 2) * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(covariances[0]))
                sampled_points = self.shape.evaluate(t_vals_a, shape_params)  # (N', d)
                diff = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
                dist_sq = np.einsum('mni,ij,mnj->mn', diff, inv_cov, diff)  # (M, N')
            else:
                inv_covs = np.linalg.inv(covariances)
                sign, logdet = np.linalg.slogdet(covariances)
                log_norm = (d / 2) * np.log(2 * np.pi) + 0.5 * logdet  # (N',)
                sampled_points = self.shape.evaluate(t_vals_a, shape_params)  # (N', d)
                diff = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
                dist_sq = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)  # (M, N')

            log_gauss = -0.5 * dist_sq - log_norm  # (M, N')
            integrand = np.where(log_gauss > -40,
                                 np.exp(log_gauss) * pi_vals_a[None, :],
                                 0.0)  # (M, N')

            # GL quadrature: dot with combined node weights
            values = integrand @ quad_weights_a  # (M,)

        else:
            raise NotImplementedError(f"log_likelihood not implemented for parameter_dim={param_dim}")

        return np.log(values + 1e-200)