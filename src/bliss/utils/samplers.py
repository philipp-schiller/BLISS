# samplers.py

from abc import ABC, abstractmethod
from typing import override

import numpy as np


class Sampler(ABC):
    """Decides which parameter-space points t to evaluate the integrand at."""

    @abstractmethod
    def sample(
        self,
        shape,
        source_density,
        noise_function,
        measurement:   np.ndarray,   # (M, d)
        shape_params:  np.ndarray,
        source_params,
        noise_params,
        t_vals_full:   np.ndarray,   # (N, param_dim) — full grid
        pi_vals_full:  np.ndarray,   # (N,)
        domain_low:    np.ndarray,
        domain_high:   np.ndarray,
        param_dim:     int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Returns (t_vals, pi_vals, mask_or_None).
        mask is only needed by grid-based integrators for scattering back.
        Measurement-aware samplers (e.g. ClosestPoint) return None for mask.
        """
        ...

    @property
    def per_measurement(self):
        return False

    @property
    def trivial_integral(self):
        return False


class GridSampler(Sampler):
    """
    Uniform grid, threshold by pi. The original approach.
    Returns one shared set of t_vals for all measurements.
    """

    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def sample(self, shape, source_density, noise_function, measurement,
               shape_params, source_params, noise_params,
               t_vals_full, pi_vals_full, domain_low, domain_high, param_dim):
        mask    = pi_vals_full >= self.eps
        return t_vals_full[mask], pi_vals_full[mask], mask


class TopKSampler(Sampler):
    """
    Keeps only the top-k pi values. Useful when support is very sparse
    and you want a fixed evaluation budget regardless of eps.
    """

    def __init__(self, k: int = 200):
        self.k = k

    def sample(self, shape, source_density, noise_function, measurement,
               shape_params, source_params, noise_params,
               t_vals_full, pi_vals_full, domain_low, domain_high, param_dim):
        k    = min(self.k, len(pi_vals_full))
        idx  = np.argpartition(pi_vals_full, -k)[-k:]
        mask = np.zeros(len(pi_vals_full), dtype=bool)
        mask[idx] = True
        return t_vals_full[mask], pi_vals_full[mask], mask


class ClosestPointWindowSampler(Sampler):
    """
    Per-measurement: samples t in a local window around the closest point
    on the shape to each measurement x_m. Concentrates points where the
    Gaussian kernel is large, ignoring far-away regions entirely.

    Returns t_vals of shape (M, S, param_dim) — one set per measurement.
    mask is None since these points don't map back to the full grid.
    """

    def __init__(self, n_samples: int = 50, n_std: float = 3.0):
        self.n_samples = n_samples
        self.n_std     = n_std

    def sample(self, shape, source_density, noise_function, measurement,
               shape_params, source_params, noise_params,
               t_vals_full, pi_vals_full, domain_low, domain_high, param_dim):

        M = measurement.shape[0]

        # Estimate noise scale from the full grid
        sample_covs = noise_function.evaluate(t_vals_full, params=noise_params)
        mean_std    = np.sqrt(np.mean(np.diagonal(sample_covs, axis1=1, axis2=2)))
        param_range = domain_high - domain_low
        window      = np.clip(self.n_std * mean_std / np.linalg.norm(param_range),
                              0.05, 0.5) * param_range   # (param_dim,)

        t_per_meas  = []
        pi_per_meas = []

        for m in range(M):
            _, t_star = shape.get_closest_point(
                shape_params, measurement[m], return_params=True
            )                                                    # (param_dim,)

            local_low  = np.clip(t_star - window, domain_low, domain_high)
            local_high = np.clip(t_star + window, domain_low, domain_high)

            if param_dim == 1:
                s      = self.n_samples
                t_loc  = np.linspace(local_low[0], local_high[0], s)[:, None]
            else:
                s      = int(np.sqrt(self.n_samples))
                u_loc  = np.linspace(local_low[0], local_high[0], s)
                v_loc  = np.linspace(local_low[1], local_high[1], s)
                uu, vv = np.meshgrid(u_loc, v_loc)
                t_loc  = np.stack([uu.ravel(), vv.ravel()], axis=-1)  # (s², 2)

            pi_loc = source_density.evaluate(t_loc, params=source_params)
            t_per_meas.append(t_loc)
            pi_per_meas.append(pi_loc)

        # Stack: (M, S, param_dim) and (M, S)
        return (np.stack(t_per_meas, axis=0),
                np.stack(pi_per_meas, axis=0),
                None)   # no mask — doesn't map to full grid

    @property
    def per_measurement(self):
        return True

class AdaptiveGridSampler(Sampler):
    """
    Two-pass: coarse grid to locate the support, fine grid only there.
    """
    def __init__(self, n_coarse: int = 20, n_fine: int = 50, eps: float = 1e-6):
        self.n_coarse = n_coarse
        self.n_fine   = n_fine
        self.eps      = eps

    def sample(self, shape, source_density, noise_function, measurement,
               shape_params, source_params, noise_params,
               t_vals_full, pi_vals_full, domain_low, domain_high, param_dim):

        # ── pass 1: coarse grid to find support ────────────────────────
        coarse = shape.sample_parameter_space_grid(n_points_per_dim=self.n_coarse)
        pi_c   = source_density.evaluate(coarse, params=source_params)
        hot    = coarse[pi_c >= self.eps]                    # support cells

        if len(hot) == 0:
            mask = pi_vals_full >= self.eps
            return t_vals_full[mask], pi_vals_full[mask], mask

        # ── pass 2: fine grid within support bounding box ─────────────
        low  = hot.min(axis=0)
        high = hot.max(axis=0)

        if param_dim == 1:
            t_fine = np.linspace(low[0], high[0], self.n_fine)[:, None]
        else:
            u = np.linspace(low[0], high[0], self.n_fine)
            v = np.linspace(low[1], high[1], self.n_fine)
            uu, vv = np.meshgrid(u, v)
            t_fine = np.stack([uu.ravel(), vv.ravel()], axis=-1)

        pi_fine = source_density.evaluate(t_fine, params=source_params)
        mask_fine = pi_fine >= self.eps
        return t_fine[mask_fine], pi_fine[mask_fine], None  # None: local grid

class ClosestPointSampler(Sampler):
    """
    Just retrieves the closest point (delta approximation).
    Produces exactly one sample per measurement.
    """

    def sample(self, shape, source_density, noise_function, measurement,
               shape_params, source_params, noise_params,
               t_vals_full, pi_vals_full, domain_low, domain_high, param_dim):

        M = measurement.shape[0]

        t_per_meas  = []
        pi_per_meas = []

        for m in range(M):
            _, t_star = shape.get_closest_point(
                shape_params, measurement[m], return_params=True
            )                                                # (param_dim,)

            t_star = np.atleast_2d(t_star)                   # (1, param_dim)
            pi     = source_density.evaluate(t_star,
                                             params=source_params)  # (1,)

            t_per_meas.append(t_star)
            pi_per_meas.append(pi)

        return (
            np.stack(t_per_meas,  axis=0),   # (M, 1, param_dim)
            np.stack(pi_per_meas, axis=0),   # (M, 1)
            None
        )

    @property
    def per_measurement(self):
        return True

    @property
    def trivial_integral(self):
        return True
