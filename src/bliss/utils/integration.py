# integrators.py

from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import simpson


class Integrator(ABC):
    """Decides how to integrate an (M, N') integrand matrix."""

    def build_integrand(
        self,
        measurement:            np.ndarray,   # (M, d)
        sampled_points:         np.ndarray,   # (N', d)
        point_covariances:      np.ndarray,   # (N', d, d)
        normals:                np.ndarray | None,  # (M, d)
        sampled_normals:        np.ndarray | None,  # (N', d)
        normals_covariances:    np.ndarray | None,  # (N', d, d)
        pi_vals:                np.ndarray,   # (N',)

    ) -> np.ndarray:                  # (M, N')
        """Cholesky kernel — shared by all integrators by default."""

        d  = measurement.shape[1]
        L  = np.linalg.cholesky(point_covariances)
        log_det  = 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=-1)
        log_norm = 0.5 * log_det + (d / 2.0) * np.log(2 * np.pi)

        diff   = measurement[:, None, :] - sampled_points[None, :, :]  # (M, N', d)
        diff_T = diff.transpose(1, 2, 0)                                # (N', d, M)
        z      = np.linalg.solve(L, diff_T)                            # (N', d, M)
        dist_sq = np.einsum('nim,nim->nm', z, z).T                     # (M, N')

        log_integrand = (
            -0.5 * dist_sq
            - log_norm[None, :]
            + np.log(np.clip(pi_vals[None, :], 1e-300, None))
        )

        # now compute the normals

        if normals is not None and sampled_normals is not None:
            # normalize to be safe
            n_obs = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            n_model = sampled_normals / np.linalg.norm(sampled_normals, axis=1, keepdims=True)

            if normals_covariances is None:
                # simple cosine similarity model (fast + robust)
                cos_sim = np.einsum('md,nd->mn', n_obs, n_model)  # (M, N')

                # clamp for numerical safety
                cos_sim = np.clip(cos_sim, -1.0, 1.0)

                # convert to penalty (1 - cos theta)
                normal_dist = 1.0 - cos_sim

                # weight factor (tune this!)
                kappa = 10.0

                log_integrand += -kappa * normal_dist

            else:
                # Gaussian model in normal space
                # normals: (M, d)
                # sampled_normals: (N', d)
                # normals_covariances: (M, d, d)

                Ln = np.linalg.cholesky(normals_covariances)  # (M, d, d)

                log_det_n = 2.0 * np.sum(
                    np.log(np.diagonal(Ln, axis1=1, axis2=2)), axis=-1
                )  # (M,)

                log_norm_n = 0.5 * log_det_n + (d / 2.0) * np.log(2 * np.pi)

                # pairwise differences
                ndiff = normals[:, None, :] - sampled_normals[None, :, :]  # (M, N', d)

                # reorder to match solve: (M, d, N')
                ndiff_T = ndiff.transpose(0, 2, 1)  # (M, d, N')

                # solve using measurement covariances
                zn = np.linalg.solve(Ln, ndiff_T)  # (M, d, N')

                # squared Mahalanobis distance
                ndist_sq = np.einsum('mdi,mdi->mi', zn, zn)  # (M, N')

                log_integrand += (
                        -0.5 * ndist_sq
                        - log_norm_n[:, None]  # broadcast over N'
                )
        return np.exp(log_integrand)

    @abstractmethod
    def integrate(
        self,
        integrand:   np.ndarray,        # (M, N')
        t_vals:      np.ndarray,        # (N', param_dim)
        mask:        np.ndarray | None, # (N_full,) bool or None
        domain_low:  np.ndarray,
        domain_high: np.ndarray,
        n_per_dim:   int,
        param_dim:   int,
    ) -> np.ndarray:                    # (M,)
        ...


class SimpsonIntegrator(Integrator):

    def integrate(self, integrand, t_vals, mask, domain_low, domain_high,
                  n_per_dim, param_dim):
        M = integrand.shape[0]

        # ── no mask: local points (e.g. ClosestPointSampler) ──────────
        if mask is None:
            if param_dim == 1:
                return simpson(integrand, x=t_vals[:, 0], axis=-1)
            else:
                s     = int(np.sqrt(integrand.shape[1]))
                u_loc = np.unique(t_vals[:, 0])
                v_loc = np.unique(t_vals[:, 1])
                grid  = integrand.reshape(M, s, s)
                return simpson(simpson(grid, x=u_loc, axis=-1), x=v_loc, axis=-1)

        # ── mask provided: scatter back onto full grid ─────────────────
        if param_dim == 1:
            full          = np.zeros((M, n_per_dim))
            full[:, mask] = integrand
            return simpson(full, x=np.linspace(domain_low[0], domain_high[0], n_per_dim), axis=-1)
        else:
            full          = np.zeros((M, n_per_dim * n_per_dim))
            full[:, mask] = integrand
            grid          = full.reshape(M, n_per_dim, n_per_dim)
            u_vals        = np.linspace(domain_low[0], domain_high[0], n_per_dim)
            v_vals        = np.linspace(domain_low[1], domain_high[1], n_per_dim)
            return simpson(simpson(grid, x=u_vals, axis=-1), x=v_vals, axis=-1)


class TrapezoidIntegrator(Integrator):

    def integrate(self, integrand, t_vals, mask, domain_low, domain_high,
                  n_per_dim, param_dim):
        M = integrand.shape[0]

        if mask is None:
            if param_dim == 1:
                return np.trapezoid(integrand, x=t_vals[:, 0], axis=-1)
            else:
                s     = int(np.sqrt(integrand.shape[1]))
                u_loc = np.unique(t_vals[:, 0])
                v_loc = np.unique(t_vals[:, 1])
                grid  = integrand.reshape(M, s, s)
                return np.trapezoid(np.trapezoid(grid, x=u_loc, axis=-1), x=v_loc, axis=-1)

        if param_dim == 1:
            full          = np.zeros((M, n_per_dim))
            full[:, mask] = integrand
            return np.trapezoid(full, x=np.linspace(domain_low[0], domain_high[0], n_per_dim), axis=-1)
        else:
            full          = np.zeros((M, n_per_dim * n_per_dim))
            full[:, mask] = integrand
            grid          = full.reshape(M, n_per_dim, n_per_dim)
            u_vals        = np.linspace(domain_low[0], domain_high[0], n_per_dim)
            v_vals        = np.linspace(domain_low[1], domain_high[1], n_per_dim)
            return np.trapezoid(np.trapezoid(grid, x=u_vals, axis=-1), x=v_vals, axis=-1)


class MonteCarloIntegrator(Integrator):

    def integrate(self, integrand, t_vals, mask, domain_low, domain_high,
                  n_per_dim, param_dim):
        # MC never needs the grid — mask is irrelevant either way
        volume = float(np.prod(domain_high - domain_low))
        return volume * integrand.mean(axis=-1)


class OriginalIntegrator(Integrator):

    def build_integrand(self, measurement, pi_vals, sampled_points, point_covariances):
        d        = measurement.shape[1]
        inv_covs = np.linalg.inv(point_covariances)
        sigmasq  = (2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(point_covariances))
        diff     = measurement[:, None, :] - sampled_points[None, :, :]
        dist_sq  = np.einsum('mni,nij,mnj->mn', diff, inv_covs, diff)
        return np.exp(-0.5 * dist_sq) / sigmasq[None, :] * pi_vals[None, :]

    def integrate(self, integrand, t_vals, mask, domain_low, domain_high,
                  n_per_dim, param_dim):
        M = integrand.shape[0]

        if mask is None:
            if param_dim == 1:
                return simpson(integrand, x=t_vals[:, 0], axis=-1)
            else:
                s     = int(np.sqrt(integrand.shape[1]))
                u_loc = np.unique(t_vals[:, 0])
                v_loc = np.unique(t_vals[:, 1])
                grid  = integrand.reshape(M, s, s)
                return simpson(simpson(grid, x=u_loc, axis=-1), x=v_loc, axis=-1)

        if param_dim == 1:
            full          = np.zeros((M, n_per_dim))
            full[:, mask] = integrand
            return simpson(full, x=np.linspace(domain_low[0], domain_high[0], n_per_dim), axis=-1)
        else:
            full          = np.zeros((M, n_per_dim * n_per_dim))
            full[:, mask] = integrand
            grid          = full.reshape(M, n_per_dim, n_per_dim)
            u_vals        = np.linspace(domain_low[0], domain_high[0], n_per_dim)
            v_vals        = np.linspace(domain_low[1], domain_high[1], n_per_dim)
            return simpson(simpson(grid, x=u_vals, axis=-1), x=v_vals, axis=-1)

class SupportOnlyIntegrator(Integrator):
    """
    Detects the non-zero support of the integrand and integrates only there.
    Skips the full-grid scatter entirely — ideal when integrand is near-zero
    almost everywhere (as shown by the tiny cluster in your data).
    """

    def __init__(self, eps: float = 1e-300, method: str = "trapezoid"):
        self.eps    = eps
        self.method = method   # "trapezoid" or "simpson"

    def integrate(self, integrand, t_vals, mask, domain_low, domain_high,
                  n_per_dim, param_dim):
        M  = integrand.shape[0]
        fn = np.trapezoid if self.method == "trapezoid" else simpson

        # ── find columns where ANY measurement has a non-negligible value ──
        active = np.any(integrand > self.eps, axis=0)   # (N',)

        if not np.any(active):
            return np.zeros(M)

        integrand_active = integrand[:, active]          # (M, K)  K << N'
        t_active         = t_vals[active]                # (K, param_dim)

        if param_dim == 1:
            return fn(integrand_active, x=t_active[:, 0], axis=-1)

        else:
            # recover u/v coordinates of active points
            u_active = t_active[:, 0]
            v_active = t_active[:, 1]

            u_unique = np.unique(u_active)
            v_unique = np.unique(v_active)

            # scatter into tight bounding grid
            u_idx = np.searchsorted(u_unique, u_active)
            v_idx = np.searchsorted(v_unique, v_active)

            n_u = len(u_unique)
            n_v = len(v_unique)

            # ensure odd counts for simpson
            if self.method == "simpson":
                if n_u % 2 == 0: u_unique, n_u = u_unique[:-1], n_u - 1
                if n_v % 2 == 0: v_unique, n_v = v_unique[:-1], n_v - 1
                keep  = (u_idx < n_u) & (v_idx < n_v)
                integrand_active = integrand_active[:, keep]
                u_idx, v_idx     = u_idx[keep], v_idx[keep]

            grid = np.zeros((M, n_v, n_u))
            grid[:, v_idx, u_idx] = integrand_active

            return fn(fn(grid, x=u_unique, axis=-1), x=v_unique, axis=-1)
