import time

import cma
import numpy as np

from .map_estimation import MAPResult

class CMAESOptimizer:
    """
    CMA-ES optimization for MAP estimation.
    Gradient-free, global-ish optimization method.
    Does NOT require a good initial guess.

    Parameters
    ----------
    sigma0 : float
        Initial search scale (step size). Typical: 0.5 – 2.0.
    maxiter : int
        Maximum number of generations (iterations).
    population : int
        CMA-ES population size (lambda). Default None uses CMA defaults.
    verbose : bool
        Print progress.
    """

    def __init__(self, sigma0=1.0, maxiter=200, population=None, verbose=False):
        self.sigma0 = sigma0
        self.maxiter = maxiter
        self.population = population
        self.verbose = verbose

    # ----------------------------------------------------------------------
    def optimize(
        self,
        posterior,
        shape_init,
        source_init=None,
        noise_init=None,
        shape_bounds=None,
        source_bounds=None,
        noise_bounds=None,
        CMA_stds=None
    ):

        def _ensure_scalar(x):
            x_arr = np.asarray(x)
            return float(x_arr.sum()) if x_arr.ndim > 0 else float(x_arr)

        # -------------------------------
        # Decide blocks
        # -------------------------------
        optimize_shape = True
        optimize_source = source_init is not None
        optimize_noise = noise_init is not None

        shape_init = np.asarray(shape_init, float).ravel()
        if optimize_source:
            source_init = np.asarray(source_init, float).ravel()
        if optimize_noise:
            noise_init = np.asarray(noise_init, float).ravel()

        # -------------------------------
        # Build concatenated x0 and bounds
        # -------------------------------
        blocks = [shape_init]
        bounds_blocks = [shape_bounds]

        if optimize_source:
            blocks.append(source_init)
            bounds_blocks.append(source_bounds)

        if optimize_noise:
            blocks.append(noise_init)
            bounds_blocks.append(noise_bounds)

        x0 = np.concatenate(blocks)

        # Create bound arrays for CMA-ES
        if any(b is not None for b in bounds_blocks):
            lower = []
            upper = []
            for b, v in zip(bounds_blocks, blocks):
                if b is None:
                    # unconstrained block
                    lower.extend([-np.inf] * len(v))
                    upper.extend([np.inf] * len(v))
                else:
                    # user bounds
                    if len(b) != len(v):
                        raise ValueError("Bounds length mismatch.")
                    lo, hi = zip(*b)  # unzip
                    lower.extend([x if x is not None else -np.inf for x in lo])
                    upper.extend([x if x is not None else  np.inf for x in hi])
            bounds = [lower, upper]
        else:
            bounds = None

        # -------------------------------
        # Split function
        # -------------------------------
        def unpack(flat):
            idx = 0
            s = flat[:len(shape_init)]
            idx += len(shape_init)

            if optimize_source:
                src = flat[idx:idx + len(source_init)]
                idx += len(source_init)
            else:
                src = None

            if optimize_noise:
                noi = flat[idx:idx + len(noise_init)]
            else:
                noi = None

            return s, src, noi

        # -------------------------------
        # Objective = - log posterior
        # -------------------------------
        def objective(x):
            s, src, noi = unpack(x)
            lp = posterior.log_posterior(
                shape_params=s, source_params=src, noise_params=noi
            )
            return -_ensure_scalar(lp)

        # -------------------------------
        # CMA-ES RUN
        # -------------------------------
        t0 = time.perf_counter()

        es = cma.CMAEvolutionStrategy(
            x0=x0.tolist(),
            sigma0=self.sigma0,
            inopts={
                "bounds": bounds,
                "CMA_stds": CMA_stds,
                "popsize": self.population,
                "verbose": self.verbose
            }
        )

        # Main loop
        for _ in range(self.maxiter):
            solutions = es.ask()
            es.tell(solutions, [objective(s) for s in solutions])
            if es.stop():
                break

        theta_opt = np.array(es.best.x)
        log_posterior = -float(es.best.f)
        runtime = time.perf_counter() - t0

        # -------------------------------
        # Unpack back
        # -------------------------------
        shape_hat, source_hat, noise_hat = unpack(theta_opt)

        return MAPResult(
            theta_map=theta_opt,
            log_posterior=log_posterior,
            success=True,
            n_iterations=es.result.iterations,
            n_function_evals=es.result.evaluations,
            runtime=runtime,
            message="CMA-ES finished",
            optimizer_result=es.result,
            param_history=None,  # could fill manually
        )