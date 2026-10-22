import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict
from .base_optimizer import Optimizer, OptimizationResult

import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class ECMResult(OptimizationResult):
    shape_params: np.ndarray
    source_params: np.ndarray
    noise_params: np.ndarray
    log_posterior_final: float

    n_iterations: int = 0
    n_function_evals: int = 0
    runtime: float = 0.0
    converged: bool = False
    message: str = ""

    log_posterior_history: list = field(default_factory=list)
    shape_history: list = field(default_factory=list)
    source_history: list = field(default_factory=list)
    noise_history: list = field(default_factory=list)


# ------------------------------------------------------------------
# ECM optimizer
# ------------------------------------------------------------------

class ECMOptimizer(Optimizer):

    def __init__(
        self,
        methods: Optional[Dict[str, str]] = None,
        scale_factor: float = 1,
        max_iter: int = 200,
        inner_max_iter: int = 500,
        ftol: float = 1e-6,
        tol_lp: float = 1e-6,
        tol_params: float = 1e-3,
        verbose: bool = False,
    ):
        super().__init__(
            max_iter=max_iter,
            tol_lp=tol_lp,
            tol_params=tol_params,
            verbose=verbose,
        )

        self.methods = methods or {
            "shape": "L-BFGS-B",
            "source": "Powell",
            "noise": "Powell",
        }

        self.scale_factor = scale_factor
        self.inner_max_iter = inner_max_iter
        self.ftol = ftol

    # ------------------------------------------------------------------
    # Safe objective wrapper (CRITICAL)
    # ------------------------------------------------------------------

    def _safe_neg_lp(self, lp_func, bounds):

        def neg_lp(x):
            # --- HARD bounds check BEFORE calling posterior ---
            if bounds is not None:
                for i, (lo, hi) in enumerate(bounds):
                    if lo is not None and x[i] < lo:
                        print("Error with bounds occured!")
                        return 1e20
                    if hi is not None and x[i] > hi:
                        print("Error with bounds occured!")
                        return 1e20

            # Safe evaluation
            return -self.scale_factor * lp_func(x)

        return neg_lp


    # ------------------------------------------------------------------
    # CM step
    # ------------------------------------------------------------------

    def _cm_step(
        self,
        target,
        shape,
        source,
        noise,
        posterior,
        shape_bounds,
        source_bounds,
        noise_bounds,
    ):
        method = self.methods[target]

        if target == 'shape':
            x0, bounds = shape.copy(), shape_bounds
            def lp_func(x):
                return posterior.log_posterior(x, source, noise)

        elif target == 'source':
            x0, bounds = source.copy(), source_bounds
            def lp_func(x):
                return posterior.log_posterior(shape, x, noise)

        elif target == 'noise':
            x0, bounds = noise.copy(), noise_bounds
            def lp_func(x):
                return posterior.log_posterior(shape, source, x)

        else:
            raise ValueError(target)

        neg_lp = self._safe_neg_lp(lp_func, bounds)

        # ---- options ----
        options = {'maxiter': self.inner_max_iter}

        if method in ('L-BFGS-B', 'BFGS'):
            options['ftol'] = self.ftol
            options["gtol"] = 1e-4

        if method in ("Powell"):
            options["ftol"] = 1e-6
            options["xtol"] = 1e-6

        def callback(xk):
            print("Current params:", xk)

        # ---- run optimizer ----
        result = minimize(
            neg_lp,
            x0,
            method=method,
            bounds=bounds if method in ('L-BFGS-B', 'TNC', 'SLSQP', "Powell") else None,
            options=options,
            #callback=callback
        )

        if not result.success:
            warnings.warn(
                f"{target} step did not fully converge: {result.message}",
                UserWarning,
            )

        new_params = result.x
        lp = lp_func(new_params)

        return new_params, lp, result.nfev


    # ------------------------------------------------------------------
    # Main optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        posterior,
        shape_init,
        source_init=None,
        noise_init=None,
        shape_bounds=None,
        source_bounds=None,
        noise_bounds=None,
    ):

        shape = np.asarray(shape_init, float)
        source = np.asarray(source_init, float) if source_init is not None else None
        noise = np.asarray(noise_init, float) if noise_init is not None else None

        lp = posterior.log_posterior(shape, source, noise)
        lp_history = [lp]
        total_nfev = 1

        # histories
        shape_history = [shape.copy()]
        source_history = [source.copy() if source is not None else None]
        noise_history = [noise.copy() if noise is not None else None]

        if self.verbose:
            print(f"ECM (scale={self.scale_factor}, methods={self.methods})")
            print(f"Initial lp: {lp:.6f}")

        start_time = time.time()
        converged = False

        for i in range(1, self.max_iter + 1):

            shape_old = shape.copy()
            source_old = source.copy() if source is not None else None
            noise_old = noise.copy() if noise is not None else None
            lp_old = lp

            # --- ECM steps ---

            if noise is not None:
                noise, lp, nfev = self._cm_step(
                    'noise', shape, source, noise,
                    posterior, shape_bounds, source_bounds, noise_bounds
                )
                total_nfev += nfev

            if source is not None:
                source, lp, nfev = self._cm_step(
                    'source', shape, source, noise,
                    posterior, shape_bounds, source_bounds, noise_bounds
                )
                total_nfev += nfev


            shape, lp, nfev = self._cm_step(
                'shape', shape, source, noise,
                posterior, shape_bounds, source_bounds, noise_bounds
            )
            total_nfev += nfev

            # --- store history ---
            shape_history.append(shape.copy())
            source_history.append(source.copy() if source is not None else None)
            noise_history.append(noise.copy() if noise is not None else None)
            lp_history.append(lp)

            # --- convergence ---
            delta_lp = abs(lp - lp_old)

            deltas = [np.max(np.abs(shape - shape_old))]
            if source is not None:
                deltas.append(np.max(np.abs(source - source_old)))
            if noise is not None:
                deltas.append(np.max(np.abs(noise - noise_old)))

            delta_params = max(deltas)

            if self.verbose:
                print(f"{i:3d} | lp={lp:.6f} | Δlp={delta_lp:.2e} | Δp={delta_params:.2e}")

                # --- print parameters (nicely formatted) ---
                print(f"      shape : {np.array2string(shape, precision=3, suppress_small=True)}")

                if source is not None:
                    print(f"      source: {np.array2string(source, precision=3, suppress_small=True)}")

                if noise is not None:
                    print(f"      noise : {np.array2string(noise, precision=7, suppress_small=True)}")

            if delta_lp < self.tol_lp and delta_params < self.tol_params:
                converged = True
                break

        runtime = time.time() - start_time

        return ECMResult(
            shape_params=shape,
            source_params=source,
            noise_params=noise,
            log_posterior_final=lp,

            log_posterior_history=lp_history,
            shape_history=shape_history,
            source_history=source_history,
            noise_history=noise_history,

            n_iterations= i,
            n_function_evals=total_nfev,
            runtime=runtime,
            converged=converged,
            message="Converged" if converged else "Max iterations reached",
        )