import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

#from ..posterior import MarginalPosterior

@dataclass
class ECMResult:
    """
    Results from ECM optimization.

    Attributes
    ----------
    shape_params : np.ndarray
        MAP estimate of the shape parameters (primary interest).
    source_params : np.ndarray
        Final source parameters at convergence.
    noise_params : np.ndarray
        Final noise parameters at convergence.
    log_posterior_final : float
        Log-posterior value at convergence.
    log_posterior_history : list of float
        Log-posterior value after each outer ECM iteration.
        Index 0 is the initial evaluation.
    n_iterations : int
        Number of outer ECM iterations completed.
    n_function_evals : int
        Total number of log-posterior evaluations across all CM steps.
    runtime : float
        Wall-clock time in seconds.
    converged : bool
        Whether the convergence criterion was met.
    message : str
        Human-readable termination message.
    """
    shape_params: np.ndarray
    source_params: np.ndarray
    noise_params: np.ndarray
    log_posterior_final: float
    log_posterior_history: list = field(default_factory=list)
    n_iterations: int = 0
    n_function_evals: int = 0
    runtime: float = 0.0
    converged: bool = False
    message: str = ""


class ECMOptimizer:
    """
    Expectation Conditional Maximization (ECM) optimizer.

    Finds the MAP estimate of shape_params by alternating conditional
    maximization steps over each parameter group:

      1. shape_params  — parameters of primary interest
      2. source_params — nuisance parameters
      3. noise_params  — nuisance parameters

    Each CM step is a lower-dimensional optimization with the other
    two groups held fixed, which is cheaper than joint optimization
    and guaranteed to monotonically increase the log-posterior.

    Parameters
    ----------
    method : str, default='L-BFGS-B'
        Optimization method used for each inner CM step. Options:
        - 'L-BFGS-B'   : Limited-memory BFGS with box constraints (recommended)
        - 'BFGS'       : BFGS without constraints
        - 'Nelder-Mead': Simplex, derivative-free (auto-selected if bounds absent)
        - 'Powell'     : Powell's method, derivative-free
    max_iter : int, default=200
        Maximum number of outer ECM iterations.
    inner_max_iter : int, default=500
        Maximum iterations for each inner scipy.optimize.minimize call.
    ftol : float, default=1e-9
        Function-value tolerance passed to inner optimizers.
    tol_lp : float, default=1e-5
        Outer convergence tolerance on the absolute change in log-posterior.
    tol_params : float, default=1e-5
        Outer convergence tolerance on the max absolute parameter change.
    verbose : bool, default=False
        Print per-iteration progress and a convergence summary.

    Examples
    --------
    >>> optimizer = ECMOptimizer(method='L-BFGS-B', verbose=True)
    >>> result = optimizer.optimize(
    ...     posterior,
    ...     shape_init=np.zeros(4),
    ...     source_init=np.zeros(2),
    ...     noise_init=np.array([1.0]),
    ... )
    >>> print(f"Shape MAP estimate: {result.shape_params}")
    >>> print(f"Final log-posterior: {result.log_posterior_final:.4f}")
    """

    def __init__(
            self,
            method: str = 'L-BFGS-B',
            max_iter: int = 200,
            inner_max_iter: int = 500,
            ftol: float = 1e-9,
            tol_lp: float = 1e-5,
            tol_params: float = 1e-5,
            verbose: bool = False,
    ):
        self.method = method
        self.max_iter = max_iter
        self.inner_max_iter = inner_max_iter
        self.ftol = ftol
        self.tol_lp = tol_lp
        self.tol_params = tol_params
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal: single conditional maximization step
    # ------------------------------------------------------------------

    def _cm_step(
            self,
            target: str,
            shape: np.ndarray,
            source: np.ndarray,
            noise: np.ndarray,
            posterior,
            shape_bounds, source_bounds, noise_bounds,
    ) -> tuple[np.ndarray, float, int]:
        """
        Maximize log_posterior over `target`, holding the others fixed.

        Returns
        -------
        params : np.ndarray
            Updated parameter array for `target`.
        log_post : float
            Log-posterior at the new parameters.
        nfev : int
            Number of function evaluations used.
        """
        if target == 'shape':
            x0, bounds = shape.copy(), shape_bounds
            def neg_lp(x): return -posterior.log_posterior(x, source, noise)
        elif target == 'source':
            x0, bounds = source.copy(), source_bounds
            def neg_lp(x): return -posterior.log_posterior(shape, x, noise)
        elif target == 'noise':
            x0, bounds = noise.copy(), noise_bounds
            def neg_lp(x): return -posterior.log_posterior(shape, source, x)
        else:
            raise ValueError(f"Unknown target: '{target}'")

        # Nelder-Mead does not accept bounds — fall back gracefully
        method = self.method
        if method == 'Nelder-Mead' and bounds is not None:
            warnings.warn(
                f"Nelder-Mead does not support bounds; switching to L-BFGS-B "
                f"for the {target} step.",
                UserWarning,
                stacklevel=3,
            )
            method = 'L-BFGS-B'

        options = {'maxiter': self.inner_max_iter}
        if method in ('L-BFGS-B', 'BFGS'):
            options['ftol'] = self.ftol

        result = minimize(neg_lp, x0, method=method, bounds=bounds, options=options)

        if not result.success:
            warnings.warn(
                f"Inner optimizer did not fully converge on {target} step: "
                f"{result.message}",
                UserWarning,
                stacklevel=3,
            )

        return result.x, -result.fun, result.nfev

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(
            self,
            posterior,
            shape_init: np.ndarray,
            source_init: Optional[np.ndarray] = None,
            noise_init: Optional[np.ndarray] = None,
            shape_bounds: Optional[list] = None,
            source_bounds: Optional[list] = None,
            noise_bounds: Optional[list] = None,
    ) -> ECMResult:
        """
        Run ECM to find the MAP estimate of shape_params.

        Parameters
        ----------
        posterior : object
            Posterior object exposing:
                log_posterior(shape_params, source_params, noise_params) -> float
            Optionally, bounds are read from ``posterior.shape.bounds``,
            ``posterior.source.bounds``, and ``posterior.noise.bounds`` if
            not provided explicitly.
        shape_init : np.ndarray
            Initial values for shape_params.
        source_init : np.ndarray or None
            Initial values for source_params. If None, the source CM step
            is skipped entirely and source_params is passed as None to
            log_posterior throughout.
        noise_init : np.ndarray or None
            Initial values for noise_params. If None, the noise CM step
            is skipped entirely and noise_params is passed as None to
            log_posterior throughout.
        shape_bounds : list of (lo, hi) or None
            Box constraints for shape_params. Overrides posterior attribute.
        source_bounds : list of (lo, hi) or None
            Box constraints for source_params. Overrides posterior attribute.
        noise_bounds : list of (lo, hi) or None
            Box constraints for noise_params. Overrides posterior attribute.

        Returns
        -------
        ECMResult
        """
        # Resolve bounds: explicit args > posterior attributes > None
        if shape_bounds is None and hasattr(posterior, 'shape') \
                and hasattr(posterior.shape, 'bounds'):
            shape_bounds = posterior.shape.bounds
        if source_bounds is None and hasattr(posterior, 'source') \
                and hasattr(posterior.source, 'bounds'):
            source_bounds = posterior.source.bounds
        if noise_bounds is None and hasattr(posterior, 'noise') \
                and hasattr(posterior.noise, 'bounds'):
            noise_bounds = posterior.noise.bounds

        shape = np.asarray(shape_init, dtype=float).copy()
        source = np.asarray(source_init, dtype=float).copy() if source_init is not None else None
        noise = np.asarray(noise_init, dtype=float).copy() if noise_init is not None else None

        lp_init = posterior.log_posterior(shape, source, noise)
        lp = lp_init
        lp_history = [lp]
        total_nfev = 1

        converged = False
        message = "Maximum iterations reached."

        if self.verbose:
            print(f"ECM Optimization with {self.method}")
            print("=" * 70)
            print(f"Initial log-posterior : {lp_init:.6f}")
            print(f"Shape params          : {len(shape)}")
            print(f"Source params         : {len(source) if source is not None else 'None (skipped)'}")
            print(f"Noise params          : {len(noise) if noise is not None else 'None (skipped)'}")
            print(f"Max outer iterations  : {self.max_iter}")
            print()
            print(f"{'Iter':>5}  {'log-posterior':>16}  {'Δlp':>12}  {'Δparams':>12}")
            print("-" * 52)
            print(f"{'init':>5}  {lp:>16.6f}")

        start_time = time.time()

        for i in range(1, self.max_iter + 1):
            shape_old = shape.copy()
            source_old = source.copy() if source is not None else None
            noise_old = noise.copy() if noise is not None else None
            lp_old = lp

            # --- CM steps: shape → source → noise (skipped if None) ---
            shape, lp, nfev = self._cm_step(
                'shape', shape, source, noise, posterior,
                shape_bounds, source_bounds, noise_bounds,
            )
            total_nfev += nfev

            if source is not None:
                source, lp, nfev = self._cm_step(
                    'source', shape, source, noise, posterior,
                    shape_bounds, source_bounds, noise_bounds,
                )
                total_nfev += nfev

            if noise is not None:
                noise, lp, nfev = self._cm_step(
                    'noise', shape, source, noise, posterior,
                    shape_bounds, source_bounds, noise_bounds,
                )
                total_nfev += nfev

            lp_history.append(lp)

            delta_lp = abs(lp - lp_old)
            deltas = [np.max(np.abs(shape - shape_old))]
            if source is not None:
                deltas.append(np.max(np.abs(source - source_old)))
            if noise is not None:
                deltas.append(np.max(np.abs(noise - noise_old)))
            delta_params = max(deltas)

            if self.verbose:
                print(f"{i:>5}  {lp:>16.6f}  {delta_lp:>12.2e}  {delta_params:>12.2e}")

            # ECM guarantees monotone increase — warn if violated
            if lp < lp_old - 1e-8:
                warnings.warn(
                    f"Log-posterior decreased at iteration {i} "
                    f"({lp_old:.6f} → {lp:.6f}). "
                    "This may indicate numerical issues in log_posterior.",
                    UserWarning,
                )

            if delta_lp < self.tol_lp and delta_params < self.tol_params:
                converged = True
                message = (
                    f"Converged after {i} iterations "
                    f"(Δlp={delta_lp:.2e}, Δparams={delta_params:.2e})."
                )
                break

        runtime = time.time() - start_time

        if self.verbose:
            print()
            print("=" * 70)
            print("ECM Optimization Complete")
            print("=" * 70)
            print(f"Converged            : {converged}")
            print(f"Message              : {message}")
            print(f"Final log-posterior  : {lp:.6f}")
            print(f"Improvement          : {lp - lp_init:.6f}")
            print(f"Outer iterations     : {i}")
            print(f"Total function evals : {total_nfev}")
            print(f"Runtime              : {runtime:.3f}s")

            if not converged:
                print()
                print("⚠ Warning: ECM did not converge within the iteration limit.")
                print(f"  Consider increasing max_iter (current: {self.max_iter}) or")
                print(f"  relaxing tol_lp / tol_params.")
            print()

        return ECMResult(
            shape_params=shape,
            source_params=source,
            noise_params=noise,
            log_posterior_final=lp,
            log_posterior_history=lp_history,
            n_iterations=i,
            n_function_evals=total_nfev,
            runtime=runtime,
            converged=converged,
            message=message,
        )
