"""
MAP estimation for a Posterior

"""

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Tuple, Type
import numpy as np
import math
from scipy.optimize import minimize, OptimizeResult, approx_fprime
import time
from enum import Enum

from src.bayes_shape_fitting.inference.posterior import Posterior, MarginalPosterior


# ============================================================================
# MAP Estimation
# ============================================================================


def diagnose_scipy_minimize(
    result,
    log_posterior: float,
    log_post_init: float,
    runtime: float,
    *,
    min_improvement: float = 1e-6,
    max_runtime: Optional[float] = None,
    require_success: bool = True,
    grad_tol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Diagnose a SciPy `minimize` OptimizeResult.

    Parameters
    ----------
    result : OptimizeResult
        Result of scipy.optimize.minimize. Expects .success, .message, .status,
        and ideally .x, .fun, .jac, .nit, .nfev.
    log_posterior : float
        Final objective (your log posterior).
    log_post_init : float
        Initial objective value before optimization.
    runtime : float
        Runtime in seconds.
    min_improvement : float
        Minimal ∆ objective considered "meaningful".
    max_runtime : float, optional
        If set, marks run abnormal if runtime exceeds this.
    require_success : bool
        If True, a falsey `result.success` is considered abnormal.
    grad_tol : float, optional
        If provided, checks ||grad||_∞ <= grad_tol (when gradient available).

    Returns
    -------
    dict with keys:
        is_abnormal: bool
        reasons: list[str]
        summary: str
        extras: dict  # includes computed grad_norm, improvement, etc.
    """
    reasons = []
    extras = {}

    # -- Success & message
    success = bool(getattr(result, "success", False))
    message = str(getattr(result, "message", "")).strip()
    status  = getattr(result, "status", None)
    extras["success"] = success
    extras["message"] = message
    extras["status"]  = status

    if require_success and not success:
        # Common SciPy patterns
        if message:
            reasons.append(f"Optimizer reported failure (status={status}): {message}")
        else:
            reasons.append(f"Optimizer reported failure (status={status}, no message)")

    # -- Improvement
    improvement = float(log_posterior) - float(log_post_init)
    extras["improvement"] = improvement
    if improvement < min_improvement:
        reasons.append(f"Insufficient improvement: Δ={improvement:.6g} < {min_improvement:g}")

    # -- Runtime limit
    if max_runtime is not None and runtime > max_runtime:
        reasons.append(f"Runtime exceeded limit: {runtime:.3f}s > {max_runtime:.3f}s")

    # -- NaN/Inf checks

    final_fun = getattr(result, "fun", log_posterior)
    if final_fun is None or not math.isfinite(float(final_fun)):
        reasons.append("Final objective is NaN/inf")

    x = getattr(result, "x", None)
    if x is not None:
        x = np.asarray(x)
        if not np.all(np.isfinite(x)):
            reasons.append("Solution vector contains NaN/inf")

    # -- Gradient diagnostics (when available)
    jac = getattr(result, "jac", None)
    grad_norm = None
    if jac is not None:
        try:
            grad_norm = float(np.linalg.norm(np.asarray(jac), ord=np.inf))
            extras["grad_norm_inf"] = grad_norm
        except Exception:
            grad_norm = None

    if grad_tol is not None and grad_norm is not None and grad_norm > grad_tol:
        reasons.append(f"First-order optimality not satisfied: ||g||_inf={grad_norm:.3e} > tol={grad_tol:.3e}")

    # -- Method-specific hints (based on message/status heuristics)
    m = (message or "").lower()
    if "max" in m and ("iter" in m or "iteration" in m):
        reasons.append("Terminated by maximum iterations")
    if "precision" in m or "tolerance" in m:
        if not success:
            reasons.append("Terminated due to tolerance without convergence")

    is_abnormal = len(reasons) > 0
    summary = "OK" if not is_abnormal else "; ".join(reasons)
    return {"is_abnormal": is_abnormal, "reasons": reasons, "summary": summary, "extras": extras}


@dataclass
class MAPResult:
    """
    Result from MAP optimization.

    Attributes
    ----------
    theta_map : np.ndarray
        MAP estimate: argmax p(θ | X)
    log_posterior : float
        Log posterior at MAP: log p(θ_MAP | X)
    success : bool
        Whether optimization converged successfully
    n_iterations : int
        Number of optimization iterations
    n_function_evals : int
        Number of function evaluations
    runtime : float
        Total runtime in seconds
    message : str
        Convergence message from optimizer
    optimizer_result : OptimizeResult
        Full scipy optimization result
    """

    theta_map: np.ndarray
    log_posterior: float
    success: bool
    n_iterations: int
    n_function_evals: int
    runtime: float
    message: str
    optimizer_result: OptimizeResult
    param_history: list[np.ndarray] = None


class MAPOptimizer:
    """
    Maximum a posteriori (MAP) estimation for PosteriorFixed.

    Finds θ_MAP = argmax p(θ | X) via numerical optimization.

    Parameters
    ----------
    method : str, default='L-BFGS-B'
        Optimization method. Options:
        - 'L-BFGS-B': Limited-memory BFGS with box constraints (recommended)
        - 'BFGS': BFGS without constraints
        - 'Nelder-Mead': Simplex method (derivative-free)
        - 'Powell': Powell's method (derivative-free)
        - 'trust-constr': Trust-region constrained
    maxiter : int, default=1000
        Maximum number of iterations
    ftol : float, default=1e-9
        Tolerance for function value convergence
    gtol : float, default=1e-5
        Tolerance for gradient convergence (gradient-based methods)
    verbose : bool, default=False
        Print optimization progress

    Examples
    --------
    >>> optimizer = MAPOptimizer(method='L-BFGS-B', verbose=True)
    >>> result = optimizer.optimize(posterior, initial_theta)
    >>> print(f"MAP estimate: {result.theta_map}")
    >>> print(f"Log posterior: {result.log_posterior}")
    """

    def __init__(
            self,
            method: str = 'L-BFGS-B',
            maxiter: int = 1000,
            ftol: float = 1e-6,
            gtol: float = 1e-4,
            verbose: bool = False
    ):
        self.method = method
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose

    def optimize(
            self,
            posterior: MarginalPosterior,
            initial_theta: np.ndarray,
            bounds: Optional[list] = None,
    ) -> MAPResult:
        """
        Find MAP estimate.

        Parameters
        ----------
        posterior : PosteriorFixed
            Posterior distribution object
        initial_theta : np.ndarray
            Initial parameter values
        bounds : list of tuples, optional
            Box constraints [(low_1, high_1), ..., (low_n, high_n)]
            If None, uses bounds from posterior.shape if available

        Returns
        -------
        result : MAPResult
            MAP estimation results
        """
        # Get bounds from shape if not provided
        if bounds is None and hasattr(posterior.shape, 'bounds'):
            bounds = posterior.shape.bounds

        # Define objective function (negative log posterior)
        def objective(theta):
            log_post = posterior.log_posterior(theta)
            return -log_post  # Minimize negative

        # Set up options
        options = {'maxiter': self.maxiter, 'disp': self.verbose}

        if self.method in ['L-BFGS-B', 'BFGS']:
            options['ftol'] = self.ftol
            options['gtol'] = self.gtol

        # here we store the parameter history
        param_history = []

        def callback(params):
            param_history.append(np.array(params.copy()))

        # Initial evaluation
        if self.verbose:
            log_post_init = posterior.log_posterior(initial_theta)
            print(f"MAP Optimization with {self.method}")
            print("=" * 70)
            print(f"Initial log posterior: {log_post_init:.6f}")
            print(f"Number of parameters: {len(initial_theta)}")
            if bounds is not None:
                print(f"Using box constraints: {len(bounds)} bounds")
            print()

        # Run optimization
        start_time = time.time()

        result = minimize(
            fun=objective,
            x0=initial_theta,
            method=self.method,
            bounds=bounds,
            options=options,
            callback=callback
        )

        runtime = time.time() - start_time

        # Extract results
        theta_map = result.x
        log_posterior = -result.fun  # Convert back from negative

        if self.verbose:
            print()
            print("=" * 70)
            print("Optimization Complete")
            print("=" * 70)
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Final log posterior: {log_posterior:.6f}")
            print(f"Improvement: {log_posterior - log_post_init:.6f}")
            print(f"Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
            print(f"Function evaluations: {result.nfev}")
            print(f"Runtime: {runtime:.3f}s")

            diag = diagnose_scipy_minimize(
                result=result,
                log_posterior=log_posterior,
                log_post_init=log_post_init,
                runtime=runtime,
                min_improvement=1e-6,  # tune this to your scale
                max_runtime=None,  # e.g., 60.0 to flag > 1 minute
                require_success=True,
                grad_tol=self.gtol  # set to your first-order optimality tolerance, if relevant
            )

            if diag["is_abnormal"]:
                print("-" * 70)
                print("⚠ Abnormal termination detected")
                for r in diag["reasons"]:
                    print(f"• {r}")
                # Optional extra info
                grad_norm = diag["extras"].get("grad_norm_inf", None)
                if grad_norm is not None:
                    print(f"Gradient infinity-norm: {grad_norm:.3e}")
            print()

        return MAPResult(
            theta_map=theta_map,
            log_posterior=log_posterior,
            success=result.success,
            n_iterations=result.nit if hasattr(result, 'nit') else -1,
            n_function_evals=result.nfev,
            runtime=runtime,
            message=result.message,
            optimizer_result=result,
            param_history=param_history
        )

    def multi_start_optimize(
            self,
            posterior: MarginalPosterior,
            n_starts: int = 5,
            bounds: Optional[list] = None,
            seed: Optional[int] = None
    ) -> MAPResult:
        """
        Multi-start optimization to avoid local optima.

        Runs optimization from multiple random initializations and
        returns the best result.

        Parameters
        ----------
        posterior : PosteriorFixed
            Posterior distribution
        n_starts : int, default=5
            Number of random initializations
        bounds : list of tuples, optional
            Parameter bounds
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        best_result : MAPResult
            Best result across all starts
        """
        rng = np.random.default_rng(seed)

        # Get bounds
        if bounds is None and hasattr(posterior.shape, 'bounds'):
            bounds = posterior.shape.bounds

        results = []

        if self.verbose:
            print(f"\nMulti-start optimization with {n_starts} starts")
            print("=" * 70)

        for i in range(n_starts):
            if self.verbose:
                print(f"\nStart {i + 1}/{n_starts}")
                print("-" * 70)

            # Generate random initialization
            if bounds is not None:
                # Sample uniformly within bounds
                n_params = len(bounds)
                initial = np.array([
                    rng.uniform(low, high)
                    if np.isfinite(low) and np.isfinite(high)
                    else rng.normal(0, 1)
                    for low, high in bounds
                ])
            else:
                # Small random initialization
                n_params = posterior.shape.get_n_params()
                initial = rng.normal(0, 0.1, size=n_params)

            # Optimize
            result = self.optimize(posterior, initial, bounds)
            results.append(result)

        # Find best result
        best_idx = np.argmax([r.log_posterior for r in results])
        best_result = results[best_idx]

        if self.verbose:
            print("\n" + "=" * 70)
            print("Multi-start Summary")
            print("=" * 70)
            print(f"Best result from start {best_idx + 1}")
            print(f"Best log posterior: {best_result.log_posterior:.6f}")
            print(f"Successful runs: {sum(r.success for r in results)}/{n_starts}")
            print()

        return best_result