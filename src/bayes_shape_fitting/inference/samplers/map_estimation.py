"""
MAP estimation for a Posterior

"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from scipy.optimize import minimize, OptimizeResult


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
            posterior,  # MarginalPosterior
            shape_init: np.ndarray,  # required
            source_init: Optional[np.ndarray] = None,  # optional
            noise_init: Optional[np.ndarray] = None,  # optional
            shape_bounds: Optional[List[Tuple[float, float]]] = None,
            source_bounds: Optional[List[Tuple[float, float]]] = None,
            noise_bounds: Optional[List[Tuple[float, float]]] = None,
        ):
        """
        Find MAP estimate for the subset of parameters that are provided.
        - Always optimizes 'shape'.
        - Also optimizes 'source' and/or 'noise' if their initial vectors are provided.
        """

        def _ensure_scalar(x):
            """Make sure objective returns a scalar float (sum if vector)."""
            x_arr = np.asarray(x)
            return float(x_arr.sum()) if x_arr.ndim > 0 else float(x_arr)

        # ---------------------------
        # Decide which blocks to optimize
        # ---------------------------
        optimize_shape = True
        optimize_source = source_init is not None
        optimize_noise = noise_init is not None

        # Validate 1D inputs (common for scipy.optimize)
        shape_init = np.asarray(shape_init, dtype=float).ravel()
        if optimize_source:
            source_init = np.asarray(source_init, dtype=float).ravel()
        if optimize_noise:
            noise_init = np.asarray(noise_init, dtype=float).ravel()

        # ---------------------------
        # Build x0 and bounds vector for scipy
        # ---------------------------
        blocks = []
        bounds_blocks = []

        # order: shape, source, noise
        blocks.append(shape_init)
        bounds_blocks.append(shape_bounds)

        if optimize_source:
            blocks.append(source_init)
            bounds_blocks.append(source_bounds)

        if optimize_noise:
            blocks.append(noise_init)
            bounds_blocks.append(noise_bounds)

        # Concatenate initial vector
        x0 = np.concatenate(blocks) if len(blocks) else np.array([], dtype=float)

        # Concatenate bounds (None → unbounded for that variable)
        combined_bounds = None
        if any(b is not None for b in bounds_blocks):
            combined_bounds = []
            for b, v in zip(bounds_blocks, blocks):
                if b is None:
                    # No bounds for this block → use (None, None) per parameter
                    combined_bounds.extend([(None, None)] * len(v))
                else:
                    if len(b) != len(v):
                        raise ValueError(
                            "Bounds length does not match parameter length: "
                            f"expected {len(v)}, got {len(b)}"
                        )
                    combined_bounds.extend(b)

        # ---------------------------
        # Helper: unpack a flat vector back to blocks
        # ---------------------------
        def unpack_params(flat):
            idx = 0
            # shape
            s_len = len(shape_init)
            shape_params = flat[idx:idx + s_len]
            idx += s_len

            # source
            if optimize_source:
                src_len = len(source_init)
                source_params = flat[idx:idx + src_len]
                idx += src_len
            else:
                source_params = None  # fixed as None

            # noise
            if optimize_noise:
                n_len = len(noise_init)
                noise_params = flat[idx:idx + n_len]
                idx += n_len
            else:
                noise_params = None  # fixed as None

            return shape_params, source_params, noise_params

        # ---------------------------
        # Objective (negative log posterior)
        # ---------------------------
        def objective(flat_params):
            shape_params, source_params, noise_params = unpack_params(flat_params)
            log_post = posterior.log_posterior(
                shape_params=shape_params,
                source_params=source_params,
                noise_params=noise_params
            )
            return -_ensure_scalar(log_post)

        # ---------------------------
        # Callback & history (store only the optimized vector)
        # ---------------------------
        param_history = []

        def callback(flat_params):
            # store a copy to avoid mutation
            param_history.append(np.array(flat_params, copy=True))

        # ---------------------------
        # Initial evaluation (for verbose logging)
        # ---------------------------
        if self.verbose:
            log_post_init = posterior.log_posterior(
                shape_params=shape_init,
                source_params=(source_init if optimize_source else None),
                noise_params=(noise_init if optimize_noise else None)
            )
            log_post_init = _ensure_scalar(log_post_init)

            print(f"MAP Optimization with {self.method}")
            print("=" * 70)
            print(f"Initial log posterior: {log_post_init:.6f}")
            print(f"Optimizing blocks: "
                  f"{'shape ' if optimize_shape else ''}"
                  f"{'source ' if optimize_source else ''}"
                  f"{'noise' if optimize_noise else ''}".strip())
            print(f"Number of optimized parameters: {len(x0)}")
            if combined_bounds is not None:
                print(f"Using box constraints: {len(combined_bounds)} bounds")
            print()

        # ---------------------------
        # SciPy options
        # ---------------------------
        options = {'maxiter': self.maxiter, 'disp': self.verbose}
        # Tolerances: use where supported; they’re silently ignored by some methods
        if self.method in ['L-BFGS-B', 'BFGS', 'CG', 'TNC', 'Powell', 'SLSQP', 'trust-constr', 'Nelder-Mead']:
            if getattr(self, 'ftol', None) is not None:
                options['ftol'] = self.ftol
            # 'gtol' is used e.g. by CG/BFGS; not by L-BFGS-B (which uses 'pgtol' internally)
            if getattr(self, 'gtol', None) is not None and self.method in ['CG', 'BFGS']:
                options['gtol'] = self.gtol

        # ---------------------------
        # Run optimization
        # ---------------------------
        t0 = time.perf_counter()
        result = minimize(
            fun=objective,
            x0=x0,
            method=self.method,
            bounds=combined_bounds,
            options=options,
            callback=callback
        )
        runtime = time.perf_counter() - t0

        # ---------------------------
        # Extract results
        # ---------------------------
        # Final optimized subset vector
        theta_opt = result.x

        # Map back to block structure
        shape_hat, source_hat_opt, noise_hat_opt = unpack_params(theta_opt)

        # If a block was not optimized, keep it as None (consistent with inputs)
        source_hat = source_hat_opt if optimize_source else None
        noise_hat = noise_hat_opt if optimize_noise else None

        # Convert back to the types you prefer (np.ndarray)
        shape_hat = np.asarray(shape_hat)
        if source_hat is not None:
            source_hat = np.asarray(source_hat)
        if noise_hat is not None:
            noise_hat = np.asarray(noise_hat)

        log_posterior = -float(result.fun)  # back from negative

        if self.verbose:
            print()
            print("=" * 70)
            print("Optimization Complete")
            print("=" * 70)
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Final log posterior: {log_posterior:.6f}")
            if 'log_post_init' in locals():
                print(f"Improvement: {log_posterior - log_post_init:.6f}")
            print(f"Iterations: {getattr(result, 'nit', 'N/A')}")
            print(f"Function evaluations: {result.nfev}")
            print(f"Runtime: {runtime:.3f}s")

            # Optional: your custom diagnostic (kept as in your code)
            try:
                diag = diagnose_scipy_minimize(
                    result=result,
                    log_posterior=log_posterior,
                    log_post_init=(log_post_init if 'log_post_init' in locals() else None),
                    runtime=runtime,
                    min_improvement=1e-6,
                    max_runtime=None,
                    require_success=True,
                    grad_tol=getattr(self, 'gtol', None)
                )
                if diag.get("is_abnormal", False):
                    print("-" * 70)
                    print("⚠ Abnormal termination detected")
                    for r in diag.get("reasons", []):
                        print(f"• {r}")
                    grad_norm = diag.get("extras", {}).get("grad_norm_inf")
                    if grad_norm is not None:
                        print(f"Gradient infinity-norm: {grad_norm:.3e}")
            except NameError:
                # diagnose_scipy_minimize not available; safely ignore
                pass
            print()

        # ---------------------------
        # Build return object
        # ---------------------------
        # If your MAPResult expects a flat vector, we return result.x (optimized subset).
        # If you want the structured blocks too, you can extend MAPResult to include them.
        return MAPResult(
            theta_map=theta_opt,
            log_posterior=log_posterior,
            success=result.success,
            n_iterations=getattr(result, 'nit', -1),
            n_function_evals=result.nfev,
            runtime=runtime,
            message=result.message,
            optimizer_result=result,
            param_history=param_history,

        )
