import numpy as np
from scipy.optimize import minimize


def closest_point(
    shape: "Shape",
    params: np.ndarray,
    x: np.ndarray,
    n_restarts: int = 8,
    tol: float = 1e-7,
    return_params: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Find the closest point(s) on the shape to query point(s) x.

    Solves: t* = argmin_{t ∈ D} ||γ(t; θ) - x||²

    Uses L-BFGS-B with multiple random restarts seeded from a coarse grid
    to avoid local minima on non-convex shapes.

    Parameters
    ----------
    shape : Shape
        A Shape instance with shape parameters already set (i.e., parametrization
        is callable as shape.parametrization(t, *params) or shape.parametrization(t)).
    x : np.ndarray, shape (n,) or (M, n)
        One or more query points in ambient space.
    n_restarts : int, default=8
        Number of optimizer restarts per query point. More restarts = more
        robust against local minima, but slower. For smooth convex shapes,
        n_restarts=1 may suffice.
    tol : float, default=1e-7
        Convergence tolerance passed to the optimizer.
    return_params : bool, default=False
        If True, also return the optimal parameter values t*.

    Returns
    -------
    closest : np.ndarray, shape (n,) or (M, n)
        Closest point(s) on the shape. Matches input shape convention.
    params : np.ndarray, shape (k,) or (M, k)  [only if return_params=True]
        Optimal parameter values t* achieving the minimum distance.

    Notes
    -----
    - Complexity: O(M * n_restarts * iters_per_opt * cost(γ))
    - For repeated calls with slowly-moving x, pass warm_start_params to seed
      restarts near the previous solution (not shown here, but trivial to add).
    - If the shape has a jacobian_fn, gradient info is used automatically via
      finite differences inside scipy (or you can extend this to pass jac=True).
    """
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 1
    if scalar_input:
        x = x[np.newaxis, :]  # (1, n)

    M = x.shape[0]
    k = shape.parameter_dim

    low = shape.domain_low    # (k,)
    high = shape.domain_high  # (k,)
    periodic = shape.periodic  # (k,) bool

    bounds_opt = [
        (None, None) if periodic[i] else (low[i], high[i])
        for i in range(k)
    ]

    # --- Coarse grid for seeding restarts ---
    # Sample a grid of n_restarts^k points (capped to avoid explosion)
    pts_per_dim = max(1, int(round(n_restarts ** (1.0 / k))))
    grid_1d = [np.linspace(low[i], high[i], pts_per_dim + 2)[1:-1] for i in range(k)]
    grid = np.stack(np.meshgrid(*grid_1d, indexing="ij"), axis=-1).reshape(-1, k)
    # Pad with random restarts if grid is small
    if len(grid) < n_restarts:
        extra = np.random.uniform(low, high, size=(n_restarts - len(grid), k))
        grid = np.vstack([grid, extra])

    # --- Per-point optimization ---
    best_t = np.empty((M, k))
    best_pts = np.empty((M, x.shape[1]))

    def objective(t_flat, xi):
        t = t_flat.reshape(1, k)
        # Wrap periodic dims into [low, high]
        for i in range(k):
            if periodic[i]:
                span = high[i] - low[i]
                t[0, i] = low[i] + (t[0, i] - low[i]) % span
        pt = shape.parametrization(t, params)  # (1, n)
        diff = pt[0] - xi
        return float(np.dot(diff, diff))

    for m in range(M):
        xi = x[m]
        best_val = np.inf
        best_t_m = grid[0].copy()

        for t0 in grid[:n_restarts]:
            result = minimize(
                objective,
                t0.copy(),
                args=(xi,),
                method="L-BFGS-B",
                bounds=bounds_opt,
                options={"ftol": tol**2, "gtol": tol, "maxiter": 200},
            )
            if result.fun < best_val:
                best_val = result.fun
                best_t_m = result.x

        # Final eval at best t
        t_star = best_t_m.reshape(1, k)
        for i in range(k):
            if periodic[i]:
                span = high[i] - low[i]
                t_star[0, i] = low[i] + (t_star[0, i] - low[i]) % span
        best_t[m] = t_star[0]
        best_pts[m] = shape.parametrization(t_star, params)[0]

    if scalar_input:
        return (best_pts[0], best_t[0]) if return_params else best_pts[0]
    return (best_pts, best_t) if return_params else best_pts