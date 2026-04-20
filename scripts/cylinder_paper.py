"""
Benchmark: Bayesian MAP vs baseline cylinder fitting algorithms
---------------------------------------------------------------
Methods compared:
  1. MAP (fixed source density)
  2. MAP (variable source density + prior)
  3. PCA-based axis estimation
  4. Least squares (nonlinear, point-to-surface distance)
  5. Algebraic Kåsa (direct linear least squares)
  6. Algebraic Pratt (constrained, less biased)

Metrics per run:
  - Radius RMSE
  - Radius bias (mean signed error)
  - Axis angle error (degrees)
  - Axis-to-axis distance
All runs use the same ground-truth theta; only the noise realization varies.
"""

import time
import numpy as np
import pandas as pd
import scipy.linalg
from tqdm import tqdm
from scipy.optimize import minimize

from bayes_shape_fitting.inference.samplers.map_estimation_global import CMAESOptimizer
from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity, PartiallyFixedDensity
from bayes_shape_fitting.models.shapes.shape_examples import shape_cases
from bayes_shape_fitting.utils.sample_data import simulate_measurement

# ── fixed experimental setup ──────────────────────────────────────────────────
PARAM_NAMES = ["x_c", "y_c", "z_c", "r", "h", "polar", "azim"]
N_RUNS      = 100
N_OBS       = 100
NOISE_STD   = 0.1
N_LIK       = 100       # likelihood MC samples

case        = shape_cases[5]
shape       = case["shape"]
#theta_true  = case["theta"]    # [0.0, 1.0, 0.0, 1.2, 10.0, 0.5, 0.3]
theta_true  = np.array([0.0, 1.0, 0.0, 2, 10.0, 0.5, 0.3])
init_theta  = np.array([0.0, 1.0, 0.5, 2.2, 9.5, 0.6, 0.2])
   # [0.2, 1.2,-0.3, 2.0,  8.5, 0.6, 0.1]

source_density_true = UniformSourceDensity(
    n_dim=2,
    domain_low=shape.domain_low,
    domain_high=shape.domain_high,
    fixed_params=np.array([0.5, 0.5, 0.6, 0.15]),
    periodic=shape.periodic,
)
noise_function = IsotropicNoise(
    dim_param=shape.parameter_dim,
    dim_ambient=shape.ambient_dim,
    fixed_params=NOISE_STD,
)

# MAP optimiser (shared across runs)
#optimizer = MAPOptimizer(method="L-BFGS-B", verbose=True, ftol=1e-10)

# CMA-ES with per-parameter scales
CMA_stds = [0.3, 0.3, 0.3,   # center xyz
            0.5,              # r
            0.4,              # h
            0.05, 0.1]

optimizer = CMAESOptimizer(sigma0=0.1, maxiter=20, verbose=True)

# ── cylinder geometry helpers ─────────────────────────────────────────────────

def _cylinder_frame(polar, azim):
    """Return axis unit vector and two orthogonal cross-section basis vectors."""
    sp, cp = np.sin(polar), np.cos(polar)
    sa, ca = np.sin(azim),  np.cos(azim)
    axis = np.array([sp * ca, sp * sa, cp])
    z    = np.array([0.0, 0.0, 1.0])
    e1   = np.cross(axis, z)
    norm = np.linalg.norm(e1)
    if norm < 1e-9:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 /= norm
    e2 = np.cross(axis, e1)
    return axis, e1, e2


def axis_angle_error_deg(theta_est, theta_true):
    """
    Angle (in degrees) between the estimated and true cylinder axes.
    Uses the unsigned angle so flipped axes (180°) are treated as 0°.
    """
    _, _, _, _, _, polar_e, azim_e = theta_est
    _, _, _, _, _, polar_t, azim_t = theta_true
    axis_e, _, _ = _cylinder_frame(polar_e, azim_e)
    axis_t, _, _ = _cylinder_frame(polar_t, azim_t)
    cos_angle = np.clip(np.abs(axis_e @ axis_t), 0.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def axis_to_axis_distance(theta_est, theta_true):
    """
    Shortest distance between the two infinite lines defined by the true
    and estimated cylinder axes (distance between skew lines).
    """
    xc_e, yc_e, zc_e, _, _, polar_e, azim_e = theta_est
    xc_t, yc_t, zc_t, _, _, polar_t, azim_t = theta_true

    p_e = np.array([xc_e, yc_e, zc_e])
    p_t = np.array([xc_t, yc_t, zc_t])
    d_e, _, _ = _cylinder_frame(polar_e, azim_e)
    d_t, _, _ = _cylinder_frame(polar_t, azim_t)

    w     = p_e - p_t
    cross = np.cross(d_e, d_t)
    denom = np.linalg.norm(cross)

    if denom < 1e-10:
        return np.linalg.norm(w - (w @ d_t) * d_t)

    return abs(w @ cross) / denom


# ── algebraic helpers ─────────────────────────────────────────────────────────

def _project_to_cross_section(pts, axis, e1, e2):
    """Project 3-D points onto the 2-D cross-section plane of the cylinder."""
    center = pts.mean(axis=0)
    v = pts - center
    return np.column_stack([v @ e1, v @ e2]), center


def _pca_axis(pts):
    """Estimate cylinder axis as the dominant PCA direction."""
    mu = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mu, full_matrices=False)
    axis = Vt[0] / np.linalg.norm(Vt[0])
    polar = np.arccos(np.clip(axis[2], -1, 1))
    azim  = np.arctan2(axis[1], axis[0]) % (2 * np.pi)
    return axis, polar, azim


def _circle_to_theta(cx, cy, r_est, center, e1, e2, axis, pts):
    """Shared back-projection: 2-D circle → 7-D cylinder parameter vector."""
    circle_center_3d = center + cx * e1 + cy * e2
    axial = (pts - circle_center_3d) @ axis
    h_est = axial.max() - axial.min()
    bottom_center = circle_center_3d + axial.min() * axis
    polar = np.arccos(np.clip(axis[2], -1, 1))
    azim  = np.arctan2(axis[1], axis[0]) % (2 * np.pi)
    return np.array([
        bottom_center[0], bottom_center[1], bottom_center[2],
        r_est, h_est, polar, azim,
    ])


# ── Baseline 1: PCA ───────────────────────────────────────────────────────────

def fit_pca(pts):
    """
    Estimate cylinder axis as the dominant direction of the point cloud via PCA,
    then recover center, r, h from projections.
    """
    center_est = pts.mean(axis=0)
    cov = np.cov((pts - center_est).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    axial = (pts - center_est) @ axis
    h_est = axial.max() - axial.min()
    bottom_center = center_est + axial.min() * axis

    v          = pts - center_est
    axial_comp = np.outer(axial, axis)
    radial_vec = v - axial_comp
    r_est      = np.mean(np.linalg.norm(radial_vec, axis=1))

    ax        = axis / np.linalg.norm(axis)
    polar_est = np.arccos(np.clip(ax[2], -1, 1))
    azim_est  = np.arctan2(ax[1], ax[0]) % (2 * np.pi)

    return np.array([
        bottom_center[0], bottom_center[1], bottom_center[2],
        r_est, h_est, polar_est, azim_est,
    ])


# ── Baseline 2: Least squares ─────────────────────────────────────────────────

def fit_least_squares(pts, init=None):
    """
    Nonlinear least squares: minimise sum of squared point-to-surface distances.
    Uses the PCA estimate as initialisation when init is None.
    """
    if init is None:
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        init = fit_pca(pts)

    def _pt_to_cyl_dist(pts_, theta_):
        xc, yc, zc, r, _, polar, azim = theta_
        center = np.array([xc, yc, zc])
        axis, e1, e2 = _cylinder_frame(polar, azim)
        v   = pts_ - center
        rad = np.sqrt((v @ e1) ** 2 + (v @ e2) ** 2)
        return np.abs(rad - r)

    bounds = [
        (-10, 10), (-10, 10), (-10, 10),
        (0.1, 10), (0.5, 20),
        (0, np.pi), (0, 2 * np.pi),
    ]

    bounds = [
        (-2, 2), (-2, 2), (-2, 2),
        (1, 4), (9, 11),
        (0, 1), (0, 1),
    ]

    def objective(theta):
        return np.sum(_pt_to_cyl_dist(pts, theta) ** 2)

    result = minimize(
        objective, x0=init, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return result.x


# ── Algebraic method 1: Kåsa (direct linear LS) ───────────────────────────────

def fit_algebraic_kasa(pts):
    """
    Direct algebraic (Kåsa) cylinder fit.

    1. Estimate axis direction via PCA (largest eigenvector).
    2. Project points onto the perpendicular cross-section plane.
    3. Solve the linear system  [2u, 2v, 1] c = u²+v²  (least squares)
       where c = [cx, cy, cx²+cy²-r²]  →  recover cx, cy, r.
    4. Map the 2-D circle centre back to 3-D; estimate h from axial extent.

    Note: minimises algebraic distance → tends to over-estimate r when
    points cover less than the full arc.
    """
    axis, polar_est, azim_est = _pca_axis(pts)
    _, e1, e2 = _cylinder_frame(polar_est, azim_est)
    Y, center = _project_to_cross_section(pts, axis, e1, e2)

    A = np.column_stack([2 * Y[:, 0], 2 * Y[:, 1], np.ones(len(Y))])
    b = Y[:, 0] ** 2 + Y[:, 1] ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    cx, cy = c[0], c[1]
    r_est  = np.sqrt(max(c[2] + cx ** 2 + cy ** 2, 1e-12))

    return _circle_to_theta(cx, cy, r_est, center, e1, e2, axis, pts)


# ── Algebraic method 2: Pratt (constrained, less biased) ─────────────────────

def fit_algebraic_pratt(pts):
    """
    Pratt-constrained algebraic cylinder fit.

    Same axis-estimation and projection as fit_algebraic_kasa, but the circle
    parameters are found by solving the generalised eigenproblem

        S v = λ M v,   S = Z^T Z,   Z = [u²+v², u, v, 1]

    with the Pratt constraint matrix M = diag(0, 1, 1, -4) (normalised so
    the coefficient of u²+v² is 1 after dividing through).  The solution is
    the eigenvector with the smallest *positive* eigenvalue, which enforces
    the geometric constraint that the fitted conic is a real circle.

    Reference:
        Pratt V. (1987) "Direct least-squares fitting of algebraic surfaces",
        SIGGRAPH Comput. Graph. 21(4), pp. 145-152.
    """
    axis, polar_est, azim_est = _pca_axis(pts)
    _, e1, e2 = _cylinder_frame(polar_est, azim_est)
    Y, center = _project_to_cross_section(pts, axis, e1, e2)

    u, v = Y[:, 0], Y[:, 1]
    Z = np.column_stack([u ** 2 + v ** 2, u, v, np.ones(len(u))])
    S = Z.T @ Z

    # Pratt constraint matrix
    M = np.array([
        [ 0,  0,  0, -2],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [-2,  0,  0,  0],
    ], dtype=float)

    try:
        eigvals, eigvecs = scipy.linalg.eigh(S, M)
    except scipy.linalg.LinAlgError:
        return fit_algebraic_kasa(pts)

    pos_mask = eigvals > 1e-12
    if not pos_mask.any():
        return fit_algebraic_kasa(pts)

    idx = np.where(pos_mask)[0][np.argmin(eigvals[pos_mask])]
    vec = eigvecs[:, idx]

    # Normalise so the leading coefficient (u²+v²) equals 1
    a, b, c, d = vec / vec[0]
    # Circle:  (u²+v²) + b·u + c·v + d = 0
    #        → (u + b/2)² + (v + c/2)² = (b²+c²)/4 - d
    cx    = -b / 2
    cy    = -c / 2
    r_est = np.sqrt(max(cx ** 2 + cy ** 2 - d, 1e-12))

    return _circle_to_theta(cx, cy, r_est, center, e1, e2, axis, pts)


# ── Single-run helper ─────────────────────────────────────────────────────────

def run_once(seed):
    np.random.seed(seed)
    measurement = simulate_measurement(
        shape, theta_true, source_density_true, noise_function, num_samples=N_OBS
    )
    pts = measurement

    results = {}

    init_theta_run = theta_true + 0.01 * np.random.randn(theta_true.size)

    shape_bounds = [
        (-2, 2), (-2, 2), (-2, 2),
        (1, 4), (9, 11),
        (0, np.pi), (0, 2 * np.pi),
    ]

    # ── MAP fixed ─────────────────────────────────────────────────────────────
    likelihood_fixed = MarginalLikelihood(
        shape=shape,
        source_density=source_density_true,
        noise_function=noise_function,
        num_samples=N_LIK,
    )


    posterior_fixed = MarginalPosterior(
        shape=shape, measurement=measurement, likelihood=likelihood_fixed
    )
    t0 = time.perf_counter()
    res_fixed = optimizer.optimize(posterior_fixed, init_theta_run, shape_bounds=shape_bounds)
    results["Fixed Prior"] = {
        "theta": res_fixed.theta_map,
        "time":  time.perf_counter() - t0,
    }

    # ── MAP variable ──────────────────────────────────────────────────────────

    source_prior_strong = GaussianPrior(mean=np.array([0.15]), cov = np.array([0.000016]))

    source_density_var_strong = UniformSourceDensity(
        n_dim=2,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
        prior=source_prior_strong,
    )

    source_density_partial_strong = PartiallyFixedDensity(source_density_var_strong, fixed_params={0:0.5,1:0.5,2:0.6})

    likelihood_var_strong = MarginalLikelihood(
        shape=shape,
        source_density=source_density_partial_strong,
        noise_function=noise_function,
        num_samples=N_LIK,
    )

    posterior_var_strong = MarginalPosterior(
        shape=shape, measurement=measurement, likelihood=likelihood_var_strong
    )

    source_init = np.array([0.15])
    source_bounds = [(0.125,0.175)]

    t0 = time.perf_counter()
    res_var = optimizer.optimize(
        posterior_var_strong,
        shape_bounds=shape_bounds,
        shape_init=init_theta_run,
        source_init=source_init,
        source_bounds=source_bounds,
    )
    results["Strong Prior"] = {
        "theta": res_var.theta_map[: shape.get_n_params()],
        "time":  time.perf_counter() - t0,
    }

    # ── MAP variable ──────────────────────────────────────────────────────────

    source_prior_weak = GaussianPrior(mean=np.array([0.15]), cov=np.array([0.00016]))

    source_density_var_weak = UniformSourceDensity(
        n_dim=2,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
        prior=source_prior_weak,
    )

    source_density_partial_weak = PartiallyFixedDensity(source_density_var_weak, fixed_params={0: 0.5, 1: 0.5, 2: 0.6})

    likelihood_var_weak = MarginalLikelihood(
        shape=shape,
        source_density=source_density_partial_weak,
        noise_function=noise_function,
        num_samples=N_LIK,
    )

    posterior_var_weak = MarginalPosterior(
        shape=shape, measurement=measurement, likelihood=likelihood_var_weak
    )

    source_init = np.array([0.15])
    source_bounds = [(0.1, 0.2)]

    t0 = time.perf_counter()
    res_var = optimizer.optimize(
        posterior_var_weak,
        shape_bounds=shape_bounds,
        shape_init=init_theta_run,
        source_init=source_init,
        source_bounds=source_bounds,
    )
    results["Weak Prior"] = {
        "theta": res_var.theta_map[: shape.get_n_params()],
        "time": time.perf_counter() - t0,
    }



    # ── Least squares ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    results["Least squares"] = {
        "theta": fit_least_squares(pts, init_theta_run),
        "time":  time.perf_counter() - t0,
    }

    # ── PCA ───────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    results["PCA"] = {
        "theta": fit_pca(pts),
        "time":  time.perf_counter() - t0,
    }

    # ── Algebraic Kåsa ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    results["Algebraic (Kåsa)"] = {
        "theta": fit_algebraic_kasa(pts),
        "time":  time.perf_counter() - t0,
    }


    return results


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark():
    print(f"Running benchmark: {N_RUNS} runs × 6 methods")
    print(f"Ground truth θ = {theta_true}\n")

    METHOD_ORDER = [
        "Least squares",
        "PCA",
        "Algebraic (Kåsa)",
        "Weak Prior",
        "Strong Prior",
        "Fixed Prior",
    ]
    all_records = []
    geo_records = []
    timing      = {m: [] for m in METHOD_ORDER}

    for i in tqdm(range(N_RUNS), desc="Benchmark runs", unit="run"):
        run_results = run_once(seed=100 + i)

        for method, res in run_results.items():
            theta_hat = np.asarray(res["theta"])
            timing[method].append(res["time"])
            for j, pname in enumerate(PARAM_NAMES):
                all_records.append({
                    "method":    method,
                    "param":     pname,
                    "error":     theta_hat[j] - theta_true[j],
                    "abs_error": abs(theta_hat[j] - theta_true[j]),
                })
            geo_records.append({
                "method":         method,
                "axis_angle_deg": axis_angle_error_deg(theta_hat, theta_true),
                "axis_dist":      axis_to_axis_distance(theta_hat, theta_true),
            })

    df     = pd.DataFrame(all_records)
    geo_df = pd.DataFrame(geo_records)

    # ── radius RMSE ───────────────────────────────────────────────────────────
    r_df   = df[df["param"] == "r"].copy()
    r_rmse = (
        r_df.groupby("method")["error"]
        .apply(lambda e: np.sqrt((e ** 2).mean()))
        .rename("radius_rmse")
    )

    # ── radius bias (mean signed error) ──────────────────────────────────────
    r_bias = (
        r_df.groupby("method")["error"]
        .mean()
        .rename("radius_bias")
    )

    # ── geometry metrics (RMSE + mean) ─────────────────────────────────────────
    geo_summary = geo_df.groupby("method").agg(
        axis_angle_rmse=("axis_angle_deg",
                         lambda e: np.sqrt((e ** 2).mean())),
        axis_angle_mean=("axis_angle_deg", "mean"),

        axis_dist_rmse=("axis_dist",
                        lambda e: np.sqrt((e ** 2).mean())),
        axis_dist_mean=("axis_dist", "mean"),
    )

    # ── combined summary ──────────────────────────────────────────────────────
    summary = pd.concat([r_rmse, r_bias, geo_summary], axis=1).loc[METHOD_ORDER]

    # ── timing ────────────────────────────────────────────────────────────────
    timing_df = pd.DataFrame(
        {m: {"mean_time_s": np.mean(v), "std_time_s": np.std(v)}
         for m, v in timing.items()}
    ).T.loc[METHOD_ORDER]

    # ── print ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(summary.round(4).to_string())

    print("\n" + "=" * 70)
    print("MEAN RUNTIME (seconds)")
    print("=" * 70)
    print(timing_df.round(4).to_string())

    # ── save ──────────────────────────────────────────────────────────────────
    summary.to_csv("benchmark_summary.csv")
    timing_df.to_csv("benchmark_timing.csv")
    df.to_csv("benchmark_raw_errors.csv", index=False)
    print("\nResults saved to benchmark_summary.csv / benchmark_timing.csv / benchmark_raw_errors.csv")

    return summary, timing_df, df


# ── Quick visual check (single measurement) ──────────────────────────────────

if __name__ == "__main__":
    """plotter     = ShapePlotter3D()
    measurement = simulate_measurement(
        shape, theta_true, source_density_true, noise_function
    )

    shape_bounds = [
        (-10, 10), (-10, 10), (-10, 10),
        (1, 4), (9, 11),
        (0, np.pi), (0, 2*np.pi),
    ]

    # ── MAP fixed ─────────────────────────────────────────────────────────────
    likelihood_fixed = MarginalLikelihood(
        shape=shape,
        source_density=source_density_true,
        noise_function=noise_function,
        num_samples=N_LIK,
    )

    posterior_fixed = MarginalPosterior(
        shape=shape, measurement=measurement, likelihood=likelihood_fixed
    )
    t0 = time.perf_counter()

    CMA_stds = [0.3, 0.3, 0.3,  # center xyz
                0.5,  # r
                0.4,  # h
                0.05, 0.1]

    res_fixed = optimizer.optimize(posterior_fixed, init_theta, shape_bounds=shape_bounds, CMA_stds = CMA_stds)

    source_density_var = UniformSourceDensity(
        n_dim=2,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
        # prior=source_prior,
    )

    source_density_partial = PartiallyFixedDensity(source_density_var, fixed_params={0: 0.5, 1: 0.5, 2: 0.6})

    likelihood_var = MarginalLikelihood(
        shape=shape,
        source_density=source_density_partial,
        noise_function=noise_function,
        num_samples=N_LIK,
    )

    posterior_var = MarginalPosterior(
        shape=shape, measurement=measurement, likelihood=likelihood_var
    )

    source_init = np.array([0.1])
    source_bounds = [(0.05, 0.15)]

    CMA_stds = [0.3, 0.3, 0.3,  # center xyz
                0.5,  # r
                0.4,  # h
                0.05, 0.1, 0.05]

    t0 = time.perf_counter()
    res_var = optimizer.optimize(
        posterior_var,
        shape_bounds=shape_bounds,
        shape_init=init_theta,
        source_init=source_init,
        source_bounds=source_bounds, CMA_stds=CMA_stds
    )

    print(res_var.theta_map)

    fig, ax = plotter.plot_shape(shape, theta_true, label="Ground Truth")
    plotter.plot_shape(shape, res_fixed.theta_map, ax=ax, label = "Fixed", color = "black")
    plotter.plot_shape(shape, res_var.theta_map[: shape.get_n_params()], ax=ax, label = "VAR", color = "brown")
    plotter.plot_shape(shape, fit_pca(measurement),              ax=ax, label="PCA",              color="green")
    plotter.plot_shape(shape, fit_least_squares(measurement, init_theta),    ax=ax, label="LS",               color= "yellow")
    plotter.plot_shape(shape, fit_algebraic_kasa(measurement),   ax=ax, label="Kåsa",             color="orange")
    plotter.plot_shape(shape, fit_algebraic_pratt(measurement),  ax=ax, label="Pratt",            color="purple")
    plotter.plot_observations(observations=measurement, ax=ax)
    plt.show()"""

    summary, timing_df, df = run_benchmark()