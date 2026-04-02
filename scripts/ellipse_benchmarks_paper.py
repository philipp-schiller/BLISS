"""
Ellipse Fitting Benchmarking Framework
========================================
Compares multiple ellipse fitting methods on synthetic 2D data.
Focus: partial arc coverage (ill-posed scenario) and noise robustness.
No outliers — data is clean but incomplete.

Methods included:
  - Bookstein (1979)       — algebraic, constraint ||[a,c]||=1
  - Fitzgibbon (1999)      — algebraic, ellipse-specific constraint 4ac-b²=1
  - Halir & Flusser (1998) — numerically stable Fitzgibbon variant
  - Taubin (1991)          — gradient-weighted algebraic fit
  - Geometric LM           — minimizes orthogonal distance (LM optimizer)
  - HyperLS (Kanatani)     — bias-corrected algebraic fit, state-of-the-art accuracy
  - MCC-VC (2023)          — Maximum Correntropy Criterion with variable center

Usage:
    python ellipse_benchmark.py

Requirements:
    pip install numpy scipy matplotlib
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize

from examples.basic.fit_ellipse import source_density
from src.bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from src.bayes_shape_fitting.inference.posterior import MarginalPosterior
from src.bayes_shape_fitting.inference.priors import GaussianPrior
from src.bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from src.bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from src.bayes_shape_fitting.models.shapes.ellipse import Ellipse
from tests.fixtures.sample_data import simulate_measurement


# ─────────────────────────────────────────────
# 1. ELLIPSE PARAMETERISATION HELPERS
# ─────────────────────────────────────────────
#
# We use two representations interchangeably:
#
# (A) General conic:  ax² + bxy + cy² + dx + ey + f = 0
#     stored as coeffs = [a, b, c, d, e, f]
#
# (B) Geometric:      (cx, cy, semi_a, semi_b, angle)
#     center, semi-axes, rotation angle in radians
#
# All fitting methods return a dict with both forms filled in.

def conic_to_geometric(coeffs):
    """Convert general conic [a,b,c,d,e,f] to geometric params."""
    a, b, c, d, e, f = [float(v) for v in coeffs]

    # Discriminant check: must be an ellipse
    if b**2 - 4*a*c >= 0:
        return None

    # Center via solving the 2x2 linear system
    # 2a*cx + b*cy + d = 0
    # b*cx + 2c*cy + e = 0
    denom = 4*a*c - b**2   # > 0 for ellipse
    cx = (b*e - 2*c*d) / denom
    cy = (b*d - 2*a*e) / denom

    # Value of conic at center (used to get axes)
    f_prime = a*cx**2 + b*cx*cy + c*cy**2 + d*cx + e*cy + f
    if f_prime == 0:
        return None

    # Eigenvalues of [[a, b/2], [b/2, c]] give reciprocal squared semi-axes
    M = np.array([[a, b/2], [b/2, c]])
    evals, evecs = np.linalg.eigh(M)  # evals in ascending order

    # Semi-axes: axis_i = sqrt(-f_prime / eval_i)
    if f_prime > 0:
        # axes require -f_prime/eval < 0, so evals must be negative — wrong sign
        # Flip: normalise so that f_prime < 0
        a, b, c, d, e, f = -a, -b, -c, -d, -e, -f
        f_prime = -f_prime
        M = -M
        evals, evecs = np.linalg.eigh(M)

    vals = -f_prime / evals
    if np.any(vals <= 0):
        return None

    semi_axes = np.sqrt(vals)
    # eigh returns ascending eigenvalues:
    #   evals[0] = smallest eigenvalue → largest semi-axis = semi_a
    #   evals[1] = largest  eigenvalue → smallest semi-axis = semi_b
    semi_a_val = semi_axes[0]
    semi_b_val = semi_axes[1]

    # Rotation angle from eigenvector of larger semi-axis (smaller eigenvalue → larger axis)
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])

    return {
        "center": np.array([cx, cy]),
        "semi_a": semi_a_val,
        "semi_b": semi_b_val,
        "angle":  angle % np.pi,
        "coeffs": np.array([a, b, c, d, e, f]),
    }


def geometric_to_conic(cx, cy, semi_a, semi_b, angle):
    """Convert geometric params to general conic coefficients."""
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    a2, b2 = semi_a**2, semi_b**2
    A = cos_t**2/a2 + sin_t**2/b2
    B = 2*cos_t*sin_t*(1/a2 - 1/b2)
    C = sin_t**2/a2 + cos_t**2/b2
    D = -2*A*cx - B*cy
    E = -B*cx - 2*C*cy
    F = A*cx**2 + B*cx*cy + C*cy**2 - 1
    return np.array([A, B, C, D, E, F])


def ellipse_point_distances(params, points):
    """Approximate algebraic distances for residual computation."""
    a, b, c, d, e, f = params
    x, y = points[:, 0], points[:, 1]
    return a*x**2 + b*x*y + c*y**2 + d*x + e*y + f


# ─────────────────────────────────────────────
# 2. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_ellipse_points(
    center=(0.0, 0.0),
    semi_a=3.0,
    semi_b=1.5,
    angle=np.pi / 6,
    n_points=100,
    noise_std=0.0,
    outlier_ratio=0.0,
    arc_fraction=1.0,   # 1.0 = full ellipse, 0.5 = half arc (harder)
    seed=42
):
    """
    Generate 2D points sampled on an ellipse perimeter.

    arc_fraction < 1.0 simulates partial/occluded ellipses — a key
    ill-posed scenario where many methods struggle.
    """
    rng = np.random.default_rng(seed)
    cx, cy = center

    t_max = 2 * np.pi * arc_fraction
    t = rng.uniform(0, t_max, n_points)

    # Points on axis-aligned ellipse
    x = semi_a * np.cos(t)
    y = semi_b * np.sin(t)

    # Rotate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = cos_a * x - sin_a * y + cx
    yr = sin_a * x + cos_a * y + cy

    points = np.column_stack([xr, yr])

    if noise_std > 0:
        points += rng.normal(0, noise_std, points.shape)

    if outlier_ratio > 0:
        n_out = int(n_points * outlier_ratio)
        lo = points.min(axis=0) - semi_a
        hi = points.max(axis=0) + semi_a
        outliers = rng.uniform(lo, hi, (n_out, 2))
        points = np.vstack([points, outliers])

    ground_truth = {
        "center": np.array([cx, cy]),
        "semi_a": semi_a,
        "semi_b": semi_b,
        "angle":  angle,
        "coeffs": geometric_to_conic(cx, cy, semi_a, semi_b, angle),
    }
    return points, ground_truth


# ─────────────────────────────────────────────
# 2. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_ellipse_points(
    center=(0.0, 0.0),
    semi_a=3.0,
    semi_b=1.5,
    angle=np.pi / 6,
    n_points=100,

    sourc_density = None,
    noise_function = None,

    seed=42
):
    theta=np.hstack([center,semi_a,semi_b,angle])
    mesurement = simulate_measurement(shape=Ellipse(), theta=theta, source_density=sourc_density, noise_function=noise_function, num_samples=n_points)
    return measurement


# ─────────────────────────────────────────────
# 3. FITTING METHODS
# ─────────────────────────────────────────────

def fit_bookstein(points):
    """
    Bookstein (1979) algebraic fit.
    Constraint: a² + c² = 1  (norm of [a,c])
    Minimizes algebraic distance. Not ellipse-specific.

    Reference: Bookstein, F.L. (1979). "Fitting conic sections to scattered data."
    Computer Graphics and Image Processing, 9(1), 56-71.
    """
    x, y = points[:, 0], points[:, 1]
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones(len(x))])

    # Constraint matrix: a²+c²=1 → columns 0 and 2
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 0] = 1
    C[2, 2] = 1

    try:
        evals, evecs = np.linalg.eig(np.linalg.inv(S) @ C)
    except np.linalg.LinAlgError:
        return None

    # Try all real eigenvectors and return the first valid ellipse
    real_mask = np.isreal(evals)
    for idx in np.where(real_mask)[0]:
        coeffs = np.real(evecs[:, idx])
        coeffs /= np.linalg.norm(coeffs)
        result = conic_to_geometric(coeffs)
        if result is not None:
            return result
    return None


def fit_fitzgibbon(points):
    """
    Fitzgibbon, Pilu & Fisher (1999) — Direct Least Squares Ellipse Fitting.
    Constraint: 4ac - b² = 1  (guarantees ellipse solution).
    The most widely cited algebraic ellipse fitter.

    Reference: Fitzgibbon, A., Pilu, M., Fisher, R.B. (1999).
    "Direct least square fitting of ellipses."
    IEEE TPAMI 21(5), 476-480.
    """
    x, y = points[:, 0], points[:, 1]
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones(len(x))])
    S = D.T @ D

    # Constraint matrix for 4ac - b² = 1
    C = np.zeros((6, 6))
    C[0, 2] = 2; C[2, 0] = 2; C[1, 1] = -1

    try:
        evals, evecs = np.linalg.eig(np.linalg.inv(S) @ C)
    except np.linalg.LinAlgError:
        return None

    # Ellipse-specific: pick eigenvector with positive finite eigenvalue
    valid = np.isreal(evals) & (np.real(evals) > 0) & np.isfinite(np.real(evals))
    if not np.any(valid):
        return None
    idx = np.argmax(np.where(valid, np.real(evals), -np.inf))
    coeffs = np.real(evecs[:, idx])

    return conic_to_geometric(coeffs)

def fit_taubin(points):
    """
    Taubin (1991) — gradient-weighted algebraic distance fit.
    Reduces the bias of pure algebraic methods by normalizing by the squared
    gradient magnitude, making the cost closer to geometric distance.
    Solves: S v = lambda * M v  (smallest eigenvalue)
    where M = (1/n) sum_i [Jx_i^T Jx_i + Jy_i^T Jy_i]
    and Jx, Jy are the Jacobians of the design vector d_i w.r.t. x and y.

    Reference: Taubin, G. (1991). IEEE TPAMI 13(11), 1115-1138.
    """
    x, y = points[:, 0], points[:, 1]
    n = len(x)
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones(n)])
    S = D.T @ D

    # Jacobian of d_i = [xi², xiyi, yi², xi, yi, 1] w.r.t. xi and yi
    # dD/dxi = [2xi, yi,  0,  1,  0,  0]
    # dD/dyi = [0,   xi, 2yi, 0,  1,  0]
    Jx = np.column_stack([2*x, y,            np.zeros(n), np.ones(n),  np.zeros(n), np.zeros(n)])
    Jy = np.column_stack([np.zeros(n), x,    2*y,         np.zeros(n), np.ones(n),  np.zeros(n)])
    M = (Jx.T @ Jx + Jy.T @ Jy) / n
    # The constant term (f) has zero gradient w.r.t. x and y,
    # making M rank-deficient. Add a small ridge on the last diagonal entry.
    M[5, 5] += 1e-6

    try:
        evals, evecs = np.linalg.eig(np.linalg.solve(M, S))
    except np.linalg.LinAlgError:
        return None

    order = np.argsort(np.where(np.isreal(evals), np.real(evals), np.inf))
    for idx in order:
        coeffs = np.real(evecs[:, idx])
        result = conic_to_geometric(coeffs)
        if result is not None:
            return result
    return None

def fit_hyperls(points):
    """
    HyperLS — Kanatani & Sugaya (2010/2011).
    Bias-corrected algebraic fitting. Corrects the statistical bias of
    least-squares algebraic methods to first order using the noise covariance
    structure. Regarded as the most accurate non-iterative algebraic ellipse
    fitter. No free parameters or iterations required.

    Reference: Kanatani, K., Sugaya, Y. (2010).
    "Compact algorithm for strictly ML ellipse fitting." Proc. BMVC.
    Also: Kanatani, Sugaya, Niitsuma (2011), BMVC.
    """
    x, y = points[:, 0], points[:, 1]
    n = len(x)

    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones(n)])
    S = D.T @ D / n

    # Estimate sigma² from a preliminary Fitzgibbon fit
    init = fit_fitzgibbon(points)
    if init is None:
        return fit_taubin(points)

    a0, b0, c0, d0, e0, f0 = init["coeffs"]
    gx = 2*a0*x + b0*y + d0
    gy = b0*x + 2*c0*y + e0
    grad_sq = gx**2 + gy**2
    alg_vals = a0*x**2 + b0*x*y + c0*y**2 + d0*x + e0*y + f0
    sigma2 = np.mean(alg_vals**2) / (np.mean(grad_sq) + 1e-12)

    # Build noise correction matrix L = sigma² * (1/n) sum_i J_i J_i^T
    # where J_i is the Jacobian of [xi², xiyi, yi², xi, yi, 1] w.r.t. (xi, yi)
    L = np.zeros((6, 6))
    for xi, yi in zip(x, y):
        J = np.array([
            [2*xi, 0   ],
            [yi,   xi  ],
            [0,    2*yi],
            [1,    0   ],
            [0,    1   ],
            [0,    0   ]
        ])
        L += J @ J.T
    L = sigma2 * L / n

    # Solve generalised eigenproblem (S - L) v = lambda * C v
    C = np.zeros((6, 6))
    C[0, 2] = 2; C[2, 0] = 2; C[1, 1] = -1

    try:
        evals, evecs = np.linalg.eig(np.linalg.inv(S - L) @ C)
    except np.linalg.LinAlgError:
        return init

    valid = np.isreal(evals) & (np.real(evals) > 0) & np.isfinite(np.real(evals))
    if not np.any(valid):
        return init

    idx = np.argmax(np.where(valid, np.real(evals), -np.inf))
    coeffs = np.real(evecs[:, idx])
    result = conic_to_geometric(coeffs)
    return result if result is not None else init

def fit_bayes_fixed(points):
    """
    Replace this with your own ellipse fitting method.
    Must return a dict with keys:
        center  : np.array([cx, cy])
        semi_a  : float  (larger semi-axis)
        semi_b  : float  (smaller semi-axis)
        angle   : float  (rotation in radians)
        coeffs  : np.array([a,b,c,d,e,f])  (general conic form)
    """

    shape = Ellipse()
    source_density = UniformSourceDensity(n_dim = 1, domain_low = shape.domain_low, domain_high = shape.domain_high, periodic = shape.periodic)
    noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)
    likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=100)

    # let's also define a prior on the Source density
    mean = np.array([0.3, 0.2])  # the first parameter resembles the interval center c, and the second the half-width w
    cov = np.diag([0.1, 0.1])  # we set a prior independently on those
    prior = GaussianPrior(mean=mean, cov=cov)

    posterior = MarginalPosterior(shape=shape, measurement=points, likelihood=likelihood)
    optimizer = MAPOptimizer(method='Powell', verbose=True, ftol=1e-6, gtol=1e-6)

    # compute the inital_fit via fitzgibbon
    init = fit_taubin(points)
    init_theta = conic_to_geometric(init["coeffs"])
    # bring everything to the correct form + add initial guess for source_density
    init_coeffs = np.hstack([init_theta["center"], init_theta["semi_a"], init_theta["semi_b"], init_theta["angle"]])
    source_init =  np.array([0.2, 0.15])

    result = optimizer.optimize(posterior, shape_init = init_coeffs, source_init = source_init, shape_bounds = shape.bounds, source_bounds =  [(0.01,1),(0.01,0.5)])

    cx, cy, sa, sb, ang = result.theta_map[0:5]
    sa, sb = abs(sa), abs(sb)
    # Normalise angle to [0, pi)
    ang = ang % np.pi
    coeffs = geometric_to_conic(cx, cy, sa, sb, ang)
    return {"center": np.array([cx, cy]),
            "semi_a": sa, "semi_b": sb,
            "angle": ang, "coeffs": coeffs}

def fit_bayes_variable(points):
    """
    Replace this with your own ellipse fitting method.
    Must return a dict with keys:
        center  : np.array([cx, cy])
        semi_a  : float  (larger semi-axis)
        semi_b  : float  (smaller semi-axis)
        angle   : float  (rotation in radians)
        coeffs  : np.array([a,b,c,d,e,f])  (general conic form)
    """

    shape = Ellipse()

    # let's also define a prior on the Source density
    mean = np.array([0.3, 0.2])  # the first parameter resembles the interval center c, and the second the half-width w
    cov = np.diag([0.1, 0.1])  # we set a prior independently on those
    prior = GaussianPrior(mean=mean, cov=cov)

    source_density = UniformSourceDensity(n_dim = 1, domain_low = shape.domain_low, domain_high = shape.domain_high, periodic = shape.periodic, prior=prior)
    noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.1)
    likelihood = MarginalLikelihood(shape=shape, source_density=source_density, noise_function=noise_function, num_samples=100)

    posterior = MarginalPosterior(shape=shape, measurement=points, likelihood=likelihood)
    optimizer = MAPOptimizer(method='Powell', verbose=True, ftol=1e-6, gtol=1e-6)

    # compute the inital_fit via fitzgibbon
    init = fit_taubin(points)
    init_theta = conic_to_geometric(init["coeffs"])
    # bring everything to the correct form + add initial guess for source_density
    init_coeffs = np.hstack([init_theta["center"], init_theta["semi_a"], init_theta["semi_b"], init_theta["angle"]])
    source_init =  np.array([0.2, 0.15])

    result = optimizer.optimize(posterior, shape_init = init_coeffs, source_init = source_init, shape_bounds = shape.bounds, source_bounds =  [(0.01,1),(0.01,0.5)])

    cx, cy, sa, sb, ang = result.theta_map[0:5]
    sa, sb = abs(sa), abs(sb)
    # Normalise angle to [0, pi)
    ang = ang % np.pi
    coeffs = geometric_to_conic(cx, cy, sa, sb, ang)
    return {"center": np.array([cx, cy]),
            "semi_a": sa, "semi_b": sb,
            "angle": ang, "coeffs": coeffs}

# ─────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────

def compute_metrics(fitted, ground_truth, points):
    """
    Compute comparison metrics between a fitted ellipse and ground truth.

    Metrics:
      - rms_algebraic  : RMS of normalized algebraic distance (all methods comparable)
      - center_error   : Euclidean distance between centers
      - semi_axis_err  : Mean absolute error of semi-axes (sorted to handle swaps)
      - angle_error_deg: Angular difference in orientation (mod 180°)
    """
    nan_result = {k: np.nan for k in
                  ["rms_algebraic", "center_error", "semi_axis_err", "angle_error_deg"]}

    if fitted is None:
        return nan_result

    # RMS algebraic distance (Taubin-normalized)
    a, b, c, d, e, f = fitted["coeffs"]
    x, y = points[:, 0], points[:, 1]
    vals = a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
    grad_norm = np.sqrt((2*a*x + b*y + d)**2 + (b*x + 2*c*y + e)**2 + 1e-12)
    rms_alg = np.sqrt(np.mean((vals / grad_norm)**2))

    # Center error
    center_err = np.linalg.norm(fitted["center"] - ground_truth["center"])

    # Semi-axis error (sort axes to handle a↔b swap)
    fit_axes = sorted([fitted["semi_a"], fitted["semi_b"]])
    gt_axes  = sorted([ground_truth["semi_a"], ground_truth["semi_b"]])
    semi_err = np.mean(np.abs(np.array(fit_axes) - np.array(gt_axes)))

    # Angle error (mod π, handle symmetry)
    da = (fitted["angle"] - ground_truth["angle"]) % np.pi
    angle_err = min(da, np.pi - da)
    angle_err_deg = np.degrees(angle_err)

    return {
        "rms_algebraic":   rms_alg,
        "center_error":    center_err,
        "semi_axis_err":   semi_err,
        "angle_error_deg": angle_err_deg,
    }


# ─────────────────────────────────────────────
# 6. BENCHMARK RUNNER
# ─────────────────────────────────────────────

METHODS = {
    "Bookstein":      fit_bookstein,
    "Fitzgibbon":     fit_fitzgibbon,
    "Taubin":         fit_taubin,
    "HyperLS":        fit_hyperls,
    "Bayes_fixed":    fit_bayes_fixes,
    "Bayes_variable": fit_bayes_variable
}

GT_PARAMS = dict(
    center=(1.0, -0.5),
    semi_a=3.0,
    semi_b=1.5,
    angle=np.pi / 6,
    n_points=100,
    source_density = UniformSourceDensity,
)


def _run_sweep(param_values, param_name, extra_kwargs, n_trials=5):
    """Run sweep, averaging metrics over multiple random seeds for robustness."""
    results = {name: {"rms": [], "center": [], "axes": [], "angle": [], "time": []}
               for name in METHODS}

    for val in param_values:
        # Accumulate per-seed results then average (ignoring NaN)
        trial_data = {name: {"rms": [], "center": [], "axes": [], "angle": [], "time": []}
                      for name in METHODS}

        for seed in range(n_trials):
            kw = {**GT_PARAMS, param_name: val, **extra_kwargs, "seed": seed}
            points, gt = generate_ellipse_points(**kw)

            for name, method in METHODS.items():
                t0 = time.perf_counter()
                try:
                    fitted = method(points)
                except NotImplementedError:
                    fitted = None
                elapsed = time.perf_counter() - t0

                m = compute_metrics(fitted, gt, points)
                trial_data[name]["rms"].append(m["rms_algebraic"])
                trial_data[name]["center"].append(m["center_error"])
                trial_data[name]["axes"].append(m["semi_axis_err"])
                trial_data[name]["angle"].append(m["angle_error_deg"])
                trial_data[name]["time"].append(elapsed)

        for name in METHODS:
            for key in ("rms", "center", "axes", "angle", "time"):
                vals = np.array(trial_data[name][key], dtype=float)
                results[name][key].append(float(np.nanmean(vals)))

    return results


def run_noise_sweep(noise_levels=None):
    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    return noise_levels, _run_sweep(noise_levels, "noise_std", {})


def run_arc_sweep(arc_fractions=None):
    """
    Sweep over arc completeness — the key ill-posed scenario for ellipse fitting.
    arc_fraction=1.0 is a full ellipse; 0.25 is a quarter arc.
    """
    if arc_fractions is None:
        arc_fractions = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    return arc_fractions, _run_sweep(arc_fractions, "arc_fraction", {})


# ─────────────────────────────────────────────
# 7. PLOTTING
# ─────────────────────────────────────────────

def plot_sweep(x_values, results, x_label, title_suffix):
    metric_keys = [
        ("rms",    "RMS Algebraic Error"),
        ("center", "Center Error"),
        ("axes",   "Semi-Axis Error"),
        ("angle",  "Angle Error (deg)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"Ellipse Fitting Benchmark — {title_suffix}", fontsize=13)

    for ax, (key, label) in zip(axes, metric_keys):
        for name, data in results.items():
            ax.plot(x_values, data[key], marker='o', label=name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_runtime(x_values, results, x_label, title_suffix):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, data in results.items():
        ax.plot(x_values, [t * 1000 for t in data["time"]], marker='o', label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(f"Runtime — {title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_example_fits(noise_std=0.1, arc_fraction=0.3):
    """Visual overlay of all method fits on a single partial-arc example."""
    points, gt = generate_ellipse_points(
        **GT_PARAMS, noise_std=noise_std, arc_fraction=arc_fraction
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(points[:, 0], points[:, 1], s=12, color='gray', alpha=0.6, label='Data (partial arc)')

    theta = np.linspace(0, 2 * np.pi, 300)

    # Ground truth (full ellipse, dashed)
    cx, cy = gt["center"]
    sa, sb, ang = gt["semi_a"], gt["semi_b"], gt["angle"]
    xe = cx + sa*np.cos(theta)*np.cos(ang) - sb*np.sin(theta)*np.sin(ang)
    ye = cy + sa*np.cos(theta)*np.sin(ang) + sb*np.sin(theta)*np.cos(ang)
    ax.plot(xe, ye, 'k--', lw=2, label='Ground Truth')

    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    for (name, method), color in zip(METHODS.items(), colors):
        try:
            fitted = method(points)
        except NotImplementedError:
            continue
        if fitted is None:
            continue
        cx_f, cy_f = fitted["center"]
        sa_f, sb_f, ang_f = fitted["semi_a"], fitted["semi_b"], fitted["angle"]
        xe_f = cx_f + sa_f*np.cos(theta)*np.cos(ang_f) - sb_f*np.sin(theta)*np.sin(ang_f)
        ye_f = cy_f + sa_f*np.cos(theta)*np.sin(ang_f) + sb_f*np.sin(theta)*np.cos(ang_f)
        ax.plot(xe_f, ye_f, lw=1.5, color=color, label=name)

    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.set_title(f"Visual Fit Comparison — partial arc (arc={arc_fraction}, noise={noise_std})")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def print_summary_table(noise_std=0.1, arc_fraction=0.4):
    print(f"\n── Summary Table (noise={noise_std}, arc_fraction={arc_fraction}) ──")
    points_n, gt_n = generate_ellipse_points(**GT_PARAMS, noise_std=noise_std)
    points_a, gt_a = generate_ellipse_points(**GT_PARAMS, noise_std=noise_std,
                                              arc_fraction=arc_fraction)

    header = (f"{'Method':<18} {'RMS(full arc)':<16} {'RMS(partial arc)':<18} "
              f"{'CenterErr':<12} {'AxisErr':<10} {'AngleErr°':<11} {'Time(ms)'}")
    print(header)
    print("─" * len(header))

    for name, method in METHODS.items():
        try:
            t0 = time.perf_counter()
            f_n = method(points_n)
            t_ms = (time.perf_counter() - t0) * 1000
            f_a = method(points_a)
            mn = compute_metrics(f_n, gt_n, points_n)
            ma = compute_metrics(f_a, gt_a, points_a)
            print(f"{name:<18} {mn['rms_algebraic']:<16.5f} {ma['rms_algebraic']:<18.5f} "
                  f"{mn['center_error']:<12.5f} {mn['semi_axis_err']:<10.5f} "
                  f"{mn['angle_error_deg']:<11.4f} {t_ms:.3f}")
        except NotImplementedError:
            print(f"{name:<18} (not implemented)")


if __name__ == "__main__":
    print("Running noise sweep (full arc)...")
    noise_levels, noise_results = run_noise_sweep()
    plot_sweep(noise_levels, noise_results, "Noise Std Dev", "Noise Sweep (full arc)")
    plot_runtime(noise_levels, noise_results, "Noise Std Dev", "Noise Sweep")

    print("Running arc completeness sweep (key ill-posed scenario)...")
    arc_fracs, arc_results = run_arc_sweep()
    plot_sweep(arc_fracs, arc_results, "Arc Fraction (1.0=full)", "Arc Completeness Sweep")
    plot_runtime(arc_fracs, arc_results, "Arc Fraction (1.0=full)", "Arc Completeness Sweep")

    print("Plotting visual fit overlay (60% arc, noise=0.05)...")
    plot_example_fits(noise_std=0.1, arc_fraction=0.3)

    print_summary_table()
    plt.show()
