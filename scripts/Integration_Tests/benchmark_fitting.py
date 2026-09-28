# benchmark_fitting.py

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.inference.samplers.map_estimation_global import CMAESOptimizer

from bayes_shape_fitting.utils.samplers import GridSampler, TopKSampler, ClosestPointSampler
from bayes_shape_fitting.utils.integration import (
    SimpsonIntegrator, TrapezoidIntegrator, MonteCarloIntegrator, OriginalIntegrator
)
from bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class FittingResult:
    def __init__(
        self,
        sampler_name:    str,
        integrator_name: str,
        num_samples:     int,
        theta_map:       np.ndarray,
        theta_true:      np.ndarray,
        log_lik_final:   float,
        total_time_s:    float,
        n_iter:          int,
        converged:       bool,
    ):
        self.sampler_name    = sampler_name
        self.integrator_name = integrator_name
        self.num_samples     = num_samples
        self.theta_map       = theta_map
        self.theta_true      = theta_true
        self.log_lik_final   = log_lik_final
        self.total_time_s    = total_time_s
        self.n_iter          = n_iter
        self.converged       = converged

    @property
    def theta_error(self) -> float:
        return float(np.linalg.norm(self.theta_map - self.theta_true))

    @property
    def label(self) -> str:
        return f"{self.sampler_name}\n+{self.integrator_name}\nn={self.num_samples}"

    @property
    def short_label(self) -> str:
        return (f"{self.sampler_name.replace('Sampler','')}+"
                f"{self.integrator_name.replace('Integrator','')}"
                f" n={self.num_samples}")

    def __repr__(self):
        return (
            f"{self.sampler_name:25s} + {self.integrator_name:25s} | "
            f"n={self.num_samples:4d} | "
            f"time={self.total_time_s:7.2f}s | "
            f"||Δθ||={self.theta_error:.4f} | "
            f"log_lik={self.log_lik_final:.4f} | "
            f"converged={self.converged}"
        )


# ---------------------------------------------------------------------------
# Single fitting run
# ---------------------------------------------------------------------------

def _run_single(
    shape,
    measurement,
    source_density,
    noise_function,
    init_theta,
    theta_true,
    sampler,
    integrator,
    num_samples,
    optimizer_method,
) -> FittingResult:

    likelihood = MarginalLikelihood(
        shape          = shape,
        source_density = source_density,
        noise_function = noise_function,
        num_samples    = num_samples,
        sampler        = sampler,
        integrator     = integrator,
    )
    posterior = MarginalPosterior(
        shape       = shape,
        measurement = measurement,
        likelihood  = likelihood,
    )
    optimizer = MAPOptimizer(method=optimizer_method, verbose=False)
    #optimizer = CMAESOptimizer()

    t0     = time.perf_counter()
    result = optimizer.optimize(posterior, init_theta)
    elapsed = time.perf_counter() - t0

    # log-likelihood at the MAP estimate
    log_lik_final = float(
        likelihood.log_likelihood(measurement, result.theta_map).sum()
    )

    return FittingResult(
        sampler_name    = type(sampler).__name__,
        integrator_name = type(integrator).__name__,
        num_samples     = num_samples,
        theta_map       = result.theta_map,
        theta_true      = theta_true,
        log_lik_final   = log_lik_final,
        total_time_s    = elapsed,
        n_iter          = getattr(result, 'n_iter', -1),
        converged       = getattr(result, 'success', True),
    )


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def benchmark_fitting(
    shape,
    measurement:     np.ndarray,
    source_density,
    noise_function,
    init_theta:      np.ndarray,
    theta_true:      np.ndarray,
    samplers:        list  = None,
    integrators:     list  = None,
    num_samples_list: list = None,
    optimizer_method: str  = "L-BFGS-B",
) -> list[FittingResult]:

    if samplers is None:
        samplers = [GridSampler(), TopKSampler(k=200), ClosestPointSampler(n_samples=50)]
    if integrators is None:
        integrators = [SimpsonIntegrator(), TrapezoidIntegrator(), MonteCarloIntegrator()]
    if num_samples_list is None:
        num_samples_list = [25, 50, 100]

    results = []

    combos = list(itertools.product(samplers, integrators, num_samples_list))
    print(f"Running {len(combos)} fitting experiments...\n")
    print("=" * 90)

    for sampler, integrator, num_samples in combos:
        print(f"  {type(sampler).__name__:25s} + {type(integrator).__name__:25s} | n={num_samples} ...",
              end=" ", flush=True)
        try:
            res = _run_single(
                shape, measurement, source_density, noise_function,
                init_theta, theta_true,
                sampler, integrator, num_samples, optimizer_method,
            )
            results.append(res)
            print(f"done  ||Δθ||={res.theta_error:.4f}  t={res.total_time_s:.2f}s  "
                  f"converged={res.converged}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("=" * 90)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fitting_benchmark(
    results:    list[FittingResult],
    theta_true: np.ndarray,
):
    # ── colour by sampler, marker by integrator ────────────────────────────
    samplers    = list(dict.fromkeys(r.sampler_name    for r in results))
    integrators = list(dict.fromkeys(r.integrator_name for r in results))
    num_samples = sorted(set(r.num_samples for r in results))

    s_colors  = plt.cm.Set2(np.linspace(0, 0.8, len(samplers)))
    i_markers = ['o', 's', '^', 'D', 'x']
    i_lines   = ['-', '--', ':', '-.']

    color_map  = {s: c for s, c in zip(samplers,    s_colors)}
    marker_map = {i: m for i, m in zip(integrators, i_markers)}
    line_map   = {i: l for i, l in zip(integrators, i_lines)}

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Fitting Benchmark — Sampler × Integrator × num_samples",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── A. theta error vs num_samples ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for sampler, integrator in itertools.product(samplers, integrators):
        sub = [r for r in results
               if r.sampler_name == sampler and r.integrator_name == integrator]
        if not sub:
            continue
        sub.sort(key=lambda r: r.num_samples)
        ax_a.plot([r.num_samples for r in sub],
                  [r.theta_error  for r in sub],
                  color=color_map[sampler],
                  marker=marker_map[integrator],
                  linestyle=line_map[integrator],
                  linewidth=1.5, markersize=6,
                  label=f"{sampler.replace('Sampler','')}+{integrator.replace('Integrator','')}")
    ax_a.set_title("||θ_map − θ_true|| vs num_samples")
    ax_a.set_xlabel("num_samples")
    ax_a.set_ylabel("parameter error")
    ax_a.set_xscale("log")
    ax_a.legend(fontsize=6, ncol=2)

    # ── B. total time vs num_samples ───────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    for sampler, integrator in itertools.product(samplers, integrators):
        sub = [r for r in results
               if r.sampler_name == sampler and r.integrator_name == integrator]
        if not sub:
            continue
        sub.sort(key=lambda r: r.num_samples)
        ax_b.plot([r.num_samples    for r in sub],
                  [r.total_time_s   for r in sub],
                  color=color_map[sampler],
                  marker=marker_map[integrator],
                  linestyle=line_map[integrator],
                  linewidth=1.5, markersize=6)
    ax_b.set_title("Optimization time vs num_samples")
    ax_b.set_xlabel("num_samples")
    ax_b.set_ylabel("seconds")
    ax_b.set_xscale("log")

    # ── C. accuracy vs speed scatter (one point per run) ──────────────────
    ax_c = fig.add_subplot(gs[1, :])
    for r in results:
        ax_c.scatter(r.total_time_s, r.theta_error,
                     color=color_map[r.sampler_name],
                     marker=marker_map[r.integrator_name],
                     s=60 + r.num_samples,    # size encodes num_samples
                     alpha=0.8, edgecolors="white", linewidths=0.5)
        ax_c.annotate(f"n={r.num_samples}",
                      (r.total_time_s, r.theta_error),
                      fontsize=5, alpha=0.7,
                      xytext=(3, 3), textcoords="offset points")
    ax_c.set_title("Accuracy vs Speed  (marker=integrator, colour=sampler, size=num_samples)")
    ax_c.set_xlabel("total optimization time (s)")
    ax_c.set_ylabel("||θ_map − θ_true||")

    # ── D. per-parameter error heatmap ────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, 0])
    n_params = len(theta_true)
    # one row per result, one col per parameter
    error_matrix = np.array([np.abs(r.theta_map - r.theta_true) for r in results])
    im = ax_d.imshow(error_matrix.T, aspect="auto", cmap="YlOrRd")
    ax_d.set_yticks(np.arange(n_params))
    ax_d.set_yticklabels([f"θ_{i}" for i in range(n_params)], fontsize=7)
    ax_d.set_xticks(np.arange(len(results)))
    ax_d.set_xticklabels([r.short_label for r in results],
                          fontsize=5, rotation=45, ha="right")
    ax_d.set_title("|θ_map − θ_true| per parameter")
    plt.colorbar(im, ax=ax_d, shrink=0.8)

    # ── E. log-likelihood at convergence ──────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 1])
    for r in results:
        ax_e.scatter(r.short_label, r.log_lik_final,
                     color=color_map[r.sampler_name],
                     marker=marker_map[r.integrator_name],
                     s=70, alpha=0.85, edgecolors="white")
    ax_e.set_title("Log-likelihood at convergence")
    ax_e.set_ylabel("Σ log p(y | θ_map)")
    ax_e.tick_params(axis='x', labelsize=5, rotation=45)

    # ── global legend ──────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = (
        [Patch(facecolor=color_map[s], label=s.replace("Sampler", ""))
         for s in samplers] +
        [Line2D([0], [0], color="gray", marker=marker_map[i],
                linestyle=line_map[i], label=i.replace("Integrator", ""))
         for i in integrators]
    )
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8,
               title="Sampler / Integrator", title_fontsize=9,
               bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()
    return fig

def plot_fits(
    shape,
    measurement:  np.ndarray,
    init_theta:   np.ndarray,
    theta_true:   np.ndarray,
    results:      list[FittingResult],
):
    """
    One 3D subplot per fitting result, each showing ground truth,
    initial guess, and the fitted shape side by side.
    """
    n     = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig   = plt.figure(figsize=(7 * ncols, 6 * nrows))
    fig.suptitle("Fitted shapes per configuration", fontsize=14, fontweight="bold")

    plotter = ShapePlotter3D()

    for idx, r in enumerate(results):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")

        plotter.plot_shape(shape, theta_true,  ax=ax, label="Ground truth", color="blue")
        plotter.plot_shape(shape, init_theta,  ax=ax, label="Initial",      color="orange")
        plotter.plot_shape(shape, r.theta_map, ax=ax, label="Fitted",       color="green")
        plotter.plot_observations(observations=measurement, ax=ax)

        ax.set_title(
            f"{r.sampler_name.replace('Sampler', '')}"
            f"+{r.integrator_name.replace('Integrator', '')}"
            f"  n={r.num_samples}\n"
            f"||Δθ||={r.theta_error:.4f}  t={r.total_time_s:.1f}s",
            fontsize=8,
        )
        ax.legend(fontsize=6)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fitting_benchmark(
    shape,
    measurement:      np.ndarray,
    source_density,
    noise_function,
    init_theta:       np.ndarray,
    theta_true:       np.ndarray,
    samplers:         list  = None,
    integrators:      list  = None,
    num_samples_list: list  = None,
    optimizer_method: str   = "L-BFGS-B",
    save_fig:         str   = None,
) -> list[FittingResult]:

    results = benchmark_fitting(
        shape, measurement, source_density, noise_function,
        init_theta, theta_true,
        samplers, integrators, num_samples_list, optimizer_method,
    )

    fig_metrics = plot_fitting_benchmark(results, theta_true)
    fig_shapes  = plot_fits(shape, measurement, init_theta, theta_true, results)

    if save_fig:
        base = save_fig.replace(".png", "")
        fig_metrics.savefig(f"{base}_metrics.png", dpi=150, bbox_inches="tight")
        fig_shapes.savefig(f"{base}_shapes.png",  dpi=150, bbox_inches="tight")
        print(f"Figures saved → {base}_metrics.png, {base}_shapes.png")

    plt.show()
    return results

