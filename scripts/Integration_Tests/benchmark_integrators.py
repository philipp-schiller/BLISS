# benchmark.py

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bayes_shape_fitting.utils.samplers import GridSampler, TopKSampler, ClosestPointWindowSampler, Sampler
from bayes_shape_fitting.utils.integration import SimpsonIntegrator, TrapezoidIntegrator, MonteCarloIntegrator, OriginalIntegrator, Integrator
from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood


# ---------------------------------------------------------------------------
# Timed wrappers
# ---------------------------------------------------------------------------

class TimedSampler(Sampler):

    def __init__(self, sampler: Sampler):
        self.sampler    = sampler
        self.call_times = []

    @property
    def name(self) -> str:
        return type(self.sampler).__name__

    def sample(self, *args, **kwargs):
        t0     = time.perf_counter()
        result = self.sampler.sample(*args, **kwargs)
        self.call_times.append(time.perf_counter() - t0)
        return result

    def reset(self):
        self.call_times.clear()

    @property
    def per_measurement(self):
        return self.sampler.per_measurement

    @property
    def trivial_integral(self):
        return self.sampler.trivial_integral


class TimedIntegrator(Integrator):

    def __init__(self, integrator: Integrator):
        self.integrator  = integrator
        self.call_times  = []

    @property
    def name(self) -> str:
        return type(self.integrator).__name__

    def build_integrand(self, *args, **kwargs):
        return self.integrator.build_integrand(*args, **kwargs)

    def integrate(self, *args, **kwargs):
        t0     = time.perf_counter()
        result = self.integrator.integrate(*args, **kwargs)
        self.call_times.append(time.perf_counter() - t0)
        return result

    def reset(self):
        self.call_times.clear()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class BenchmarkResult:
    def __init__(self, sampler_name, integrator_name, log_lik, sampler_time, integrator_time):
        self.sampler_name     = sampler_name
        self.integrator_name  = integrator_name
        self.log_lik          = log_lik            # (M,)
        self.sampler_time     = sampler_time        # mean seconds
        self.integrator_time  = integrator_time     # mean seconds

    @property
    def total_time(self):
        return self.sampler_time + self.integrator_time

    @property
    def label(self):
        return f"{self.sampler_name}\n+{self.integrator_name}"

    def __repr__(self):
        return (
            f"{self.sampler_name:25s} + {self.integrator_name:25s} | "
            f"sampler={self.sampler_time*1e3:7.2f}ms  "
            f"integrator={self.integrator_time*1e3:7.2f}ms  "
            f"total={self.total_time*1e3:7.2f}ms  "
            f"log_lik[:6]={self.log_lik[:6]}"
        )

# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark(
    ml_factory,             # Callable[[TimedSampler, TimedIntegrator], MarginalLikelihood]
    measurement:   np.ndarray,
    shape_params:  np.ndarray,
    source_params: np.ndarray = None,
    noise_params:  np.ndarray = None,
    samplers:      list       = None,
    integrators:   list       = None,
    n_repeats:     int        = 5,
) -> list[BenchmarkResult]:

    if samplers is None:
        samplers = [GridSampler(), TopKSampler(k=200), ClosestPointWindowSampler(n_samples=50)]
    if integrators is None:
        integrators = [SimpsonIntegrator(), TrapezoidIntegrator(), MonteCarloIntegrator(), OriginalIntegrator()]

    results = []

    for base_sampler, base_integrator in itertools.product(samplers, integrators):
        timed_s = TimedSampler(base_sampler)
        timed_i = TimedIntegrator(base_integrator)
        ml      = ml_factory(timed_s, timed_i)

        # Warm-up
        ml.log_likelihood(measurement, shape_params, source_params, noise_params)
        timed_s.reset()
        timed_i.reset()

        # Timed runs
        log_liks = []
        for _ in range(n_repeats):
            ll = ml.log_likelihood(measurement, shape_params, source_params, noise_params)
            log_liks.append(ll)

        result = BenchmarkResult(
            sampler_name     = timed_s.name,
            integrator_name  = timed_i.name,
            log_lik          = np.mean(log_liks, axis=0),
            sampler_time     = np.mean(timed_s.call_times),
            integrator_time  = np.mean(timed_i.call_times),
        )
        results.append(result)
        print(result)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmark(results: list[BenchmarkResult], reference_name: str = "SimpsonIntegrator"):
    n       = len(results)
    x       = np.arange(n)

    # Colour by sampler, hatch by integrator
    samplers    = list(dict.fromkeys(r.sampler_name    for r in results))
    integrators = list(dict.fromkeys(r.integrator_name for r in results))
    s_colors    = plt.cm.Set2(np.linspace(0, 0.8, len(samplers)))
    i_hatches   = ['', '///', '...', 'xxx', '---']

    color_map = {s: c for s, c in zip(samplers,    s_colors)}
    hatch_map = {i: h for i, h in zip(integrators, i_hatches)}

    # ── layout: 3 rows, 2 cols ─────────────────────────────────────────
    # row 0: [A. total time]        [B. time split]
    # row 1: [C. error vs ref] (full width)
    # row 2: [D. log-lik]     (full width)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Sampler × Integrator Benchmark", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    bar_kw = dict(edgecolor="white", width=0.6)
    labels = [r.label for r in results]

    # ── A. Total time ──────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(results):
        ax_a.bar(i, r.total_time * 1e3,
                 color=color_map[r.sampler_name],
                 hatch=hatch_map[r.integrator_name], **bar_kw)
    ax_a.set_title("Total time per call")
    ax_a.set_ylabel("ms")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=6, rotation=35, ha="right")

    # ── B. Sampler vs integrator time split (stacked bar) ─────────────
    ax_b = fig.add_subplot(gs[0, 1])
    s_times = [r.sampler_time    * 1e3 for r in results]
    i_times = [r.integrator_time * 1e3 for r in results]
    ax_b.bar(x, s_times, label="Sampler",    color="#5C85D6", **bar_kw)
    ax_b.bar(x, i_times, label="Integrator", color="#E07B54",
             bottom=s_times, **bar_kw)
    ax_b.set_title("Time split: sampler vs integrator")
    ax_b.set_ylabel("ms")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=6, rotation=35, ha="right")
    ax_b.legend(fontsize=9)

    # ── C. Max |Δ log-lik| vs reference (full width) ──────────────────
    ax_c = fig.add_subplot(gs[1, :])
    ref = next((r for r in results
                if reference_name in r.integrator_name
                and "Grid" in r.sampler_name), None)

    if ref is not None:
        devs, dev_labels, dev_colors = [], [], []
        for r in results:
            if r is ref:
                continue
            devs.append(np.max(np.abs(r.log_lik - ref.log_lik)))
            dev_labels.append(r.label.replace("\n", " "))
            dev_colors.append(color_map[r.sampler_name])
        ax_c.bar(np.arange(len(devs)), devs, color=dev_colors, edgecolor="white", width=0.6)
        ax_c.set_title(f"Max |Δ log-lik| vs reference: {ref.sampler_name} + {ref.integrator_name}")
        ax_c.set_ylabel("absolute error")
        ax_c.set_xticks(np.arange(len(devs)))
        ax_c.set_xticklabels(dev_labels, fontsize=7, rotation=35, ha="right")
    else:
        ax_c.set_visible(False)

    # ── D. Log-likelihood per measurement (full width) ────────────────
    ax_d = fig.add_subplot(gs[2, :])
    M = len(results[0].log_lik)
    for r in results:
        ax_d.plot(np.arange(M), r.log_lik,
                  label=r.label.replace("\n", " "),
                  color=color_map[r.sampler_name],
                  linestyle="--" if hatch_map[r.integrator_name] else "-",
                  linewidth=1.5, alpha=0.8)
    ax_d.set_title("Log-likelihood per measurement point")
    ax_d.set_xlabel("Measurement index")
    ax_d.set_ylabel("log p(y)")
    ax_d.legend(fontsize=7, ncol=3, loc="lower right")

    # ── legend: sampler colours + integrator hatches ──────────────────
    from matplotlib.patches import Patch, Rectangle
    legend_elements  = [Patch(facecolor=color_map[s], label=s) for s in samplers]
    legend_elements += [Rectangle((0, 0), 1, 1, fill=False,
                                   hatch=hatch_map[i], edgecolor="black",
                                   label=i) for i in integrators]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8,
               title="Sampler / Integrator", title_fontsize=9,
               bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    ml_factory,
    measurement:   np.ndarray,
    shape_params:  np.ndarray,
    source_params: np.ndarray = None,
    noise_params:  np.ndarray = None,
    samplers:      list       = None,
    integrators:   list       = None,
    n_repeats:     int        = 5,
    save_fig:      str        = None,
):
    print("=" * 80)
    print("BENCHMARK  —  single log_likelihood evaluation")
    print("=" * 80)

    results = benchmark(
        ml_factory, measurement, shape_params,
        source_params, noise_params,
        samplers, integrators, n_repeats,
    )

    fig = plot_benchmark(results)

    if save_fig:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {save_fig}")

    plt.show()
    return results