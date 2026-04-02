# runner.py
import argparse
import functools
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy

from scripts.Benchmarking.Fitters.Ellipse_fitters.base import BaseEllipseFitter
from scripts.Benchmarking.ground_truth_config import GroundTruthConfig
from scripts.Benchmarking.plot_utils import plot_summary, plot_result, plot_fit
from scripts.Benchmarking.storing_utils import save_result, load_result
from src.bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from results import SweepResult
from suite import BenchmarkCase, build_suite

from src.bayes_shape_fitting.models.shapes.ellipse import Ellipse

np.random.seed(42)

# ── sweep & plot (unchanged logic) ───────────────────────────────────────────

def _run_sweep(case: BenchmarkCase) -> SweepResult:
    n_fitters = len(case.fitters)  + 3 # number of Static fitters + Bayesian Fitters
    fitting_params = np.zeros((case.num_it, n_fitters, 5))

    m = 1 if isinstance(case.ground_truth_config.noise_params,float) else len(case.ground_truth_config.noise_params)

    # store per-iteration ground truth
    shape_gt_all = np.zeros((case.num_it, 5))
    source_gt_all = np.zeros((case.num_it, len(case.ground_truth_config.source_params)))
    noise_gt_all = np.zeros((case.num_it, m))

    # store the fitting times as well
    fit_times = np.zeros((case.num_it, n_fitters))

    for it in tqdm(range(case.num_it), desc=case.name):

        case.ground_truth_config.resample()  # no-op if nothing is random
        shape_gt_all[it] = case.ground_truth_config.shape_params
        source_gt_all[it] = case.ground_truth_config.source_params
        noise_gt_all[it] = case.ground_truth_config.noise_params

        # sample points from the shape
        points = case.ground_truth_config.get_measurement(case.num_points)

        # build the per-iteration fitter list
        fitters = case.fitters + case.build_bayes_fitters()

        for i, fitter in enumerate(fitters):
            try:
                start = time.perf_counter()

                params_fit = fitter.fit(
                    points,
                    uncertainties=case.ground_truth_config.noise_function.evaluate(
                        np.zeros(case.num_points), params=case.ground_truth_config.noise_params
                    ),
                    shape_init=None,
                )

                end = time.perf_counter()
                fit_times[it, i] = end - start

                fitting_params[it, i, :] = params_fit
            except Exception as e:
                fitting_params[it, i, :] = np.nan
                fit_times[it, i] = np.nan
                print(f"  {fitter.__class__.__name__} failed: {e}")

    return SweepResult(
        case_name=case.name,
        noise_name=case.noise_name,
        source_name=case.source_name,
        ground_truth_config=case.ground_truth_config,
        fitter_labels=case.labels,
        fitting_params=fitting_params,
        fitting_times=fit_times,
        shape_gt=shape_gt_all,
        source_gt=source_gt_all,
        noise_gt=noise_gt_all
    )


# ── entry points ─────────────────────────────────────────────────────────────

def run(cache_dir: str = "results_cache"):
    suite = build_suite()
    for case in suite:
        print(f"\n--- Running: {case.name} ---")
        result = _run_sweep(case)
        save_result(result, save_dir=cache_dir)


def plot(cache_dir: str = "results_cache", plot_dir: str = "plots"):
    stems = [
        os.path.join(cache_dir, f[:-5])          # strip ".json"
        for f in os.listdir(cache_dir)
        if f.endswith(".json")
    ]
    results = [load_result(s) for s in stems]

    print("\n--- Plotting & Summary ---")
    for r in results:
        plot_result(r, save_dir=plot_dir)
        plot_fit(r, save_dir=plot_dir)
        for label in r.fitter_labels:
            print(
                f"{r.case_name} | {label}: "
                f"MSE={r.get_mse(label):.4f}, "
                f"failures={r.get_failure_rate(label):.0%}"
                f"Time={r.get_fitting_time(label):.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["run", "plot", "run-and-plot"],
        help="run: execute sweeps and cache; plot: load cache and plot; run-and-plot: both",
    )
    parser.add_argument("--cache-dir", default="results_cache")
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args()

    if args.mode in ("run", "run-and-plot"):
        run(cache_dir=args.cache_dir)
    if args.mode in ("plot", "run-and-plot"):
        #plot_summary(cache_dir=args.cache_dir, plot_dir=args.plot_dir)
        plot(cache_dir=args.cache_dir, plot_dir=args.plot_dir)
