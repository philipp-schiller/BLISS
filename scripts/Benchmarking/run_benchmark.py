"""
The main file for running benchmarks
"""

# runner.py
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scripts.Benchmarking.Fitters.Ellipse_fitters.base import BaseEllipseFitter
from scripts.Benchmarking.ground_truth_config import GroundTruthConfig
from src.bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from results import SweepResult
from suite import BenchmarkCase, build_suite

np.random.seed(42)

def _run_sweep(case: BenchmarkCase) -> SweepResult:
    shape = case.ground_truth_config.shape
    shape_gt = case.ground_truth_config.shape_params
    n_fitters = len(case.fitters)

    fitting_params = np.zeros((case.num_it, n_fitters, 5))

    for it in tqdm(range(case.num_it), desc=case.name):
        points = case.ground_truth_config.get_measurement(case.num_points)

        for i, fitter in enumerate(case.fitters):
            try:
                params_fit = fitter.fit(
                    points,
                    uncertainties = case.ground_truth_config.noise_function.evaluate(np.zeros(case.num_points)),
                    #source_density=case.ground_truth_config.source_density,
                    #noise_function=case.ground_truth_config.noise_function,
                    shape_init=None,
                )
                fitting_params[it, i, :] = params_fit
            except Exception as e:
                fitting_params[it, i, :] = np.nan
                print(f"  {fitter.__class__.__name__} failed: {e}")

    return SweepResult(
        case_name=case.name,
        ground_truth_config=case.ground_truth_config,
        fitter_labels=case.labels,
        fitting_params=fitting_params,
        shape_gt=shape_gt,
    )


def _plot_result(result: SweepResult, save_dir: str = "plots"):
    shape = result.ground_truth_config.shape
    shape_gt = result.shape_gt
    fitting_params = result.fitting_params
    labels = result.fitter_labels
    n_fitters = len(labels)

    cols = math.ceil(n_fitters / 2)
    rows = 2
    plotter = ShapePlotter2D()

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4), sharex=True, sharey=True)
    fig.set_size_inches(19.2, 10.8)
    fig.suptitle(result.case_name, fontsize=16)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        plt.sca(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=25)
        ax.grid(True, alpha=1)

        margin_factor_x = 1.1
        margin_factor_y = 2.5
        x_range = shape_gt[2] * margin_factor_x
        y_range = shape_gt[3] * margin_factor_y
        ax.set_xlim(shape_gt[0] - x_range, shape_gt[0] + x_range)
        ax.set_ylim(shape_gt[1] - y_range, shape_gt[1] + y_range)

        # plot all valid fitted ellipses
        valid = ~np.isnan(fitting_params[:, i, :]).any(axis=1)
        for params_fit in fitting_params[valid, i, :]:
            plotter.plot_shape(shape, params_fit, label="Fitted", color="red", ax=ax)

        # ground truth ellipse and center
        plotter.plot_shape(shape, shape_gt, label="Ground Truth", color="black", ax=ax)
        ax.plot(shape_gt[0], shape_gt[1], 'ro', color="black", label="True Center")

        # averaged center
        ax.plot(
            np.mean(fitting_params[valid, i, 0]),
            np.mean(fitting_params[valid, i, 1]),
            "*", color="blue", label="Averaged Center",
        )

    # hide any unused subplots
    for ax in axes[n_fitters:]:
        ax.set_visible(False)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{result.case_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {save_path}")


if __name__ == "__main__":
    suite = build_suite()
    results = []

    for case in suite:
        print(f"\n--- Running: {case.name} ---")
        result = _run_sweep(case)
        _plot_result(result, save_dir="plots")
        results.append(result)

    # summary
    print("\n--- Summary ---")
    for r in results:
        for label in r.fitter_labels:
            print(f"{r.case_name} | {label}: MSE={r.get_mse(label):.4f}, failures={r.get_failure_rate(label):.0%}")