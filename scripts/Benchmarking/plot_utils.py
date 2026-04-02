import os
import numpy as np
import math
from scripts.Benchmarking.results import SweepResult
from scripts.Benchmarking.storing_utils import load_result
import matplotlib.pyplot as plt
import copy
from src.bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D


def plot_result(result: SweepResult, save_dir: str = "plots"):

    # this only makes sense if there is a fixed shape

    shape = result.ground_truth_config.shape
    shape_gt = result.shape_gt[0]
    fitting_params = result.fitting_params
    labels = result.fitter_labels
    n_fitters = len(labels)

    cols = math.ceil(n_fitters / 2)
    rows = 2
    plotter = ShapePlotter2D()

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4), sharex=True, sharey=True)
    fig.set_size_inches(19.2, 9.8)
    fig.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.1)
    #fig.suptitle(result.case_name, fontsize=16)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        plt.sca(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=20)
        ax.grid(True, alpha=1)

        ax.set_xlabel("")
        ax.set_ylabel("")

        margin_factor_x = 1.5
        margin_factor_y = 2
        x_range = shape_gt[2] * margin_factor_x
        y_range = shape_gt[3] * margin_factor_y
        ax.set_xlim(shape_gt[0] - x_range, shape_gt[0] + x_range)
        ax.set_ylim(shape_gt[1] - y_range, shape_gt[1] + y_range)

        valid = ~np.isnan(fitting_params[:, i, :]).any(axis=1)
        for params_fit in fitting_params[valid, i, :]:
            plotter.plot_shape(shape, params_fit, label="Fitted", color="red", ax=ax)

        plotter.plot_shape(shape, shape_gt, label="Ground Truth", color="black", ax=ax)
        ax.plot(shape_gt[0], shape_gt[1], "ro", color="black", label="True Center")
        ax.plot(
            np.mean(fitting_params[valid, i, 0]),
            np.mean(fitting_params[valid, i, 1]),
            "*", color="blue", label="Averaged Center",
        )

    # in the last spot we add a plot of an example setup
    ax = axes[-1]
    plt.sca(ax)
    #plt.gca().set_aspect("equal", adjustable="box")
    plotter.plot_shape(shape, shape_gt, label="Ground Truth", color="black", ax=axes[-1])
    ax.plot(shape_gt[0], shape_gt[1], "ro", color="black", label="True Center")


    fixed_sigma = result.noise_gt[0][0]
    new_noise = copy.copy(result.ground_truth_config.noise_function)
    new_noise.fixed_params = fixed_sigma

    source_params = result.source_gt[0]
    new_source = copy.copy(result.ground_truth_config.source_density)
    new_source.fixed_params = source_params

    # compute observations
    parameter_samples = new_source.sample(n=100)
    covariances = new_noise.evaluate(parameter_samples, params = fixed_sigma)
    source_points = shape.parametrization(parameter_samples, shape_gt)
    errors = np.array([np.random.multivariate_normal(np.zeros(shape.ambient_dim), cov) for cov in covariances])
    observations = source_points + errors

    ax.scatter(
        observations[:, 0],
        observations[:, 1], s=8, alpha=0.8)

    ax.set_xlabel("")
    ax.set_ylabel("")

    plotter.plot_covariances(shape, shape_gt, new_source, new_noise)
    ax.set_title("Setup", fontsize=20)

    for ax in axes[n_fitters + 1:]:
        ax.set_visible(False)

    # adjust difference between height
    fig.subplots_adjust(hspace=0.01)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{result.case_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {save_path}")

    print("Now plot the table: ")
    print(result.generate_latex_table_Ec_Ea_Earea())


def plot_fit(result: SweepResult, idx=2, save_dir: str = "plots"):
    """
    Plot a single fit (iteration `idx`) showing ALL fitters in the same plot.

    Shows:
        - Ground truth ellipse
        - All fitted ellipses (one per method)
        - Noisy observations
        - Covariance ellipses
        - Centers (GT + mean fitted per method)
    """

    shape = result.ground_truth_config.shape
    shape_gt = result.shape_gt[idx]
    source_params = result.source_gt[idx]
    noise_params = result.noise_gt[idx]

    fitting_params = result.fitting_params
    labels = result.fitter_labels
    n_fitters = len(labels)

    plotter = ShapePlotter2D()

    # ------------------------------------------------------------
    #  Create figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"All Fitters — Iteration {idx}", fontsize=22)
    ax.grid(True, alpha=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # ------------------------------------------------------------
    # 1. Plot ground truth ellipse
    # ------------------------------------------------------------
    plotter.plot_shape(shape, shape_gt, color="black", ax=ax, linewidth=2, label="Ground Truth")
    ax.plot(shape_gt[0], shape_gt[1], "o", color="black")

    # ------------------------------------------------------------
    # 2. Plot each fitted ellipse
    # ------------------------------------------------------------
    color_cycle = plt.cm.tab10(np.linspace(0, 1, n_fitters))

    for i, label in enumerate(labels):
        params_fit = fitting_params[idx, i]

        if np.isnan(params_fit).any():
            continue  # skip failed fit

        plotter.plot_shape(shape, params_fit,
                           color=color_cycle[i],
                           ax=ax,
                           linewidth=2.5,
                           label=label)

    # ------------------------------------------------------------
    # 3. Generate noisy observations (same as in plot_result)
    # ------------------------------------------------------------
    # source
    new_source = copy.copy(result.ground_truth_config.source_density)
    new_source.fixed_params = source_params

    # noise
    new_noise = copy.copy(result.ground_truth_config.noise_function)
    new_noise.fixed_sigma = noise_params[0]

    # sample measurement
    parameter_samples = new_source.sample(n=100)
    covariances = new_noise.evaluate(parameter_samples, params=np.array([noise_params[0]]))
    source_points = shape.parametrization(parameter_samples, shape_gt)

    errors = np.array([
        np.random.multivariate_normal(np.zeros(shape.ambient_dim), cov)
        for cov in covariances
    ])
    observations = source_points + errors

    # plot points
    ax.scatter(observations[:, 0], observations[:, 1], s=10, alpha=0.3, color="gray")

    # ------------------------------------------------------------
    # 4. Plot covariance ellipses
    # ------------------------------------------------------------
    # plotter.plot_covariances(shape, shape_gt, new_source, new_noise)

    # ------------------------------------------------------------
    # 5. Auto scale
    # ------------------------------------------------------------
    margin_factor_x = 1.2
    margin_factor_y = 3
    x_range = shape_gt[2] * margin_factor_x
    y_range = shape_gt[3] * margin_factor_y
    ax.set_xlim(shape_gt[0] - x_range, shape_gt[0] + x_range)
    ax.set_ylim(shape_gt[1] - y_range, shape_gt[1] + y_range)

    ax.legend(loc="upper right", fontsize=10)

    # ------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{result.case_name}_all_fit_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined fit plot: {save_path}")


def plot_summary(cache_dir: str = "results_cache", plot_dir: str = "plots", shape_name = "ellipse_fixed"):
    """ Plots the summary statistics of a sweep"""

    stems = [
        os.path.join(cache_dir, f[:-5])          # strip ".json"
        for f in os.listdir(cache_dir)
        if f.endswith(".json")
        if shape_name in f
    ]
    results = [load_result(s) for s in stems]

    results.sort(key=lambda r: r.noise_gt[0])

    noise_levels = [r.noise_gt[0] for r in results]
    noise_levels = np.array(noise_levels, dtype=float).flatten()

    fitter_labels = results[0].fitter_labels
    n_fitters = len(fitter_labels)
    n_params = 4  # Center, Major, Minor, Rotation

    # Allocate array: [noise_levels, fitters, params]
    all_mse = np.zeros((len(results), n_fitters, n_params))
    all_bias = np.zeros_like(all_mse)
    all_std = np.zeros((len(results), n_fitters, 4))

    # Fill array
    for ni, r in enumerate(results):
        for fi, label in enumerate(fitter_labels):
            mse = r.get_all_mse(label)  # shape (4,)
            all_mse[ni, fi] = mse
            all_bias[ni, fi] = r.get_all_bias(label)
            all_std[ni, fi] = r.get_all_std(label)

    # Plotting
    param_names = ['Center Deviation', 'Major Axis', 'Minor Axis',  'Rotation']
    colors = plt.cm.tab10(np.linspace(0, 1, n_fitters))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))  # bigger figure

    # ---------- Row 1: MSE (log scale, ±1 std shading) ----------
    for p in range(4):
        ax = axes[0, p]

        for fi, label in enumerate(fitter_labels):
            mse_vals = all_mse[:, fi, p]
            std_vals = all_std[:, fi, p]

            # mean line
            ax.plot(
                noise_levels,
                mse_vals,
                marker='o',
                color=colors[fi],
                label=label,
                linewidth=2
            )


        ax.set_yscale("log")
        ax.set_title(f"MSE – {param_names[p]}", fontsize=12)
        ax.set_xlabel("Noise σ")
        ax.grid(True, linestyle='--', alpha=0.5)

    # ---------- Row 2: Bias (normal linear scale) ----------
    for p in range(4):
        ax = axes[1, p]

        for fi, label in enumerate(fitter_labels):
            bias_vals = all_bias[:, fi, p]
            std_vals = all_std[:, fi, p]

            # signed bias curve
            ax.plot(
                noise_levels,
                bias_vals,
                marker='s',
                color=colors[fi],
                linewidth=2,
                alpha=0.9
            )

        ax.set_title(f"Bias – {param_names[p]}", fontsize=12)
        ax.set_xlabel("Noise σ")
        ax.grid(True, linestyle='--', alpha=0.5)

    # Legend at the top right
    axes[0, -1].legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=10
    )

    #fig.suptitle("Ellipse Fitting: MSE (log, top) and Bias (bottom)", fontsize=16)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "metrics_vs_noise.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Saved MSE metric plot to {save_path}")

