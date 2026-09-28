# plot_results.py
from bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from scripts.Benchmarking.Experiments.cylinder_experiment import CylinderExperiment
from scripts.Benchmarking.Experiments.random_ellipse import RandomExperiment, make_random_experiment
from scripts.Benchmarking.Experiments.random_ellipse_noise import make_random_experiment_noise
from scripts.Benchmarking.Experiments.unisotropic_noise import LaserExperiment, make_laser_experiment
from scripts.Benchmarking.results import SweepResult
from scripts.Benchmarking.Experiments.experiment_config import ExperimentConfig
from scripts.Benchmarking.Experiments.isotropic_noise import IsotropicExperiment
from scripts.Benchmarking.Experiments.isotropic_illustrative import IsotropicIllustrative
import numpy as np
import copy
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    save_dir = "plots"

    nums = [0,1,2]
    ws = [0.1,0.2,0.3]
    indices = [7,13,18]

    fig, axes = plt.subplots(
        1,
        len(indices),
        figsize=(3 * len(indices), 3),
        squeeze=False
    )
    axes = axes.ravel()

    plotter = ShapePlotter2D()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for w, fit_idx, num in zip(ws, indices, nums):
        config_dict = make_random_experiment(w = w)
        stem = "results/Random_Experiment_" + str(int(w*200))

        # rebuild config
        config = ExperimentConfig.get_from_dict(config_dict)

        # load sweep result
        result = SweepResult.load(stem, experiment_config=config)

        shape = config.shape

        legend_handles = None
        legend_labels = None

        max_radius_x = max(shape_gt[2] for shape_gt in result.shape_gt[indices])
        max_radius_y = max(shape_gt[3] for shape_gt in result.shape_gt[indices])


        shape_gt = result.shape_gt[fit_idx]
        source_params = result.source_gt[fit_idx]
        noise_params = result.noise_gt[fit_idx]

        fitting_params = result.fitting_params
        labels = result.experiment_config.labels

        ax = axes[num]

        ax.grid(True, alpha=1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        margin_factor_x = 1.7
        margin_factor_y = 1.9

        ax.set_xlim(shape_gt[0] - margin_factor_x * max_radius_x,
                    shape_gt[0] + margin_factor_x * max_radius_x)

        ax.set_ylim(shape_gt[1] - margin_factor_y * max_radius_y,
                    shape_gt[1] + margin_factor_y * max_radius_y)
        ax.set_box_aspect(1)

        plotter.plot_shape(
            shape,
            shape_gt,
            color="black",
            ax=ax,
            linewidth=2,
            linestyle="dashed",
            label="Ground Truth"
        )

        for i, label in enumerate(labels):
            params_fit = fitting_params[fit_idx, i]

            if np.isnan(params_fit).any():
                continue

            plotter.plot_shape(
                shape,
                params_fit,
                color=color_cycle[i],
                ax=ax,
                linewidth=2.5,
                label=label
            )

        # ------------------------------------------------------------
        # 3. Generate noisy observations (same as in plot_result)
        # ------------------------------------------------------------
        # source

        observations = np.load(f"Pointclouds/pointcloud_Random_Experiment_{int(w*200)}_{fit_idx}.npy")

        # plot points
        ax.scatter(observations[:, 0], observations[:, 1], s=10, alpha=0.3, color="gray")

        ax.set_xlabel(f"Coverage: {int(w*200)}%", fontsize = 10)

        # Save legend entries once
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # ------------------------------------------------------------------
    # Combined legend
    # ------------------------------------------------------------------
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=10,
    )

    # Leave room for legend
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"arc_qualitative.eps"
    )

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)