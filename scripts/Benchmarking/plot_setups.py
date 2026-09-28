from bayes_shape_fitting.inference.noise_functions import LaserDependentNoise
from scripts.Benchmarking.Experiments.cylinder_experiment import CylinderExperiment
from scripts.Benchmarking.Experiments.random_ellipse import RandomExperiment, make_random_experiment
from scripts.Benchmarking.Experiments.random_ellipse_noise import make_random_experiment_noise
from scripts.Benchmarking.Experiments.unisotropic_noise import LaserExperiment, make_laser_experiment
from scripts.Benchmarking.results import SweepResult
from scripts.Benchmarking.Experiments.experiment_config import ExperimentConfig
from scripts.Benchmarking.Experiments.isotropic_noise import IsotropicExperiment
from scripts.Benchmarking.Experiments.isotropic_illustrative import IsotropicIllustrative
from bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import copy

np.random.seed(42)

save_dir = "plots"

results = []
config = ExperimentConfig.get_from_dict(IsotropicIllustrative)
stem = "results/Isotropic_illustrative"
result = SweepResult.load(stem, experiment_config=config)
results.append(result)

sigma_0 = 0.05
config_dict = make_laser_experiment(sigma_0=sigma_0, sigma_0_prior=sigma_0)
stem = "results/Laser_Experiment_" + str(sigma_0) + "prior_" + str(sigma_0)
config = ExperimentConfig.get_from_dict(config_dict)
result = SweepResult.load(stem, experiment_config=config)
results.append(result)

# we plot the setups

fig, axes = plt.subplots(figsize=(7, 5), ncols=len(results))

for i, result in enumerate(results):

        ax = axes[i]

        shape = result.experiment_config.shape
        shape_gt = result.shape_gt[0]
        fitting_params = result.fitting_params

        plotter = ShapePlotter2D()

        ax.set_aspect("equal", adjustable="box")
        plotter.plot_shape(shape, shape_gt, label="Ground Truth", color="black", ax=ax)
        #ax.plot(shape_gt[0], shape_gt[1], "ro", color="black", label="True Center")

        fixed_sigma = result.noise_gt[0]
        # fixed_sigma = result.noise_gt[0][0] # for fixed
        noise_model = copy.copy(result.experiment_config.noise_function)
        noise_model.fixed_params = fixed_sigma

        source_params = result.source_gt[0]
        new_source = copy.copy(result.experiment_config.source_density)
        new_source.fixed_params = source_params

        # compute observations
        parameter_samples = new_source.sample(n=100)

        # dirty for laser
        if i == 1:
            noise_model = LaserDependentNoise(dim_param=1, dim_ambient=2, shape=shape, scanner=np.array([-6,2]), fixed_params = np.array([-1,0,3,2,np.pi/4,fixed_sigma[0],0.08]))
            covariances = noise_model.evaluate(parameter_samples)
        else: covariances = noise_model.evaluate(parameter_samples, params=fixed_sigma)

        source_points = shape.parametrization(parameter_samples, shape_gt)
        errors = np.array([np.random.multivariate_normal(np.zeros(shape.ambient_dim), cov) for cov in covariances])
        observations = source_points + errors

        ax.scatter(
            observations[:, 0],
            observations[:, 1], s=8, alpha=0.8, label = "Observations", color = "red", zorder=4)

        ax.set_xlabel("")
        ax.set_ylabel("")

        # now plot covariances
        # sample_tvals

        c,w = source_params

        t_arc = np.linspace(
            c -  w,
            c + w,
            300
        )

        arc_points = shape.parametrization(t_arc, shape_gt)

        """ax.plot(
            arc_points[:, 0],
            arc_points[:, 1],
            color="dimgray",
            linewidth=6,
            alpha=0.9,
            label="Observed Arc",
            zorder=5,
        )"""

        # we sample more source points for the covariances
        parameter_samples_plot = new_source.sample(n=250)
        # dirty for laser
        if i == 1:
            noise_model = LaserDependentNoise(dim_param=1, dim_ambient=2, shape=shape, scanner=np.array([-6, 2]),
                                              fixed_params=np.array([-1, 0, 3, 2, np.pi / 4, fixed_sigma[0], 0.08]))
            covariances_plot = noise_model.evaluate(parameter_samples_plot)
        else:
            covariances_plot = noise_model.evaluate(parameter_samples_plot, params=fixed_sigma)

        source_points_plot = shape.parametrization(parameter_samples_plot, shape_gt)

        for i in range(len(parameter_samples_plot)):
            cov = covariances_plot[i]
            center = source_points_plot[i]

            # Eigen-decomposition of covariance matrix
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            # Compute ellipse angle in degrees
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            # Width and height = 2 * sqrt(eigenvalues)
            width, height = 2 * np.sqrt(vals)

            # Create and add ellipse
            ellip = patches.Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor="green",
                                    facecolor="green", lw=0.1, alpha=0.3, label="Noise model", zorder=1)
            ax.add_patch(ellip)


# we want the same size
# Make both panels have identical limits
xmins, xmaxs, ymins, ymaxs = [], [], [], []

for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xmins.append(xmin)
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)

xmin, xmax = min(xmins), max(xmaxs)
ymin, ymax = min(ymins), max(ymaxs)

for ax in axes:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


# adjust difference between height
fig.subplots_adjust(hspace=0.01)

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))

fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="lower center",
    bbox_to_anchor=(0.5, 0.1),
    ncol=4,
    fontsize=12,
    frameon=False,
)

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"plot_setups.eps")
plt.savefig(save_path, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"  Saved plot: {save_path}")