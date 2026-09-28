# plot_results.py
from scripts.Benchmarking.Experiments.cylinder_experiment import CylinderExperiment
from scripts.Benchmarking.Experiments.random_ellipse_noise import make_random_experiment_noise
from scripts.Benchmarking.Experiments.unisotropic_noise import LaserExperiment
from scripts.Benchmarking.results import SweepResult
from scripts.Benchmarking.Experiments.experiment_config import ExperimentConfig
from scripts.Benchmarking.Experiments.isotropic_noise import IsotropicExperiment
from scripts.Benchmarking.Experiments.isotropic_illustrative import IsotropicIllustrative
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import trim_mean

coverages = []
area_errors = {}
center_errors = {}
deviation_errors = {}

sigma_sq = 0.05

#for sigma_prior in [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35]:#,0.15,0.2,0.25, 0.3, 0.35]:
for sigma_sq_prior in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]:#,0.15,0.2,0.25, 0.3, 0.35]:
#for sigma_sq_prior in [0.05,0.06,0.07,0.08,0.09]:#,0.15,0.2,0.25, 0.3, 0.35]:

    coverage = sigma_sq_prior
    coverages.append(coverage)

    config = ExperimentConfig.get_from_dict(
        make_random_experiment_noise(sigma_sq=sigma_sq, sigma_sq_prior=sigma_sq_prior)
    )

    stem = "results/Random_Experiment_Noise_" + str(sigma_sq) + "_prior_" + str(sigma_sq_prior)

    result = SweepResult.load(
        stem,
        experiment_config=config
    )

    methods = result.experiment_config.labels

    for method in methods:
        Ec_list = result.get_Ec(method)
        Earea_list = result.get_Earea(method)
        Ea_list = result.get_Ea(method)

        Ec = trim_mean(Ec_list, 0.01)
        Earea = trim_mean(np.abs(Earea_list), 0.01)
        Ea = trim_mean(Ea_list, 0.01)

        center_errors.setdefault(method, []).append(Ec)
        area_errors.setdefault(method, []).append(Earea)
        deviation_errors.setdefault(method, []).append(Ea)

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axs = plt.subplots(1, 3, figsize=(10, 4))

# Area error
for method, errors in area_errors.items():
    axs[0].plot(
        coverages,
        errors,
        marker="o",
        linewidth=2,
        label=method
    )

axs[0].set_xlabel(r"Noise level $\sigma^2$")
axs[0].set_ylabel(r"$E_{\mathrm{area}}$")
axs[0].set_yscale("log")
axs[0].set_title("Area Error")
axs[0].grid(alpha=0.3)

# Center error
for method, errors in center_errors.items():
    axs[1].plot(
        coverages,
        errors,
        marker="o",
        linewidth=2,
        label=method
    )

axs[1].set_xlabel(r"Noise level $\sigma^2$")
axs[1].set_ylabel(r"$E_c$")
axs[1].set_yscale("log")
axs[1].set_title("Center Error")
axs[1].grid(alpha=0.3)

# Center error
for method, errors in deviation_errors.items():
    axs[2].plot(
        coverages,
        errors,
        marker="o",
        linewidth=2,
        label=method
    )

axs[2].set_xlabel(r"Noise level $\sigma^2$")
axs[2].set_ylabel(r"$E_a$")
axs[2].set_yscale("log")
axs[2].set_title("Deviation Error")
axs[2].grid(alpha=0.3)


# Shared legend
handles, labels = axs[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

plt.savefig(
    "Random_Noise_level_wrong_prior.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

