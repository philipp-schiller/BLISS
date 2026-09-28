# plot_results.py
from scripts.Benchmarking.Experiments.cylinder_experiment import CylinderExperiment
from scripts.Benchmarking.Experiments.unisotropic_noise import LaserExperiment, make_laser_experiment
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
fitting_time = {}


for sigma_0 in [0.01,0.02,0.03,0.04,0.05]:

    coverage = sigma_0
    coverages.append(coverage)

    config = ExperimentConfig.get_from_dict(
        make_laser_experiment(sigma_0=sigma_0, sigma_0_prior=sigma_0)
    )

    stem = "results/Laser_Experiment_" + str(sigma_0)+"prior_"+str(sigma_0)

    result = SweepResult.load(
        stem,
        experiment_config=config
    )

    methods = result.experiment_config.labels

    for i,method in enumerate(methods):
        Ec_list = result.get_Ec(method)
        Earea_list = result.get_Earea(method)
        Ea_list = result.get_Ea(method)

        Ec = trim_mean(Ec_list, 0.01)
        Earea = trim_mean(np.abs(Earea_list), 0.01)
        Ea = trim_mean(Ea_list, 0.01)

        center_errors.setdefault(method, []).append(Ec)
        area_errors.setdefault(method, []).append(Earea)
        deviation_errors.setdefault(method, []).append(Ea)
        fitting_time.setdefault(method, []).append(np.nanmean(result.fitting_times[:,i], axis=0))

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axs = plt.subplots(1, 4, figsize=(12, 3.8))

# Area error
for method, errors in area_errors.items():
    axs[0].plot(
        coverages,
        errors,
        marker="o",
        linewidth=2,
        label=method
    )

axs[0].set_xlabel(r"Base Noise $\sigma_0$", fontsize = 12)
axs[0].set_ylabel(r"Area deviation $E_{\mathrm{area}}$",fontsize = 12)
#axs[0].set_yscale("log")
#axs[0].set_title("Area Error")
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

axs[1].set_xlabel(r"Base Noise $\sigma_0$", fontsize = 12)
axs[1].set_ylabel(r"Centre deviation $E_c$", fontsize = 12)
#axs[1].set_yscale("log")
#axs[1].set_title("Center Error")
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

axs[2].set_xlabel(r"Base Noise $\sigma_0$", fontsize = 12)
axs[2].set_ylabel(r"Shape deviation $E_{\mathrm{shape}}$", fontsize = 12)
#axs[2].set_yscale("log")
#axs[2].set_title("Deviation Error")
axs[2].grid(alpha=0.3)

# Fitting time
for method, errors in fitting_time.items():
    axs[3].plot(
        coverages,
        errors,
        marker="o",
        linewidth=2,
        label=method
    )

axs[3].set_xlabel(r"Base Noise $\sigma_0$", fontsize = 12)
axs[3].set_ylabel("Avg comp. time (s)", fontsize = 12)
#axs[3].set_yscale("log")
#axs[0].set_title("Area Error")
axs[3].grid(alpha=0.3)

# Shared legend
handles, labels = axs[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
    fontsize = 12
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)

plt.savefig(
    "Laser_sweep.eps",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

