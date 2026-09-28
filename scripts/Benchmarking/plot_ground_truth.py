# plot_results.py
from scripts.Benchmarking.Experiments.cylinder_experiment import CylinderExperiment
from scripts.Benchmarking.Experiments.random_ellipse import make_random_experiment
from scripts.Benchmarking.Experiments.unisotropic_noise import LaserExperiment
from scripts.Benchmarking.results import SweepResult
from scripts.Benchmarking.Experiments.experiment_config import ExperimentConfig
from scripts.Benchmarking.Experiments.isotropic_noise import IsotropicExperiment
from scripts.Benchmarking.Experiments.isotropic_illustrative import IsotropicIllustrative
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import trim_mean

coverages = []
area_errors = {}
center_errors = {}
deviation_errors = {}
fitting_time = {}

for w in [0.1, 0.15, 0.2, 0.25, 0.3]:

    coverage = int(w * 200)   # 20, 30, 40, 50
    coverages.append(coverage)

    config = ExperimentConfig.get_from_dict(
        make_random_experiment(w=w)
    )

    stem = f"results/Random_Experiment_{coverage}"

    result = SweepResult.load(
        stem,
        experiment_config=config
    )

    methods = result.experiment_config.labels

    for method in methods:

        Ec_list = result.get_Ec(method)
        Earea_list = result.get_Earea(method)
        Ea_list = result.get_Ea(method)

        Ec = np.nanmedian(Ec_list)
        Earea = np.nanmedian(Earea_list)
        Ea = np.nanmedian(Ea_list)

        Ec = trim_mean(Ec_list, 0.1)
        Earea = trim_mean(np.abs(Earea_list), 0.1)
        Ea = trim_mean(Ea_list, 0.1)

        center_errors.setdefault(method, []).append(Ec)
        area_errors.setdefault(method, []).append(Earea)
        deviation_errors.setdefault(method, []).append(Ea)
        fitting_time.setdefault(method, []).append(np.nanmean(result.fitting_times, axis=0))

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.8))

# Area error
for method, errors in area_errors.items():
    axs[0].plot(
        coverages,
        errors,
        marker="o",
        linewidth=2,
        label=method
    )

axs[0].set_xlabel("Arc Coverage (%)", fontsize=12)
axs[0].set_ylabel(r"Area deviation $E_{\mathrm{area}}$",fontsize = 12)
axs[0].set_yscale("log")
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

axs[1].set_xlabel("Arc Coverage (%)", fontsize =12)
axs[1].set_ylabel(r"Centre deviation $E_c$", fontsize = 12)
axs[1].set_yscale("log")
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

axs[2].set_xlabel("Arc Coverage (%)", fontsize=12)
axs[2].set_ylabel(r"Shape deviation $E_{\mathrm{shape}}$", fontsize = 12)
axs[2].set_yscale("log")
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

axs[3].set_xlabel("Arc Coverage (%)", fontsize = 12)
axs[3].set_ylabel("Avg comp. time (s)", fontsize = 12)
#axs[3].set_yscale("log")
#axs[0].set_title("Area Error")
axs[3].grid(alpha=0.3)


# Optional: show increasing ill-posedness from left → right
for ax in axs:
    ax.invert_xaxis()

# Shared legend
handles, labels = axs[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
    fontsize = 12,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.savefig(
    "Random_Arc_Coverage.png",
    dpi=600,
    bbox_inches="tight"
)
plt.savefig(
    "Random_Arc_Coverage.eps",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

