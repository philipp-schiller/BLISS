# plot_results.py

from scripts.Benchmarking.Experiments.random_ellipse import make_random_experiment

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

w = 0.2

for w_prior in [0.1, 0.15, 0.2, 0.25, 0.3]:

    coverage = int(w_prior * 200)   # 20, 30, 40, 50
    coverages.append(coverage)

    config = ExperimentConfig.get_from_dict(
        make_random_experiment(w=w, w_prior=w_prior)
    )

    stem = f"results/Random_Experiment_{int(w * 200)}prior_{int(w_prior*200)}"

    result = SweepResult.load(
        stem,
        experiment_config=config
    )

    methods = result.experiment_config.labels
    selected = [methods[0], methods[3], methods[4]]

    for method in selected:

        Ec_list = result.get_Ec(method)
        Earea_list = result.get_Earea(method)
        Ea_list = result.get_Ea(method)

        Ec = trim_mean(Ec_list, 0.1)
        Earea = trim_mean(np.abs(Earea_list), 0.1)
        Ea = trim_mean(Ea_list, 0.1)

        #Ec = np.nanmean(Ec_list)
        #Earea = np.nanmean(Earea_list)
        #Ea = np.nanmedian(Ea_list)

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

axs[0].set_xlabel("Arc Coverage (%)", fontsize=15)
axs[0].set_ylabel(r"Area deviation $E_{\mathrm{area}}$",fontsize = 15)
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

axs[1].set_xlabel("Arc Coverage (%)", fontsize =15)
axs[1].set_ylabel(r"Center deviation $E_c$", fontsize = 15)
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

axs[2].set_xlabel("Arc Coverage (%)", fontsize=15)
axs[2].set_ylabel(r"Shape deviation $E_a$", fontsize = 15)
#axs[2].set_yscale("log")
axs[2].grid(alpha=0.3)


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
    fontsize = 15
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)

plt.savefig(
    "Wrong_prior.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

