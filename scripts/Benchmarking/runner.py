
# run_experiments.py
import time
from pathlib import Path

import numpy as np

from scripts.Benchmarking.Experiments.cylinder_experiment import CylinderExperiment
from scripts.Benchmarking.Experiments.experiment_config import ExperimentConfig
from scripts.Benchmarking.Experiments.isotropic_noise import IsotropicExperiment
from scripts.Benchmarking.Experiments.random_ellipse import RandomExperiment
from scripts.Benchmarking.Experiments.unisotropic_noise import LaserExperiment

np.random.seed(876)

def run_experiment(config: ExperimentConfig, save_dir="results"):
    Path(save_dir).mkdir(exist_ok=True)

    print(f"\n--- Running experiment: {config.name} ---")

    t0 = time.perf_counter()
    result = config.run()
    dt = time.perf_counter() - t0

    print(f"Finished in {dt:.2f} seconds.")

    stem = Path(save_dir) / config.name.replace(" ", "_")
    result.save(save_dir)  # saves <stem>.npz and <stem>.json

    print(f"Saved result to {stem}.npz and {stem}.json")
    return result


if __name__ == "__main__":
    #config = ExperimentConfig.get_from_dict(IsotropicExperiment)
    config = ExperimentConfig.get_from_dict(LaserExperiment)
    #config = ExperimentConfig.get_from_dict(RandomExperiment)
    #config = ExperimentConfig.get_from_dict(CylinderExperiment)

    run_experiment(config, save_dir="results")

