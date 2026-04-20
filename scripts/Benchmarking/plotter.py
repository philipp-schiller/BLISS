# plot_results.py
from scripts.Benchmarking.Experiments.cylinder_experiment import CylinderExperiment
from scripts.Benchmarking.Experiments.random_ellipse import RandomExperiment
from scripts.Benchmarking.Experiments.unisotropic_noise import LaserExperiment
from scripts.Benchmarking.results import SweepResult
from scripts.Benchmarking.Experiments.experiment_config import ExperimentConfig
from scripts.Benchmarking.Experiments.isotropic_noise import IsotropicExperiment
import numpy as np

def load_and_plot(stem: str, config_dict):
    # rebuild config
    config = ExperimentConfig.get_from_dict(config_dict)

    # load sweep result
    result = SweepResult.load(stem, experiment_config=config)

    print("Average fitting times:")
    print(np.nanmean(result.fitting_times, axis=0))

    # plot
    result.plot_result(save_dir="plots")

    #result.plot_cylinder()
    #result.print_cylinder_comparison_table()

    #result.plot_individual_fits()

    # print table
    print(result.generate_latex_table())
    #print(result.generate_latex_table_Ec_Ea_Earea())

if __name__ == "__main__":
    config_dict = IsotropicExperiment
    stem = "results/isotropic_Experiment"
    #config_dict = LaserExperiment
    #stem = "results/Laser_Experiment"  # without extension
    #stem = "results/isotropic_Experiment"  # without extension

    #config_dict = RandomExperiment
    #stem = "results/Random_Experiment"

    #config_dict = CylinderExperiment
    #stem = "results/Cylinder_Experiment"

    load_and_plot(stem, config_dict)