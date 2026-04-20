import numpy as np

from scripts.Benchmarking.Fitters.Cylinder_fitters.least_squares_cylinder import LeastSquaresCylinderFitter
from scripts.Benchmarking.Fitters.Cylinder_fitters.pca_cylinder import PCACylinderFitter
from scripts.Benchmarking.Fitters.Cylinder_fitters.ransac_cylinder import RANSACCylinderFitter
from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.cylinder import Cylinder


shape = Cylinder()

CylinderExperiment = {

        "name": "Cylinder_Experiment",

        "iterations": 100,
        "num_points": 100,

        "shape": shape,


        # now define source density and noise function of the ground truth
        "source_density": UniformSourceDensity(n_dim=2, domain_low=shape.domain_low,
                                               domain_high=shape.domain_high, periodic=shape.periodic),

        "noise_function": IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim),


        # define the ground-truth-params
        "shape_params": np.array([0.0, 1.0, 0.0, 1.2, 10.0, 0.5, 0.3]),
        "source_params": np.array([0.5,0.2, 0.6, 0.3]),
        "noise_params": np.array([0.1]),

        # let's define the priors on source and noise
        "source_prior_weak": GaussianPrior(mean=np.array([0.5, 0.2, 0.6, 0.3]), cov=np.diag([10, 0.0001, 10, 0.0001])),
        "source_prior_strong": GaussianPrior(mean=np.array([0.5, 0.2, 0.6, 0.3]), cov=np.diag([10, 0.001, 10, 0.001])),
        "source_bounds_weak": [(0, 1), (0.15, 0.25), (0, 1), (0.25,0.35)],
        "source_bounds_strong": [(0, 1), (0.175, 0.225),(0, 1), (0.275, 0.325)],

        "noise_prior_weak": GaussianPrior(mean=np.array([0.0225]), cov=np.array([0.01])),
        "noise_prior_strong": GaussianPrior(mean=np.array([0.0225]), cov=np.array([0.001])),
        "noise_bounds_weak": [(0.001, 0.04)],
        "noise_bounds_strong": [(0.001, 0.04)],

        # now we define init_values for source_density and noise function
        "source_init": np.array([0.3, 0.2, 0.6, 0.3]), #  + 0.01 * np.random.randn(4),
        "noise_init": np.array(0.1),

        # specify the baseline fitters
        "fitters_static": [
            # we define standard fitters
            LeastSquaresCylinderFitter(),
            RANSACCylinderFitter(),
            PCACylinderFitter()
        ],

        # and their labels
        "labels" : [
            "LS",
            "RANSAC",
            "PCA",
            r"$\mathbf{Weak\ Prior}$",
            r"$\mathbf{Strong\ Prior}$",
            r"$\mathbf{Fixed\ Prior}$",
        ]
}