import numpy as np

from bayes_shape_fitting.models.shapes.ellipsoid import Ellipsoid_R

from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity, PartiallyFixedDensity
from bayes_shape_fitting.models.shapes.cylinder import Cylinder


shape = Ellipsoid_R()

# now define source density
source_density_full = UniformSourceDensity(n_dim=2, domain_low=shape.domain_low, domain_high=shape.domain_high)
source_density = PartiallyFixedDensity(source_density_full, fixed_params={0: 0.5, 1: 0.5})


source_bounds = [(0, 1), (0.1, 1)]

Ellipsoid_Experiment = {

        "name": "Ellipsoid_Experiment",

        "iterations": 2,
        "num_points": 100,

        "shape": shape,

        "source_density": source_density,
        "noise_function": IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.05),

        # define the ground-truth-params
        "shape_params": np.array([0.0, 1.0, 0.0, 3, 2/3, 1/3, 0.3,0.5,0.1]),
        "source_params": np.array([0.3, 0.35]),
        "noise_params": np.array([0.05]),

        # let's define the priors on source and noise
        "source_prior_weak": GaussianPrior(mean=np.array([0,0.35]), cov=np.diag([10, 0.0001])),
        "source_prior_strong": GaussianPrior(mean=np.array([0, 0.35]), cov=np.diag([10, 0.001])),
        "source_bounds_weak": source_bounds,
        "source_bounds_strong": source_bounds,

        "noise_prior_weak": GaussianPrior(mean=np.array([0.0225]), cov=np.array([0.01])),
        "noise_prior_strong": GaussianPrior(mean=np.array([0.0225]), cov=np.array([0.001])),
        "noise_bounds_weak": [(0.04,0.06)],
        "noise_bounds_strong": [(0.04,0.06)],

        # now we define init_values for source_density and noise function
        "source_init": np.array([0, 0.35]), #  + 0.01 * np.random.randn(4),
        "noise_init": np.array(0.05),

        # specify the baseline fitters
        "fitters_static": [
            # we define standard fitters

        ],

        # and their labels
        "labels" : [

            r"$\mathbf{Weak\ Prior}$",
            r"$\mathbf{Strong\ Prior}$",
            r"$\mathbf{Fixed\ Prior}$",
        ]
}