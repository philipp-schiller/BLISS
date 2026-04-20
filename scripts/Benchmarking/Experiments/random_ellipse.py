# axes.py
import numpy as np

from scripts.Benchmarking.Fitters.Ellipse_fitters.HEIV import HEIVFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.bayes_fitter import BayesFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.direct_ellipse_fit import DirectEllipseFit
from scripts.Benchmarking.Fitters.Ellipse_fitters.hyper_renormalization import HyperRenormalizationFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.least_squares import LeastSquaresFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.self_normalized import SelfNormalizedFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.taubin import TaubinFitter

from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.ellipse import Ellipse


shape = Ellipse()

RandomExperiment = {

        "name": "Random_Experiment",

        "iterations": 100,
        "num_points": 100,

        "shape": shape,


        # now define source density and noise function of the ground truth
        "source_density": UniformSourceDensity(n_dim=1, domain_low=shape.domain_low,
                                               domain_high=shape.domain_high, periodic=shape.periodic),

        "noise_function": IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim),


        # define the ground-truth-params
        "shape_params": None,
        "source_params":None,
        "noise_params": None,

        # let's define the priors on source and noise
        "source_prior_weak": GaussianPrior(mean=np.array([0.5, 0.20]), cov=np.diag([10, 0.00065])),
        "source_prior_strong": GaussianPrior(mean=np.array([0.5, 0.20]), cov=np.diag([10, 0.00016])),
        "source_bounds_weak": [(0, 1), (0.15, 0.25)],
        "source_bounds_strong": [(0, 1), (0.175, 0.225)],

        "noise_prior_weak": GaussianPrior(mean=np.array([0.0225]), cov=np.array([0.00001])),
        "noise_prior_strong": GaussianPrior(mean=np.array([0.0225]), cov=np.array([0.000001])),
        "noise_bounds_weak": [(0.001, 0.04)],
        "noise_bounds_strong": [(0.001, 0.04)],

        # now we define init_values for source_density and noise function
        "source_init": np.array([0.5, 0.20]) + 0.01 * np.random.randn(2),
        "noise_init": np.array(0.01),

        # specify the baseline fitters
        "fitters_static": [
            # we define standard fitters
            DirectEllipseFit(),
            HyperRenormalizationFitter(),
            SelfNormalizedFitter(),
            HEIVFitter(),
        ],

        # and their labels
        "labels" : [
            "DirectFit",
            "HR",
            "HLS",
            "HEIV",
            r"$\mathbf{Weak\ Prior}$",
            r"$\mathbf{Strong\ Prior}$",
            r"$\mathbf{Fixed\ Prior}$",
        ]
}