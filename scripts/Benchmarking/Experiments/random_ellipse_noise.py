# axes.py
import numpy as np

from bayes_shape_fitting.models.shapes.ellipse_reparametrized import Ellipse_R
from scripts.Benchmarking.Fitters.Ellipse_fitters.HEIV import HEIVFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.bayes_fitter import BayesFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.direct_ellipse_fit import DirectEllipseFit
from scripts.Benchmarking.Fitters.Ellipse_fitters.hyper_renormalization import HyperRenormalizationFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.least_squares import LeastSquaresFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.self_normalized import SelfNormalizedFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.taubin import TaubinFitter

from scripts.Benchmarking.Fitters.Ellipse_fitters.zhao_bayes import BayesianEllipseFitterZhao
from scripts.Benchmarking.Fitters.Ellipse_fitters.geometric_ls import GeometricEllipseFitter

from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.ellipse import Ellipse


shape = Ellipse()

w = 0.2 # 40% arc coverage

def make_random_experiment_noise(sigma_sq = 0.05, sigma_sq_prior = None):
    #sigma_sq = sigma**2
    if sigma_sq_prior is None:
        name = "Random_Experiment_Noise_"+ str(sigma_sq)
        sigma_sq_prior = sigma_sq
    else:
        name = "Random_Experiment_Noise_"+ str(sigma_sq)+"_prior_"+str(sigma_sq_prior)

    return {

        "name": name,

        "iterations": 100,
        "num_points": 100,

        "shape": shape,

        # now define source density and noise function of the ground truth
        "source_density": UniformSourceDensity(n_dim=1, domain_low=shape.domain_low,
                                               domain_high=shape.domain_high, periodic=shape.periodic),

        "noise_function": IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim),

        # define the ground-truth-params
        "shape_params": None,
        "source_params":np.array([None, w]),
        "noise_params": sigma_sq,

        # let's define the priors on source and noise
        #"source_prior_weak": GaussianPrior(mean=np.array([0.5, w]), cov=np.diag([1000, 0.00065])),
        #"source_prior_weak": GaussianPrior(mean=np.array([0.5, w]), cov=np.diag([1000, 1000])),
        #"source_prior_strong": GaussianPrior(mean=np.array([0.5, w]), cov=np.diag([1000, 0.00016])),
        #"source_prior_strong": GaussianPrior(mean=np.array([0.5, w]), cov=np.diag([1000, 1000])),
        "source_bounds_weak": [(0, 1), (w - 0.05, w + 0.05)],
        "source_bounds_strong": [(0, 1), (w - 0.025, w + 0.025)],

        "noise_prior_weak": GaussianPrior(mean=np.array(sigma_sq_prior), cov=np.array([2.6e-5])),
        "noise_prior_strong": GaussianPrior(mean=np.array(sigma_sq_prior), cov=np.array([6e-6])),

        "noise_bounds_weak": [
        (max(0.0, sigma_sq_prior - 1.96 * np.sqrt(2.6e-5)),
         sigma_sq_prior + 1.96 * np.sqrt(2.6e-5))
        ],

        "noise_bounds_strong": [
        (max(0.0, sigma_sq_prior - 1.96 * np.sqrt(6e-6)),
         sigma_sq_prior + 1.96 * np.sqrt(6e-6))
        ],

    # now we define init_values for source_density and noise function
        "source_init": np.array([0.5, w]),
        "noise_init": np.array(sigma_sq_prior),

        # specify the baseline fitters
        "fitters_static": [
            # we define standard fitters
            GeometricEllipseFitter(),
            HyperRenormalizationFitter(),
            BayesianEllipseFitterZhao()
        ],

        # and their labels
        "labels" : [
            "GLS",
            "HR",
            "Zhao",
            r"$\mathbf{BLISS\text{-}weak}$",
            r"$\mathbf{BLISS\text{-}strong}$",
            r"$\mathbf{BLISS\text{-}exact}$",
        ]
}
