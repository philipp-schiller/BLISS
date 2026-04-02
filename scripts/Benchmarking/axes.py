# axes.py
import numpy as np

from scripts.Benchmarking.Fitters.Ellipse_fitters.HEIV import HEIVFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.bayes_fitter import BayesFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.direct_ellipse_fit import DirectEllipseFit
from scripts.Benchmarking.Fitters.Ellipse_fitters.hyper_renormalization import HyperRenormalizationFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.least_squares import LeastSquaresFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.self_normalized import SelfNormalizedFitter
from scripts.Benchmarking.Fitters.Ellipse_fitters.taubin import TaubinFitter

from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise, LaserDependentNoise, PartiallyFixedNoise
from src.bayes_shape_fitting.inference.priors import Prior, GaussianPrior
from src.bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from src.bayes_shape_fitting.models.shapes.ellipse import Ellipse
from src.bayes_shape_fitting.models.shapes.ellipse_reparametrized import Ellipse_R


# define the ellipse shape
shape = Ellipse()
"""
SHAPE_CASES = [
    {
        "name": "ellipse_fixed",
        "shape": shape,
        "shape_params": np.array([-1, 0, 3, 2, np.pi / 4]),
    }]

"""
SHAPE_CASES = [
  {
        "name": "ellipse_random",
        "shape": shape,
        "shape_params": None  #None # resembles random shape params
    }
]


"""{
      "name": "isotropic_fixed_0.01",
      "noise_function": IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.01)
  },"""

fixed_params=np.array([-1, 0, 3, 2, np.pi / 4, 0.01,0.01])
base_noise = LaserDependentNoise(dim_param=1, dim_ambient=2, shape=shape, scanner=np.array([-5,5]))
fixed_params = {0: -1, 1: 0, 2: 3,3: 2, 4: np.pi/4, 5:0.0001}
laser_noise = PartiallyFixedNoise(base_noise, fixed_params=fixed_params)

"""NOISE_FUNCTION_CASES = [
    {
    "name": "laser_noise",
    # the first params of the parameters resemble the shape, only the last two are sigma_0, gamma_0
    "noise_function" : laser_noise
    }
]
"""
NOISE_FUNCTION_CASES = [
    {
      "name": "isotropic_random",
      "noise_function": IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.01)
    }
]

"""
,
    {
        "name": "isotropic_fixed_0.15",
        "noise_function": IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_sigma=0.15)
    }
"""


"""SOURCE_DENSITY_CASES = [
    {
        "name": "uniform_fixed",
        "source_density": UniformSourceDensity(n_dim=1, fixed_params=np.array([0.3, 0.25]), domain_low=shape.domain_low,
                                               domain_high=shape.domain_high, periodic=shape.periodic)
    }
]"""

SOURCE_DENSITY_CASES = [
    {
        "name": "uniform_fixed",
        "source_density": UniformSourceDensity(n_dim=1, fixed_params=np.array([0.3, 0.25]), domain_low=shape.domain_low,
                                               domain_high=shape.domain_high, periodic=shape.periodic)
    }
]

SOURCE_DENSITY_CASES = [
    {
        "name": "uniform_random",
        "source_density": UniformSourceDensity(n_dim=1, domain_low=shape.domain_low,
                                               domain_high=shape.domain_high, periodic=shape.periodic)
    }
]
FITTERS_STATIC = [
    # we define standard fitters
    DirectEllipseFit(),
    HyperRenormalizationFitter(),
    SelfNormalizedFitter(),
    HEIVFitter(),
]

LABELS = [
    "DirectFit",
    "HR",
    "HLS",
    "HEIV",
    r"$\mathbf{Weak\ Prior}$",
    r"$\mathbf{Strong\ Prior}$",
    r"$\mathbf{Fixed\ Prior}$",
]