# axes.py
import numpy as np

from examples.cylinder_fit_vem import Cylinder

from src.bayes_shape_fitting.inference.noise_functions import IsotropicNoise, LaserDependentNoise
from src.bayes_shape_fitting.inference.priors import Prior, GaussianPrior
from src.bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from src.bayes_shape_fitting.models.shapes.ellipse import Ellipse
from src.bayes_shape_fitting.models.shapes.ellipse_reparametrized import Ellipse_R


# define the ellipse shape
shape = Cylinder()

SHAPE_CASES = [
    {
        "name": "cylinder_fixed",
        "shape": shape,
        "shape_params": np.array([-1, 0, 3, 2, np.pi / 4]),
    }]


NOISE_FUNCTION_CASES = [
    {
        "name": "isotropic_fixed_0.005",
        "noise_function": IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, fixed_params=0.005)
    }]


SOURCE_DENSITY_CASES = [
    {
        "name": "uniform_fixed",
        "source_density": UniformSourceDensity(n_dim = 2, domain_low=shape.domain_low, domain_high=shape.domain_high, fixed_params=np.array([0.5,0.2, 0.6, 0.3]), periodic=shape.periodic)
    }
]

"""SOURCE_DENSITIES = [
    UniformSourceDensity(n_dim=1, fixed_params=np.array([0.3, 0.2]), domain_low=shape.domain_low,
                                               domain_high=shape.domain_high, periodic=shape.periodic),

    #UniformSourceDensity(n_dim=1, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic),

    #UniformSourceDensity(n_dim=1, fixed_params=np.array([0.4, 0.3]), domain_low=shape.domain_low,
    #                                           domain_high=shape.domain_high, periodic=shape.periodic)
]"""

FITTERS_STATIC = [
    # we define standard fitters
    DirectEllipseFit(),
    HyperRenormalizationFitter(),
    SelfNormalizedFitter(),
    HEIVFitter(),
]

LABELS = ["DirectFit", "HR", "SelfN", "HEIV", "Bayes-weak", "Bayes-strong","Bayes-exact"]