"""
This file provides concrete example shapes, including curves and surfaces
"""
import numpy as np

from .bezier_curve import BezierCurve
from .circle import Circle
from .cylinder import Cylinder
from .ellipse import Ellipse
from .polygon2D import Polygon2D
from .sphere import Sphere
from .generalized_cylinder import GeneralizedCylinder
from .ellipsoid import Ellipsoid_R

# Define shape cases
shape_cases = [
    {
        "name": "circle",
        "shape": Circle(),
        "theta": np.array([-1, 0, 2,]),
        "init_params": np.array([-1.2, 0.3, 1.5]),
        "markers": ["circle", "curve"]
    },
    {
        "name": "ellipse",
        "shape": Ellipse(),
        "theta": np.array([-1, 0, 3, 2, np.pi / 4]),
        "init_params": np.array([-2, 1, 2, 1.5, 0]),
        "markers": ["ellipse", "curve"]
    },
    {
        "name": "polygon2D",
        "shape": Polygon2D(n=4),
        "theta": np.array([0,0, 1,0, 1,1, 0,1]),
        "init_params": np.array([0.1,-0.2, 1.1,0.5, 1.1,1, 0.23,1.2]),
        "markers": ["polygon2D", "curve"]
    },
    {
        "name": "sphere",
        "shape": Sphere(),
        "theta": np.array([-1, 0, 2, 3]),
        "init_params": np.array([-1.2, 0.1, 2.3, 3.1]),
        "markers": ["sphere", "surface"]
    },
    {
        "name": "bezier_curve",
        "shape": BezierCurve(degree = 3),
        "theta": np.array([0.0, 1.0, 2.0, 3.0, 0.0, 2.0, -1.0, 1.0]),
        "init_params": np.array([0.0, 1.0, 2.0, 2.5, 0.0, 2.1, -1.3, 1.0]),
        "markers": ["bezier_curve", "curve"]
    },
    {
        "name": "cylinder",
        "shape": Cylinder(),
        "theta": np.array([0.0, 1.0, 0.0, 1.2, 10.0, 0.5, 0.3]), # x_c, y_c, z_c, r, h, polar, azim
        "init_params":  np.array([0.2, 1.2, -0.3, 2.0, 8.5, 0.6, 0.1]),
        "markers": ["cylinder", "surface"]
    },

    {
        "name": "generalized_cylinder",
        "shape": GeneralizedCylinder(
            n_spine_ctrl=4,
            n_radius_ctrl=3,
            degree_spine=3,
            degree_radius=2,
        ),

        "theta": np.concatenate([
            # x-spine: gentle lean with slight curvature
            np.array([0.0, 0.12, 0.20, 0.32]),

            # y-spine: small sideways drift (e.g. wind)
            np.array([0.0, 0.05, 0.07,  0.05]),

            # z-spine: mostly vertical growth
            5*np.array([0.0, 0.40, 0.65, 1.0]),

            # radius: thicker base, gentle taper
            np.array([0.25, 0.2, 0.14]),
        ]),

        # Initial guess
        "init_params": np.concatenate([
            0.05 * np.random.randn(4),      # noisy x
            0.05 * np.random.randn(4),      # noisy y
            5*np.linspace(0, 1, 4),           # straight initial spine
            0.20 * np.ones(3),              # roughly correct radius
        ]),


        "markers": ["generalized_cylinder", "surface"]
    },
    {
        "name": "ellipsoid",
        "shape": Ellipsoid_R(),
        "theta": np.array([
            -1.0,   # x_c
             0.5,   # y_c
             1.0,   # z_c
             3.0,   # a (semi-major axis)
             0.6,   # r1 = b/a
             0.4,   # r2 = c/a
             np.pi / 6,   # α (rotation around z)
             np.pi / 8,   # β (rotation around y)
             np.pi / 10   # γ (rotation around x)
        ]),
        "init_params": np.array([
            -1.5,  0.0,  0.8,   # center guess
             2.5,          # a
             0.7, 0.5,     # axis ratios
             0.0, 0.0, 0.0 # initial rotation guess
        ]),
        "markers": ["ellipsoid", "surface"]
    }
]
