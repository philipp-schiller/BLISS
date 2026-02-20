"""
This file provides concrete example shapes, including curves and surfaces
"""
from .bezier_curve import BezierCurve
from .circle import Circle
from .sphere import Sphere
from .ellipse import Ellipse
from .polygon2D import Polygon2D
from .cylinder import Cylinder
import numpy as np

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
        "theta": np.array([-1, 0, 2, 3, np.pi / 4]),
        "init_params": np.array([-2, 1, 1.5, 2, 0]),
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
    }
]
