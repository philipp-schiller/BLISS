"""
This file provides concrete example shapes, including curves and surfaces
"""
from .bezier_curve import BezierCurve
from .circle import Circle
from .sphere import Sphere
from .ellipse import Ellipse
from .polygon2D import Polygon2D
import numpy as np

# Define shape cases
shape_cases = [
    {
        "name": "circle",
        "shape": Circle(),
        "theta": np.array([-1, 0, 2,]),
        "init_params": np.array([-2, 1, 1.5]),
        "markers": ["circle", "curve"]
    },
    {
        "name": "ellipse",
        "shape": Ellipse(),
        "theta": np.array([-1, 0, 2, 3, np.pi / 4]),
        "init_params": np.array([-2, 1, 1.5, 4, 0]),
        "markers": ["ellipse", "curve"]
    },
    {
        "name": "polygon2D",
        "shape": Polygon2D(n=5),
        "theta": np.array([-1, 0, -2, 0, -2, 1, 1, 0.5, 0.7, 0.8]),
        "init_params": np.array([-1, 0, -2, 1, -2, 1, 1, 0.5, 0.7, 0.8]),
        "markers": ["polygon2D", "curve"]
    },
    {
        "name": "sphere",
        "shape": Sphere(),
        "theta": np.array([-1, 0, 2, 3]),
        "init_params": np.array([-2, 1, 1.5, 4]),
        "markers": ["sphere", "surface"]
    },
    {
        "name": "bezier_curve",
        "shape": BezierCurve(degree = 3),
        "theta": np.array([0.0, 1.0, 2.0, 3.0, 0.0, 2.0, -1.0, 1.0]),
        "init_params": np.array([-1, 0, 1.5, -0.3, 0, 0.2, 1.5, 0.2]),
        "markers": ["bezier_curve", "curve"]
    },
]
