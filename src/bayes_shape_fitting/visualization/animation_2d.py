"""
Animation extensions for ShapePlotter2D.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple, Callable
from pathlib import Path
from src.bayes_shape_fitting.inference.likelihoods import Likelihood
from src.bayes_shape_fitting.inference.source_densitites import SourceDensity
from src.bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D


class ShapePlotterAnimations:
    """
    Animation methods for 2D shape plotting.
    """

    def animate_param_history(self, shape, param_history: List[np.ndarray], ax=None):
        """
        Animates a sequence of shape parameters

        Parameters
        ----------
        shape
        param_history

        Returns
        -------
        """

        if ax is None:
            ax = plt.gcf().gca()

        # we start with plotting the first shape
        plotter = ShapePlotter2D()

        # we use the first element in the history
        params = param_history[0]

        fig, ax, line = plotter.plot_shape(shape=shape, params=params, ax = ax)
        # now get the artist of the shape-plot

        # define the parameter values
        t = np.linspace(
            float(np.atleast_1d(shape.domain_low)[0]),
            float(np.atleast_1d(shape.domain_high)[0]),
            300
        ).reshape(-1, 1)

        def update(frame):
            params = param_history[frame]
            points = shape.parametrization(t, params)
            line.set_data(points[:, 0], points[:, 1])
            return line,

        ani = FuncAnimation(fig, update, frames=len(param_history), blit=True, interval=500)
        plt.show()
        return ani