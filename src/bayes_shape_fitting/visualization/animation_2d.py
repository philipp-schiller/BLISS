"""
Animation extensions for ShapePlotter2D.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..inference.source_densitites import SourceDensity
from ..models.shapes.base_shape import Shape
from ..visualization.plotting_2d import ShapePlotter2D


class ShapePlotterAnimations:
    """
    Animation methods for 2D shape plotting.
    """

    def animate_param_history(self, shape: Shape, param_history: List[np.ndarray], ax=None, show = False):
        """
        Animates a sequence of shape parameters

        Parameters
        ----------
        shape
        param_history: List of parameters

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

        if show:
            plt.show()
        return ani


    def animate_source_density(self, shape: Shape, source_density: SourceDensity, param_history: List[np.ndarray], ax=None, show = False):
        """
        Animates a list of source_densities

        Parameters
        ----------
        source_density
        shape
        param_history
        ax
        show

        Returns
        -------
        """
        if ax is None:
            ax = plt.gcf().gca()

        fig = plt.gcf()

        # we use the first element in the history
        params = param_history[0]

        # define the parameter values
        t = np.linspace(
            float(np.atleast_1d(shape.domain_low)[0]),
            float(np.atleast_1d(shape.domain_high)[0]),
            300
        ).reshape(-1, 1)

        density_values = source_density.evaluate(t, params)

        (line,) = ax.plot(t, density_values, linewidth=2, color='purple')

        def update(frame):
            params = param_history[frame]
            density_values = source_density.evaluate(t, params)
            line.set_data(t.ravel(), density_values)
            return (line,)

        ani = FuncAnimation(fig, update, frames=len(param_history), blit=True, interval=500)

        if show:
            plt.show()
        return ani