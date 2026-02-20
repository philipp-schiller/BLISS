"""
Animation extensions for ShapePlotter3D.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple
from src.bayes_shape_fitting.models.shapes.base_shape import Shape
from src.bayes_shape_fitting.inference.likelihoods import Likelihood
from src.bayes_shape_fitting.inference.source_densitites import SourceDensity
from src.bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D


class ShapePlotterAnimations3D:
    """
    Animation methods for 3D shape plotting.
    """

    def animate_param_history(self,
                               shape: Shape,
                               param_history: List[np.ndarray],
                               ax: Optional[Axes3D] = None,
                               n_points: int = 50,
                               show: bool = False) -> FuncAnimation:
        """
        Animates a sequence of shape parameters over a 3D surface.

        Parameters
        ----------
        shape : Shape
            Shape object to plot.
        param_history : List[np.ndarray]
            List of parameter arrays, one per frame.
        ax : Axes3D, optional
            Axes to plot on. Creates new figure if None.
        n_points : int, default=50
            Number of points per parameter dimension.
        show : bool, default=False
            If True, call plt.show() after creating animation.

        Returns
        -------
        ani : FuncAnimation
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        domain_low = np.atleast_1d(shape.domain_low)
        domain_high = np.atleast_1d(shape.domain_high)

        u_vals = np.linspace(domain_low[0], domain_high[0], n_points)
        v_vals = np.linspace(domain_low[1], domain_high[1], n_points)
        U, V = np.meshgrid(u_vals, v_vals)
        t = np.stack([U.ravel(), V.ravel()], axis=-1)  # (M, 2)

        # Draw initial frame
        points = shape.parametrization(t, param_history[0])
        X = points[:, 0].reshape(U.shape)
        Y = points[:, 1].reshape(U.shape)
        Z = points[:, 2].reshape(U.shape)

        # plot_surface returns a Poly3DCollection — we remove and redraw each frame
        # since Poly3DCollection does not support in-place updates like Line2D
        surf_container = [ax.plot_surface(X, Y, Z, color='blue', alpha=0.4, linewidth=0.5)]

        # Fix axis limits from first frame so they don't jump around
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_aspect("equal")

        def update(frame):
            # Remove previous surface
            surf_container[0].remove()

            params = param_history[frame]
            points = shape.parametrization(t, params)
            X = points[:, 0].reshape(U.shape)
            Y = points[:, 1].reshape(U.shape)
            Z = points[:, 2].reshape(U.shape)

            surf_container[0] = ax.plot_surface(X, Y, Z, color='blue', alpha=0.4, linewidth=0.5)
            ax.set_title(f'Frame {frame + 1} / {len(param_history)}')
            return surf_container[0],

        ani = FuncAnimation(fig, update, frames=len(param_history), interval=500, blit=False)
        # Note: blit=False is required for 3D axes — matplotlib's blit mechanism
        # does not support Axes3D

        if show:
            plt.show()
        return ani

    def animate_source_density(self,
                               shape: Shape,
                               source_density: SourceDensity,
                               param_history: List[np.ndarray],
                               n_points: int = 50,
                               show: bool = False) -> FuncAnimation:

        fig, ax2d = plt.subplots(figsize=(7, 6))

        domain_low = np.atleast_1d(shape.domain_low)
        domain_high = np.atleast_1d(shape.domain_high)

        u_vals = np.linspace(domain_low[0], domain_high[0], n_points)
        v_vals = np.linspace(domain_low[1], domain_high[1], n_points)
        U, V = np.meshgrid(u_vals, v_vals)
        t = np.stack([U.ravel(), V.ravel()], axis=-1)  # (M, 2)

        du = u_vals[1] - u_vals[0]
        dv = v_vals[1] - v_vals[0]

        # Precompute all densities
        all_densities = []
        for params in param_history:
            density = source_density.evaluate(t, params)
            density = density / (density.sum() * du * dv)
            all_densities.append(density)

        # Global norm across all frames
        global_min = min(d.min() for d in all_densities)
        global_max = max(d.max() for d in all_densities)
        norm = plt.Normalize(vmin=global_min, vmax=global_max)
        cmap = plt.cm.viridis

        # Initial frame
        D = all_densities[0].reshape(U.shape)
        contour_container = [ax2d.contourf(U, V, D, levels=50, cmap=cmap, norm=norm)]

        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax2d, label='Source density')
        ax2d.set_xlabel('u')
        ax2d.set_ylabel('v')

        def update(frame):
            for coll in contour_container[0].collections:
                coll.remove()
            D = all_densities[frame].reshape(U.shape)
            contour_container[0] = ax2d.contourf(U, V, D, levels=50, cmap=cmap, norm=norm)
            ax2d.set_title(f'Frame {frame + 1} / {len(param_history)}')
            return []

        ani = FuncAnimation(fig, update, frames=len(param_history), interval=500, blit=False)

        if show:
            plt.show()
        return ani