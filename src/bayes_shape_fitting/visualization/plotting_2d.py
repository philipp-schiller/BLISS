import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from src.bayes_shape_fitting.inference.source_densitites import SourceDensity


@dataclass
class PlotConfig:
    """Configuration for plot aesthetics."""
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 100
    shape_color: str = 'blue'
    shape_linewidth: float = 2.0
    obs_color: str = 'red'
    obs_alpha: float = 0.5
    obs_size: float = 30
    grid_alpha: float = 0.3
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10

class ShapePlotter2D:
    """
    Plotting utilities for 2D shapes (curves).

    Provides methods to visualize:
    - Shape curves
    - Observations
    - Source densities
    - Multiple shapes (e.g., true vs fitted)

    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize plotter.

        Parameters
        ----------
        config : PlotConfig, optional
            Plot configuration. Uses defaults if not provided.
        """
        self.config = config or PlotConfig()

    def plot_shape(self,
                   shape: 'Shape',
                   params: np.ndarray,
                   ax: Optional[plt.Axes] = None,
                   n_points: int = 200,
                   **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single shape curve.

        Parameters
        ----------
        shape : Shape
            Shape object to plot.
        params : np.ndarray
            Shape parameters.
        ax : plt.Axes, optional
            Axes to plot on. Creates new figure if None.
        n_points : int, default=200
            Number of points to evaluate on the curve.
        **kwargs
            Additional arguments passed to plt.plot.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()

        # Generate parameter values
        t = np.linspace(
            float(np.atleast_1d(shape.domain_low)[0]),
            float(np.atleast_1d(shape.domain_high)[0]),
            n_points
        ).reshape(-1, 1)

        # Evaluate shape
        points = shape.parametrization(t, params)

        # Set default plotting kwargs
        plot_kwargs = {
            'color': self.config.shape_color,
            'linewidth': self.config.shape_linewidth,
        }
        plot_kwargs.update(kwargs)

        # Plot
        ax.plot(points[:, 0], points[:, 1], **plot_kwargs)

        ax.set_xlabel('x', fontsize=self.config.label_fontsize)
        ax.set_ylabel('y', fontsize=self.config.label_fontsize)
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.set_aspect('equal', adjustable='box')

        return fig, ax

    def plot_observations(self,     observations: np.ndarray,
                                    ax: Optional[plt.Axes] = None,
                                    title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot shape with observations overlaid.

        Parameters
        ----------
        observations : np.ndarray, shape (n_obs, 2)
            Observed points.
        ax : plt.Axes, optional
            Axes to plot on.
        title : str, optional
            Plot title.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes
        """
        #fig, ax = self.plot_shape(shape, params, ax=ax, label='Fitted shape')
        fig = plt.gcf()
        ax = fig.gca()

        # Plot observations
        ax.scatter(
            observations[:, 0],
            observations[:, 1],
            c=self.config.obs_color,
            alpha=self.config.obs_alpha,
            s=self.config.obs_size,
            label='Observations',
            edgecolors='black',
            linewidths=0.5
        )

        if title:
            ax.set_title(title, fontsize=self.config.title_fontsize)

        ax.legend(fontsize=self.config.legend_fontsize)

        return fig, ax

    def plot_multiple_shapes(self,
                             shapes_and_params: List[Tuple['Shape', np.ndarray, str, Dict]],
                             observations: Optional[np.ndarray] = None,
                             ax: Optional[plt.Axes] = None,
                             title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot multiple shapes (e.g., true vs MAP vs posterior samples).

        Parameters
        ----------
        shapes_and_params : list of tuples
            Each tuple is (shape, params, label, plot_kwargs).
            Example: [(circle, true_params, 'True', {'color': 'green', 'linestyle': '--'}),
                     (circle, map_params, 'MAP', {'color': 'red'})]
        observations : np.ndarray, optional
            Observed points to overlay.
        ax : plt.Axes, optional
            Axes to plot on.
        title : str, optional
            Plot title.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes

        Examples
        --------
        >>> plotter.plot_multiple_shapes([
        ...     (circle, true_params, 'True', {'color': 'green', 'linestyle': '--'}),
        ...     (circle, map_params, 'MAP', {'color': 'red'}),
        ...     (circle, mean_params, 'Posterior Mean', {'color': 'blue'})
        ... ], observations=obs)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()

        # Plot each shape
        for shape, params, label, plot_kwargs in shapes_and_params:
            self.plot_shape(shape, params, ax=ax, label=label, **plot_kwargs)

        # Plot observations if provided
        if observations is not None:
            ax.scatter(
                observations[:, 0],
                observations[:, 1],
                c=self.config.obs_color,
                alpha=self.config.obs_alpha,
                s=self.config.obs_size,
                label='Observations',
                edgecolors='black',
                linewidths=0.5,
                zorder=10
            )

        if title:
            ax.set_title(title, fontsize=self.config.title_fontsize)

        ax.legend(fontsize=self.config.legend_fontsize)
        ax.set_xlabel('x', fontsize=self.config.label_fontsize)
        ax.set_ylabel('y', fontsize=self.config.label_fontsize)
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.set_aspect('equal', adjustable='box')

        return fig, ax


    def plot_source_density(self,
                            shape: 'Shape',
                            source_density: 'SourceDensity',
                            params: np.ndarray,
                            ax: Optional[plt.Axes] = None,
                            n_points: int = 500,
                            plot_on_shape: bool = False,
                            shape_params: Optional[np.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot source density function.

        Can plot either:
        1. As function of parameter t (1D plot)
        2. Color-coded on the shape itself (2D plot)

        Parameters
        ----------
        shape : Shape
            Shape object.
        source_density : ParametricFunction
            Source density function.
        params : np.ndarray
            Parameters for source density function.
        ax : plt.Axes, optional
            Axes to plot on.
        n_points : int, default=500
            Number of points to evaluate.
        plot_on_shape : bool, default=False
            If True, plot as colors on the shape. If False, plot as 1D curve.
        shape_params : np.ndarray, optional
            Shape parameters (required if plot_on_shape=True).

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()

        # Generate parameter values
        t = np.linspace(
            float(np.atleast_1d(shape.domain_low)[0]),
            float(np.atleast_1d(shape.domain_high)[0]),
            n_points
        ).reshape(-1, 1)

        # Evaluate source density
        density_values = source_density.evaluate(t, params)

        # Normalize to integrate to 1
        dt = (t[1, 0] - t[0, 0])
        density_values = density_values / (density_values.sum() * dt)

        if plot_on_shape:
            if shape_params is None:
                raise ValueError("shape_params required when plot_on_shape=True")

            # Plot as colored line on shape
            points = shape.parametrization(t, shape_params)

            # Create line collection with colors
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
            norm = plt.Normalize(vmin=density_values.min(), vmax=density_values.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=3)
            lc.set_array(density_values[:-1])

            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax, label='Source density')

            ax.set_xlabel('x', fontsize=self.config.label_fontsize)
            ax.set_ylabel('y', fontsize=self.config.label_fontsize)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('Source Density on Shape', fontsize=self.config.title_fontsize)

        else:
            # Plot as 1D function
            ax.plot(t, density_values, linewidth=2, color='purple')
            ax.fill_between(t.ravel(), density_values, alpha=0.3, color='purple')
            ax.set_xlabel('Parameter t', fontsize=self.config.label_fontsize)
            ax.set_ylabel('Density', fontsize=self.config.label_fontsize)
            ax.set_title('Source Density Function', fontsize=self.config.title_fontsize)

        ax.grid(True, alpha=self.config.grid_alpha)

        return fig, ax