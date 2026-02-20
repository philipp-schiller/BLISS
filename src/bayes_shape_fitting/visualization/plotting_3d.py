import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Optional, Tuple
from dataclasses import dataclass

from src.bayes_shape_fitting.inference.likelihoods import Likelihood
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


class ShapePlotter3D:
    """
    Plotting utilities for 3D shapes (surfaces).

    Provides methods to visualize:
    - Shape surfaces (as wireframes)
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
                   ax: Optional[Axes3D] = None,
                   n_points: int = 50,
                   **kwargs) -> Tuple[plt.Figure, Axes3D]:
        """
        Plot a single shape surface as a wireframe.

        Parameters
        ----------
        shape : Shape
            Shape object to plot.
        params : np.ndarray
            Shape parameters.
        ax : Axes3D, optional
            Axes to plot on. Creates new figure if None.
        n_points : int, default=50
            Number of points per parameter dimension.
        **kwargs
            Additional arguments passed to plot_wireframe.

        Returns
        -------
        fig : plt.Figure
        ax : Axes3D
        """
        if ax is None:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        domain_low = np.atleast_1d(shape.domain_low)
        domain_high = np.atleast_1d(shape.domain_high)

        u_vals = np.linspace(domain_low[0], domain_high[0], n_points)
        v_vals = np.linspace(domain_low[1], domain_high[1], n_points)
        U, V = np.meshgrid(u_vals, v_vals)

        # Flatten to (M, 2) for parametrization, then reshape back
        t = np.stack([U.ravel(), V.ravel()], axis=-1)
        points = shape.parametrization(t, params)  # (M, 3)

        X = points[:, 0].reshape(U.shape)
        Y = points[:, 1].reshape(U.shape)
        Z = points[:, 2].reshape(U.shape)

        plot_kwargs = {
            'color': self.config.shape_color,
            'linewidth': 0.5,
            'alpha': 0.4,
        }
        plot_kwargs.update(kwargs)

        ax.plot_wireframe(X, Y, Z, **plot_kwargs)

        ax.set_xlabel('x', fontsize=self.config.label_fontsize)
        ax.set_ylabel('y', fontsize=self.config.label_fontsize)
        ax.set_zlabel('z', fontsize=self.config.label_fontsize)
        ax.grid(True, alpha=self.config.grid_alpha)

        ax.set_aspect("equal")

        return fig, ax

    def plot_observations(self,
                          observations: np.ndarray,
                          ax: Optional[Axes3D] = None,
                          title: Optional[str] = None) -> Tuple[plt.Figure, Axes3D]:
        """
        Plot observations in 3D.

        Parameters
        ----------
        observations : np.ndarray, shape (n_obs, 3)
            Observed points.
        ax : Axes3D, optional
            Axes to plot on.
        title : str, optional
            Plot title.

        Returns
        -------
        fig : plt.Figure
        ax : Axes3D
        """
        fig = plt.gcf()
        if ax is None:
            ax = fig.gca()

        ax.scatter(
            observations[:, 0],
            observations[:, 1],
            observations[:, 2],
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

    def plot_source_density(self,
                            shape: 'Shape',
                            source_density: 'SourceDensity',
                            params: np.ndarray,
                            ax: Optional[Axes3D] = None,
                            n_points: int = 50,
                            plot_on_shape: bool = False,
                            shape_params: Optional[np.ndarray] = None) -> Tuple[plt.Figure, Axes3D]:
        """
        Plot source density function.

        Can plot either:
        1. As a 2D heatmap over the (u, v) parameter space
        2. Color-coded on the shape surface itself

        Parameters
        ----------
        shape : Shape
            Shape object.
        source_density : SourceDensity
            Source density function.
        params : np.ndarray
            Parameters for source density function.
        ax : Axes3D, optional
            Axes to plot on.
        n_points : int, default=50
            Number of points per parameter dimension.
        plot_on_shape : bool, default=False
            If True, plot as colors on the shape surface.
            If False, plot as 2D heatmap over parameter space.
        shape_params : np.ndarray, optional
            Shape parameters (required if plot_on_shape=True).

        Returns
        -------
        fig : plt.Figure
        ax : Axes3D or plt.Axes
        """
        domain_low = np.atleast_1d(shape.domain_low)
        domain_high = np.atleast_1d(shape.domain_high)

        u_vals = np.linspace(domain_low[0], domain_high[0], n_points)
        v_vals = np.linspace(domain_low[1], domain_high[1], n_points)
        U, V = np.meshgrid(u_vals, v_vals)
        t = np.stack([U.ravel(), V.ravel()], axis=-1)  # (M, 2)

        # Evaluate source density
        density_values = source_density.evaluate(t, params)  # (M,)

        # Normalize to integrate to 1 over the 2D parameter domain
        du = u_vals[1] - u_vals[0]
        dv = v_vals[1] - v_vals[0]
        density_values = density_values / (density_values.sum() * du * dv)

        if plot_on_shape:
            if shape_params is None:
                raise ValueError("shape_params required when plot_on_shape=True")

            if ax is None:
                fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig = ax.get_figure()

            # Evaluate shape points
            points = shape.parametrization(t, shape_params)  # (M, 3)

            X = points[:, 0].reshape(U.shape)
            Y = points[:, 1].reshape(U.shape)
            Z = points[:, 2].reshape(U.shape)
            D = density_values.reshape(U.shape)

            norm = plt.Normalize(vmin=D.min(), vmax=D.max())
            colors = plt.cm.viridis(norm(D))  # (n, n, 4)

            surf = ax.plot_surface(X, Y, Z, facecolors=colors, shade=False, alpha=0.8)

            # Colorbar via ScalarMappable
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='Source density', shrink=0.5, pad=0.1)

            ax.set_xlabel('x', fontsize=self.config.label_fontsize)
            ax.set_ylabel('y', fontsize=self.config.label_fontsize)
            ax.set_zlabel('z', fontsize=self.config.label_fontsize)
            ax.set_title('Source Density on Shape', fontsize=self.config.title_fontsize)

        else:
            # Plot as 2D heatmap over (u, v) parameter space
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

            D = density_values.reshape(U.shape)
            im = ax.contourf(U, V, D, levels=50, cmap='viridis')
            fig.colorbar(im, ax=ax, label='Source density')

            ax.set_xlabel('u', fontsize=self.config.label_fontsize)
            ax.set_ylabel('v', fontsize=self.config.label_fontsize)
            ax.set_title('Source Density over Parameter Space', fontsize=self.config.title_fontsize)

        ax.grid(True, alpha=self.config.grid_alpha)

        return fig, ax

    def plot_likelihood(self,
                        likelihood: Likelihood,
                        theta: np.ndarray,
                        ax: Optional[Axes3D] = None,
                        x_min: float = -5, x_max: float = 5,
                        y_min: float = -5, y_max: float = 5,
                        z_min: float = -5, z_max: float = 5,
                        n_grid: int = 30,
                        plot_measurement: bool = False,
                        measurement: Optional[np.ndarray] = None) -> Tuple[plt.Figure, Axes3D]:
        """
        Plot log-likelihood as a volume in 3D using a scatter of grid points
        colored by their log-likelihood value.

        Parameters
        ----------
        likelihood : Likelihood
            Likelihood object.
        theta : np.ndarray
            Shape/likelihood parameters.
        ax : Axes3D, optional
            Axes to plot on.
        x_min, x_max, y_min, y_max, z_min, z_max : float
            Bounds of the 3D grid.
        n_grid : int, default=30
            Number of grid points per dimension.
        plot_measurement : bool, default=False
            If True, overlay measurement points.
        measurement : np.ndarray, optional
            Observed points, shape (N, 3). Required if plot_measurement=True.

        Returns
        -------
        fig : plt.Figure
        ax : Axes3D
        """
        if ax is None:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # Build 3D grid
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.linspace(y_min, y_max, n_grid)
        z_grid = np.linspace(z_min, z_max, n_grid)
        Xg, Yg, Zg = np.meshgrid(x_grid, y_grid, z_grid)
        sources = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)  # (M, 3)

        likelihood_values = likelihood.log_likelihood(theta=theta, measurement=sources)  # (M,)

        norm = plt.Normalize(vmin=likelihood_values.min(), vmax=likelihood_values.max())
        colors = plt.cm.viridis(norm(likelihood_values))

        sc = ax.scatter(
            sources[:, 0], sources[:, 1], sources[:, 2],
            c=likelihood_values, cmap='viridis',
            alpha=0.1, s=5
        )
        fig.colorbar(sc, ax=ax, label='Log-likelihood', shrink=0.5, pad=0.1)

        if plot_measurement:
            if measurement is None:
                raise ValueError("measurement required when plot_measurement=True")

            likelihood_measurement = likelihood.log_likelihood(theta=theta, measurement=measurement)

            ax.scatter(
                measurement[:, 0], measurement[:, 1], measurement[:, 2],
                c=likelihood_measurement, cmap='plasma',
                s=self.config.obs_size, edgecolors='black',
                linewidths=0.5, label='Measurements'
            )
            ax.legend(fontsize=self.config.legend_fontsize)

        ax.set_xlabel('x', fontsize=self.config.label_fontsize)
        ax.set_ylabel('y', fontsize=self.config.label_fontsize)
        ax.set_zlabel('z', fontsize=self.config.label_fontsize)
        ax.set_title('Log-Likelihood Volume', fontsize=self.config.title_fontsize)
        ax.grid(True, alpha=self.config.grid_alpha)

        return fig, ax