from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..inference.likelihoods import Likelihood
from ..inference.noise_functions import NoiseFunction
from ..inference.source_densitites import SourceDensity
from ..models.shapes.base_shape import Shape


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
                   **kwargs) -> Tuple[plt.Figure, plt.Axes, plt.Line2D]:
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
        line, = ax.plot(points[:, 0], points[:, 1], **plot_kwargs)

        ax.grid(True, alpha=self.config.grid_alpha)
        ax.set_aspect('equal', adjustable='box')

        return fig, ax, line

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
        if ax is None:
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

        # ax.legend(fontsize=self.config.legend_fontsize)

        return fig, ax

    def plot_normals_(
            self,
            measurement: np.ndarray,
            normals: np.ndarray,
            normal_var: np.ndarray = None,  # (N,) optional — from BayesianNormalResult
            ax=None,
            scale: float = 0.1,
            every: int = 1,
            **kwargs,
    ):
        N, d = measurement.shape
        idx = np.arange(0, N, every)
        pts = measurement[idx]
        nrm = normals[idx]

        bbox_diag = np.linalg.norm(measurement.max(axis=0) - measurement.min(axis=0))
        arrow_len = scale * bbox_diag
        nrm_scaled = nrm * arrow_len

        base_color = kwargs.pop("color", self.config.shape_color)

        if d == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            else:
                fig = ax.get_figure()

            if normal_var is not None:
                # colour arrows by uncertainty: low var = blue, high var = red
                var_sub = normal_var[idx]
                var_norm = (var_sub - var_sub.min()) / (var_sub.max() - var_sub.min() + 1e-10)
                colors = plt.cm.RdYlBu_r(var_norm)
                for j in range(len(idx)):
                    ax.annotate(
                        "", xy=(pts[j, 0] + nrm_scaled[j, 0],
                                pts[j, 1] + nrm_scaled[j, 1]),
                        xytext=(pts[j, 0], pts[j, 1]),
                        arrowprops=dict(arrowstyle="->", color=colors[j], lw=1.2),
                    )
                sm = plt.cm.ScalarMappable(cmap="RdYlBu_r",
                                           norm=plt.Normalize(var_sub.min(), var_sub.max()))
                plt.colorbar(sm, ax=ax, label="normal variance", shrink=0.6)
            else:
                ax.quiver(pts[:, 0], pts[:, 1],
                          nrm_scaled[:, 0], nrm_scaled[:, 1],
                          angles="xy", scale_units="xy", scale=1,
                          color=base_color, alpha=0.7, **kwargs)

            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=base_color, alpha=0.5, zorder=3)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=self.config.grid_alpha)

        else:  # d == 3
            if ax is None:
                fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig = ax.get_figure()

            if normal_var is not None:
                var_sub = normal_var[idx]
                var_norm = (var_sub - var_sub.min()) / (var_sub.ptp() + 1e-10)
                colors = plt.cm.RdYlBu_r(var_norm)
            else:
                colors = base_color

            ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2],
                      nrm_scaled[:, 0], nrm_scaled[:, 1], nrm_scaled[:, 2],
                      colors=colors, alpha=0.7, **kwargs)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=8, color=base_color, alpha=0.5, zorder=3)

        return fig, ax

    def plot_normals(
            self,
            measurement: np.ndarray,  # (N, d)
            normals: np.ndarray,  # (N, d)
            uncertainties: list | None = None,  # list of uncertainty objects
            ax=None,
            scale: float = 0.1,
            every: int = 1,
            color: str = "#4C72B0",
            cmap: str = "plasma",
            **kwargs,
    ):
        """
        Plot normals (and optional uncertainty) for a 2D or 3D point cloud.

        Uncertainty is extracted uniformly via ``unc.to_covariance()`` and
        ``unc.scalar_uncertainty()``, which every built-in uncertainty type
        implements.  Custom estimators only need to expose those two methods.

        Parameters
        ----------
        measurement    : (N, d) point cloud.
        normals        : (N, d) unit normals from compute_normals_bayes().
        uncertainties  : list of N uncertainty objects (BinghamUncertainty,
                         DeltaUncertainty, PlanarityUncertainty, or custom).
                         If None, normals are drawn without uncertainty.
        ax             : existing matplotlib Axes (2D) or Axes3D (3D).  Created
                         if None.
        scale          : normal arrow length as a fraction of the point-cloud
                         bounding-box diagonal.
        every          : plot every n-th point (subsampling for dense clouds).

        color          : base color for points and arrows.
        cmap           : matplotlib colormap name for scalar uncertainty encoding.

        Returns
        -------
        fig, ax
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        measurement = np.asarray(measurement)
        normals = np.asarray(normals)
        N, d = measurement.shape

        if d not in (2, 3):
            raise ValueError("Only 2D/3D supported.")

        idx = np.arange(0, N, every)
        pts = measurement[idx]
        nrm = normals[idx]

        bbox_diag = np.linalg.norm(measurement.max(axis=0) - measurement.min(axis=0))
        arrow_len = scale * bbox_diag

        # ------------------------------------------------------------------
        # Extract per-point covariance and scalar uncertainty (if available)
        # ------------------------------------------------------------------
        covs = None
        scalars = None

        if uncertainties is not None:
            unc_sub = [uncertainties[i] for i in idx]

            covs = np.array([u.to_covariance() for u in unc_sub])  # (M, d, d)
            scalars = np.array([u.scalar_uncertainty() for u in unc_sub])  # (M,)

            # Log-scale for better colormap contrast
            log_scalars = np.log(scalars + 1e-12)
            norm_scalar = mcolors.Normalize(vmin=log_scalars.min(),
                                            vmax=log_scalars.max())
            cmap_obj = cm.get_cmap(cmap)

        # ------------------------------------------------------------------
        # 2D
        # ------------------------------------------------------------------
        if d == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            else:
                fig = ax.get_figure()

            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.5, zorder=2)

            for j in range(len(idx)):
                p = pts[j]
                n = nrm[j]
                tip = p + arrow_len * n

                arrow_color = (cmap_obj(norm_scalar(log_scalars[j]))
                               if covs is not None else color)

                ax.annotate(
                    "",
                    xy=(tip[0], tip[1]),
                    xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                    zorder=3,
                )

                if covs is not None:
                    C = covs[j]  # (2, 2)
                    eigvals_c, eigvecs_c = np.linalg.eigh(C)
                    order = np.argsort(eigvals_c)[::-1]
                    eigvals_c = eigvals_c[order]
                    eigvecs_c = eigvecs_c[:, order]

                    radii = np.sqrt(np.maximum(eigvals_c, 0.0)) * arrow_len

                    theta = np.linspace(0, 2 * np.pi, 60)
                    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
                    ellipse = circle @ np.diag(radii) @ eigvecs_c.T + tip

                    ax.fill(ellipse[:, 0], ellipse[:, 1],
                            color=arrow_color, alpha=0.18, zorder=1)
                    ax.plot(ellipse[:, 0], ellipse[:, 1],
                            color=arrow_color, alpha=0.6, lw=0.8, zorder=1)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        # Colorbar
        if uncertainties is not None:
            sm = cm.ScalarMappable(norm=norm_scalar, cmap=cmap_obj)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Uncertainty (log scale)", shrink=0.6)

        return fig, ax

    def plot_normals_old(
            self,
            measurement: np.ndarray,
            normals: np.ndarray,
            normal_cov: np.ndarray = None,  # (N,d,d)
            ax=None,
            scale: float = 0.1,
            every: int = 1,
            cone_scale: float = 1.0,
            n_cone_samples: int = 20,
            **kwargs,
    ):

        N, d = measurement.shape
        idx = np.arange(0, N, every)

        pts = measurement[idx]
        nrm = normals[idx]

        bbox_diag = np.linalg.norm(measurement.max(axis=0) - measurement.min(axis=0))
        arrow_len = scale * bbox_diag
        nrm_scaled = normals[idx] * arrow_len

        base_color = kwargs.pop("color", self.config.shape_color)

        # =========================
        # ======== 2D CASE ========
        # =========================
        if d == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            else:
                fig = ax.get_figure()

            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=base_color, alpha=0.5)

            for j in range(len(idx)):
                p = pts[j]
                n = nrm[j]

                tip = p + arrow_len * n

                # draw normal
                ax.plot([p[0], tip[0]], [p[1], tip[1]], color=base_color)


                if normal_cov is not None:
                    import matplotlib.cm as cm
                    import matplotlib.colors as mcolors

                    C = normal_cov[idx[j]]

                    unc = np.array([np.trace(normal_cov[i]) for i in idx])

                    # optional (recommended): log-scale
                    unc = np.log(unc + 1e-12)

                    norm = mcolors.Normalize(vmin=unc.min(), vmax=unc.max())
                    cmap = cm.viridis

                    eigvals, eigvecs = np.linalg.eigh(C)
                    order = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]

                    radii = np.sqrt(np.maximum(eigvals, 0)) * arrow_len

                    theta = np.linspace(0, 2 * np.pi, 50)
                    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

                    ellipse = circle @ np.diag(radii) @ eigvecs.T

                    # ✅ centered at tip
                    ellipse += (p + nrm_scaled[j])

                    #ax.plot(ellipse[:, 0], ellipse[:, 1], color="red", alpha=0.7)

                    color = cmap(norm(unc[j])) if normal_cov is not None else base_color

                    ax.annotate(
                        "",
                        xy=(p[0] + nrm_scaled[j, 0], p[1] + nrm_scaled[j, 1]),
                        xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    )

            ax.set_aspect("equal")
            ax.grid(True, alpha=self.config.grid_alpha)

        # =========================
        # ======== 3D CASE ========
        # =========================
        elif d == 3:
            if ax is None:
                fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig = ax.get_figure()

            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=8, color=base_color, alpha=0.5)

            for j in range(len(idx)):
                p = pts[j]
                n = nrm[j]

                tip = p + arrow_len * n

                # normal line
                ax.plot([p[0], tip[0]],
                        [p[1], tip[1]],
                        [p[2], tip[2]],
                        color=base_color)

                if normal_cov is not None:
                    C = normal_cov[idx[j]]

                    # --- compute tangent basis ---
                    # pick arbitrary perpendicular vectors
                    u = np.cross(n, np.array([1, 0, 0]))
                    if np.linalg.norm(u) < 1e-6:
                        u = np.cross(n, np.array([0, 1, 0]))
                    u /= np.linalg.norm(u)

                    v = np.cross(n, u)

                    # project covariance to tangent plane
                    T = np.stack([u, v], axis=1)  # (3,2)
                    C_tan = T.T @ C @ T

                    # eigen decomposition in tangent plane
                    eigvals, eigvecs = np.linalg.eigh(C_tan)

                    # ellipse radii → angular uncertainty
                    radii = np.sqrt(np.maximum(eigvals, 0)) * cone_scale

                    # sample ellipse in tangent plane
                    theta = np.linspace(0, 2 * np.pi, n_cone_samples)

                    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
                    ellipse = circle @ np.diag(radii) @ eigvecs.T

                    # map back to 3D directions
                    directions = (T @ ellipse.T).T + n

                    # normalize directions
                    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

                    # scale + translate
                    cone_pts = tip + arrow_len * 0.5 * directions

                    # draw fan / cone boundary
                    ax.plot(cone_pts[:, 0],
                            cone_pts[:, 1],
                            cone_pts[:, 2],
                            color="red", alpha=0.7)

                    # optional: fill cone (light)
                    for k in range(len(cone_pts)):
                        ax.plot(
                            [tip[0], cone_pts[k, 0]],
                            [tip[1], cone_pts[k, 1]],
                            [tip[2], cone_pts[k, 2]],
                            color="green", alpha=0.2
                        )

        else:
            raise ValueError("Only 2D/3D supported")

        if normal_cov is not None:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Normal uncertainty (log trace)")

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

    def plot_covariances(self, shape: Shape, shape_params: np.ndarray, source_density: SourceDensity, noise_function: NoiseFunction, color = "green",  label ="Uncertainty"):

        # sample_tvals
        t_vals = source_density.sample(n = 100)
        # compute source points
        points = shape.parametrization(t_vals, shape_params)

        covariances = noise_function.evaluate(t_vals)

        for i in range(len(points)):
            cov = covariances[i]
            center = points[i]

            # Eigen-decomposition of covariance matrix
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            # Compute ellipse angle in degrees
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            # Width and height = 2 * sqrt(eigenvalues)
            width, height = 2 * np.sqrt(vals)

            ellip_label = label if i == 0 else None

            # Create and add ellipse
            ellip = patches.Ellipse(xy=center, width=width, height=height, angle=angle,
                                    edgecolor=color, facecolor=color, lw=0.1, alpha=0.1, label=ellip_label)
            plt.gca().add_patch(ellip)

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

    def plot_likelihood(self, likelihood: Likelihood, theta: np.ndarray, ax=None, x_min = -5, x_max = 5, y_min = -5, y_max = 5, plot_measurement = False, measurement = None) -> Tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()

        # Resolution of grid
        n_grid = 150
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.linspace(y_min, y_max, n_grid)
        X, Y = np.meshgrid(x_grid, y_grid)
        sources = np.stack([X.ravel(), Y.ravel()], axis=1)  # (M, 2)

        likelihood_values = np.exp(likelihood.log_likelihood(shape_params=theta, measurement=sources))

        likelihood_values = likelihood_values.reshape(X.shape)

        # --- Plot on the provided axes ---
        im = ax.contourf(X, Y, likelihood_values, levels=50, cmap='viridis')

        plt.colorbar(im, fraction=0.046)

        if plot_measurement:
            if measurement is None:
                raise ValueError("measurement required when plot_measurement=True")

            # first, plot the measurement
            sc = ax.scatter(measurement[:, 0], measurement[:, 1], s=7, c='brown',
                            edgecolors='k', linewidths=0.5, label="Measurements")

            # compute the likelihoods of the measurement
            likelihood_measurement = np.exp(likelihood.log_likelihood(shape_params= theta, measurement=measurement))

            # we want to be able to hover over points to see their respective likelihood
            tooltip_css = """
                        .mpld3-tooltip {
                            background: white;
                            color: #222;                 /* text color */
                            border: 1px solid rgba(0,0,0,0.25);
                            border-radius: 6px;
                            padding: 6px 8px;
                            font: 12px/1.3 monospace;    /* match your code-like style */
                            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                            pointer-events: none;        /* don't block mouse events */
                            z-index: 9999;
                        }
                        """

            labels = [
                f"""<div>
                                   idx: {i}<br>
                                   x: {x:.4g}<br>
                                   y: {y:.4g}<br>
                                   ℓ: <b>{l:.6g}</b>
                                </div>"""
                for i, ((x, y), l) in enumerate(zip(measurement, likelihood_measurement))
            ]

            # Attach mpld3 tooltip plugin
            from mpld3 import plugins

            tooltip = plugins.PointHTMLTooltip(
                sc, labels=labels, voffset=10, hoffset=10, css=tooltip_css
            )
            plugins.connect(ax.figure, tooltip)

            # Create a single annotation to reuse on hover
            annot = ax.annotate(
                "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="0.5")
            )
            annot.set_visible(False)

            offsets = sc.get_offsets()  # Nx2
            N = offsets.shape[0]

            def update_annotation(ind):
                # ind is a dict from PathCollection.contains(event)
                # Use the first index under the cursor
                idx = ind["ind"][0]
                (x, y) = offsets[idx]
                # Update annotation position and text
                annot.xy = (x, y)
                # Format text; adjust precision as you prefer
                text = f"index: {idx}\n(x, y)=({x:.4g}, {y:.4g})\nℓ={likelihood_measurement[idx]:.6g}"
                annot.set_text(text)

            def hover(event):
                if event.inaxes != ax:
                    # Hide if cursor leaves the axes
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
                    return

                # Check if the mouse is over any scatter point
                contains, ind = sc.contains(event)
                if contains and ind and "ind" in ind and len(ind["ind"]) > 0:
                    update_annotation(ind)
                    if not annot.get_visible():
                        annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    # Hide annotation if not hovering any point
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

            # Connect the motion event to the hover function
            fig.canvas.mpl_connect("motion_notify_event", hover)

        return fig, ax