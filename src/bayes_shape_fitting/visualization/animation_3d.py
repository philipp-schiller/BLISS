"""
Animation extensions for ShapePlotter3D.
"""

from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from ..inference.source_densitites import SourceDensity
from ..models.shapes.base_shape import Shape

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np

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



    def animate_optimization_result(
            self,
            result,
            shape,
            source_density=None,
            ax=None,
            n_points=50,
            interval=300,
            show=False,
    ):

        if result.shape_history is None:
            raise ValueError(
                "OptimizationResult does not contain shape_history"
            )

        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()

        # --------------------------------------------------
        # parameter domain
        # --------------------------------------------------

        domain_low = np.atleast_1d(shape.domain_low)
        domain_high = np.atleast_1d(shape.domain_high)

        u_vals = np.linspace(
            domain_low[0],
            domain_high[0],
            n_points,
        )

        v_vals = np.linspace(
            domain_low[1],
            domain_high[1],
            n_points,
        )

        U, V = np.meshgrid(u_vals, v_vals)

        t = np.stack(
            [U.ravel(), V.ravel()],
            axis=-1
        )

        du = u_vals[1] - u_vals[0]
        dv = v_vals[1] - v_vals[0]

        # --------------------------------------------------
        # density normalization across all frames
        # --------------------------------------------------

        density_min = None
        density_max = None

        if (
                source_density is not None
                and result.source_history is not None
        ):
            densities = []

            for source_params in result.source_history:
                density = source_density.evaluate(
                    t,
                    source_params,
                ).ravel()

                density /= density.sum() * du * dv

                densities.append(density)

            density_min = min(
                np.min(d)
                for d in densities
            )

            density_max = max(
                np.max(d)
                for d in densities
            )

            norm = Normalize(
                density_min,
                density_max,
            )

            cmap = plt.cm.viridis

            sm = ScalarMappable(
                norm=norm,
                cmap=cmap,
            )

            fig.colorbar(
                sm,
                ax=ax,
                label="Source density",
            )

        # --------------------------------------------------
        # initial surface
        # --------------------------------------------------

        shape_params = result.shape_history[0]

        points = shape.parametrization(
            t,
            shape_params,
        )

        X = points[:, 0].reshape(U.shape)
        Y = points[:, 1].reshape(U.shape)
        Z = points[:, 2].reshape(U.shape)

        if (
                source_density is not None
                and result.source_history is not None
        ):
            density = densities[0].reshape(U.shape)

            facecolors = cmap(
                norm(density)
            )

        else:
            facecolors = "cornflowerblue"

        surface_container = [
            ax.plot_surface(
                X,
                Y,
                Z,
                facecolors=facecolors,
                alpha=0.8,
                linewidth=0,
                shade=False,
            )
        ]

        # --------------------------------------------------
        # text
        # --------------------------------------------------

        txt = ax.text2D(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
            ),
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # --------------------------------------------------
        # fix axis limits
        # --------------------------------------------------

        all_points = []

        for params in result.shape_history:
            p = shape.parametrization(
                t,
                params,
            )
            all_points.append(p)

        all_points = np.vstack(all_points)

        ax.set_xlim(
            all_points[:, 0].min(),
            all_points[:, 0].max(),
        )

        ax.set_ylim(
            all_points[:, 1].min(),
            all_points[:, 1].max(),
        )

        ax.set_zlim(
            all_points[:, 2].min(),
            all_points[:, 2].max(),
        )

        ax.set_box_aspect(
            (
                np.ptp(all_points[:, 0]),
                np.ptp(all_points[:, 1]),
                np.ptp(all_points[:, 2]),
            )
        )

        # --------------------------------------------------
        # animation
        # --------------------------------------------------

        def update(frame):

            surface_container[0].remove()

            shape_params = result.shape_history[frame]

            points = shape.parametrization(
                t,
                shape_params,
            )

            X = points[:, 0].reshape(U.shape)
            Y = points[:, 1].reshape(U.shape)
            Z = points[:, 2].reshape(U.shape)

            kwargs = {}

            if (
                    source_density is not None
                    and result.source_history is not None
            ):
                density = densities[frame].reshape(
                    U.shape
                )

                kwargs["facecolors"] = cmap(
                    norm(density)
                )

                kwargs["shade"] = False

            else:
                kwargs["color"] = "cornflowerblue"

            surface_container[0] = ax.plot_surface(
                X,
                Y,
                Z,
                alpha=0.8,
                linewidth=0,
                **kwargs,
            )

            info = f"Iteration: {frame}"

            if (
                    hasattr(
                        result,
                        "log_posterior_history"
                    )
                    and result.log_posterior_history
            ):
                info += (
                    f"\nlog posterior: "
                    f"{result.log_posterior_history[frame]:.3f}"
                )

            if (
                    hasattr(
                        result,
                        "shape_history"
                    )
                    and result.shape_history
            ):
                info += (
                    f"\nShape params: "
                    f"{result.shape_history[frame]}"
                )

            if (
                    hasattr(
                        result,
                        "source_history"
                    )
                    and result.shape_history
            ):
                info += (
                    f"\nSource params: "
                    f"{result.source_history[frame]}"
                )

            if (
                    result.noise_history is not None
                    and frame < len(result.noise_history)
            ):
                noise = result.noise_history[frame]

                if noise is not None:
                    info += (
                            "\nnoise: "
                            + np.array2string(
                        noise,
                        precision=3,
                    )
                    )

            txt.set_text(info)

            return (
                surface_container[0],
                txt,
            )

        ani = FuncAnimation(
            fig,
            update,
            frames=len(
                result.shape_history
            ),
            interval=interval,
            blit=False,  # required for 3D
        )

        if show:
            plt.show()

        return ani