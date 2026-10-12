"""
Animation extensions for ShapePlotter2D.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from ..inference.source_densitites import SourceDensity
from ..models.shapes.base_shape import Shape
from ..visualization.plotting_2d import ShapePlotter2D


class ShapePlotterAnimations:
    """
    Animation methods for 2D shape plotting.
    """

    from typing import List
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    def animate_param_history(
            self,
            shape: Shape,
            param_history: List[np.ndarray],
            source_history: List[np.ndarray] = None,
            noise_history: List[np.ndarray] = None,
            ax=None,
            show=False
    ):
        """
        Animates shape evolution and displays source/noise parameters.

        Parameters
        ----------
        shape
        shape_history : list of shape parameter arrays
        source_history : list or None
        noise_history : list or None
        """

        if ax is None:
            ax = plt.gcf().gca()

        fig = ax.get_figure()
        plotter = ShapePlotter2D()
        params = param_history[0]
        _, _, line = plotter.plot_shape(shape=shape, params=params, ax=ax)


        # parameter grid for plotting shape
        t = np.linspace(
            float(np.atleast_1d(shape.domain_low)[0]),
            float(np.atleast_1d(shape.domain_high)[0]),
            300
        ).reshape(-1, 1)

        # ---- text box for source + noise ----
        text = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        def update(frame):
            # --- update shape ---
            params = param_history[frame]
            points = shape.parametrization(t, params)
            line.set_data(points[:, 0], points[:, 1])

            # --- build text ---
            txt = f"Iter: {frame}"

            if source_history is not None:
                s = source_history[frame]
                txt += f"\nsource: {np.array2string(s, precision=3)}"

            if noise_history is not None:
                n = noise_history[frame]
                txt += f"\nnoise: {np.array2string(n, precision=3)}"

            text.set_text(txt)

            return line, text

        ani = FuncAnimation(
            fig,
            update,
            frames=len(param_history),
            blit=True,
            interval=500
        )

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

    def animate_optimization_result(
        self,
        result,
        shape,
        source_density=None,
        noise_function=None,
        ax=None,
        interval=300,
        show=False,
    ):

        if result.shape_history is None:
            raise ValueError("OptimizationResult does not contain shape_history")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()

        # --------------------------------------------------
        # parameter values
        # --------------------------------------------------

        t = np.linspace(
            float(np.atleast_1d(shape.domain_low)[0]),
            float(np.atleast_1d(shape.domain_high)[0]),
            500,
        ).reshape(-1, 1)

        # --------------------------------------------------
        # initial shape
        # --------------------------------------------------

        shape_params = result.shape_history[0]
        points = shape.parametrization(t, shape_params)

        # line collection
        segments = np.concatenate(
            [points[:-1, None, :], points[1:, None, :]],
            axis=1
        )

        if source_density is not None:

            source_params = result.source_history[0]

            density = source_density.evaluate(
                t,
                source_params
            ).ravel()

            density /= np.trapezoid(density, t.ravel())

        else:

            density = np.ones(len(t))

        norm = plt.Normalize(
            vmin=np.min(density),
            vmax=np.max(density)
        )

        lc = LineCollection(
            segments,
            cmap="viridis",
            norm=norm,
            linewidth=4,
        )

        lc.set_array(density[:-1])

        ax.add_collection(lc)

        cbar = fig.colorbar(
            lc,
            ax=ax,
            label="Source density"
        )

        # --------------------------------------------------
        # covariance ellipses
        # --------------------------------------------------

        cov_ellipses = []

        if noise_function is not None:

            t_cov = np.linspace(
                float(np.atleast_1d(shape.domain_low)[0]),
                float(np.atleast_1d(shape.domain_high)[0]),
                50,
            ).reshape(-1, 1)

            covariances = noise_function.evaluate(
                t_cov,
                params = result.noise_history[0],
                shape_params=shape_params,  # if required
            )

            sources = shape.parametrization(t_cov, shape_params)

            for point, cov in zip(sources, covariances):
                vals, vecs = np.linalg.eigh(cov)

                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]

                angle = np.degrees(
                    np.arctan2(vecs[1, 0], vecs[0, 0])
                )

                ell = Ellipse(
                    xy=point,
                    width=2 * np.sqrt(vals[0]),
                    height=2 * np.sqrt(vals[1]),
                    angle=angle,
                    edgecolor="green",
                    facecolor="green",
                    alpha=0.15,
                    lw=0.5,
                )

                ax.add_patch(ell)
                cov_ellipses.append(ell)

        # --------------------------------------------------
        # text info
        # --------------------------------------------------

        txt = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.8
            ),
        )

        ax.set_aspect("equal")
        ax.autoscale_view()

        # --------------------------------------------------
        # animation
        # --------------------------------------------------

        def update(frame):

            shape_params = result.shape_history[frame]

            points = shape.parametrization(
                t,
                shape_params
            )

            segments = np.concatenate(
                [points[:-1, None, :],
                 points[1:, None, :]],
                axis=1
            )

            lc.set_segments(segments)

            if (
                source_density is not None
                and result.source_history is not None
            ):

                source_params = result.source_history[frame]

                density = source_density.evaluate(
                    t,
                    source_params
                ).ravel()

                density /= np.trapezoid(
                    density,
                    t.ravel()
                )

                lc.set_array(density[:-1])

            if noise_function is not None:

                covariances = noise_function.evaluate(
                    t_cov,
                    params = result.noise_history[frame],
                    shape_params=shape_params,  # if required
                )

                sources = shape.parametrization(t_cov, shape_params)

                for ell, point, cov in zip(
                        cov_ellipses,
                        sources,
                        covariances,
                ):
                    vals, vecs = np.linalg.eigh(cov)

                    order = vals.argsort()[::-1]
                    vals = vals[order]
                    vecs = vecs[:, order]

                    angle = np.degrees(
                        np.arctan2(vecs[1, 0], vecs[0, 0])
                    )

                    ell.center = point
                    ell.width = 2 * np.sqrt(vals[0])
                    ell.height = 2 * np.sqrt(vals[1])
                    ell.angle = angle

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

            return (lc, txt, *cov_ellipses)

        ani = FuncAnimation(
            fig,
            update,
            frames=len(result.shape_history),
            interval=interval,
            blit=True,
        )

        if show:
            plt.show()

        return ani
