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

    def animate_optimization_result_(
        self,
        result,
        shape,
        source_density=None,
        noise_function=None,
        source_prior=None,
        noise_prior=None,
        ax=None,
        interval=300,
        show=False,
    ):

        if result.shape_history is None:
            raise ValueError("OptimizationResult does not contain shape_history")

        if source_prior is not None or noise_prior is not None:

            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(
                2,
                2,
                width_ratios=[2.5, 1]
            )

            ax = fig.add_subplot(gs[:, 0])

            ax_source_prior = (
                fig.add_subplot(gs[0, 1])
                if source_prior is not None
                else None
            )

            ax_noise_prior = (
                fig.add_subplot(gs[1, 1])
                if noise_prior is not None
                else None
            )

        else:

            fig, ax = plt.subplots(figsize=(8, 8))


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

    def animate_optimization_result(
            self,
            result,
            shape,
            source_density=None,
            noise_function=None,
            source_prior=None,
            noise_prior=None,
            source_bounds=None,
            noise_bounds=None,
            source_parameter_names=None,
            noise_parameter_names=None,
            ground_truth_params=None,
            observations=None,
            ax=None,
            interval=300,
            show=False,
    ):
        """
        Animate an optimization result together with source and noise priors.

        The main axis shows:
            - the current shape,
            - source density as the line color,
            - covariance ellipses from the noise model,
            - current ellipse, source, and noise parameters.

        The two right-hand axes show:
            - marginal Gaussian source priors,
            - marginal Gaussian noise priors,
            - the optimization trajectory evaluated on each marginal PDF,
            - a marker for the current parameter value.

        Assumptions
        -----------
        For an ellipse, the first five shape parameters are assumed to be:

            [center_x, center_y, semi_axis_a, semi_axis_b, angle]

        The angle is assumed to be in radians.

        Parameters
        ----------
        result
            Optimization result containing shape_history and optionally
            source_history, noise_history, and log_posterior_history.

        shape
            Shape object exposing:
                - domain_low
                - domain_high
                - parametrization(t, shape_params)

        source_density
            Optional source-density object exposing:
                evaluate(t, source_params)

        noise_function
            Optional noise object exposing:
                evaluate(t, params=noise_params, shape_params=shape_params)

        source_prior
            Optional Gaussian prior with:
                - mean
                - cov

        noise_prior
            Optional Gaussian prior with:
                - mean
                - cov

        ax
            Optional axis for the shape animation. This is supported when no
            prior panels are requested. If priors are supplied, the function
            creates a dedicated multi-panel figure.

        interval
            Delay between animation frames in milliseconds.

        show
            Whether to call plt.show().

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Ellipse

        # --------------------------------------------------
        # validate histories
        # --------------------------------------------------

        if result.shape_history is None:
            raise ValueError(
                "OptimizationResult does not contain shape_history."
            )

        shape_history = [
            np.asarray(values, dtype=float).reshape(-1)
            for values in result.shape_history
        ]

        if len(shape_history) == 0:
            raise ValueError("shape_history is empty.")

        n_frames = len(shape_history)

        source_history_raw = getattr(result, "source_history", None)
        noise_history_raw = getattr(result, "noise_history", None)

        source_history = None
        noise_history = None

        if source_history_raw is not None:
            source_history = [
                source_density.fixed_params
                if values is None
                else np.asarray(values, dtype=float).reshape(-1)
                for values in source_history_raw
            ]

        if noise_history_raw is not None:
            noise_history = [
                noise_function.fixed_params
                if values is None
                else np.asarray(values, dtype=float).reshape(-1)
                for values in noise_history_raw
            ]

        if source_prior is not None and source_history is None:
            raise ValueError(
                "source_prior was supplied, but result.source_history is None."
            )

        if noise_prior is not None and noise_history is None:
            raise ValueError(
                "noise_prior was supplied, but result.noise_history is None."
            )

        # --------------------------------------------------
        # figure layout
        # --------------------------------------------------

        has_prior_panels = (
                source_prior is not None
                or noise_prior is not None
        )

        if has_prior_panels:
            # A dedicated layout is required for the prior subplots.
            fig = plt.figure(figsize=(15, 8.5))

            grid = fig.add_gridspec(
                nrows=2,
                ncols=2,
                width_ratios=(2.4, 1.25),
                height_ratios=(1.0, 1.0),
                wspace=0.28,
                hspace=0.35,
            )

            shape_ax = fig.add_subplot(grid[:, 0])

            source_prior_ax = (
                fig.add_subplot(grid[0, 1])
                if source_prior is not None
                else None
            )

            noise_prior_ax = (
                fig.add_subplot(grid[1, 1])
                if noise_prior is not None
                else None
            )

            # If only one prior exists, allow it to occupy the full right side.
            if source_prior is not None and noise_prior is None:
                source_prior_ax.remove()
                source_prior_ax = fig.add_subplot(grid[:, 1])

            elif noise_prior is not None and source_prior is None:
                noise_prior_ax.remove()
                noise_prior_ax = fig.add_subplot(grid[:, 1])

        else:
            if ax is None:
                fig, shape_ax = plt.subplots(figsize=(9, 8))
            else:
                shape_ax = ax
                fig = shape_ax.get_figure()

            source_prior_ax = None
            noise_prior_ax = None

        # --------------------------------------------------
        # parameter-domain values
        # --------------------------------------------------

        domain_low = float(
            np.atleast_1d(shape.domain_low)[0]
        )

        domain_high = float(
            np.atleast_1d(shape.domain_high)[0]
        )

        t = np.linspace(
            domain_low,
            domain_high,
            500,
        ).reshape(-1, 1)

        # --------------------------------------------------
        # Static ground truth and observations
        # --------------------------------------------------

        ground_truth_artist = None
        observation_artist = None

        if observations is not None:
            observations_array = np.asarray(
                observations,
                dtype=float,
            )

            observation_artist = shape_ax.scatter(
                observations_array[:, 0],
                observations_array[:, 1],
                s=14,
                color="black",
                alpha=0.45,
                label="Observations",
                zorder=1,
            )

        if ground_truth_params is not None:
            ground_truth_points = np.asarray(
                shape.parametrization(
                    t,
                    np.asarray(
                        ground_truth_params,
                        dtype=float,
                    ),
                )
            )

            ground_truth_artist, = shape_ax.plot(
                ground_truth_points[:, 0],
                ground_truth_points[:, 1],
                color="black",
                linestyle="--",
                linewidth=2.5,
                alpha=0.9,
                label="Ground truth",
                zorder=3,
            )

        # --------------------------------------------------
        # helper functions
        # --------------------------------------------------

        def normalize_parameter_names(
                names,
                dimension,
                default_prefix,
        ):
            """Return validated parameter names."""

            if names is None:
                return [
                    f"{default_prefix}[{index}]"
                    for index in range(dimension)
                ]

            if len(names) != dimension:
                raise ValueError(
                    f"Expected {dimension} {default_prefix} parameter names, "
                    f"but received {len(names)}."
                )

            return [
                str(name)
                for name in names
            ]

        def normalize_bounds(bounds, dimension, name):
            """
            Validate bounds and return a normalized list.

            Each element is either:
                (lower, upper)

            Both lower and upper may be None.
            """

            if bounds is None:
                return [None] * dimension

            if len(bounds) != dimension:
                raise ValueError(
                    f"{name} contains {len(bounds)} bounds, but the "
                    f"corresponding parameter vector has dimension {dimension}."
                )

            normalized = []

            for index, bound in enumerate(bounds):
                if bound is None:
                    normalized.append((None, None))
                    continue

                if len(bound) != 2:
                    raise ValueError(
                        f"{name}[{index}] must be a tuple "
                        f"(lower, upper)."
                    )

                lower, upper = bound

                lower = (
                    None
                    if lower is None
                    else float(lower)
                )

                upper = (
                    None
                    if upper is None
                    else float(upper)
                )

                if (
                        lower is not None
                        and upper is not None
                        and lower > upper
                ):
                    raise ValueError(
                        f"{name}[{index}] has lower bound {lower}, "
                        f"which is larger than upper bound {upper}."
                    )

                normalized.append((lower, upper))

            return normalized

        def value_within_bounds(value, bound):
            """Return True if a value is inside its bounds."""

            if bound is None:
                return True

            lower, upper = bound

            if lower is not None and value < lower:
                return False

            if upper is not None and value > upper:
                return False

            return True

        def format_bound(bound):
            """Format one bound for the parameter-information panel."""

            if bound is None:
                return "unbounded"

            lower, upper = bound

            lower_text = (
                "-inf"
                if lower is None
                else f"{lower:.4g}"
            )

            upper_text = (
                "inf"
                if upper is None
                else f"{upper:.4g}"
            )

            return f"[{lower_text}, {upper_text}]"

        def normalized_density(parameters):
            """Evaluate and normalize the source density."""

            density_values = np.asarray(
                source_density.evaluate(
                    t,
                    parameters,
                ),
                dtype=float,
            ).reshape(-1)

            integral = np.trapezoid(
                density_values,
                t.ravel(),
            )

            if (
                    np.isfinite(integral)
                    and abs(integral) > np.finfo(float).eps
            ):
                density_values = density_values / integral

            return density_values

        def valid_history_value(history, frame):
            """Safely retrieve a parameter vector from a history."""

            if history is None:
                return None

            if frame >= len(history):
                return None

            return history[frame]

        def format_array(values, precision=3):
            """Format a one-dimensional parameter vector compactly."""

            return np.array2string(
                np.asarray(values),
                precision=precision,
                suppress_small=True,
                separator=", ",
                max_line_width=38,
            )

        def build_parameter_text(frame):
            """Create a neatly aligned parameter display."""

            current_shape = np.asarray(
                shape_history[frame],
                dtype=float,
            ).reshape(-1)

            # Fixed column widths for monospaced text.
            label_width = 17
            value_width = 10

            def parameter_row(
                    label,
                    value,
                    bound=None,
                    unit="",
                    precision=4,
            ):
                """Format one aligned parameter row."""

                value_text = f"{float(value):.{precision}f}"

                if unit:
                    value_text += f" {unit}"

                row = (
                    f"{label:<{label_width}}"
                    f"{value_text:>{value_width}}"
                )

                if bound is not None and bound != (None, None):
                    row += f"    {format_bound(bound)}"

                return row

            def section_heading(title):
                """Format a section heading."""

                return [
                    "",
                    title,
                    "-" * len(title),
                ]

            lines = [
                f"{'Iteration':<{label_width}}"
                f"{frame + 1:>{value_width - 5}} / {n_frames}",
            ]

            # --------------------------------------------------
            # Posterior
            # --------------------------------------------------

            posterior_history = getattr(
                result,
                "log_posterior_history",
                None,
            )

            if (
                    posterior_history is not None
                    and frame < len(posterior_history)
                    and posterior_history[frame] is not None
            ):
                posterior_value = float(
                    posterior_history[frame]
                )

                lines.append(
                    parameter_row(
                        label="Log posterior",
                        value=posterior_value,
                        precision=4,
                    )
                )

            # --------------------------------------------------
            # Ellipse
            # --------------------------------------------------

            lines.extend(
                section_heading("ELLIPSE")
            )

            if len(current_shape) >= 5:
                center_x = current_shape[0]
                center_y = current_shape[1]
                semi_axis_a = current_shape[2]
                semi_axis_b = current_shape[3]

                angle_degrees = np.degrees(
                    current_shape[4]
                )

                lines.extend([
                    parameter_row(
                        "Center x",
                        center_x,
                    ),
                    parameter_row(
                        "Center y",
                        center_y,
                    ),
                    parameter_row(
                        "Semi-axis a",
                        semi_axis_a,
                    ),
                    parameter_row(
                        "Semi-axis b",
                        semi_axis_b,
                    ),
                    parameter_row(
                        "Angle",
                        angle_degrees,
                        unit="deg",
                        precision=2,
                    ),
                ])

                # Preserve additional shape parameters, if present.
                for index, value in enumerate(
                        current_shape[5:],
                        start=5,
                ):
                    lines.append(
                        parameter_row(
                            f"Shape {index}",
                            value,
                        )
                    )

            else:
                for index, value in enumerate(current_shape):
                    lines.append(
                        parameter_row(
                            f"Shape {index}",
                            value,
                        )
                    )

            # --------------------------------------------------
            # Source parameters
            # --------------------------------------------------

            current_source = valid_history_value(
                source_history,
                frame,
            )

            if current_source is not None:
                current_source = np.asarray(
                    current_source,
                    dtype=float,
                ).reshape(-1)

                displayed_source_bounds = normalize_bounds(
                    source_bounds,
                    len(current_source),
                    "source_bounds",
                )

                source_names = normalize_parameter_names(
                    names=source_parameter_names,
                    dimension=len(current_source),
                    default_prefix="source",
                )

                lines.extend(
                    section_heading("SOURCE")
                )

                for index, value in enumerate(current_source):
                    bound = displayed_source_bounds[index]

                    name = source_names[index]

                    if not value_within_bounds(value, bound):
                        name = f"! {name}"

                    lines.append(
                        parameter_row(
                            label=name,
                            value=value,
                            bound=bound,
                        )
                    )

            # --------------------------------------------------
            # Noise parameters
            # --------------------------------------------------

            current_noise = valid_history_value(
                noise_history,
                frame,
            )

            if current_noise is not None:
                current_noise = np.asarray(
                    current_noise,
                    dtype=float,
                ).reshape(-1)

                displayed_noise_bounds = normalize_bounds(
                    noise_bounds,
                    len(current_noise),
                    "noise_bounds",
                )

                noise_names = normalize_parameter_names(
                    names=noise_parameter_names,
                    dimension=len(current_noise),
                    default_prefix="noise",
                )

                lines.extend(
                    section_heading("NOISE")
                )

                for index, value in enumerate(current_noise):
                    bound = displayed_noise_bounds[index]

                    name = noise_names[index]

                    if not value_within_bounds(value, bound):
                        name = f"! {name}"

                    lines.append(
                        parameter_row(
                            label=name,
                            value=value,
                            bound=bound,
                        )
                    )

            return "\n".join(lines)

        def create_prior_panel(
                prior_ax,
                prior,
                history,
                bounds,
                title,
                parameter_prefix,
                parameter_names=None,
        ):
            """
            Create marginal Gaussian priors, parameter trajectories, and bounds.

            For every parameter:
                - the Gaussian marginal is shown as a curve,
                - the feasible interval is shown near the bottom,
                - lower and upper bounds are shown as dotted lines,
                - the optimization path is drawn on the Gaussian density,
                - the current parameter value is shown as a marker.

            The shared x-axis covers exactly the union of the supplied bounds.

            Example
            -------
            source_bounds = [(0.0, 1.0), (0.0, 0.5)]

            The shared source-prior axis will span [0.0, 1.0].
            """

            if prior_ax is None or prior is None:
                return [], [], [], []

            if history is None or len(history) == 0:
                raise ValueError(
                    f"{title}: no parameter history is available."
                )

            # --------------------------------------------------
            # Prior dimensions
            # --------------------------------------------------

            mean = np.asarray(
                prior.mean,
                dtype=float,
            ).reshape(-1)

            covariance = np.asarray(
                prior.cov,
                dtype=float,
            )

            number_of_parameters = mean.size

            if number_of_parameters == 0:
                raise ValueError(
                    f"{title}: prior.mean is empty."
                )

            display_names = normalize_parameter_names(
                names=parameter_names,
                dimension=number_of_parameters,
                default_prefix=parameter_prefix,
            )

            if covariance.ndim == 0:
                covariance = covariance.reshape(1, 1)

            elif covariance.ndim == 1:
                if covariance.size != number_of_parameters:
                    raise ValueError(
                        f"{title}: prior.mean contains "
                        f"{number_of_parameters} parameters, but prior.cov "
                        f"contains {covariance.size} diagonal entries."
                    )

                covariance = np.diag(covariance)

            expected_covariance_shape = (
                number_of_parameters,
                number_of_parameters,
            )

            if covariance.shape != expected_covariance_shape:
                raise ValueError(
                    f"{title}: prior covariance has shape "
                    f"{covariance.shape}, but expected "
                    f"{expected_covariance_shape}."
                )

            variances = np.diag(covariance)

            if np.any(~np.isfinite(variances)):
                raise ValueError(
                    f"{title}: covariance diagonal contains non-finite values."
                )

            if np.any(variances <= 0.0):
                invalid_indices = np.flatnonzero(
                    variances <= 0.0
                ).tolist()

                raise ValueError(
                    f"{title}: covariance diagonal entries must be positive. "
                    f"Invalid parameter indices: {invalid_indices}."
                )

            standard_deviations = np.sqrt(variances)

            # --------------------------------------------------
            # Validate and normalize history
            # --------------------------------------------------

            normalized_history = []

            for frame, values in enumerate(history):
                if values is None:
                    normalized_history.append(None)
                    continue

                values = np.asarray(
                    values,
                    dtype=float,
                ).reshape(-1)

                if values.size != number_of_parameters:
                    raise ValueError(
                        f"{title}: {parameter_prefix}_history[{frame}] "
                        f"contains {values.size} parameters, but the prior "
                        f"contains {number_of_parameters} parameters.\n"
                        f"History value: {values}\n"
                        f"Prior mean: {mean}"
                    )

                normalized_history.append(values)

            first_available = next(
                (
                    values
                    for values in normalized_history
                    if values is not None
                ),
                None,
            )

            if first_available is None:
                raise ValueError(
                    f"{title}: history contains no parameter values."
                )

            # --------------------------------------------------
            # Validate and normalize bounds
            # --------------------------------------------------

            normalized_parameter_bounds = normalize_bounds(
                bounds=bounds,
                dimension=number_of_parameters,
                name=f"{parameter_prefix}_bounds",
            )

            # Determine the exact shared x-axis.
            #
            # When all parameters have finite two-sided bounds, the axis is
            # exactly the union of those bounds. For one-sided or unbounded
            # parameters, use mean +/- 4 standard deviations only for the
            # missing side.

            lower_axis_candidates = []
            upper_axis_candidates = []

            for index in range(number_of_parameters):
                lower, upper = normalized_parameter_bounds[index]

                displayed_lower = (
                    float(lower)
                    if lower is not None
                    else float(
                        mean[index]
                        - 4.0 * standard_deviations[index]
                    )
                )

                displayed_upper = (
                    float(upper)
                    if upper is not None
                    else float(
                        mean[index]
                        + 4.0 * standard_deviations[index]
                    )
                )

                if displayed_lower >= displayed_upper:
                    raise ValueError(
                        f"{parameter_prefix}_bounds[{index}] produces an "
                        f"invalid display range "
                        f"({displayed_lower}, {displayed_upper})."
                    )

                lower_axis_candidates.append(displayed_lower)
                upper_axis_candidates.append(displayed_upper)

            overall_x_min = min(lower_axis_candidates)
            overall_x_max = max(upper_axis_candidates)

            if not (
                    np.isfinite(overall_x_min)
                    and np.isfinite(overall_x_max)
            ):
                raise ValueError(
                    f"{title}: the calculated x-axis limits are not finite: "
                    f"({overall_x_min}, {overall_x_max})."
                )

            if overall_x_min >= overall_x_max:
                raise ValueError(
                    f"{title}: invalid shared x-axis range "
                    f"({overall_x_min}, {overall_x_max})."
                )

            # All priors are evaluated over the same displayed x-axis.
            x_values = np.linspace(
                overall_x_min,
                overall_x_max,
                500,
            )

            # --------------------------------------------------
            # Prepare artists
            # --------------------------------------------------

            colors = plt.cm.tab10(
                np.linspace(
                    0.0,
                    1.0,
                    max(number_of_parameters, 2),
                )
            )

            prior_curves = []
            trajectory_lines = []
            current_markers = []
            bound_artists = []

            maximum_pdf = 0.0

            # --------------------------------------------------
            # Plot marginal Gaussian priors
            # --------------------------------------------------

            for index in range(number_of_parameters):
                standard_deviation = standard_deviations[index]

                pdf_values = (
                        np.exp(
                            -0.5
                            * (
                                    (x_values - mean[index])
                                    / standard_deviation
                            ) ** 2
                        )
                        / (
                                standard_deviation
                                * np.sqrt(2.0 * np.pi)
                        )
                )

                color = colors[index]

                prior_curve, = prior_ax.plot(
                    x_values,
                    pdf_values,
                    color=color,
                    linewidth=2.0,
                    alpha=0.65,
                    #label=f"{parameter_prefix}[{index}] prior",
                    label=f"{display_names[index]} prior",
                )

                trajectory_line, = prior_ax.plot(
                    [],
                    [],
                    color=color,
                    linewidth=2.5,
                    alpha=1.0,
                    zorder=5,
                )

                current_marker, = prior_ax.plot(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=1.0,
                    markersize=8,
                    zorder=7,
                )

                prior_curves.append(prior_curve)
                trajectory_lines.append(trajectory_line)
                current_markers.append(current_marker)

                maximum_pdf = max(
                    maximum_pdf,
                    float(np.max(pdf_values)),
                )

            if not np.isfinite(maximum_pdf) or maximum_pdf <= 0.0:
                maximum_pdf = 1.0

            # --------------------------------------------------
            # Plot bounds
            # --------------------------------------------------

            for index in range(number_of_parameters):
                lower, upper = normalized_parameter_bounds[index]
                color = colors[index]

                if lower is not None:
                    lower_line = prior_ax.axvline(
                        float(lower),
                        color=color,
                        linestyle=":",
                        linewidth=1.8,
                        alpha=0.9,
                        #label=f"{parameter_prefix}[{index}] bounds",
                        label=f"{display_names[index]} bounds",
                    )

                    bound_artists.append(lower_line)

                if upper is not None:
                    upper_line = prior_ax.axvline(
                        float(upper),
                        color=color,
                        linestyle=":",
                        linewidth=1.8,
                        alpha=0.9,
                    )

                    bound_artists.append(upper_line)

                # Show the feasible interval as a thick horizontal segment.
                displayed_lower = (
                    overall_x_min
                    if lower is None
                    else float(lower)
                )

                displayed_upper = (
                    overall_x_max
                    if upper is None
                    else float(upper)
                )

                # Place each parameter's feasible segment at a slightly
                # different height so overlapping intervals remain visible.
                bound_level = maximum_pdf * (
                        0.025 + 0.025 * index
                )

                feasible_segment, = prior_ax.plot(
                    [displayed_lower, displayed_upper],
                    [bound_level, bound_level],
                    color=color,
                    linewidth=5,
                    alpha=0.45,
                    solid_capstyle="butt",
                    zorder=3,
                )

                bound_artists.append(feasible_segment)

            # --------------------------------------------------
            # Axis formatting
            # --------------------------------------------------

            prior_ax.axhline(
                0.0,
                color="0.25",
                linewidth=0.8,
            )

            # This is exactly the union of the supplied finite bounds.
            prior_ax.set_xlim(
                overall_x_min,
                overall_x_max,
            )

            prior_ax.set_ylim(
                bottom=0.0,
                top=maximum_pdf * 1.15,
            )

            prior_ax.set_title(
                title,
                fontweight="bold",
            )

            prior_ax.set_xlabel("Parameter value")
            prior_ax.set_ylabel("Marginal prior density")

            prior_ax.grid(
                alpha=0.2,
                linewidth=0.8,
            )

            # --------------------------------------------------
            # Remove duplicate legend entries
            # --------------------------------------------------

            handles, labels = prior_ax.get_legend_handles_labels()

            unique_entries = {}

            for handle, label in zip(handles, labels):
                if label not in unique_entries:
                    unique_entries[label] = handle

            prior_ax.legend(
                unique_entries.values(),
                unique_entries.keys(),
                loc="best",
                fontsize=8,
                framealpha=0.9,
            )

            return (
                prior_curves,
                trajectory_lines,
                current_markers,
                bound_artists,
            )

        def update_prior_panel(
                prior,
                history,
                bounds,
                frame,
                trajectory_lines,
                current_markers,
        ):
            """
            Update prior trajectories and current-value markers.

            A current marker receives a red edge when the parameter is outside
            its specified bounds.
            """

            if (
                    prior is None
                    or history is None
                    or not trajectory_lines
            ):
                return []

            mean = np.asarray(
                prior.mean,
                dtype=float,
            ).reshape(-1)

            covariance = np.asarray(
                prior.cov,
                dtype=float,
            )

            if covariance.ndim == 0:
                covariance = covariance.reshape(1, 1)

            elif covariance.ndim == 1:
                covariance = np.diag(covariance)

            normalized_parameter_bounds = normalize_bounds(
                bounds=bounds,
                dimension=len(mean),
                name="prior bounds",
            )

            updated_artists = []

            for index, (trajectory, marker) in enumerate(
                    zip(trajectory_lines, current_markers)
            ):
                standard_deviation = np.sqrt(
                    covariance[index, index]
                )

                values_until_frame = np.asarray(
                    [
                        values[index]
                        for values in history[:frame + 1]
                        if values is not None
                    ],
                    dtype=float,
                )

                if len(values_until_frame) == 0:
                    trajectory.set_data([], [])
                    marker.set_data([], [])

                else:
                    pdf_until_frame = (
                            np.exp(
                                -0.5
                                * (
                                        (
                                                values_until_frame
                                                - mean[index]
                                        )
                                        / standard_deviation
                                ) ** 2
                            )
                            / (
                                    standard_deviation
                                    * np.sqrt(2.0 * np.pi)
                            )
                    )

                    trajectory.set_data(
                        values_until_frame,
                        pdf_until_frame,
                    )

                    current_value = values_until_frame[-1]

                    marker.set_data(
                        [current_value],
                        [pdf_until_frame[-1]],
                    )

                    inside_bounds = value_within_bounds(
                        current_value,
                        normalized_parameter_bounds[index],
                    )

                    marker.set_markeredgecolor(
                        "black" if inside_bounds else "red"
                    )

                    marker.set_markeredgewidth(
                        1.0 if inside_bounds else 2.5
                    )

                    marker.set_markersize(
                        8 if inside_bounds else 10
                    )

                updated_artists.extend([
                    trajectory,
                    marker,
                ])

            return updated_artists

        # --------------------------------------------------
        # initial shape
        # --------------------------------------------------

        initial_shape_params = shape_history[0]

        initial_points = np.asarray(
            shape.parametrization(
                t,
                initial_shape_params,
            )
        )

        segments = np.concatenate(
            [
                initial_points[:-1, None, :],
                initial_points[1:, None, :],
            ],
            axis=1,
        )

        # --------------------------------------------------
        # initial source density
        # --------------------------------------------------

        initial_source = valid_history_value(
            source_history,
            0,
        )

        if (
                source_density is not None
                and initial_source is not None
        ):
            initial_density = normalized_density(
                initial_source
            )
            colorbar_label = "Normalized source density"

        else:
            initial_density = np.ones(len(t))
            colorbar_label = "Source density"

        density_min = float(np.nanmin(initial_density))
        density_max = float(np.nanmax(initial_density))

        if np.isclose(density_min, density_max):
            density_min -= 0.5
            density_max += 0.5

        density_norm = plt.Normalize(
            vmin=density_min,
            vmax=density_max,
        )

        line_collection = LineCollection(
            segments,
            cmap="viridis",
            norm=density_norm,
            linewidth=4,
        )

        line_collection.set_array(
            initial_density[:-1]
        )

        shape_ax.add_collection(line_collection)

        colorbar = fig.colorbar(
            line_collection,
            ax=shape_ax,
            label=colorbar_label,
            fraction=0.046,
            pad=0.04,
        )

        # --------------------------------------------------
        # covariance ellipses
        # --------------------------------------------------

        covariance_ellipses = []

        t_covariance = np.linspace(
            domain_low,
            domain_high,
            50,
        ).reshape(-1, 1)

        initial_noise = valid_history_value(
            noise_history,
            0,
        )

        if (
                noise_function is not None
                and initial_noise is not None
        ):
            initial_covariances = noise_function.evaluate(
                t_covariance,
                params=initial_noise,
                shape_params=initial_shape_params,
            )

            covariance_sources = shape.parametrization(
                t_covariance,
                initial_shape_params,
            )

            for point, covariance in zip(
                    covariance_sources,
                    initial_covariances,
            ):
                covariance = np.asarray(
                    covariance,
                    dtype=float,
                )

                eigenvalues, eigenvectors = np.linalg.eigh(
                    covariance
                )

                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

                # Prevent small negative values caused by numerical error.
                eigenvalues = np.maximum(
                    eigenvalues,
                    0.0,
                )

                angle = np.degrees(
                    np.arctan2(
                        eigenvectors[1, 0],
                        eigenvectors[0, 0],
                    )
                )

                ellipse = Ellipse(
                    xy=point,
                    width=2.0 * np.sqrt(eigenvalues[0]),
                    height=2.0 * np.sqrt(eigenvalues[1]),
                    angle=angle,
                    edgecolor="green",
                    facecolor="green",
                    alpha=0.15,
                    linewidth=0.5,
                )

                shape_ax.add_patch(ellipse)
                covariance_ellipses.append(ellipse)

        # --------------------------------------------------
        # parameter information panel
        # --------------------------------------------------

        parameter_text = shape_ax.text(
            -0.03,
            0.98,
            build_parameter_text(0),
            transform=shape_ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            family="monospace",
            fontsize=8.5,
            linespacing=1.25,
            color="0.15",
            clip_on=False,
            zorder=20,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "white",
                "edgecolor": "0.4",
                "linewidth": 0.8,
                "alpha": 0.95,
            },
        )

        shape_ax.set_title(
            "Optimization",
            fontweight="bold",
        )

        shape_ax.set_aspect(
            "equal",
            adjustable="datalim",
        )

        shape_ax.autoscale_view()
        shape_ax.margins(0.08)
        shape_ax.grid(alpha=0.15)

        # Keep the initial view fixed throughout the animation.
        shape_xlim = shape_ax.get_xlim()
        shape_ylim = shape_ax.get_ylim()

        # --------------------------------------------------
        # prior panels
        # --------------------------------------------------

        (
            source_prior_curves,
            source_trajectory_lines,
            source_current_markers,
            source_bound_artists,
        ) = create_prior_panel(
            prior_ax=source_prior_ax,
            prior=source_prior,
            history=source_history,
            bounds=source_bounds,
            title="Source-parameter priors and bounds",
            parameter_prefix="source",
            parameter_names=source_parameter_names,
        )

        (
            noise_prior_curves,
            noise_trajectory_lines,
            noise_current_markers,
            noise_bound_artists,
        ) = create_prior_panel(
            prior_ax=noise_prior_ax,
            prior=noise_prior,
            history=noise_history,
            bounds=noise_bounds,
            title="Noise-parameter priors and bounds",
            parameter_prefix="noise",
            parameter_names=noise_parameter_names,
        )

        def reserve_parameter_text_space(
                fig,
                parameter_text,
                build_text,
                number_of_frames,
                padding=0.015,
        ):
            """
            Automatically reserve enough left margin for the parameter box.

            All animation frames are measured so that the panel does not move
            or become cropped when parameter values change.
            """

            original_text = parameter_text.get_text()

            # Render once so that Matplotlib has a renderer available.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            maximum_width_pixels = 0.0

            # Measure the parameter text for every animation frame.
            for frame_index in range(number_of_frames):
                parameter_text.set_text(
                    build_text(frame_index)
                )

                bounding_box = parameter_text.get_window_extent(
                    renderer=renderer
                )

                maximum_width_pixels = max(
                    maximum_width_pixels,
                    bounding_box.width,
                )

            # Restore the initial animation text.
            parameter_text.set_text(original_text)

            figure_width_pixels = fig.get_window_extent(
                renderer=renderer
            ).width

            text_width_fraction = (
                    maximum_width_pixels
                    / figure_width_pixels
            )

            # Account for the small gap between the text and shape axis.
            axes_width_fraction = (
                shape_ax.get_position().width
            )

            anchor_gap_fraction = (
                    0.03 * axes_width_fraction
            )

            required_left_margin = (
                    text_width_fraction
                    + anchor_gap_fraction
                    + padding
            )

            # Retain an existing larger margin, if applicable.
            required_left_margin = max(
                fig.subplotpars.left,
                required_left_margin,
            )

            # Leave a safety limit so that the remaining subplots do not vanish.
            required_left_margin = min(
                required_left_margin,
                0.40,
            )

            fig.subplots_adjust(
                left=required_left_margin
            )

            # Re-render after changing the subplot positions.
            fig.canvas.draw_idle()

        reserve_parameter_text_space(
            fig=fig,
            parameter_text=parameter_text,
            build_text=build_parameter_text,
            number_of_frames=n_frames,
        )

        # --------------------------------------------------
        # animation update
        # --------------------------------------------------

        def update(frame):
            updated_artists = []

            current_shape_params = shape_history[frame]

            current_points = np.asarray(
                shape.parametrization(
                    t,
                    current_shape_params,
                )
            )

            current_segments = np.concatenate(
                [
                    current_points[:-1, None, :],
                    current_points[1:, None, :],
                ],
                axis=1,
            )

            line_collection.set_segments(
                current_segments
            )

            updated_artists.append(line_collection)

            # ----------------------------------------------
            # update source density
            # ----------------------------------------------

            current_source = valid_history_value(
                source_history,
                frame,
            )

            if (
                    source_density is not None
                    and current_source is not None
            ):
                current_density = normalized_density(
                    current_source
                )

                line_collection.set_array(
                    current_density[:-1]
                )

            # ----------------------------------------------
            # update covariance ellipses
            # ----------------------------------------------

            current_noise = valid_history_value(
                noise_history,
                frame,
            )

            if (
                    noise_function is not None
                    and current_noise is not None
                    and covariance_ellipses
            ):
                current_covariances = noise_function.evaluate(
                    t_covariance,
                    params=current_noise,
                    shape_params=current_shape_params,
                )

                current_sources = shape.parametrization(
                    t_covariance,
                    current_shape_params,
                )

                for ellipse, point, covariance in zip(
                        covariance_ellipses,
                        current_sources,
                        current_covariances,
                ):
                    covariance = np.asarray(
                        covariance,
                        dtype=float,
                    )

                    eigenvalues, eigenvectors = np.linalg.eigh(
                        covariance
                    )

                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]

                    eigenvalues = np.maximum(
                        eigenvalues,
                        0.0,
                    )

                    angle = np.degrees(
                        np.arctan2(
                            eigenvectors[1, 0],
                            eigenvectors[0, 0],
                        )
                    )

                    ellipse.center = point
                    ellipse.width = (
                            2.0 * np.sqrt(eigenvalues[0])
                    )
                    ellipse.height = (
                            2.0 * np.sqrt(eigenvalues[1])
                    )
                    ellipse.angle = angle

                    updated_artists.append(ellipse)

            # ----------------------------------------------
            # update text
            # ----------------------------------------------

            parameter_text.set_text(
                build_parameter_text(frame)
            )

            updated_artists.append(parameter_text)

            # ----------------------------------------------
            # update prior trajectories
            # ----------------------------------------------

            updated_artists.extend(
                update_prior_panel(
                    prior=source_prior,
                    history=source_history,
                    bounds=source_bounds,
                    frame=frame,
                    trajectory_lines=source_trajectory_lines,
                    current_markers=source_current_markers,
                )
            )

            updated_artists.extend(
                update_prior_panel(
                    prior=noise_prior,
                    history=noise_history,
                    bounds=noise_bounds,
                    frame=frame,
                    trajectory_lines=noise_trajectory_lines,
                    current_markers=noise_current_markers,
                )
            )

            # Prevent the main view from changing between frames.
            shape_ax.set_xlim(shape_xlim)
            shape_ax.set_ylim(shape_ylim)

            return tuple(updated_artists)

        # Initialize prior paths and markers at frame zero.
        update(0)

        # blit=False is more reliable here because several axes, patches,
        # text objects, and a LineCollection are updated simultaneously.
        animation = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=False,
            repeat=False,
        )

        if show:
            plt.show()

        return animation