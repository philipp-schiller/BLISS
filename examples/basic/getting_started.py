"""
Minimal BLISS example.

Generate noisy observations from an ellipse, fit the ellipse using the
high-level fit() interface, and visualize the result.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bliss.inference.fit import fit

from bliss.visualization.animation_2d import ShapePlotterAnimations

from bliss.models.shapes.ellipse import Ellipse
from bliss.visualization.plotting_2d import ShapePlotter2D


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

np.random.seed(42)


# ---------------------------------------------------------------------
# Define the ground-truth ellipse
# ---------------------------------------------------------------------

shape = Ellipse()

# Parameter order:
# [center_x, center_y, semi_axis_1, semi_axis_2, rotation]
true_shape_params = np.array([
    0.0,  # center x-coordinate
    0.0,  # center y-coordinate
    3.0,  # first semi-axis
    2.0,  # second semi-axis
    0.3,  # rotation angle in radians
])


# ---------------------------------------------------------------------
# Generate noisy observations from part of the ellipse
# ---------------------------------------------------------------------

rng = np.random.default_rng(42)

num_points = 150

# Sample 40% of the ellipse parameter domain.
arc_start = 0.10
arc_end = 0.50

t = np.linspace(
    arc_start,
    arc_end,
    num_points,
).reshape(-1, 1)

points_without_noise = shape.evaluate(
    t,
    true_shape_params,
)

noise_standard_deviation = 0.15

points = points_without_noise + rng.normal(
    loc=0.0,
    scale=noise_standard_deviation,
    size=points_without_noise.shape,
)

# ---------------------------------------------------------------------
# Fit the ellipse
# ---------------------------------------------------------------------

shape_init = shape.initial_fit(points)

result = fit(
    points,
    shape,
    shape_init=shape_init,
)

print("True shape parameters:")
print(true_shape_params)

print("\nInitial shape parameters:")
print(shape_init)

print("\nEstimated shape parameters:")
print(result.shape_params)

print("\nEstimated noise variance:")
print(result.noise_params)

print("\nOptimization diagnostics:")
print(f"  Final log-posterior: {result.log_posterior_final:.6f}")
print(f"  Converged:           {result.converged}")
print(f"  Runtime:             {result.runtime:.3f} s")
print(f"  Message:             {result.message}")


# ---------------------------------------------------------------------
# Visualize the result
# ---------------------------------------------------------------------

plotter = ShapePlotter2D()

fig, ax, _ = plotter.plot_shape(
    shape,
    true_shape_params,
    label="Ground truth",
    color="black",
    linestyle="--",
)

plotter.plot_shape(
    shape,
    result.shape_params,
    ax=ax,
    label="BLISS fit",
    color="orange",
)

plotter.plot_observations(
    observations=points,
    ax=ax,
)

ax.set_title("Ellipse fitting with BLISS")
ax.set_aspect("equal")
ax.legend()

fig.tight_layout()


# ---------------------------------------------------------------------
# Save the figure for the documentation
# ---------------------------------------------------------------------

repository_root = Path(__file__).resolve().parents[2]

output_path = (
    repository_root
    / "docs"
    / "_static"
    / "getting_started.png"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight",
)

print(f"\nSaved plot to: {output_path}")

plt.show()

# ---------------------------------------------------------------------
# Animate the optimization
# ---------------------------------------------------------------------

animator = ShapePlotterAnimations()

animation = animator.animate_optimization_result(
    result=result,
    shape=shape,
    source_density=shape.default_source_density(),
    noise_function=shape.default_noise_function(),
    source_parameter_names=[
        "arc center",
        "arc half-width",
    ],
    noise_parameter_names=[
        "noise variance",
    ],
    ground_truth_params=true_shape_params,
    observations=points,
    interval=1000,
    show=True,
)

from pathlib import Path

repository_root = Path(__file__).resolve().parents[2]
output_path = repository_root / "docs" / "_static" / "getting_started.gif"
output_path.parent.mkdir(parents=True, exist_ok=True)

animation.save(
    output_path,
    writer="pillow",
    fps=2,
)

print(f"Saved animation to: {output_path}")