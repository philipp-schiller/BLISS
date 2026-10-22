"""
Fit an ellipse to observations from a partially visible arc.

This example demonstrates the main BLISS workflow:

1. Generate noisy observations from one arc of an ellipse.
2. Represent knowledge about the visible region with a source-density model.
3. Estimate the ellipse, source-density, and noise parameters jointly.
4. Inspect and animate the optimization result.
"""

import numpy as np

from bliss.inference.fit import fit
from bliss.inference.noise_functions import IsotropicNoise
from bliss.inference.priors import GaussianPrior
from bliss.inference.source_densitites import (
    UniformSourceDensity,
)

from bliss.visualization.animation_2d import ShapePlotterAnimations

from bliss.models.shapes.ellipse import Ellipse

from bliss.visualization.plotting_2d import ShapePlotter2D

from pathlib import Path
# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

np.random.seed(63)


# ---------------------------------------------------------------------
# Define the ground-truth ellipse
# ---------------------------------------------------------------------

shape = Ellipse()

# Ground-truth ellipse parameters:
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
# Specify the observation model used for fitting
# ---------------------------------------------------------------------

# Imagine that we know the measurements originate from one contiguous
# arc, but we do not know the exact center or width of that arc.
source_density = UniformSourceDensity(
    n_dim=shape.parameter_dim,
    domain_low=shape.domain_low,
    domain_high=shape.domain_high,
    periodic=shape.periodic,
)

# Prior over [arc_center, arc_half_width].
source_density.prior = GaussianPrior(
    mean=np.array([0.50, 0.20]),
    cov=np.diag([1, 0.00065]),
)

# Assume isotropic Gaussian noise with unknown variance.
noise_function = IsotropicNoise(
    dim_param=shape.parameter_dim,
    dim_ambient=shape.ambient_dim,
)

# Prior over the noise variance.
noise_function.prior = GaussianPrior(
    mean=np.array([0.15**2]),
    cov=np.diag([0.002**2]),
)


# ---------------------------------------------------------------------
# Initial parameter estimates
# ---------------------------------------------------------------------

shape_init = shape.initial_fit(points)

# Initial [arc_center, arc_half_width].
source_init = np.array([0.50, 0.15])

# Initial noise variance.
noise_init = np.array([0.08])


# ---------------------------------------------------------------------
# Fit the BLISS model
# ---------------------------------------------------------------------

result = fit(
    points,
    shape,
    source_density=source_density,
    noise_function=noise_function,
    shape_init=shape_init,
    source_init=source_init,
    noise_init=noise_init,
    likelihood_kwargs={
        "num_samples": 300,
    },
)


# ---------------------------------------------------------------------
# Inspect the result
# ---------------------------------------------------------------------

print("True shape parameters:")
print(true_shape_params)

print("\nInitial shape parameters:")
print(shape_init)

print("\nEstimated shape parameters:")
print(result.shape_params)

print("\nEstimated source parameters [center, half-width]:")
print(result.source_params)

print("\nEstimated noise variance:")
print(result.noise_params)

print("\nOptimization diagnostics:")
print(f"  Final log-posterior: {result.log_posterior_final:.6f}")
print(f"  Converged:           {result.converged}")
print(f"  Iterations:          {result.n_iterations}")
print(f"  Function evaluations:{result.n_function_evals}")
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
    / "fit_ellipse_partial_with_priors.png"
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


# ---------------------------------------------------------------------
# Animate the optimization
# ---------------------------------------------------------------------

animator = ShapePlotterAnimations()

animation = animator.animate_optimization_result(
    result=result,
    shape=shape,
    source_density=source_density,
    noise_function=noise_function,
    source_prior=source_density.prior,
    noise_prior=noise_function.prior,
    source_bounds=source_density.get_parameter_bounds(),
    noise_bounds=noise_function.get_bounds(),
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
    show=False,
)

from pathlib import Path

repository_root = Path(__file__).resolve().parents[2]
output_path = repository_root / "docs" / "_static" / "fit_ellipse_partial_with_priors.gif"
output_path.parent.mkdir(parents=True, exist_ok=True)

animation.save(
    output_path,
    writer="pillow",
    fps=2,
)

print(f"Saved animation to: {output_path}")