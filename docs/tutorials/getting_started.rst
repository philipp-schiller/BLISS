Getting Started
===============

This tutorial introduces the public ``fit()`` interface and demonstrates
how to fit an ellipse to noisy observations covering only part of its
boundary.

The complete executable example is available in
``examples/basic/getting_started.py``.

Installation
------------

Clone the repository and install BLISS in editable mode:

.. code-block:: bash

   git clone https://github.com/philipp-schiller/BLISS
   cd BLISS
   python -m pip install -e .

Defining the Ellipse
--------------------

We begin by defining an ellipse that will be used to generate synthetic
observations.

.. code-block:: python

   import numpy as np

   from bliss.models.shapes.ellipse import Ellipse

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

Generating Partial Observations
-------------------------------

Instead of observing the complete ellipse, we sample points from only
one part of its parameter domain.

.. code-block:: python

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

The interval from ``0.10`` to ``0.50`` covers 40% of the normalized
parameter domain. This does not necessarily correspond to exactly 40%
of the geometric arc length because the ellipse parameterization is not
uniform with respect to arc length.

We add independent Gaussian noise to both coordinates:

.. code-block:: python

   noise_standard_deviation = 0.15

   points = points_without_noise + rng.normal(
       loc=0.0,
       scale=noise_standard_deviation,
       size=points_without_noise.shape,
   )

The resulting ``points`` array has shape ``(150, 2)``. Each row contains
one noisy two-dimensional observation.

In practical applications, ``points`` can instead contain measured data,
for example:

.. code-block:: python

   points = np.loadtxt("measurements.txt")

Fitting the Ellipse
-------------------

We first use the simplest possible BLISS call:

.. code-block:: python

   from bliss.inference.fit import fit

   result = fit(
       points,
       shape,
   )

With no additional configuration, BLISS uses:

* a fixed uniform source density over the complete parameter domain,
* an isotropic Gaussian noise model with an estimated variance,
* a marginal likelihood, and
* ECM optimization.

The ellipse initialization and initial noise variance are determined
automatically.

Inspecting the Result
---------------------

The returned result contains the estimated parameters and optimization
diagnostics:

.. code-block:: python

   print("Estimated shape parameters:")
   print(result.shape_params)

   print("\nEstimated noise variance:")
   print(result.noise_params)

   print("\nOptimization diagnostics:")
   print(f"  Final log-posterior: {result.log_posterior_final:.6f}")
   print(f"  Converged:           {result.converged}")
   print(f"  Runtime:             {result.runtime:.3f} s")
   print(f"  Message:             {result.message}")

The main fields are:

``shape_params``
   Estimated geometric parameters.

``source_params``
   Estimated source parameters, or ``None`` when the source density is
   fixed. The default source density used here is fixed.

``noise_params``
   Estimated noise parameters. For the default isotropic model, this
   contains the estimated variance.

``log_posterior_final``
   Final value of the log-posterior objective.

``converged``
   Whether the optimizer reported convergence.

``message``
   Status message returned by the optimizer.

``runtime``
   Total optimization runtime.

Visualizing the Reconstruction
------------------------------

BLISS provides plotting utilities for two-dimensional shapes:

.. code-block:: python

   import matplotlib.pyplot as plt

   from bliss.visualization.plotting_2d import (
       ShapePlotter2D,
   )

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

   ax.set_title("Default fit from partial observations")
   ax.set_aspect("equal")
   ax.legend()

   fig.tight_layout()
   plt.show()

The resulting reconstruction is shown below:

.. image:: /_static/getting_started.png
   :alt: Default ellipse fit from partial observations
   :align: center
   :width: 75%

The fitted ellipse follows the measurements reasonably closely in the
observed region. In the unobserved region, however, the reconstruction
differs substantially from the ground truth.

This is not surprising. A limited noisy arc can be explained by several
different complete ellipses. Moreover, the default source model assumes
that observations may originate from the complete parameter domain,
although the data in this example cover only one interval.

This raises a natural question:

**Can the reconstruction be improved if we include information about
which part of the ellipse was observed?**

The :doc:`partial-observation tutorial <ellipses_reconstruction>`
investigates this question.

Running the Example
-------------------

Run the complete example from the repository root:

.. code-block:: bash

   python examples/basic/getting_started.py

The script saves the generated figure to:

.. code-block:: text

   docs/_static/getting_started.png