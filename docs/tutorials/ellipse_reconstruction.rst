Improving Ellipse Reconstruction with Observation Information
==============================================================

The :doc:`Getting Started <getting_started>` tutorial fitted an ellipse
to observations covering only one part of its boundary.

The baseline fit used the default source model, which assumes that
observations may originate uniformly from the complete parameter domain.
The fitted ellipse explained the visible measurements reasonably well
but differed from the ground truth in the unobserved region.

This tutorial investigates whether information about the observation
process can reduce that ambiguity.

The tutorial considers two cases:

#. The observed arc and noise variance are known exactly.
#. The observed arc and noise variance are unknown, but prior information
   about them is available.

The corresponding executable examples are available in:

.. code-block:: text

   examples/basic/fit_ellipse_partial.py
   examples/basic/fit_ellipse_partial_with_priors.py

Problem Setup
-------------

We use the same ground-truth ellipse and observations as in the Getting
Started example.

.. code-block:: python

   import numpy as np

   from bayes_shape_fitting.models.shapes.ellipse import Ellipse

   shape = Ellipse()

   true_shape_params = np.array([
       0.0,  # center x-coordinate
       0.0,  # center y-coordinate
       3.0,  # first semi-axis
       2.0,  # second semi-axis
       0.3,  # rotation angle in radians
   ])

   rng = np.random.default_rng(42)

   num_points = 150
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

The sampled interval has center

.. math::

   c = \frac{0.10 + 0.50}{2} = 0.30

and half-width

.. math::

   w = \frac{0.50 - 0.10}{2} = 0.20.

Using the Known Arc
-------------------

Suppose first that the center and half-width of the observed interval
are known exactly.

We represent the known arc with a fixed uniform source density:

.. code-block:: python

   from bayes_shape_fitting.inference.source_densitites import (
       UniformSourceDensity,
   )

   known_source_density = UniformSourceDensity(
       n_dim=shape.parameter_dim,
       domain_low=shape.domain_low,
       domain_high=shape.domain_high,
       periodic=shape.periodic,
       fixed_params=np.array([
           0.30,  # known arc center
           0.20,  # known arc half-width
       ]),
   )

Because ``fixed_params`` is supplied, these source parameters are not
estimated during fitting.

We also assume that the measurement-noise variance is known. The
observations were generated with standard deviation ``0.15``, so the
variance is:

.. math::

   \sigma^2 = 0.15^2.

.. code-block:: python

   from bayes_shape_fitting.inference.noise_functions import IsotropicNoise

   known_noise_function = IsotropicNoise(
       dim_param=shape.parameter_dim,
       dim_ambient=shape.ambient_dim,
       fixed_params=0.15**2,
   )

The ellipse is fitted using these known observation-process parameters:

.. code-block:: python

   from bayes_shape_fitting.inference.fit import fit

   result_known = fit(
       points,
       shape,
       source_density=known_source_density,
       noise_function=known_noise_function,
       likelihood_kwargs={
           "num_samples": 300,
       },
   )

The resulting reconstruction is shown below:

.. image:: /_static/fit_ellipse_partial.png
   :alt: Ellipse reconstruction with known source and noise parameters
   :align: center
   :width: 75%

This example represents an idealized situation. The exact observed
interval and noise variance are rarely known in practice. Nevertheless,
the result illustrates how information about the observation process can
reduce ambiguity in the reconstructed geometry.

When the Arc Is Unknown
-----------------------

In a realistic application, the exact center and half-width of the
observed arc may not be known.

We therefore define the same uniform source-density model without
``fixed_params``:

.. code-block:: python

   source_density = UniformSourceDensity(
       n_dim=shape.parameter_dim,
       domain_low=shape.domain_low,
       domain_high=shape.domain_high,
       periodic=shape.periodic,
   )

The source parameters

.. code-block:: text

   [arc_center, arc_half_width]

are now estimated jointly with the ellipse parameters.

Similarly, we use an isotropic noise model with unknown variance:

.. code-block:: python

   noise_function = IsotropicNoise(
       dim_param=shape.parameter_dim,
       dim_ambient=shape.ambient_dim,
   )

Adding Prior Information
------------------------

Although the exact source and noise parameters are unknown, approximate
information about the measurement process may still be available.

For example, the measurement configuration may provide information
about the approximate visible fraction of the object or the expected
sensor uncertainty.

We encode such information using Gaussian priors.

First, we place a prior on the source parameters:

.. code-block:: python

   from bayes_shape_fitting.inference.priors import GaussianPrior

   source_density.prior = GaussianPrior(
       mean=np.array([
           0.50,  # prior mean for the arc center
           0.20,  # prior mean for the arc half-width
       ]),
       cov=np.diag([
           1.0,
           0.00065,
       ]),
   )

This prior expresses relatively weak information about the arc center
and stronger information about the arc half-width.

The half-width prior reflects approximate knowledge of the visible
fraction without assuming that the location of the visible interval is
known precisely.

We also place a prior on the unknown noise variance:

.. code-block:: python

   noise_function.prior = GaussianPrior(
       mean=np.array([
           0.15**2,
       ]),
       cov=np.diag([
           0.002**2,
       ]),
   )

The prior is defined over the variance :math:`\sigma^2`, matching the
parameterization used by ``IsotropicNoise``.

Priors and Initial Values
-------------------------

Priors and initial values play different roles.

A prior contributes to the posterior objective throughout optimization.
An initial value only specifies where the numerical optimization begins.

For this example, we initialize the source parameters as:

.. code-block:: python

   source_init = np.array([
       0.50,  # initial arc center
       0.15,  # initial arc half-width
   ])

and the noise variance as:

.. code-block:: python

   noise_init = np.array([
       0.08,
   ])

These values do not fix the parameters. The parameters remain free to
change during fitting.

Joint Estimation
----------------

We now estimate the ellipse, source-density, and noise parameters
jointly:

.. code-block:: python

   result_prior = fit(
       points,
       shape,
       source_density=source_density,
       noise_function=noise_function,
       source_init=source_init,
       noise_init=noise_init,
       likelihood_kwargs={
           "num_samples": 300,
       },
   )

BLISS constructs the marginal likelihood and posterior internally. The
priors attached to ``source_density`` and ``noise_function`` are included
in the posterior objective.

The resulting reconstruction is shown below:

.. image:: /_static/fit_ellipse_partial_with_priors.png
   :alt: Ellipse reconstruction with source and noise priors
   :align: center
   :width: 75%

Inspecting the Estimated Observation Model
------------------------------------------

The estimated source and noise parameters are available in the result:

.. code-block:: python

   print("Estimated shape parameters:")
   print(result_prior.shape_params)

   print("\nEstimated source parameters [center, half-width]:")
   print(result_prior.source_params)

   print("\nEstimated noise variance:")
   print(result_prior.noise_params)

The estimated source parameters describe the interval that BLISS
considers most likely to have generated the observations.

The estimated noise parameter describes the inferred isotropic
measurement variance.

Animating the Joint Estimation
------------------------------

The optimization histories can be animated to show how the shape,
source-density, and noise estimates change during fitting:

.. image:: /_static/fit_ellipse_partial_with_priors.gif
   :alt: Joint estimation of ellipse, source-density, and noise parameters
   :align: center
   :width: 90%

The animation also displays the source and noise priors together with
the corresponding parameter trajectories.

Comparison
----------

The three reconstructions use the same observations and ellipse model.
They differ only in how the observation process is represented:

#. **Default model:** observations are assumed to originate uniformly
   from the complete parameter domain.
#. **Known observation process:** the arc center, arc half-width, and
   noise variance are supplied exactly.
#. **Prior-based observation model:** the source and noise parameters are
   estimated, while approximate information is incorporated through
   priors.

The known-parameter case demonstrates the value of accurate information
about the observation process, but it is idealized.

The prior-based model offers a more realistic alternative. It allows
uncertain information about visibility and sensor noise to guide the
reconstruction without treating those quantities as exactly known.

Running the Examples
--------------------

Run the known-arc example with:

.. code-block:: bash

   python examples/basic/fit_ellipse_partial.py

Run the prior-based example with:

.. code-block:: bash

   python examples/basic/fit_ellipse_partial_with_priors.py

