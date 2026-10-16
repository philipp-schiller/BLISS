BLISS Documentation
===================

Bayesian Latent Inference via Sampling Density and Sensor Noise
---------------------------------------------------------------

BLISS is a Python framework for Bayesian geometric shape fitting under
non-uniform sampling densities and heteroscedastic measurement noise.

BLISS models three components of the observation process:

* the geometric shape,
* the source density on the shape, and
* the measurement noise.

Getting Started
---------------

The :doc:`tutorials/getting_started` guide explains how to install
BLISS, prepare measurement points, fit a shape, and inspect the result.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   tutorials/getting_started

Tutorials
---------

The tutorials demonstrate how observation-process information can be
incorporated into more challenging reconstruction problems.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/ellipse_reconstruction

Project Resources
-----------------

* Example scripts: ``examples/basic/``
* Example notebooks: ``examples/notebooks/``
* Source code: ``src/bayes_shape_fitting/``
* Tests: ``tests/``