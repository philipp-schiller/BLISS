# BLISS

**Bayesian Latent Inference via Sampling Density and Sensor Noise**

BLISS is a Python framework for Bayesian geometric primitive fitting under non-uniform sampling densities and heteroscedastic measurement noise.

Unlike classical shape fitting approaches that estimate only geometric parameters, BLISS explicitly models the observation process through probabilistic source and noise models. This enables robust reconstruction in challenging scenarios where only a part of an object is observed and measurement noise varies across the shape.

---

## Features

- Bayesian shape fitting for geometric primitives and parametric models
- Explicit modelling of:
  - Shape parameters
  - Source distributions
  - Measurement noise

---

## Installation

Clone the repository:

```bash
git clone https://github.com/philipp-schiller/BLISS
cd BLISS
```

Install in editable mode:

```bash
pip install -e .
```

Alternatively:

```bash
pip install .
```

---

## Quick Start

The example below generates noisy points from an ellipse and fits an ellipse model to them.

```python
import numpy as np

from bayes_shape_fitting.inference.fit import fit
from bayes_shape_fitting.models.shapes.ellipse import Ellipse

# Generate synthetic ellipse observations
rng = np.random.default_rng(42)
t = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)

points = np.column_stack([
    2.0 * np.cos(t),
    1.0 * np.sin(t),
])

points += rng.normal(scale=0.05, size=points.shape)

# Fit an ellipse
shape = Ellipse()
result = fit(points, shape)

print("Estimated shape parameters:", result.shape_params)
print("Estimated noise parameters:", result.noise_params)
print("Converged:", result.converged)
```

Here, `points` is an array of shape `(N, 2)`, where each row contains one observed 2D point. For a 3D fitting problem, the expected shape is `(N, 3)`.

By default, BLISS uses:

- a fixed uniform source density over the complete parameter domain,
- an isotropic Gaussian noise model with an estimated variance,
- a marginal likelihood,
- and ECM optimization.

---

## Supported Shape Models

### Geometric Primitives

- Circle
- Sphere
- Ellipse
- Ellipsoid
- Cylinder

### Parametric Models

- B-Splines
- Bézier Curves
- Generalized Cylinders
- Polygonal Models

New shape models can be added by inheriting from the provided base classes.

---

## Examples

Example scripts can be found in:

```text
examples/basic/
```

Example notebooks can be found in:

```text
examples/notebooks/
```

---

## Documentation

The documentation introduces BLISS through a three-stage example:

1. Fit partial observations using the default source model.
2. Repeat the fit when the observed arc is known.
3. Treat the arc and noise parameters as unknown and incorporate prior information.

Start with:

- docs/tutorials/getting_started.rst
- docs/tutorials/ellipses_reconstruction.rst

Executable versions of the tutorials are available in `examples/basic/`.