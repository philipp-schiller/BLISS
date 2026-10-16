import os
import sys

sys.path.insert(
    0,
    os.path.abspath("../src"),
)

project = "BLISS"
author = "Philipp Schiller"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "alabaster"
html_static_path = ["_static"]