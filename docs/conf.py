"""Sphinx configuration for the GlassNN API reference."""

import glassnn

project = "GlassNN"
author = "Vlad Popovici"
copyright = "2026, Vlad Popovici, RECETOX - Faculty of Science, Masaryk University"
release = glassnn.__version__
version = ".".join(release.split(".")[:2])

# Base URL of the Quarto book; API pages link to chapters relative to it.
BOOK_URL = "https://bioinfo-recetox.github.io/glassnn/"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinxcontrib.bibtex",
]

# Google-style docstrings only (PLAN.md, section 6.1).
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# "Shapes:" sections list tensor shapes, rendered like "Args:".
napoleon_custom_sections = [("Shapes", "params_style")]

autodoc_typehints = "description"
autodoc_member_order = "bysource"

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"

# :book:`Title <chapters/02-autodiff.html>` links to a chapter of the book.
extlinks = {"book": (BOOK_URL + "%s", "%s")}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

rst_epilog = f"""
.. |book| replace:: GlassNN book
.. _book: {BOOK_URL}
"""

exclude_patterns = ["_build"]
html_theme = "alabaster"
html_title = f"GlassNN {release} API reference"
