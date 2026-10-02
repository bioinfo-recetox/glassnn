# GlassNN

A small, transparent neural-network library for teaching. Its source code is
meant to be read: every operation is a few lines of forward and backward code,
with the derivative written in the docstring. Names follow PyTorch
(`Module`, `Linear`, `forward`, `zero_grad`, `step`, `lr`), so code transfers
almost line by line.

GlassNN accompanies the neural-network lectures of the course *Advanced
Machine Learning* (RECETOX, Masaryk University).

- **Book** (theory, implementation notes, tutorials):
  <https://bioinfo-recetox.github.io/glassnn/>
- **API reference:** <https://bioinfo-recetox.github.io/glassnn/api/>

> Status: early development (milestone M0, project skeleton). There is no
> usable API yet.

## Installation

GlassNN needs Python 3.13 or newer and depends only on NumPy and SciPy.
Install a tagged release directly from GitHub:

```
pip install "glassnn @ git+https://github.com/bioinfo-recetox/glassnn@v0.1.0"
```

### Optional GPU backend

The optional GPU backend uses [CuPy](https://cupy.dev) with CUDA 13
(`pip install "glassnn[gpu] @ git+..."`). CuPy is the "RAPIDS option": RAPIDS
libraries (cuDF, cuML) exchange arrays with CuPy without copies through
DLPack; GlassNN itself needs only an array library and does not depend on
RAPIDS.

## Development

```
uv sync --extra dev --extra docs --extra book --extra examples
uv run pytest
```

See `CONTRIBUTING.md`.

## License and citation

MIT license, see `LICENSE`. To cite GlassNN, see `CITATION.cff`.
