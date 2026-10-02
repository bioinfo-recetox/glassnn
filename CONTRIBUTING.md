# Contributing to GlassNN

GlassNN is a teaching library: its source code is read by students.
Readability comes before speed and before cleverness.

## Setup

```
uv sync --extra dev --extra docs --extra book --extra examples
uv run pre-commit install
```

The `gpu` extra (`cupy-cuda13x`) needs an NVIDIA GPU with CUDA 13; install it
only on such a machine: `uv sync --extra gpu`.

## Checks

```
uv run pytest                                  # tests (torch and gpu markers excluded)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu  # once; not in the lock file
uv run pytest -m torch                         # PyTorch cross-checks
uv run pytest -m gpu                           # CuPy parity tests (needs CuPy and a GPU)
uv run ruff check . && uv run ruff format --check .
uv run sphinx-build -W docs docs/_build/html   # API reference
uv run quarto render book                      # the book
uv run python book/_check_api_links.py         # book -> API links (after both builds)
```

## Rules

- Write the test first, then the implementation.
- Google-style docstrings with the math, the shapes, a doctested example and
  references to `docs/references.bib`.
- Runtime dependencies are `numpy` and `scipy` only.
- Do not copy code from sources without a compatible license.
- One branch per milestone, merged through a pull request; update
  `CHANGELOG.md`.

See `CLAUDE.md` and `PLAN.md` for the full conventions.
