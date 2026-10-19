# Contributing to GlassNN

GlassNN is a teaching library: its source code is read by students.
Readability comes before speed and before cleverness.

## Setup

```
uv sync --extra dev --extra docs --extra book --extra examples
uv run pre-commit install
```

The `gpu` extra (`cupy-cuda13x[ctk]`) needs an NVIDIA GPU with a CUDA 13
driver; the CUDA compiler and headers come as wheels. Install it only on such
a machine, together with the other extras (`uv sync` removes extras that are
not named): `uv sync --extra dev --extra docs --extra book --extra examples
--extra gpu`. On a machine with several GPUs, choose one with
`CUDA_VISIBLE_DEVICES=1 uv run pytest -m gpu`. The GPU benchmark is
`uv run python benchmarks/bench_backends.py`; after a change of the backend,
re-run it with `--json book/assets/benchmark_backends.json` to refresh the
table of book chapter 13.

## Checks

```
uv run pytest                                  # tests (torch and gpu markers excluded)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu  # once; not in the lock file
uv run pytest -m torch                         # PyTorch cross-checks
uv run pytest -m gpu                           # CuPy parity tests (needs CuPy and a GPU)
uv run pytest -m network                       # Colab notebook C (needs torch, transformers, internet)
uv run ruff check . && uv run ruff format --check .
uv run sphinx-build -W docs docs/_build/html   # API reference
uv run quarto render book                      # the book
uv run python book/_check_api_links.py         # links within and between book and API (after both builds)
uv run sphinx-build -b linkcheck docs docs/_build/linkcheck  # external links, by hand before a release
```

After editing `book/assets/rosetta.toml` (the table of differences from
PyTorch), copy it into the Colab notebooks with
`uv run python examples/colab_pytorch/_sync_rosetta.py`; `tests/test_rosetta.py`
fails until then, and also when a docstring gains a "Differences from
PyTorch" note without a row in the table. The Colab notebooks are executed by
`uv run pytest -m torch` (needs the `examples` extra). Notebook C also needs
Hugging Face `transformers` (`uv pip install transformers`; like torch, not
in the lock file) and downloads a model; it is executed by
`uv run pytest -m network`, which CI does not run.

## Releasing

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`
   (the tagging day), start with a short `### Highlights` list, add an empty
   `## [Unreleased]` above it and update the comparison links at the bottom.
2. Set `version` and `date-released` in `CITATION.cff`, and the pinned tag in
   `README.md`, book chapter 1 and Colab notebook B.
3. Update `VERSION` in `tests/test_release.py` and run the checks above;
   `tests/test_release.py` fails while any of these disagree.
4. After the pull request is merged: `git tag -a vX.Y.Z -m "GlassNN X.Y.Z"`
   on `main` and `git push origin vX.Y.Z`. The `release.yml` workflow runs the
   tests, builds the sdist and the wheel, and creates the GitHub Release with
   the changelog section as notes.

## Rules

- Write the test first, then the implementation.
- Google-style docstrings with the math, the shapes, a doctested example and
  references to `docs/references.bib`.
- Runtime dependencies are `numpy` and `scipy` only.
- Do not copy code from sources without a compatible license.
- One branch per milestone, merged through a pull request; update
  `CHANGELOG.md`.

See `CLAUDE.md` and `PLAN.md` for the full conventions.
