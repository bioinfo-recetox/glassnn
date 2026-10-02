# CLAUDE.md: instructions for working in this repository

`glassnn` is a small, transparent neural-network library for teaching (MIT license). Read `PLAN.md` first: it defines scope, architecture, milestones and acceptance criteria. If this file and `PLAN.md` disagree, ask before proceeding.

## How to work

1. Work **one milestone at a time**, in the order of `PLAN.md` section 7. Before writing code for a milestone, write a short plan (files to create, the order, how each acceptance criterion will be tested) and wait for approval if anything is ambiguous.
2. Write the **test first** for every op or layer, then the implementation. A feature is not done until its tests pass.
3. Commit at the end of each milestone (small, descriptive commit messages). Do not push to `main` directly; use a branch per milestone and a pull request.
4. When a design question comes up that `PLAN.md` does not answer, list the options with their trade-offs and ask. Do not decide silently.

## Commands (using uv)

```
uv sync --extra dev --extra docs --extra examples   # create .venv (not --all-extras: the gpu extra needs CUDA)
uv run pytest                 # all tests except torch/gpu markers
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # once; `uv sync` removes it again
uv run pytest -m torch        # cross-checks against PyTorch
uv run pytest -m gpu          # CuPy parity tests (needs CuPy and a GPU)
uv run ruff check . && uv run ruff format --check .
uv run sphinx-build -W docs docs/_build/html
uv run quarto render book && uv run python book/_check_api_links.py
```

## Conventions

- **Readability beats cleverness.** This code is read by students. Short functions, clear names, no metaprogramming, no hidden global state except the documented backend, generator and grad-mode flag (`no_grad`).
- **PyTorch naming parity** for classes, functions and arguments (`Module`, `Linear`, `forward`, `zero_grad`, `step`, `lr`, `weight_decay`, `eps`, `betas`). Where behaviour differs, document it in the docstring under "Differences from PyTorch".
- **Docstrings:** Google style (rendered by `sphinx.ext.napoleon`). Each public symbol states the math in LaTeX, tensor shapes, parameters, a short doctested example, and `References` with BibTeX keys from `docs/references.bib`. Each autodiff op writes its derivative in the docstring.
- **Type hints** on the public API. Raise informative errors (shape mismatches name both shapes). No in-place tensor operations; raise `NotImplementedError` with an explanation if attempted.
- **Dependencies:** runtime = `numpy` and `scipy` only. Never import `torch` or `cupy` at module import time in `src/`. `torch` is allowed only in `tests/` (marker `torch`) and in `examples/colab_pytorch/`.
- **Randomness:** only through the generator in `backend.py` or an explicit `generator=` argument. Tests set seeds.
- **Numerics:** tests run in float64 with central finite differences; default dtype elsewhere is float32. Use stable formulations (`log_softmax` with max subtraction, `logsumexp`, `softplus` with thresholds).

## Do not

- Copy code from `johnma2006/candle` (no license) or from any source without a compatible license; implement from the literature listed in `PLAN.md`. Do not paste code from `micrograd` either; the idea is acknowledged, the code is original.
- Invent references. Every entry in `docs/references.bib` must be verified (title, authors, venue, year, DOI or arXiv id) against the publisher or arXiv page. If you cannot verify one, mark it `TODO-verify` in the bib file and tell me.
- Add runtime dependencies, features outside `PLAN.md` scope, or change the public API without updating tests, docs and `CHANGELOG.md`.
- Leave failing, skipped-without-reason, or commented-out tests.

## Definition of done (every milestone)

- All acceptance criteria of the milestone in `PLAN.md` are met and each is covered by a test.
- `pytest`, `ruff`, and the documentation build (warnings as errors) pass locally and in CI.
- New public symbols are documented with math, shapes, example and references.
- `CHANGELOG.md` is updated, and the final message of the milestone lists: what was done, what was verified and how, what was not verified, and any decision I need to take.
