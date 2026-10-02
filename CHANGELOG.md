# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [semantic versioning](https://semver.org/) through
annotated git tags.

## [Unreleased]

### Added

- Project skeleton (milestone M0): `src` layout, `pyproject.toml` with the
  version derived from git tags, MIT license, empty Sphinx API reference and
  empty Quarto book, pre-commit configuration, CI, site deployment and
  release workflows.
- IEEE citation style (CSL, CC BY-SA 3.0) for the book.
- `glassnn.backend`: active array module `xp`, `get_backend`/`set_backend`
  (NumPy; CuPy follows in M6), `get_default_dtype`/`set_default_dtype`
  (float32 by default), `asarray` with the dtype rule, `to_numpy`.
- `glassnn.Tensor` with reverse-mode automatic differentiation: `+ - * / **`
  (scalar or tensor exponent), unary minus, `@` (vectors, matrices, batches),
  `exp`, `log`, `sum`/`mean` (`dim`, `keepdim`), `reshape`, `transpose`,
  `permute`, `.T` (at most 2-D), NumPy indexing (slices, integer arrays,
  boolean masks); broadcasting in every binary operation; `backward` with an
  iterative topological sort; `retain_grad`, `detach`, `item`, `numpy`.
- `glassnn.no_grad` as context manager and decorator.
- `glassnn.gradcheck.gradcheck`: central finite differences in float64.
- In-place operations (`+=`, item assignment, ...) raise `NotImplementedError`.
- API reference: one page per module, links to the book chapters.
- Book: chapter 1 (introduction and installation) and chapter 2 (automatic
  differentiation); helpers `show_source`, `api_link`/`api_url`,
  `src_link`/`src_url`, `see_also`.
- Verified references: Rumelhart et al. 1986, Baydin et al. 2018, Griewank
  and Walther 2008, Goodfellow et al. 2016, Harris et al. 2020.

### Differences from PyTorch

- Floating-point NumPy arrays are cast to the default dtype (PyTorch keeps
  float64).
- The graph is kept after `backward()` (no `retain_graph` needed).
- `Tensor.numpy()` returns a copy.

## Later

Features deliberately kept out of scope (see `PLAN.md`, section 10):

- none yet.
