# GlassNN: a transparent teaching library for neural networks

Plan for Claude Code. Repository: `https://github.com/bioinfo-recetox/glassnn` (to be created; the name was confirmed free in the organization). Import name and distribution name: `glassnn`. License: MIT.

Revision 4. All decisions of section 12 are settled; the Python floor is verified against Google Colab and the course repository (section 9).

## 1. Purpose and scope

A small deep-learning library whose **source code is meant to be read by students**, together with a **book** (Quarto) that explains the theory, points to the implementation, and contains the tutorials. It supports the two neural-network lectures of the course *Advanced Machine Learning* (bioinformatics students who already know SVMs, kernels and statistical learning theory), and nothing else.

- **Lecture 10, training and generalization:** backpropagation, initialization, optimizers, regularization, memorization of noise, double descent, neural tangent kernel (NTK).
- **Lecture 11, architectures and representations:** 1D convolutions for sequences (DNA motifs), attention and a small transformer encoder, autoencoders.
- The library contains **all code needed by these examples**. PyTorch is used only in separate Google Colab notebooks (section 9). PyTorch is never a dependency of the package.

**Non-goals:** speed competitive with PyTorch, large models, pretrained model loading, distributed training, mixed precision, a general-purpose framework, in-place operations, higher-order derivatives (the empirical NTK needs only first derivatives).

**Design priorities, in this order:** (1) readability of the code and of the math; (2) numerical correctness, verified; (3) PyTorch-compatible naming, so that code transfers almost line by line; (4) adequate speed at toy scale (batch 32, sequences of length 60 to 200, a few thousand parameters to a few million); (5) optional GPU backend.

**Three documentation products, one bibliography:**

| Product | Tool | Contains | Audience |
|---|---|---|---|
| API reference | Sphinx | docstrings (Google style), signatures, shapes, math, references | someone using or extending the code |
| **The book** | **Quarto** | theory, derivations, pointers to the implementation, runnable tutorials, exercises | students |
| Examples and Colab notebooks | Jupyter | reference notebooks used by the lectures; PyTorch counterparts | lectures 10 and 11 |

The single bibliography file `docs/references.bib` is used by Sphinx (`sphinxcontrib-bibtex`) and by the book (Quarto citations).

## 2. Hard constraints

- **Python >= 3.13.** `.python-version` = 3.13 (the default and the CI version).
- **NumPy >= 2.5** and **SciPy >= 1.18**: the latest releases compatible with every other requirement (checked on PyPI on 2026-10-02: NumPy 2.5.3, SciPy 1.18.1 requiring `numpy>=2.0,<2.8`, CuPy 14.2.0 requiring `numpy>=2.0,<2.6`). Runtime dependencies: `numpy`, `scipy` only.
- **GPU option:** CuPy >= 14 as the single optional extra `gpu` (`cupy-cuda13x`, CUDA 13). The extra declares `numpy<2.6` because of CuPy's cap. When NumPy 2.6 appears, re-check CuPy's cap before widening any range.
- Nothing in the package may import `cupy` unless the user selects that backend.
- **Clean room:** do **not** copy code from `johnma2006/candle` (it has no license). Karpathy's `micrograd` (MIT) is an acknowledged inspiration; do not copy its code verbatim either. Implement from the literature in section 11. Third-party code, if ever included, needs a compatible license and a notice in `NOTICE`.
- **License:** MIT. `LICENSE` first line after the title: `Copyright (c) 2026 Vlad Popovici, RECETOX - Faculty of Science, Masaryk University`. `CITATION.cff` lists the same author and affiliation.
- **Versioning:** by git. Releases are annotated tags `vMAJOR.MINOR.PATCH` (semantic versioning); no PyPI upload. The package version is derived from the tag (`hatchling` with `hatch-vcs`, or `setuptools-scm`); `glassnn.__version__` reads it.
- **Reproducibility:** every source of randomness goes through one seedable generator (section 4.1).

## 3. Repository layout

```
glassnn/
  LICENSE  README.md  CITATION.cff  CHANGELOG.md  CONTRIBUTING.md  CLAUDE.md  PLAN.md  NOTICE
  pyproject.toml            # src layout; extras: gpu (cupy-cuda13x), docs, book, dev, examples
  .python-version           # 3.13
  .pre-commit-config.yaml   # ruff (lint + format), end-of-file fixer
  .github/workflows/        # ci.yml, site.yml, release.yml
  src/glassnn/
    __init__.py             # public API re-exports, __version__
    backend.py              # xp selection, to_numpy, dtype default, random generator, DLPack helpers
    tensor.py               # Tensor, no_grad, backward machinery
    functional.py           # stateless ops (the "F" namespace)
    nn/                     # Module, Parameter, layers, init, losses
    optim/                  # SGD, Adam, AdamW, RMSprop, schedulers, clip_grad_norm_
    data.py                 # DataLoader, one_hot
    analysis.py             # empirical_ntk, activation statistics, count_parameters
    gradcheck.py            # finite-difference verification
  tests/                    # pytest; mirrors src; markers "torch" and "gpu"
  docs/                     # Sphinx API reference; references.bib; one shared bibliography
  book/                     # the Quarto book (section 6)
  examples/                 # reference notebooks used by the lectures (section 9)
  examples/colab_pytorch/   # PyTorch counterparts, not part of the package
  benchmarks/               # small timing scripts, not run in CI
```

## 4. Architecture

### 4.1 Backend and randomness (`backend.py`)

- Module-level `xp` (the active array module), `set_backend("numpy" | "cupy")`, `get_backend()`, `to_numpy(x)`, `asarray(x, dtype=None)`.
- **Default dtype `float32`**; `set_default_dtype("float64")` for tests and gradient checks. Every layer takes an optional `dtype`.
- Use only the NumPy/CuPy common subset through `xp`. Special functions (`erf`, `logsumexp`) go through a small adapter: `scipy.special` on CPU, `cupyx.scipy.special` on GPU. Avoid `scipy.signal`; convolutions use `sliding_window_view` plus `tensordot`/`einsum` so they run on both backends.
- `manual_seed(seed)` creates the global generator (`numpy.random.default_rng` or `cupy.random.default_rng`). Initializers, dropout and `DataLoader` shuffling draw from it, or from an explicit `generator=` argument.
- **RAPIDS note.** RAPIDS (cuDF, cuML) is a data-frame and classical-ML suite; a neural-network library needs only an array library. The "RAPIDS option" is therefore the **CuPy** backend, which RAPIDS libraries interoperate with (zero-copy exchange through DLPack; `to_dlpack`/`from_dlpack` helpers in `backend.py`). No cuDF or cuML dependency. Say this in the book and in the README.

### 4.2 Autodiff core (`tensor.py`)

- `Tensor(data, requires_grad=False)`: attributes `data`, `grad`, `requires_grad`, `shape`, `dtype`, plus private `_parents`, `_backward`, `_op` (name, for debugging and `repr`).
- Reverse-mode automatic differentiation on a dynamically built graph; `backward()` does an iterative (non-recursive) topological sort and accumulates into `.grad` exactly like PyTorch (so `zero_grad()` is needed). `backward()` requires a scalar unless a gradient is passed.
- Operators `+ - * / ** @`, unary minus, indexing and slicing, `.T`, `.sum`, `.mean`, `.reshape`, `.transpose`, `.detach()`, `.item()`, `.numpy()`.
- Broadcasting: every binary op reduces the incoming gradient back to the operand shape (`_unbroadcast`), tested explicitly.
- `no_grad()` context manager and decorator. No in-place ops: raise a clear error.
- Each op is a few lines of forward and a few lines of backward, with the derivative written in the docstring. This is the part students read, and the book quotes it.

### 4.3 Functional ops (`functional.py`)

relu, leaky_relu, gelu (exact with erf, and the tanh approximation), tanh, sigmoid, softplus, softmax, log_softmax (stable via max subtraction), logsumexp, dropout (inverted dropout), layer_norm, batch_norm, linear, conv1d, conv2d, max_pool1d/2d, avg_pool1d/2d, embedding, scaled_dot_product_attention (with additive or boolean mask), losses: `cross_entropy` (logits and integer targets, optional label smoothing), `binary_cross_entropy_with_logits`, `mse_loss`.

Convolutions: im2col with `sliding_window_view` (Chellapilla et al. 2006); backward via the transposed correspondence. Validate against a slow, obviously correct loop implementation in the tests.

### 4.4 Modules (`nn/`)

- `Module` (parameter registration by attribute assignment, `parameters()`, `named_parameters()`, `train()`, `eval()`, `zero_grad()`, `state_dict()`, `load_state_dict()`, `apply(fn)`, `__call__` -> `forward`), `Parameter`, `Sequential`, `ModuleList`.
- Layers: `Linear` (option `parametrization="standard" | "ntk"`: the NTK variant scales the pre-activation by `1/sqrt(fan_in)` and keeps weights N(0,1), as in Jacot et al. 2018), `Embedding`, `Dropout`, `LayerNorm`, `BatchNorm1d` (running statistics, train/eval behaviour), `Conv1d`, `Conv2d`, `MaxPool1d`, `AvgPool1d`, `Flatten`, activations as modules, `MultiheadAttention`, `PositionalEncoding` (sinusoidal, Vaswani et al. 2017), `TransformerEncoderLayer` (pre-LN, GELU feed-forward), `TransformerEncoder`.
- `nn.init`: `xavier_uniform_/normal_`, `kaiming_uniform_/normal_` (fan_in and fan_out modes, gain per nonlinearity), `normal_`, `zeros_`, `ones_`. Default init of `Linear` documented and equal to PyTorch's (Kaiming uniform with a = sqrt(5)), so that behaviour matches the Colab notebooks.
- Losses as modules and as functions.

### 4.5 Optimizers (`optim/`)

`SGD(params, lr, momentum=0, nesterov=False, weight_decay=0)`, `Adam(params, lr, betas, eps, weight_decay)` (coupled L2), `AdamW` (decoupled, Loshchilov and Hutter), `RMSprop`. Schedulers: `StepLR`, `CosineAnnealingLR`, `LinearWarmup`. `clip_grad_norm_`. The argument is `lr`, as in PyTorch.

### 4.6 Data and analysis

- `data.DataLoader(X, y, batch_size, shuffle, drop_last, generator)` returning `Tensor`s; `one_hot`.
- `analysis.empirical_ntk(model, X1, X2=None, output_index=None)`: the matrix of gradient inner products `J(X1) J(X2)^T` for a scalar-output (or selected-output) model, computed with one backward pass per example (documented cost `O(n P)` memory). `analysis.activation_stats(model, X)` (mean and std of pre-activations per layer, for the initialization experiments). `analysis.count_parameters`.
- `gradcheck.gradcheck(fn, inputs, eps, atol, rtol)`: central finite differences in float64.

## 5. Quality bar

- **Gradient checks:** every op and every layer is verified against central finite differences in float64, on random shapes, including broadcasting, empty-batch edge cases and numerically hard inputs (large logits).
- **Reference implementations:** conv and attention against slow loop versions.
- **Cross-validation against PyTorch** (dev-only, marker `torch`, skipped if PyTorch is missing): same weights, same inputs, compare forward outputs and parameter gradients for Linear, Conv1d, Conv2d, LayerNorm, BatchNorm1d, MultiheadAttention, TransformerEncoderLayer, cross_entropy, and a few optimizer steps (SGD with momentum, Adam, AdamW). Tolerances stated per op.
- **Behavioural tests:** an MLP learns XOR and the moons data (test accuracy above a stated threshold); a Conv1d model detects a planted motif; an autoencoder reduces reconstruction error; training with a fixed seed is bit-for-bit reproducible on CPU.
- **Backend parity** (marker `gpu`, skipped without CuPy): NumPy and CuPy agree on forward and backward.
- **Docstring examples** are executed as doctests; **book chapters are executed** at build time, so a broken example fails the build.
- Coverage target 90 percent for `src/`. Type hints on the public API, checked with a lenient `mypy` or `pyright` configuration. `ruff` for lint and format.
- **CI** (`ci.yml`): lint, tests on Python 3.13 with the locked (latest compatible) NumPy and SciPy; a separate job installs CPU PyTorch from the CPU wheel index and runs the `torch` cross-checks. `site.yml` builds the Sphinx API reference and the Quarto book and deploys both to GitHub Pages from `main`. `release.yml` runs on a version tag: tests, build of sdist and wheel, and a **GitHub Release** with the wheel attached and the CHANGELOG section as notes. No PyPI publication.

## 6. Documentation

### 6.1 API reference (Sphinx, `docs/`)

- Extensions: `sphinx.ext.autodoc`, **`sphinx.ext.napoleon`** (configured for **Google style**: `napoleon_google_docstring = True`, `napoleon_numpy_docstring = False`), `sphinx.ext.mathjax`, `sphinx.ext.viewcode`, `sphinx.ext.doctest`, `sphinx.ext.intersphinx` (Python, NumPy, SciPy), `sphinxcontrib.bibtex`. Note: napoleon ships with Sphinx as `sphinx.ext.napoleon`; the old standalone package `sphinxcontrib-napoleon` is obsolete and must not be installed.
- **Docstring template (Google style).** Every public function and class has: one-line summary; the mathematical definition (`.. math::` blocks or `:math:` roles); `Args:` with shapes and dtypes; `Returns:`; `Raises:`; `Example:` (doctest, runnable); `Note:` for "Differences from PyTorch" where relevant; `References:` with `:cite:p:` keys from `docs/references.bib`; and a `See also:` line pointing to the book chapter (full URL built from a constant in `docs/conf.py`).
- The API reference contains no long theory: it links to the book.

### 6.2 The book (Quarto, `book/`)

A Quarto **book** project that explains the theory and the implementation side by side, and holds the tutorials. Use the Quarto version of the course slides (1.10.18) and pin it in CI.

```
book/
  _quarto.yml          # project: type: book; chapters; bibliography; format: html (pdf optional)
  index.qmd            # preface: what GlassNN is, how to read the book, installation
  chapters/NN-name.qmd # one file per chapter
  _macros.qmd          # shared LaTeX macros (included at the top of every chapter, under the first heading)
  _helpers.py          # show_source(obj), api_link(name), src_link(path, start, end), see_also(...)
  references.qmd       # generated bibliography page
  assets/              # figures, CSS
```

Settings: `bibliography: ../docs/references.bib`; a CSL style (for example `apa` or `ieee`); `execute: freeze: auto` (unchanged chapters are not re-run); `html-math-method: mathjax`; figure and equation cross-references (`@fig-`, `@eq-`, `@sec-`); callout blocks for "Where in the code", "Differences from PyTorch", "Exercise". Render with `uv run quarto render book` (so that Quarto uses the project environment, not a system Python).

**Chapter template.** Every chapter has these sections, in this order:
1. *Goals* (3 to 4 bullet points, same style as the lecture slides).
2. *Theory*: derivations with numbered equations and citations (`@key`).
3. *Where in the code* (callout): the modules and functions that implement the theory, with links to the API reference and to the source at the current tag (`src_link`), plus the **actual source** of the key functions displayed with `show_source`, so that the text never goes out of sync with the code.
4. *Try it*: runnable cells using the library, with fixed seeds.
5. *Exercises* (with the expected finding stated in a collapsed solution block).
6. *Notes and references*: pointers to the literature, and to the corresponding course lecture.

**Chapter map.**

| Part | Chapter | Main code | Lecture |
|---|---|---|---|
| I Foundations | 1 Introduction and installation | `__init__`, `backend` | 10 |
| | 2 Automatic differentiation | `tensor`, `gradcheck` | 10 |
| | 3 Backpropagation in a multilayer perceptron | `functional`, `nn.Linear` | 10 |
| | 4 Initialization and the flow of signals | `nn.init`, `analysis.activation_stats` | 10 |
| | 5 Optimization: SGD, momentum, Adam, AdamW | `optim` | 10 |
| II Generalization | 6 Regularization: weight decay, dropout, early stopping, normalization | `nn.Dropout`, `BatchNorm1d`, `LayerNorm` | 10 |
| | 7 Memorization and double descent | examples 03 and 05 | 10 |
| | 8 The neural tangent kernel and lazy training | `analysis.empirical_ntk`, `Linear(parametrization="ntk")` | 10 |
| III Architectures | 9 Convolutions for sequences | `functional.conv1d`, `nn.Conv1d` | 11 |
| | 10 Attention and the transformer encoder | `nn.MultiheadAttention`, `TransformerEncoderLayer` | 11 |
| | 11 Autoencoders and representation learning | example 08 | 11 |
| IV Toolkits | 12 From GlassNN to PyTorch (Rosetta table, Colab notebooks) | `examples/colab_pytorch` | 11 |
| | 13 Design notes: the backend, adding an operation, the GPU option | `backend` | - |
| Appendices | A Matrix-calculus cheat sheet; B API map (chapter to module); C Reading list | | |

**Cross-referencing mechanism.** `_helpers.py` provides:
- `show_source(obj)`: prints the source of a function or class as a fenced code block (via `inspect.getsource`), so the book always shows the real code.
- `api_url("glassnn.nn.Linear")`: the URL of the corresponding Sphinx page; `api_link(name, text=None)`: the same as a Markdown link.
- `src_url("src/glassnn/tensor.py", 40, 75)` or `src_url("glassnn.tensor._unbroadcast")`: a permanent link to those lines (or to the object's lines, found with `inspect`) at the current git tag (falls back to `main`); `src_link(...)`: the same as a Markdown link.
- Each Sphinx page links back to the book chapter listed in the chapter map.

Each milestone of section 7 adds or completes the book chapters that cover the code it delivers; a milestone is not finished while its chapters are missing or not executing.

## 7. Milestones and acceptance criteria

Order is dictated by what the lectures need. Commit after each milestone (one branch per milestone, merged through a pull request); do not start the next before all tests of the current one pass.

| # | Milestone | Acceptance criteria |
|---|---|---|
| M0 | Skeleton | layout, `pyproject.toml` (Python >= 3.13, NumPy >= 2.2, version from git tags), MIT license with the copyright line, CI running a trivial test, pre-commit, empty Sphinx build and empty Quarto book build both deployed by `site.yml` |
| M1 | Autodiff core | `Tensor` with arithmetic, matmul, sum/mean, reshape/transpose/indexing, broadcasting; `backward`; `no_grad`; `gradcheck`; all ops gradient-checked. Book chapters 1 and 2 |
| M2 | MLP training | activations, losses, `Module`/`Parameter`/`Linear`/`Sequential`, `init`, `SGD`/`Adam`/`AdamW`, schedulers, `DataLoader`, `manual_seed`; XOR and moons tests; PyTorch cross-checks for Linear, losses, optimizers. Book chapters 3 to 5. **Lecture 10 core is usable** |
| M3 | Generalization tools | `Dropout`, `BatchNorm1d`, `LayerNorm`, `analysis.empirical_ntk`, `activation_stats`, NTK parametrization; example notebooks 01 to 05 (section 9) with the expected findings written into the tests. Book chapters 6 to 8 |
| M4 | Convolutions | `conv1d/2d`, pooling, `Conv1d/2d`, `MaxPool`, `Flatten`; loop-reference and PyTorch cross-checks; motif-detection test. Book chapter 9 |
| M5 | Attention | `Embedding`, `PositionalEncoding`, `scaled_dot_product_attention`, `MultiheadAttention`, `TransformerEncoderLayer`, masks; cross-checks; sequence-classifier and autoencoder tests. Book chapters 10 and 11. **Lecture 11 usable** |
| M6 | GPU option | `set_backend("cupy")`, adapter for special functions, DLPack helpers, parity tests, benchmark script, extra `gpu` (`cupy-cuda13x`). Book chapter 13 |
| M7 | Toolkits and completion | Colab PyTorch notebooks, chapter 12 with the Rosetta table, appendices, README with a 5-minute tour, complete API reference with references, link check across book and API pages |
| M8 | Release 0.1.0 | CHANGELOG, `CITATION.cff`, annotated tag `v0.1.0`, GitHub Release with the wheel |

## 8. Performance expectations (smoke benchmarks, not CI gates)

At float32 on a laptop CPU, with a batch of 32: an MLP with two hidden layers of width 64 trains at well over 1000 steps per second; a two-layer Conv1d model on length-60 sequences and a one-layer transformer encoder (dimension 32) take tens of milliseconds per step. If a benchmark is 10 times slower than these orders of magnitude, profile before shipping. (A third-party NumPy implementation measured during planning ran at roughly these speeds.)

## 9. Course integration

The course repository (`adv-ml`) will depend on a **pinned tag**, for example `glassnn @ git+https://github.com/bioinfo-recetox/glassnn@v0.1.0`, in its `pyproject.toml`; this keeps the lockfile small.

**Python floor, verified.** Google Colab currently runs Python 3.13.15 (confirmed by the owner), so GlassNN installs there without workarounds. The course repository has been moved to `requires-python = ">=3.13"` with a `.python-version` file; with this floor, the five solution notebooks and the five decks produced so far were re-executed and re-rendered successfully on Python 3.13 with the same pinned package versions (NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1). Re-check the Colab Python version whenever Colab announces a runtime change.

Notebooks of the lectures come in the usual two versions (`practical-ex.ipynb` with blanks, `practical.ipynb` solution) and are built in the course repository, not here. This repository ships the **examples** below as reference notebooks, which the book chapters also use.

Lecture 10 (training and generalization):
1. `01_autodiff`: forward and reverse mode on a small expression, then backprop of a two-layer MLP checked against `gradcheck`.
2. `02_init_and_depth`: activation statistics versus depth for too-small, Xavier, He and too-large initializations; vanishing and exploding signals.
3. `03_double_descent`: test error of a one-hidden-layer network, or random-features regression with minimum-norm solution, as the width crosses the interpolation threshold, on a small sparse-signal problem.
4. `04_ntk_lazy_training`: empirical NTK at initialization and after training for increasing width; the network is close to its linearization for large width (Jacot et al. 2018; Chizat et al. 2019).
5. `05_memorizing_noise`: a network fits random labels perfectly (Zhang et al. 2017); how weight decay and early stopping change this. Links back to Rademacher complexity (lecture 2).

Lecture 11 (architectures and representations):
6. `06_cnn_dna_motif`: a 1D CNN detects the planted motif of lecture 5 (same generator); compare with the spectrum kernel.
7. `07_attention_dna_motif`: a small transformer encoder on the same data; inspect attention weights.
8. `08_autoencoder_expression`: autoencoder on simulated expression-like data with a low-dimensional latent structure; compare with PCA and kernel PCA (lectures 1 and 5).

`examples/colab_pytorch/` (separate from the package, written for Google Colab where PyTorch is preinstalled):
- A. the MLP of example 2 or 3 in PyTorch;
- B. the CNN of example 6 in PyTorch;
- C. optional: embeddings from a **pretrained protein language model** (for example a small ESM-2 checkpoint from Hugging Face) fed to a linear classifier. This needs internet access and a model download, and the exact model name must be verified before publishing. It illustrates representation learning at scale (Rives et al. 2021; Lin et al. 2023).

Each Colab notebook ends with the same Rosetta table: library name, argument, and behaviour that differ between GlassNN and PyTorch. The table is the same as in book chapter 12.

## 10. Risks

- **Numerical differences from PyTorch** (initialization defaults, epsilon in normalization, GELU variants): the cross-checks exist to find these; document every remaining difference.
- **Convolution speed** with im2col: acceptable at lecture sizes; if not, add a `numpy.einsum` path with `optimize=True` before anything fancier.
- **CuPy availability and wheel matching** differ by machine (only CUDA 13 is supported); keep the GPU backend optional and tested separately. CuPy's NumPy cap (`<2.6`) can lag behind NumPy releases.
- **Python floor** (3.13): satisfied by Colab today and by the course repository (section 9); revisit if Colab's runtime changes.
- **Book drift:** text that describes code which later changes. Mitigation: `show_source` embeds the real code, chapters are executed in CI, and links are checked.
- **Scope creep** (a "complete" framework): the non-goals list is binding; anything not needed by examples 1 to 8 goes into a "later" list in `CHANGELOG.md`.
- **Bibliographic errors:** the reference list below was written from memory and must be verified (title, venue, year, DOI) before the first release.

## 11. References (verify before release)

Automatic differentiation and backpropagation
- Rumelhart, Hinton and Williams (1986), Learning representations by back-propagating errors, *Nature* 323.
- Baydin, Pearlmutter, Radul and Siskind (2018), Automatic differentiation in machine learning: a survey, *JMLR* 18.
- Griewank and Walther (2008), *Evaluating Derivatives*, 2nd ed., SIAM.
- Goodfellow, Bengio and Courville (2016), *Deep Learning*, MIT Press.

Initialization, optimization, regularization, normalization
- Glorot and Bengio (2010), Understanding the difficulty of training deep feedforward neural networks, *AISTATS*.
- He, Zhang, Ren and Sun (2015), Delving deep into rectifiers, *ICCV*.
- Sutskever, Martens, Dahl and Hinton (2013), On the importance of initialization and momentum in deep learning, *ICML*.
- Kingma and Ba (2015), Adam: a method for stochastic optimization, *ICLR*.
- Loshchilov and Hutter (2019), Decoupled weight decay regularization, *ICLR*.
- Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov (2014), Dropout, *JMLR* 15.
- Ioffe and Szegedy (2015), Batch normalization, *ICML*.
- Ba, Kiros and Hinton (2016), Layer normalization, arXiv:1607.06450.
- Hendrycks and Gimpel (2016), Gaussian error linear units, arXiv:1606.08415.

Architectures
- LeCun, Bottou, Bengio and Haffner (1998), Gradient-based learning applied to document recognition, *Proc. IEEE* 86.
- Chellapilla, Puri and Simard (2006), High performance convolutional neural networks for document processing, *IWFHR*.
- Vaswani et al. (2017), Attention is all you need, *NeurIPS*.
- Hinton and Salakhutdinov (2006), Reducing the dimensionality of data with neural networks, *Science* 313.
- Bengio, Courville and Vincent (2013), Representation learning: a review and new perspectives, *IEEE TPAMI* 35.

Generalization
- Zhang, Bengio, Hardt, Recht and Vinyals (2017), Understanding deep learning requires rethinking generalization, *ICLR*.
- Belkin, Hsu, Ma and Mandal (2019), Reconciling modern machine-learning practice and the classical bias-variance trade-off, *PNAS* 116.
- Nakkiran et al. (2020), Deep double descent, *ICLR*.
- Bartlett, Long, Lugosi and Tsigler (2020), Benign overfitting in linear regression, *PNAS* 117.
- Jacot, Gabriel and Hongler (2018), Neural tangent kernel, *NeurIPS*.
- Chizat, Oyallon and Bach (2019), On lazy training in differentiable programming, *NeurIPS*.
- Lee et al. (2019), Wide neural networks of any depth evolve as linear models under gradient descent, *NeurIPS*.

Biology applications
- Alipanahi, Delong, Weirauch and Frey (2015), Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning, *Nature Biotechnology* 33.
- Zhou and Troyanskaya (2015), Predicting effects of noncoding variants with deep learning-based sequence model, *Nature Methods* 12.
- Rives et al. (2021), Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences, *PNAS* 118.
- Lin et al. (2023), Evolutionary-scale prediction of atomic-level protein structure with a language model, *Science* 379.

Software
- Harris et al. (2020), Array programming with NumPy, *Nature* 585.
- Okuta, Unno, Nishino, Hido and Loomis (2017), CuPy: a NumPy-compatible library for NVIDIA GPU calculations, *ML Systems Workshop at NIPS*.
- Karpathy, micrograd (MIT licensed), acknowledged as inspiration for the minimal autograd idea.

## 12. Decisions (settled)

| # | Decision | Value |
|---|---|---|
| 1 | Name | **GlassNN**; package and repository `glassnn` |
| 2 | Copyright | Vlad Popovici, RECETOX - Faculty of Science, Masaryk University |
| 3 | Default dtype | float32 |
| 4 | API documentation | Sphinx, Google-style docstrings with `sphinx.ext.napoleon` |
| 5 | Releases | annotated git tags only; no PyPI |
| 6 | Python, NumPy, SciPy | Python >= 3.13 (default 3.13); NumPy >= 2.5 and SciPy >= 1.18, the latest compatible with CuPy 14.2.0 (NumPy < 2.6) |
| 7 | Theory and tutorials | one Quarto book in `book/`, sharing the bibliography with Sphinx |
| 8 | GPU extra | only `cupy-cuda13x` (extra `gpu`) |
| 9 | Docstring style | Google style (`CLAUDE.md` updated to match) |
| 10 | Motif generator | a small copy of the lecture-5 generator lives in `examples/` |
| 11 | In-place exceptions | `nn.init.*_`, `clip_grad_norm_` and `Optimizer.step()` modify parameter data in place by design; documented as the only exceptions |
| 12 | Quarto | pinned to 1.10.18 in CI |
| 13 | Gradients kept | leaf tensors only; `retain_grad()` keeps an intermediate gradient (as in PyTorch) |
| 14 | Type of `.grad` | a `Tensor` with `requires_grad=False` (as in PyTorch) |
| 15 | dtype of new tensors | floating input (lists, scalars, arrays of any float dtype) is cast to the default dtype unless `dtype=` is given; integer and boolean input keeps its dtype. Differs from PyTorch, which keeps the dtype of a NumPy array |
| 16 | Elementwise ops in M1 | `exp` and `log` are `Tensor` methods from M1 on |
| 17 | Graph after `backward` | kept (a second `backward` works without `retain_graph`); documented as a difference from PyTorch |
| 18 | Book helper names | `api_url`/`src_url` return URLs, `api_link`/`src_link` return Markdown links; `src_url` also accepts an object name |
| 19 | Global state | backend, random generator and the grad-mode flag of `no_grad` (as in PyTorch) |
| 20 | Optimizer learning rate | one parameter group; the learning rate is `optimizer.lr`, which schedulers change. Differs from PyTorch (`param_groups`) |
| 21 | `clip_grad_norm_` | in `glassnn.optim` (PyTorch: `torch.nn.utils`); documented as a difference |
| 22 | Toy datasets | scikit-learn in the `dev`, `book` and `examples` extras (never a runtime dependency) |
| 23 | Generator before `manual_seed` | unseeded (like `numpy.random.default_rng()`); tests, examples and the book always seed |
| 24 | Non-learnable state | `Module.register_buffer`, `buffers()`, `named_buffers()`; `state_dict()`/`load_state_dict()` include buffers (as in PyTorch). `BatchNorm1d` has no `num_batches_tracked` (documented difference) |
| 25 | Bias in `Linear(parametrization="ntk")` | $z = x W^\top/\sqrt{n_\text{in}} + b$ with $W, b \sim \mathcal N(0, 1)$ (Lee et al. 2019 with $\sigma_w = \sigma_b = 1$); no extra argument |
| 26 | `activation_stats` | walks a (nested) `Sequential` and records the outputs of its `Linear` layers (later also convolutions); no forward hooks; other models raise `TypeError` |
| 27 | Example 03 (double descent) | random ReLU features (a frozen GlassNN `Linear`) with the minimum-norm least-squares readout (Rahimi and Recht 2007) |

Confirmed by the owner: the repository name `glassnn` is free in `bioinfo-recetox`; Colab runs Python 3.13.15.
