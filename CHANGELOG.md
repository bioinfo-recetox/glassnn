# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [semantic versioning](https://semver.org/) through
annotated git tags.

## [Unreleased]

### Changed

- README: installation on Google Colab, with a note on restarting the
  session when pip upgrades NumPy.

## [0.1.1] - 2026-10-19

### Highlights

- A documentation release: the book now opens with a guide to its parts,
  chapter sections, appendices and notebooks. The library is unchanged
  from 0.1.0.

### Changed

- Book: the preface has a "How to read this book" guide (parts, chapter
  sections, appendices, notebooks) instead of the "under construction"
  note; chapter 1 points to the GPU backend (chapter 13) and to the
  chapters on layers, losses, initializers and optimizers.

## [0.1.0] - 2026-10-19

### Highlights

- The first release of GlassNN, a neural-network library whose source code
  is meant to be read: reverse-mode automatic differentiation in `Tensor`,
  with every operation's derivative written in its docstring and checked
  against finite differences.
- PyTorch's names and defaults for modules, layers, losses, optimizers and
  schedulers: MLPs, 1-D and 2-D convolutions, attention and the transformer
  encoder, dropout and normalization, SGD, Adam, AdamW and RMSprop.
- Tools for the theory of the lectures: activation statistics through
  depth, the empirical neural tangent kernel, the NTK parametrization.
- An optional GPU backend (CuPy, CUDA 13) with identical results for a
  given seed, and zero-copy exchange through DLPack.
- A book with the theory, the code and exercises (13 chapters, 3
  appendices), an API reference, and eight example notebooks, plus two
  PyTorch notebooks for Google Colab and a table of every difference from
  PyTorch.
- Cross-checked against PyTorch layer by layer; runtime dependencies are
  NumPy and SciPy only.

### Added

- Project skeleton (milestone M0): `src` layout, `pyproject.toml` with the
  version derived from git tags, MIT license, empty Sphinx API reference and
  empty Quarto book, pre-commit configuration, CI, site deployment and
  release workflows.
- IEEE citation style (CSL, CC BY-SA 3.0) for the book.
- `glassnn.backend`: active array module `xp`, `get_backend`/`set_backend`
  (NumPy; CuPy from M6), `get_default_dtype`/`set_default_dtype`
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

- `glassnn.backend.manual_seed`/`get_generator` (also `glassnn.manual_seed`) and
  the `erf` adapter.
- `glassnn.functional`: `relu`, `leaky_relu`, `gelu` (exact and tanh),
  `tanh`, `sigmoid`, `softplus`, `softmax`, `log_softmax`, `logsumexp`,
  `linear`, `cross_entropy` (label smoothing, reductions),
  `binary_cross_entropy_with_logits`, `mse_loss`; stable for extreme inputs.
- `glassnn.nn`: `Module` (registration by assignment, `parameters`,
  `named_parameters`, `modules`, `apply`, `train`/`eval`, `zero_grad`,
  `state_dict`/`load_state_dict`), `Parameter`, `Sequential`, `Linear`
  (PyTorch's default initialization), activation modules (`ReLU`,
  `LeakyReLU`, `GELU`, `Tanh`, `Sigmoid`, `Softplus`, `Softmax`,
  `LogSoftmax`), loss modules (`MSELoss`, `CrossEntropyLoss`,
  `BCEWithLogitsLoss`).
- `glassnn.nn.init`: `calculate_gain`, `xavier_uniform_`/`xavier_normal_`,
  `kaiming_uniform_`/`kaiming_normal_`, `uniform_`, `normal_`, `zeros_`,
  `ones_` (optional `generator=`).
- `glassnn.optim`: `SGD` (momentum, Nesterov, weight decay), `Adam`, `AdamW`,
  `RMSprop`, schedulers `StepLR`, `CosineAnnealingLR`, `LinearWarmup`, and
  `clip_grad_norm_`.
- `glassnn.data`: `DataLoader` (arrays in memory, shuffling per epoch,
  `drop_last`, `generator=`) and `one_hot`.
- Tests: XOR and two moons (test accuracy >= 0.95) are learned; training is
  bit-for-bit reproducible with a fixed seed (float32 and float64); float32
  training stays in float32; 67 PyTorch cross-checks (marker `torch`, run in
  a separate CI job with CPU PyTorch).
- Book: chapter 3 (backpropagation in an MLP), chapter 4 (initialization and
  the flow of signals), chapter 5 (optimization); `book/_check_api_links.py`
  checks every link from the book into the API reference in CI.
- `Tensor.__repr__` aligns multi-line values and shows the class name
  (`Parameter(...)`).
- Verified references: Glorot and Bengio 2010, He et al. 2015, Sutskever et
  al. 2013, Kingma and Ba 2015, Loshchilov and Hutter 2017 and 2019,
  Hendrycks and Gimpel 2016, Szegedy et al. 2016, and the RMSprop lecture
  slides (Hinton, Srivastava and Swersky 2012).
- scikit-learn added to the `dev`, `book` and `examples` extras (not a
  runtime dependency).

- Milestone M3 (generalization tools): `glassnn.functional.dropout`
  (inverted dropout, optional `generator=`), `layer_norm` and `batch_norm`
  (shared standardization op with an explicit backward pass; running
  statistics with unbiased variance); modules `nn.Dropout`, `nn.LayerNorm`,
  `nn.BatchNorm1d` (inputs `(N, C)` and `(N, C, L)`).
- `Module.register_buffer`, `buffers`, `named_buffers`; `state_dict` and
  `load_state_dict` include buffers (after the parameters).
- `nn.Linear(parametrization="ntk")`: weights and bias from N(0, 1), weight
  scaled by `1/sqrt(in_features)` in the forward pass.
- `glassnn.analysis`: `count_parameters`, `activation_stats` (walks a
  `Sequential`, returns `LayerStats(mean, std)` per `Linear`), and
  `empirical_ntk` (one backward pass per example; existing gradients are
  restored).
- Example notebooks `01_autodiff` to `05_memorizing_noise` in `examples/`,
  executed by a new CI job (`examples/_execute.py`); their expected findings
  are asserted in `tests/test_examples.py`. `nbclient` and `nbformat` added
  to the `examples` extra.
- PyTorch cross-checks for `LayerNorm`, `BatchNorm1d` (three training steps
  with running statistics, then evaluation), dropout scaling and the
  `BatchNorm1d` state-dict keys.
- Book: chapter 6 (regularization), chapter 7 (memorization and double
  descent), chapter 8 (the neural tangent kernel and lazy training);
  chapter 4 points to `activation_stats`. API pages for
  `glassnn.nn.dropout`, `glassnn.nn.normalization`, `glassnn.analysis`
  (the "planned modules" page is removed).
- Verified references: Srivastava et al. 2014, Ioffe and Szegedy 2015, Ba
  et al. 2016, Zhang et al. 2017, Belkin et al. 2019, Nakkiran et al. 2020,
  Bartlett et al. 2020, Jacot et al. 2018, Chizat et al. 2019, Lee et al.
  2019, Rahimi and Recht 2007.

- Milestone M4 (convolutions): `glassnn.functional.conv1d`, `conv2d`
  (stride, padding as int/tuple/"valid"/"same", dilation; im2col with
  `sliding_window_view`, backward by scatter-add per kernel offset),
  `max_pool1d/2d`, `avg_pool1d/2d`; modules `nn.Conv1d`, `nn.Conv2d`
  (PyTorch's default initialization), `nn.MaxPool1d`, `nn.AvgPool1d`,
  `nn.Flatten`.
- Tests against slow loop implementations, gradient checks, and PyTorch
  cross-checks for convolution, pooling, `Flatten` and the default
  initialization of `Conv1d`.
- `examples/dna_motifs.py` (synthetic motif generator, decision 28; spectrum
  and mismatch kernel features) and notebook `06_cnn_dna_motif`; tests: the
  CNN detects and locates the planted motif, and on mutated motifs the
  mismatch kernel beats the spectrum kernel and the CNN beats both.
- `nn.MaxPool2d` and `nn.AvgPool2d` (decision 33).
- Decisions 31-33 and the extended decision 11 recorded in `PLAN.md`;
  verified references Leslie et al. 2002 (spectrum kernel) and 2004
  (mismatch kernel).
- Book: chapter 9 (convolutions for sequences); API pages for
  `glassnn.nn.conv` and `glassnn.nn.pooling`.
- Verified references: LeCun et al. 1998, Chellapilla et al. 2006,
  Alipanahi et al. 2015, Zhou and Troyanskaya 2015.

- Milestone M5 (attention): `glassnn.functional.embedding` (with
  `padding_idx`), `scaled_dot_product_attention` (boolean or float masks,
  `is_causal`, `scale`, `dropout_p`), `causal_mask`, `additive_mask`;
  modules `nn.Embedding`, `nn.PositionalEncoding` (sinusoidal, a buffer),
  `nn.MultiheadAttention` (separate q/k/v projections, padding and
  attention masks, weights per head or averaged), `nn.TransformerEncoderLayer`
  (pre-LN or post-LN, GELU or ReLU), `nn.TransformerEncoder`, `nn.ModuleList`.
- Tests against NumPy references and gradient checks; PyTorch cross-checks
  for `Embedding`, `scaled_dot_product_attention` (all mask kinds),
  `MultiheadAttention` (masks, averaged and per-head weights),
  `TransformerEncoderLayer` (four variants) and `TransformerEncoder` with a
  causal mask.
- `examples/dna_motifs.kmer_tokens`, `examples/expression.py` (decision 37)
  and notebooks `07_attention_dna_motif` (decision 38) and
  `08_autoencoder_expression`, with their findings asserted in
  `tests/test_examples.py`.
- Book: chapter 10 (attention and the transformer encoder) and chapter 11
  (autoencoders and representation learning); API pages for
  `glassnn.nn.embedding`, `glassnn.nn.attention`, `glassnn.nn.transformer`.
- Verified references: Vaswani et al. 2017, Xiong et al. 2020, Hinton and
  Salakhutdinov 2006, Bengio et al. 2013, Schölkopf et al. 1998, Bakır et
  al. 2003.
- The PyTorch cross-check of the default initialization of `Conv1d` is
  seeded and compares with the theoretical spread (it failed occasionally).
- Milestone M6 (GPU option): `glassnn.backend.set_backend("cupy")` (CuPy is
  imported only then, with an `ImportError` naming the `gpu` extra if it is
  missing); `erf` follows the backend (`cupyx.scipy.special`); `to_numpy`
  copies GPU arrays to the host; `from_dlpack` and `to_dlpack`; `asarray`
  refuses a GPU array while the NumPy backend is active.
- The random generator is a NumPy generator on both backends; dropout masks
  are copied to the device, so a seed gives the same results on CPU and GPU
  (decision 41).
- The `gpu` extra is `cupy-cuda13x[ctk]`, which brings NVRTC and the CUDA
  headers as wheels (decision 40).
- Backend parity tests (marker `gpu`): every operation, layer and optimizer,
  short training runs in float64 and float32, `empirical_ntk`,
  `activation_stats`, `gradcheck`, DLPack and the error for mixed devices.
- `benchmarks/bench_backends.py`: one training step of the MLP, Conv1d and
  transformer smoke benchmarks on both backends, at batch 32 and 1024.
- Book: chapter 13 (design notes: the backend, adding an operation, the GPU
  option), with measured times stored in `book/assets/benchmark_backends.json`
  (decision 42).
- Verified references: Okuta et al. 2017 (CuPy), DLPack.
- Milestone M7 (toolkits and completion): book chapter 12 (from GlassNN to
  PyTorch) with the Rosetta table, kept in one data file
  `book/assets/rosetta.toml` (decision 43); Colab notebooks
  `examples/colab_pytorch/A_mlp_init.ipynb` and `B_cnn_motif.ipynb`, the
  PyTorch versions of examples 02 and 06, each ending with the table
  (`_sync_rosetta.py` writes it; notebook B downloads `dna_motifs.py` at the
  pinned tag, decision 44).
- Book appendices: A (matrix calculus for backpropagation), B (API map,
  generated from the module docstrings), C (reading list of every cited
  work).
- README: a 5-minute tour, run as a doctest by `tests/test_readme.py`.
- Complete API reference (decision 46): every public symbol has a summary,
  arguments, return value and a doctested example; operations, layers,
  losses, initializers, optimizers and schedulers state their mathematics;
  methods from the literature cite it; every module links to its book
  chapter and has an API page (new page `glassnn`). Enforced by
  `tests/test_docstrings.py`.
- `book/_check_api_links.py` checks every on-site link of the book and the
  API reference in both directions, including anchors; external links are
  checked by hand with `sphinx-build -b linkcheck`.
- Tests: `tests/test_rosetta.py` (every documented difference has a row,
  notebooks hold the current table), `tests/test_colab_notebooks.py`
  (marker `torch`: the notebooks reproduce the findings of examples 02 and
  06), `tests/test_check_links.py`. The `torch` CI job installs the
  `examples` extra to run the notebooks.
- Verified references: Paszke et al. 2019 (PyTorch), Blanchard, Higham and
  Higham 2021, Goyal et al. 2017, Pascanu, Mikolov and Bengio 2013.
- Milestone M8 (release 0.1.0): this changelog section with its
  highlights, `CITATION.cff` with version and release date, the classifier
  "Development Status :: 3 - Alpha", and `tests/test_release.py`, which
  checks these against each other, runs the release-notes step of
  `release.yml`, and checks that every pinned tag names this release.

### Differences from PyTorch

The complete table is in chapter 12 of the book (`book/assets/rosetta.toml`).

- Optimizers have one parameter group; the learning rate is `optimizer.lr`.
- `clip_grad_norm_` lives in `glassnn.optim`; `one_hot` in `glassnn.data`.
- `LinearWarmup` is PyTorch's `LinearLR(start_factor, 1.0, total_iters)`.
- `DataLoader` takes arrays, not a `Dataset`.
- `manual_seed` returns a NumPy generator; the generator is unseeded until
  `manual_seed` is called.
- `Module.state_dict()` returns copies.
- `cross_entropy` supports `(N, C)` logits with integer labels only;
  `mse_loss` and `binary_cross_entropy_with_logits` reject different shapes.
- Floating-point NumPy arrays are cast to the default dtype (PyTorch keeps
  float64).
- The graph is kept after `backward()` (no `retain_graph` needed).
- `Tensor.numpy()` returns a copy.
- `BatchNorm1d` has no `num_batches_tracked` buffer and no `momentum=None`;
  `register_buffer` has no `persistent` argument; `Dropout` has no
  `inplace`; `F.dropout` accepts `generator=`.
- `Linear` has a `parametrization` argument; `glassnn.analysis` has no
  PyTorch counterpart.
- Convolutions have no `groups` and no `padding_mode`; pooling has no
  `dilation`, `ceil_mode`, `return_indices` or `count_include_pad=False`.
- Boolean attention masks are `True` where attention is forbidden, also in
  `F.scaled_dot_product_attention` (PyTorch: `True` = allowed there).
- Attention and transformer modules are batch-first only;
  `MultiheadAttention` has `q_proj`/`k_proj`/`v_proj` instead of a packed
  `in_proj_weight`; `TransformerEncoderLayer` defaults to `norm_first=True`
  and `activation="gelu"`; `is_causal=True` without a mask builds the
  causal mask; `PositionalEncoding` is not part of PyTorch.
- The device is chosen for all new tensors at once with `set_backend`, not
  per tensor (`x.to("cuda")`); one NumPy generator serves both devices;
  `backend.from_dlpack` returns an array, not a tensor.

## Later

Features deliberately kept out of scope (see `PLAN.md`, section 10):

- Colab notebook C (embeddings from a pretrained protein language model such
  as ESM-2, fed to a linear classifier): optional in `PLAN.md`, section 9;
  deferred because it needs `transformers`, a model download and a verified
  model name, and cannot run in CI (decision 45).

[Unreleased]: https://github.com/bioinfo-recetox/glassnn/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/bioinfo-recetox/glassnn/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bioinfo-recetox/glassnn/releases/tag/v0.1.0
