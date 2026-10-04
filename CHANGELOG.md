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

### Differences from PyTorch

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

## Later

Features deliberately kept out of scope (see `PLAN.md`, section 10):

- none yet.
