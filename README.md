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

## A 5-minute tour

Tensors record the operations applied to them; `backward()` computes the
gradients, exactly as in PyTorch:

```pycon
>>> from glassnn import Tensor
>>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
>>> y = (x * x).sum()
>>> y
Tensor(14., op='sum')
>>> y.backward()
>>> x.grad
Tensor([2., 4., 6.])
```

Every operation is a few lines you can read: the forward computation and a
`lambda` that maps the gradient of the output to the gradient of the input.

```pycon
>>> import inspect
>>> import glassnn.functional as F
>>> print(inspect.getsource(F.relu))
def relu(input: Tensor) -> Tensor:
    r"""Rectified linear unit, elementwise.
...
    a = input
    positive = a.data > 0
    out = backend.xp.maximum(a.data, 0)
    return _result(out, (a,), lambda grad: (grad * positive,), "relu")
```

Models, losses, optimizers and the training loop have PyTorch's names:

```pycon
>>> import numpy as np
>>> from glassnn import manual_seed, nn, optim
>>> from glassnn.data import DataLoader
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(200, 2))
>>> labels = (X[:, 0] * X[:, 1] > 0).astype(np.int64)  # an XOR-like problem
>>> _ = manual_seed(0)
>>> model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
>>> model
Sequential(
  (0): Linear(in_features=2, out_features=16, bias=True)
  (1): Tanh()
  (2): Linear(in_features=16, out_features=2, bias=True)
)
>>> optimizer = optim.Adam(model.parameters(), lr=0.05)
>>> for epoch in range(100):
...     for xb, yb in DataLoader(X, labels, batch_size=32, shuffle=True):
...         optimizer.zero_grad()
...         loss = F.cross_entropy(model(xb), yb)
...         loss.backward()
...         optimizer.step()
>>> accuracy = (model(Tensor(X)).data.argmax(axis=1) == labels).mean()
>>> bool(accuracy > 0.95)
True
```

Gradients can be checked against finite differences, which is how every
operation of the library is tested:

```pycon
>>> from glassnn.gradcheck import gradcheck
>>> w = Tensor(np.linspace(-2, 2, 5), requires_grad=True, dtype="float64")
>>> gradcheck(lambda t: (F.gelu(t) * t).sum(), [w])
True
```

Where next: the [book](https://bioinfo-recetox.github.io/glassnn/) derives
every operation and shows its code; chapter 12 lists where GlassNN and
PyTorch differ, and has PyTorch versions of two example notebooks for
Google Colab. The same code runs on an NVIDIA GPU after
`backend.set_backend("cupy")` (below).

## Installation

GlassNN needs Python 3.13 or newer and depends only on NumPy and SciPy.
Install a tagged release directly from GitHub:

```
pip install "glassnn @ git+https://github.com/bioinfo-recetox/glassnn@v0.1.1"
```

### Optional GPU backend

The optional GPU backend uses [CuPy](https://cupy.dev) with CUDA 13
(`pip install "glassnn[gpu] @ git+..."`; the extra also installs the CUDA
compiler and headers, so no system CUDA toolkit is needed). One call moves
all new tensors to the GPU:

```python
from glassnn import backend

backend.set_backend("cupy")
```

A seed gives the same results on CPU and GPU. At the batch sizes of the
lectures the CPU is as fast; the GPU pays off for large batches (book
chapter 13). CuPy is the "RAPIDS option": RAPIDS libraries (cuDF, cuML)
exchange arrays with CuPy without copies through DLPack
(`backend.from_dlpack`); GlassNN itself needs only an array library and does
not depend on RAPIDS.

## Development

```
uv sync --extra dev --extra docs --extra book --extra examples
uv run pytest
```

See `CONTRIBUTING.md`.

## License and citation

MIT license, see `LICENSE`. To cite GlassNN, see `CITATION.cff`.
