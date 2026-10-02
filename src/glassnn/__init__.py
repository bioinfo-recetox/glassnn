"""GlassNN: a transparent neural-network library for teaching.

The public API is re-exported here as it is implemented, milestone by
milestone (see ``PLAN.md``, section 7).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("glassnn")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

from glassnn import backend, functional, nn, optim
from glassnn.backend import manual_seed
from glassnn.tensor import Tensor, no_grad

__all__ = [
    "Tensor",
    "__version__",
    "backend",
    "functional",
    "manual_seed",
    "nn",
    "no_grad",
    "optim",
]
