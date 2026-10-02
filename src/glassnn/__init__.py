"""GlassNN: a transparent neural-network library for teaching.

The public API is re-exported here as it is implemented, milestone by
milestone (see ``PLAN.md``, section 7).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("glassnn")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
