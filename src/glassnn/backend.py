"""Array backend, default dtype, random generator and array conversion.

GlassNN never calls NumPy directly in its operations. It calls ``backend.xp``,
the *active array module*, which is NumPy by default. Because CuPy mirrors the
NumPy API, switching ``xp`` to CuPy (milestone M6) runs the same code on a
GPU.

Always access the array module as ``backend.xp`` at call time. Writing
``from glassnn.backend import xp`` copies the module that is active at import
time and does not follow later calls to :func:`set_backend`.

Book chapter: :book:`Introduction <chapters/01-introduction.html>`.

Example:
    >>> from glassnn import backend
    >>> backend.get_backend()
    'numpy'
    >>> backend.asarray([1.0, 2.0]).dtype == backend.get_default_dtype()
    True
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import special as _scipy_special

#: The active array module (NumPy, or CuPy from milestone M6 on).
xp: Any = np

_backend_name = "numpy"
_default_dtype = np.dtype(np.float32)

_ALLOWED_DEFAULT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

# The global random generator; unseeded until manual_seed() is called.
_generator = np.random.default_rng()


def get_backend() -> str:
    """Return the name of the active backend.

    Returns:
        ``"numpy"`` (``"cupy"`` becomes possible in milestone M6).
    """
    return _backend_name


def set_backend(name: str) -> None:
    """Select the array backend.

    Args:
        name: ``"numpy"``. The ``"cupy"`` backend is added in milestone M6.

    Raises:
        NotImplementedError: If ``name`` is ``"cupy"``.
        ValueError: If ``name`` is not a known backend.
    """
    global xp, _backend_name
    if name == "numpy":
        xp = np
        _backend_name = "numpy"
    elif name == "cupy":
        raise NotImplementedError(
            "The CuPy backend is not implemented yet (milestone M6 in PLAN.md)."
        )
    else:
        raise ValueError(f"Unknown backend {name!r}; expected 'numpy' or 'cupy'.")


def get_default_dtype() -> np.dtype:
    """Return the dtype given to new floating-point tensors.

    Returns:
        ``float32`` unless changed with :func:`set_default_dtype`.
    """
    return _default_dtype


def set_default_dtype(dtype: npt.DTypeLike) -> None:
    """Set the dtype given to new floating-point tensors.

    Use ``float64`` for gradient checks; ``float32`` (the default) is faster
    and is what PyTorch uses.

    Args:
        dtype: ``"float32"`` or ``"float64"`` (or the NumPy dtype objects).

    Raises:
        ValueError: If ``dtype`` is neither float32 nor float64.
    """
    global _default_dtype
    requested = np.dtype(dtype)
    if requested not in _ALLOWED_DEFAULT_DTYPES:
        raise ValueError(
            f"The default dtype must be float32 or float64, got {requested}."
        )
    _default_dtype = requested


def asarray(data: Any, dtype: npt.DTypeLike | None = None) -> Any:
    """Convert ``data`` to an array of the active backend.

    The dtype rule: an explicit ``dtype`` always wins. Otherwise floating-point
    input (Python floats, lists of floats, arrays of any float dtype) becomes
    the default dtype, and integer or boolean input keeps its dtype.

    Args:
        data: A scalar, a (nested) list, or an array.
        dtype: The dtype of the result, or ``None`` for the rule above.

    Returns:
        An array of the active backend (``backend.xp``).

    Raises:
        TypeError: If ``data`` does not consist of real numbers or booleans.

    Note:
        Differences from PyTorch: ``torch.tensor`` keeps the dtype of a NumPy
        float64 array; GlassNN casts it to the default dtype, so that float64
        data never silently mixes with float32 parameters.

    Example:
        >>> from glassnn import backend
        >>> backend.asarray([1, 2]).dtype.kind
        'i'
        >>> str(backend.asarray([1, 2], dtype="float64").dtype)
        'float64'
    """
    if dtype is not None:
        return xp.asarray(data, dtype=dtype)
    array = xp.asarray(data)
    if array.dtype.kind == "f":
        return array.astype(_default_dtype, copy=False)
    if array.dtype.kind in "biu":
        return array
    raise TypeError(
        f"Tensors hold real numbers or booleans; got data of dtype {array.dtype}."
    )


def to_numpy(array: Any) -> np.ndarray:
    """Return ``array`` as a NumPy array (copied to the host if needed).

    Args:
        array: An array of the active backend.

    Returns:
        A ``numpy.ndarray`` with the same values.
    """
    return np.asarray(array)


def manual_seed(seed: int) -> np.random.Generator:
    """Seed the global random generator, and return it.

    Every random choice in GlassNN (initialization, dropout, shuffling)
    draws from this generator unless an explicit ``generator=`` is given.
    Before the first call the generator is unseeded, so results differ from
    run to run, as with ``numpy.random.default_rng()``.

    Args:
        seed: A non-negative integer.

    Returns:
        The new global generator.

    Note:
        Differences from PyTorch: ``torch.manual_seed`` returns a
        ``torch.Generator`` and PyTorch's generator starts from a fixed seed;
        GlassNN uses a ``numpy.random.Generator`` that starts unseeded.

    Example:
        >>> from glassnn import backend
        >>> a = backend.manual_seed(0).random()
        >>> b = backend.manual_seed(0).random()
        >>> a == b
        True
    """
    global _generator
    _generator = xp.random.default_rng(seed)
    return _generator


def get_generator() -> np.random.Generator:
    """Return the global random generator (see :func:`manual_seed`)."""
    return _generator


def erf(x: Any) -> Any:
    r"""The error function, applied elementwise.

    .. math:: \operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}\, dt

    NumPy has no ``erf``; this adapter calls ``scipy.special.erf`` (and the
    CuPy equivalent from milestone M6 on).

    Args:
        x: An array of the active backend.

    Returns:
        An array of the same shape.
    """
    return _scipy_special.erf(x)
