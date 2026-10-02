"""Array backend, default dtype and array conversion.

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

#: The active array module (NumPy, or CuPy from milestone M6 on).
xp: Any = np

_backend_name = "numpy"
_default_dtype = np.dtype(np.float32)

_ALLOWED_DEFAULT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


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
