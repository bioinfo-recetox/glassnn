"""Array backend, default dtype, random generator and array conversion.

GlassNN never calls NumPy directly in its operations. It calls ``backend.xp``,
the *active array module*, which is NumPy by default. Because CuPy mirrors the
NumPy API :cite:p:`okuta2017cupy`, ``set_backend("cupy")`` runs the same code
on an NVIDIA GPU.

Random numbers are always drawn on the host by a NumPy generator, also with
the CuPy backend, and then copied to the device. A seed therefore gives the
same weights, dropout masks and batches on CPU and GPU.

Arrays move between libraries without copies through DLPack
:cite:p:`dlpack`: :func:`from_dlpack` takes an array from PyTorch, CuPy,
cuDF or any other library that implements the protocol.

Always access the array module as ``backend.xp`` at call time. Writing
``from glassnn.backend import xp`` copies the module that is active at import
time and does not follow later calls to :func:`set_backend`.

Book chapters: :book:`Introduction <chapters/01-introduction.html>`;
the GPU option: :book:`Design notes <chapters/13-design-notes.html>`.

Example:
    >>> from glassnn import backend
    >>> backend.get_backend()
    'numpy'
    >>> backend.asarray([1.0, 2.0]).dtype == backend.get_default_dtype()
    True
"""

import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import special as _scipy_special

#: The active array module: NumPy, or CuPy after ``set_backend("cupy")``.
xp: Any = np

_backend_name = "numpy"

# Special functions of the active backend: scipy.special or cupyx.scipy.special.
_special: Any = _scipy_special

_default_dtype = np.dtype(np.float32)

_ALLOWED_DEFAULT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

# The global random generator; unseeded until manual_seed() is called.
_generator = np.random.default_rng()


def get_backend() -> str:
    """Return the name of the active backend.

    Returns:
        ``"numpy"`` or ``"cupy"``.

    Example:
        >>> from glassnn import backend
        >>> backend.get_backend()
        'numpy'
    """
    return _backend_name


def set_backend(name: str) -> None:
    """Select the array backend.

    Tensors created afterwards live on the new backend; existing tensors stay
    where they are. Create the model and the data after this call, or move
    arrays explicitly (:func:`to_numpy`, :func:`asarray`).

    Args:
        name: ``"numpy"`` (CPU) or ``"cupy"`` (NVIDIA GPU, needs the ``gpu``
            extra: ``pip install "glassnn[gpu]"``). CuPy implements the NumPy
            interface on the GPU :cite:p:`okuta2017cupy`.

    Raises:
        ImportError: If ``name`` is ``"cupy"`` and CuPy is not installed.
        ValueError: If ``name`` is not a known backend.

    Note:
        Differences from PyTorch: PyTorch places each tensor on a device
        (``tensor.to("cuda")``); GlassNN switches one global array module, so
        all new tensors are on the same device.

    Example:
        >>> from glassnn import backend
        >>> backend.set_backend("numpy")
        >>> backend.get_backend()
        'numpy'
    """
    global xp, _special, _backend_name
    if name == "numpy":
        xp, _special = np, _scipy_special
    elif name == "cupy":
        xp, _special = _import_cupy()
    else:
        raise ValueError(f"Unknown backend {name!r}; expected 'numpy' or 'cupy'.")
    _backend_name = name


def _import_cupy() -> tuple[Any, Any]:
    """Import CuPy and its special functions only when they are asked for."""
    # CuPy is an optional extra, so a type checker may not find it.
    try:
        import cupy  # pyright: ignore[reportMissingImports]
        import cupyx.scipy.special  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ImportError(
            "The CuPy backend needs CuPy with CUDA 13 and an NVIDIA GPU; "
            'install it with: pip install "glassnn[gpu]".'
        ) from error
    return cupy, cupyx.scipy.special


def get_default_dtype() -> np.dtype:
    """Return the dtype given to new floating-point tensors.

    Returns:
        ``float32`` unless changed with :func:`set_default_dtype`.

    Example:
        >>> from glassnn import backend
        >>> str(backend.get_default_dtype())
        'float32'
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

    Example:
        >>> from glassnn import Tensor, backend
        >>> backend.set_default_dtype("float64")
        >>> str(Tensor([1.0]).dtype)
        'float64'
        >>> backend.set_default_dtype("float32")
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
    if xp is np and _is_cupy_array(data):
        raise TypeError(
            "This array is on the GPU but the active backend is NumPy; copy "
            "it with backend.to_numpy(...) or call set_backend('cupy') first."
        )
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
        array: A NumPy or CuPy array, or anything ``numpy.asarray`` accepts.

    Returns:
        A ``numpy.ndarray`` with the same values; a NumPy array is returned
        as it is, without a copy.

    Example:
        >>> from glassnn import Tensor, backend
        >>> backend.to_numpy(Tensor([1.0, 2.0]).data)
        array([1., 2.], dtype=float32)
    """
    if _is_cupy_array(array):
        return array.get()
    return np.asarray(array)


def _is_cupy_array(array: Any) -> bool:
    """Whether ``array`` is a CuPy array (without importing CuPy)."""
    cupy = sys.modules.get("cupy")
    return cupy is not None and isinstance(array, cupy.ndarray)


def from_dlpack(array: Any) -> Any:
    """Take an array from another library without copying it (DLPack).

    Args:
        array: Any object with a ``__dlpack__`` method on a device of the
            active backend: a PyTorch tensor, a CuPy or NumPy array, or a
            cuDF column on the GPU. The result shares memory with it.

    Returns:
        An array of the active backend. Wrap it with ``Tensor(...)``.

    References:
        :cite:p:`dlpack`

    Example:
        >>> import numpy as np
        >>> from glassnn import backend
        >>> x = np.arange(3.0)
        >>> np.shares_memory(backend.from_dlpack(x), x)
        True
    """
    return xp.from_dlpack(array)


def to_dlpack(array: Any) -> Any:
    """Export an array as a DLPack capsule, for libraries that expect one.

    Most libraries accept the array itself (they call ``array.__dlpack__()``
    on it); the capsule is only needed for older interfaces such as
    ``torch.utils.dlpack.from_dlpack``. A capsule can be consumed only once.

    Args:
        array: An array of the active backend, for example ``tensor.data``.

    Returns:
        A ``PyCapsule`` that shares memory with ``array``.

    References:
        :cite:p:`dlpack`

    Example:
        >>> import numpy as np
        >>> from glassnn import backend
        >>> type(backend.to_dlpack(np.arange(3.0))).__name__
        'PyCapsule'
    """
    return array.__dlpack__()


def manual_seed(seed: int) -> np.random.Generator:
    """Seed the global random generator, and return it.

    Every random choice in GlassNN (initialization, dropout, shuffling)
    draws from this generator unless an explicit ``generator=`` is given.
    Before the first call the generator is unseeded, so results differ from
    run to run, as with ``numpy.random.default_rng()``. The generator is a
    NumPy generator with both backends; with CuPy the numbers are drawn on
    the host and copied to the GPU, so a seed gives the same results on both.

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
    _generator = np.random.default_rng(seed)
    return _generator


def get_generator() -> np.random.Generator:
    """Return the global random generator (see :func:`manual_seed`).

    Returns:
        A ``numpy.random.Generator``, with both backends.

    Example:
        >>> from glassnn import backend
        >>> backend.manual_seed(0) is backend.get_generator()
        True
    """
    return _generator


def erf(x: Any) -> Any:
    r"""The error function, applied elementwise.

    .. math:: \operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}\, dt

    NumPy has no ``erf``; this adapter calls ``scipy.special.erf``, or
    ``cupyx.scipy.special.erf`` with the CuPy backend.

    Args:
        x: An array of the active backend.

    Returns:
        An array of the same shape.

    Example:
        >>> import numpy as np
        >>> from glassnn import backend
        >>> backend.erf(np.array([0.0, 3.0])).round(4)
        array([0., 1.])
    """
    return _special.erf(x)
