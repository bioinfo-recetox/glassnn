r"""The ``Tensor`` class and reverse-mode automatic differentiation.

A :class:`Tensor` wraps an array (``.data``). Every operation on tensors that
require gradients records a node of a *computational graph*: the result
remembers its inputs (``_parents``) and a function (``_backward``) that maps
the gradient of the result to the gradients of the inputs.

Notation used in the docstrings. ``L`` is the scalar on which
:meth:`Tensor.backward` is called (usually a loss). For any tensor ``z`` the
*adjoint* :math:`\bar z = \partial L / \partial z` has the shape of ``z``.
Each ``_backward`` function receives :math:`\bar z` for its output ``z`` and
returns :math:`\bar a, \bar b` for its inputs ``a, b``: a vector-Jacobian
product :cite:p:`baydin2018automatic`. :meth:`Tensor.backward` visits the
nodes in reverse topological order and adds up the contributions, which is
the chain rule :cite:p:`rumelhart1986learning,griewank2008evaluating`.

Example:
    >>> from glassnn import Tensor
    >>> x = Tensor(3.0, requires_grad=True)
    >>> y = Tensor(4.0, requires_grad=True)
    >>> z = x * y + x**2
    >>> z.backward()
    >>> x.grad, y.grad
    (Tensor(10.), Tensor(3.))
"""

import functools
import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from glassnn import backend

#: The type of a backward function: output adjoint -> one adjoint per parent.
BackwardFn = Callable[[Any], tuple[Any, ...]]

# Whether new operations are recorded in the graph; switched off by no_grad.
_grad_enabled = True


class Tensor:
    """A multidimensional array that can record operations for autodiff.

    Args:
        data: A scalar, a (nested) list, an array, or another ``Tensor``
            (its values are used; its graph is not).
        requires_grad: Whether gradients with respect to this tensor are
            computed by :meth:`backward`. Only floating-point tensors can
            require gradients.
        dtype: The dtype of the data. If ``None``, floating-point input
            becomes the default dtype (see
            :func:`glassnn.backend.set_default_dtype`) and integer or boolean
            input keeps its dtype.

    Attributes:
        data: The array of values (of the active backend).
        grad: The accumulated gradient, a ``Tensor`` of the same shape, or
            ``None`` before the first :meth:`backward`. Only leaf tensors, and
            tensors on which :meth:`retain_grad` was called, keep a gradient.
        requires_grad: See above.

    Raises:
        TypeError: If ``requires_grad`` is set on a non-floating tensor.

    Note:
        Differences from PyTorch: a floating-point NumPy array is cast to the
        default dtype (PyTorch keeps float64); the graph is kept after
        :meth:`backward`, so calling it twice works without
        ``retain_graph=True``; there are no in-place operations.

    Example:
        >>> from glassnn import Tensor
        >>> Tensor([1.0, 2.0], requires_grad=True)
        Tensor([1., 2.], requires_grad=True)
        >>> Tensor([1, 2]).dtype
        dtype('int64')
    """

    # Make NumPy hand mixed operations (``array + tensor``) to Tensor.
    __array_ufunc__ = None

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create a leaf tensor; see the class docstring."""
        if isinstance(data, Tensor):
            data = data.data
        self.data = backend.asarray(data, dtype=dtype)
        if requires_grad and self.data.dtype.kind != "f":
            raise TypeError(
                "Only floating-point tensors can require gradients; "
                f"this tensor has dtype {self.data.dtype}."
            )
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._backward: BackwardFn | None = None
        self._op = ""
        self._retains_grad = False

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the tensor."""
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the tensor."""
        return self.data.dtype

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return self.data.ndim

    @property
    def is_leaf(self) -> bool:
        """Whether the tensor was created by the user, not by an operation."""
        return self._backward is None

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        return len(self.data)

    def __repr__(self) -> str:
        """Show the values, and how the tensor takes part in autodiff."""
        values = np.array2string(backend.to_numpy(self.data), separator=", ")
        extras = ""
        if not self.is_leaf:
            extras = f", op={self._op!r}"
        elif self.requires_grad:
            extras = ", requires_grad=True"
        return f"Tensor({values}{extras})"

    def item(self) -> float | int | bool:
        """Return the value of a one-element tensor as a Python number."""
        return self.data.item()

    def numpy(self) -> np.ndarray:
        """Return a NumPy copy of the values.

        Note:
            Differences from PyTorch: ``torch.Tensor.numpy`` shares memory
            with the tensor; GlassNN returns a copy, so the tensor cannot be
            changed through it.
        """
        return backend.to_numpy(self.data).copy()

    def detach(self) -> "Tensor":
        """Return a new leaf tensor with the same values and no graph."""
        return Tensor(self.data, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Elementwise arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> "Tensor":
        r"""Elementwise sum with broadcasting.

        .. math:: z = a + b, \qquad \bar a = \bar z, \qquad \bar b = \bar z

        (each summed over the axes along which the operand was broadcast).
        """
        a, b = self, _as_tensor(other, like=self)

        def backward(grad):
            return _unbroadcast(grad, a.shape), _unbroadcast(grad, b.shape)

        return _result(a.data + b.data, (a, b), backward, "add")

    def __radd__(self, other: Any) -> "Tensor":
        """Compute ``other + self``."""
        return _as_tensor(other, like=self) + self

    def __sub__(self, other: Any) -> "Tensor":
        r"""Elementwise difference with broadcasting.

        .. math:: z = a - b, \qquad \bar a = \bar z, \qquad \bar b = -\bar z
        """
        a, b = self, _as_tensor(other, like=self)

        def backward(grad):
            return _unbroadcast(grad, a.shape), _unbroadcast(-grad, b.shape)

        return _result(a.data - b.data, (a, b), backward, "sub")

    def __rsub__(self, other: Any) -> "Tensor":
        """Compute ``other - self``."""
        return _as_tensor(other, like=self) - self

    def __mul__(self, other: Any) -> "Tensor":
        r"""Elementwise product with broadcasting.

        .. math:: z = a \odot b, \qquad \bar a = \bar z \odot b,
            \qquad \bar b = \bar z \odot a
        """
        a, b = self, _as_tensor(other, like=self)

        def backward(grad):
            return (
                _unbroadcast(grad * b.data, a.shape),
                _unbroadcast(grad * a.data, b.shape),
            )

        return _result(a.data * b.data, (a, b), backward, "mul")

    def __rmul__(self, other: Any) -> "Tensor":
        """Compute ``other * self``."""
        return _as_tensor(other, like=self) * self

    def __truediv__(self, other: Any) -> "Tensor":
        r"""Elementwise quotient with broadcasting.

        .. math:: z = a / b, \qquad \bar a = \bar z / b,
            \qquad \bar b = -\bar z \odot a / b^2
        """
        a, b = self, _as_tensor(other, like=self)

        def backward(grad):
            return (
                _unbroadcast(grad / b.data, a.shape),
                _unbroadcast(-grad * a.data / b.data**2, b.shape),
            )

        return _result(a.data / b.data, (a, b), backward, "div")

    def __rtruediv__(self, other: Any) -> "Tensor":
        """Compute ``other / self``."""
        return _as_tensor(other, like=self) / self

    def __pow__(self, other: Any) -> "Tensor":
        r"""Elementwise power with broadcasting.

        .. math:: z = a^b, \qquad \bar a = \bar z \odot b\, a^{b-1},
            \qquad \bar b = \bar z \odot a^b \ln a

        The gradient with respect to the exponent is computed only when the
        exponent requires gradients; it is defined for :math:`a > 0`.
        """
        a, b = self, _as_tensor(other, like=self)
        out = a.data**b.data

        def backward(grad):
            grad_a = _unbroadcast(grad * b.data * a.data ** (b.data - 1), a.shape)
            grad_b = None
            if b.requires_grad:
                grad_b = _unbroadcast(grad * out * backend.xp.log(a.data), b.shape)
            return grad_a, grad_b

        return _result(out, (a, b), backward, "pow")

    def __rpow__(self, other: Any) -> "Tensor":
        """Compute ``other ** self``."""
        return _as_tensor(other, like=self) ** self

    def __neg__(self) -> "Tensor":
        r"""Elementwise negation.

        .. math:: z = -a, \qquad \bar a = -\bar z
        """
        return _result(-self.data, (self,), lambda grad: (-grad,), "neg")

    def exp(self) -> "Tensor":
        r"""Elementwise exponential.

        .. math:: z = e^a, \qquad \bar a = \bar z \odot e^a = \bar z \odot z

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([0.0]).exp()
            Tensor([1.])
        """
        out = backend.xp.exp(self.data)
        return _result(out, (self,), lambda grad: (grad * out,), "exp")

    def log(self) -> "Tensor":
        r"""Elementwise natural logarithm.

        .. math:: z = \ln a, \qquad \bar a = \bar z / a

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([1.0]).log()
            Tensor([0.])
        """
        a = self
        out = backend.xp.log(a.data)
        return _result(out, (a,), lambda grad: (grad / a.data,), "log")

    # ------------------------------------------------------------------
    # Matrix multiplication
    # ------------------------------------------------------------------

    def __matmul__(self, other: Any) -> "Tensor":
        r"""Matrix product with NumPy semantics (vectors and batches).

        For matrices :math:`A \in \mathbb{R}^{m \times k}`,
        :math:`B \in \mathbb{R}^{k \times n}`:

        .. math:: Z = A B, \qquad \bar A = \bar Z B^\top,
            \qquad \bar B = A^\top \bar Z

        A 1-D left operand is treated as a row (shape ``(1, k)``) and a 1-D
        right operand as a column (``(k, 1)``); leading batch dimensions
        broadcast, and their gradients are summed back.

        Raises:
            ValueError: If an operand is 0-D or the shapes do not match (the
                message names both shapes).

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([[1.0, 2.0]]) @ Tensor([[3.0], [4.0]])
            Tensor([[11.]])
        """
        a, b = self, _as_tensor(other, like=self)
        out = _checked_matmul(a.data, b.data)
        a2 = a.data.reshape(1, -1) if a.ndim == 1 else a.data
        b2 = b.data.reshape(-1, 1) if b.ndim == 1 else b.data

        def backward(grad):
            grad2 = grad.reshape((a2 @ b2).shape)
            grad_a = grad2 @ backend.xp.swapaxes(b2, -1, -2)
            grad_b = backend.xp.swapaxes(a2, -1, -2) @ grad2
            return (
                _unbroadcast(grad_a, a2.shape).reshape(a.shape),
                _unbroadcast(grad_b, b2.shape).reshape(b.shape),
            )

        return _result(out, (a, b), backward, "matmul")

    def __rmatmul__(self, other: Any) -> "Tensor":
        """Compute ``other @ self``."""
        return _as_tensor(other, like=self) @ self

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> "Tensor":
        r"""Sum over all elements, or over the dimensions ``dim``.

        .. math:: z = \sum_i a_i, \qquad \bar a_i = \bar z

        The adjoint is copied (broadcast) back to every summed element.

        Args:
            dim: A dimension or a tuple of dimensions; ``None`` sums all.
            keepdim: Whether the summed dimensions are kept with size 1.

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([[1.0, 2.0], [3.0, 4.0]]).sum(dim=0)
            Tensor([4., 6.])
        """
        a = self
        dims = _normalize_dims(dim, a.ndim)
        out = a.data.sum(axis=dims, keepdims=keepdim)

        def backward(grad):
            if dims is not None and not keepdim:
                grad = backend.xp.expand_dims(grad, dims)
            return (backend.xp.broadcast_to(grad, a.shape),)

        return _result(out, (a,), backward, "sum")

    def mean(
        self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
    ) -> "Tensor":
        r"""Mean over all elements, or over the dimensions ``dim``.

        Built from :meth:`sum` and a division, so it needs no backward of
        its own:

        .. math:: z = \frac{1}{n} \sum_i a_i, \qquad \bar a_i = \bar z / n

        Args:
            dim: A dimension or a tuple of dimensions; ``None`` averages all.
            keepdim: Whether the averaged dimensions are kept with size 1.

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([[1.0, 2.0], [3.0, 4.0]]).mean()
            Tensor(2.5)
        """
        dims = _normalize_dims(dim, self.ndim)
        if dims is None:
            count = self.data.size
        else:
            count = math.prod(self.shape[d] for d in dims)
        return self.sum(dim=dim, keepdim=keepdim) / count

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        r"""Return the same values with a new shape.

        .. math:: \bar a = \operatorname{reshape}(\bar z, \text{shape of } a)

        Args:
            *shape: The new shape, as integers or one tuple; one entry may
                be ``-1``.

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2).shape
            (2, 2)
        """
        a = self
        out = a.data.reshape(_int_tuple(shape))
        return _result(out, (a,), lambda grad: (grad.reshape(a.shape),), "reshape")

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        r"""Swap two dimensions.

        .. math:: \bar a = \operatorname{swap}_{d_0, d_1}(\bar z)

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([[1.0, 2.0, 3.0]]).transpose(0, 1).shape
            (3, 1)
        """
        out = backend.xp.swapaxes(self.data, dim0, dim1)

        def backward(grad):
            return (backend.xp.swapaxes(grad, dim0, dim1),)

        return _result(out, (self,), backward, "transpose")

    def permute(self, *dims: int | tuple[int, ...]) -> "Tensor":
        r"""Reorder the dimensions: dimension ``i`` of the result is ``dims[i]``.

        The backward pass applies the inverse permutation to :math:`\bar z`.

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([[[1.0, 2.0, 3.0]]]).permute(2, 0, 1).shape
            (3, 1, 1)
        """
        order = [d % self.ndim for d in _int_tuple(dims)]
        inverse = [order.index(i) for i in range(self.ndim)]
        out = backend.xp.transpose(self.data, order)

        def backward(grad):
            return (backend.xp.transpose(grad, inverse),)

        return _result(out, (self,), backward, "permute")

    @property
    def T(self) -> "Tensor":
        """The transpose of a tensor with at most two dimensions.

        Raises:
            ValueError: For three or more dimensions (use :meth:`permute` or
                :meth:`transpose`).
        """
        if self.ndim > 2:
            raise ValueError(
                f".T needs at most 2 dimensions, got shape {self.shape}; "
                "use .transpose(dim0, dim1) or .permute(...)."
            )
        if self.ndim < 2:
            return self
        return self.transpose(0, 1)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, index: Any) -> "Tensor":
        r"""Select elements with NumPy indexing (slices, integers, arrays, masks).

        Each output element is a copy of one input element, so its adjoint
        is added to that input element:

        .. math:: z_j = a_{p(j)}, \qquad \bar a_i = \sum_{j : p(j) = i} \bar z_j

        The positions :math:`p(j)` are found by indexing an array that holds
        the flat position of every element; ``add.at`` then adds correctly
        even when an index repeats.

        Example:
            >>> from glassnn import Tensor
            >>> Tensor([[1.0, 2.0], [3.0, 4.0]])[:, 1]
            Tensor([2., 4.])
        """
        a = self
        index = _unwrap_index(index)

        def backward(grad):
            xp = backend.xp
            positions = xp.arange(a.data.size).reshape(a.shape)[index]
            flat_grad = xp.zeros(a.data.size, dtype=grad.dtype)
            xp.add.at(flat_grad, positions, grad)
            return (flat_grad.reshape(a.shape),)

        return _result(a.data[index], (a,), backward, "getitem")

    def __setitem__(self, index: Any, value: Any) -> None:
        """Refuse item assignment (GlassNN has no in-place operations)."""
        raise NotImplementedError(
            "GlassNN has no in-place operations, so item assignment is not "
            "supported; build a new tensor instead."
        )

    def _no_in_place(self, other: Any) -> "Tensor":
        raise NotImplementedError(
            "GlassNN has no in-place operations; write `x = x + y` instead of "
            "`x += y`, which creates a new tensor and keeps the graph correct."
        )

    __iadd__ = _no_in_place
    __isub__ = _no_in_place
    __imul__ = _no_in_place
    __itruediv__ = _no_in_place
    __ipow__ = _no_in_place
    __imatmul__ = _no_in_place

    # ------------------------------------------------------------------
    # The backward pass
    # ------------------------------------------------------------------

    def retain_grad(self) -> None:
        """Keep the gradient of this non-leaf tensor after :meth:`backward`.

        Raises:
            RuntimeError: If the tensor does not require gradients.
        """
        if not self.requires_grad:
            raise RuntimeError("retain_grad() needs a tensor with requires_grad=True.")
        self._retains_grad = True

    def backward(self, gradient: Any = None) -> None:
        r"""Compute the gradients of this tensor with respect to the leaves.

        Starting from :math:`\bar z` (``gradient``, or 1 for a scalar), the
        nodes of the graph are visited in reverse topological order: a node
        is processed only after every node that uses it, so its adjoint is
        complete. Each node's ``_backward`` turns its adjoint into
        contributions for its parents, which are added up. Leaf tensors add
        their final adjoint to ``.grad``.

        Args:
            gradient: :math:`\bar z`, with the shape of this tensor. May be
                omitted for a one-element tensor.

        Raises:
            RuntimeError: If the tensor does not require gradients, or if it
                has more than one element and ``gradient`` is not given.
            ValueError: If ``gradient`` has the wrong shape.

        Example:
            >>> from glassnn import Tensor
            >>> x = Tensor([1.0, 2.0], requires_grad=True)
            >>> (x * x).sum().backward()
            >>> x.grad
            Tensor([2., 4.])
        """
        if not self.requires_grad:
            raise RuntimeError(
                "backward() was called on a tensor with requires_grad=False; "
                "no input of the computation requires gradients."
            )
        grads: dict[Tensor, Any] = {self: self._initial_gradient(gradient)}
        for node in reversed(self._topological_order()):
            grad = grads.pop(node)
            if node.is_leaf or node._retains_grad:
                node._accumulate_grad(grad)
            if node.is_leaf:
                continue
            assert node._backward is not None
            for parent, parent_grad in zip(
                node._parents, node._backward(grad), strict=True
            ):
                if not parent.requires_grad or parent_grad is None:
                    continue
                if parent in grads:
                    grads[parent] = grads[parent] + parent_grad
                else:
                    grads[parent] = parent_grad

    def _initial_gradient(self, gradient: Any) -> Any:
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() without a gradient needs a one-element tensor; "
                    f"this one has shape {self.shape}. Pass gradient= of that shape."
                )
            return backend.xp.ones_like(self.data)
        if isinstance(gradient, Tensor):
            gradient = gradient.data
        seed = backend.asarray(gradient, dtype=self.dtype)
        if tuple(seed.shape) != self.shape:
            raise ValueError(
                f"The gradient has shape {tuple(seed.shape)}, "
                f"but the tensor has shape {self.shape}."
            )
        return seed

    def _topological_order(self) -> list["Tensor"]:
        """List the graph nodes that need gradients, each after its parents.

        An iterative depth-first search (no recursion, so long chains do not
        hit Python's recursion limit). A node is appended once all of its
        parents have been appended.
        """
        order: list[Tensor] = []
        visited: set[Tensor] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and parent not in visited:
                    stack.append((parent, False))
        return order

    def _accumulate_grad(self, grad: Any) -> None:
        if self.grad is None:
            self.grad = Tensor(backend.xp.array(grad), dtype=self.dtype)
        else:
            self.grad = Tensor(self.grad.data + grad, dtype=self.dtype)


class no_grad:
    """Disable graph recording, as a context manager or a decorator.

    Inside ``no_grad`` every operation returns a leaf tensor with
    ``requires_grad=False``. Use it for evaluation and for parameter updates.
    The switch is the module-level flag ``_grad_enabled``; it is restored on
    exit, also after an exception.

    Example:
        >>> from glassnn import Tensor, no_grad
        >>> x = Tensor([1.0], requires_grad=True)
        >>> with no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
    """

    def __enter__(self) -> None:
        """Switch recording off, remembering the previous state."""
        global _grad_enabled
        self._previous = _grad_enabled
        _grad_enabled = False

    def __exit__(self, *exc_info: object) -> None:
        """Restore the previous state."""
        global _grad_enabled
        _grad_enabled = self._previous

    def __call__(self, function: Callable) -> Callable:
        """Use ``no_grad()`` as a decorator."""

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with no_grad():
                return function(*args, **kwargs)

        return wrapper


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _result(
    data: Any, parents: tuple[Tensor, ...], backward: BackwardFn, op: str
) -> Tensor:
    """Wrap the output of an operation, recording it if gradients are needed."""
    out = Tensor(data, dtype=data.dtype)
    if _grad_enabled and any(parent.requires_grad for parent in parents):
        out.requires_grad = True
        out._parents = parents
        out._backward = backward
        out._op = op
    return out


def _as_tensor(value: Any, like: Tensor) -> Tensor:
    """Convert the other operand of a binary operation to a Tensor.

    A Python number takes the dtype of the floating tensor it is combined
    with, so ``x * 2.5`` keeps float32 data in float32.
    """
    if isinstance(value, Tensor):
        return value
    if isinstance(value, bool | int | float) and like.dtype.kind == "f":
        return Tensor(value, dtype=like.dtype)
    return Tensor(value)


def _unbroadcast(grad: Any, shape: tuple[int, ...]) -> Any:
    """Sum ``grad`` over the axes along which an operand of ``shape`` was broadcast.

    Broadcasting copies an operand along new leading axes and along axes of
    size 1. A copied value receives the sum of the adjoints of its copies.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _checked_matmul(a: Any, b: Any) -> Any:
    """Return ``a @ b``, with an error message that names both shapes."""
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError(
            "matmul needs operands with at least one dimension, got shapes "
            f"{tuple(a.shape)} and {tuple(b.shape)}."
        )
    try:
        return a @ b
    except ValueError as error:
        raise ValueError(
            f"matmul cannot multiply shapes {tuple(a.shape)} and {tuple(b.shape)}: "
            "the last dimension of the first must equal the second-to-last "
            "(or only) dimension of the second, and batch dimensions must "
            "broadcast."
        ) from error


def _normalize_dims(
    dim: int | tuple[int, ...] | None, ndim: int
) -> tuple[int, ...] | None:
    """Turn ``dim`` into a sorted tuple of non-negative dimensions."""
    if dim is None:
        return None
    dims = dim if isinstance(dim, tuple) else (dim,)
    return tuple(sorted(d % ndim for d in dims))


def _int_tuple(args: tuple[int | tuple[int, ...], ...]) -> tuple[int, ...]:
    """Accept both ``f(2, 3)`` and ``f((2, 3))``, returning ``(2, 3)``."""
    if len(args) == 1 and isinstance(args[0], tuple | list):
        return tuple(args[0])
    return cast(tuple[int, ...], args)


def _unwrap_index(index: Any) -> Any:
    """Replace Tensors inside an index by their arrays."""
    if isinstance(index, Tensor):
        return index.data
    if isinstance(index, tuple):
        return tuple(_unwrap_index(i) for i in index)
    return index
