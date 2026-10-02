"""``Parameter``: a tensor that a module learns."""

from typing import Any

import numpy.typing as npt

from glassnn.tensor import Tensor


class Parameter(Tensor):
    """A tensor that is a learnable parameter of a :class:`Module`.

    A ``Parameter`` is an ordinary :class:`~glassnn.tensor.Tensor` with two
    differences: it requires gradients by default, and assigning it to an
    attribute of a :class:`Module` registers it, so that
    :meth:`Module.parameters` finds it. Operations on parameters return plain
    tensors.

    Args:
        data: The initial values (see :class:`~glassnn.tensor.Tensor`).
        requires_grad: Whether gradients are computed (default ``True``).
        dtype: The dtype, or ``None`` for the default rule.

    Example:
        >>> from glassnn import nn
        >>> nn.Parameter([1.0, 2.0])
        Parameter([1., 2.], requires_grad=True)
    """

    def __init__(
        self,
        data: Any,
        requires_grad: bool = True,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create a parameter; see the class docstring."""
        super().__init__(data, requires_grad=requires_grad, dtype=dtype)
