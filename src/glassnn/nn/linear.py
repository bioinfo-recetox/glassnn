"""The fully connected layer ``Linear``."""

import math

import numpy.typing as npt

from glassnn import backend
from glassnn import functional as F
from glassnn.nn import init
from glassnn.nn.module import Module
from glassnn.nn.parameter import Parameter
from glassnn.tensor import Tensor


class Linear(Module):
    r"""Fully connected layer :math:`y = x W^\top + b`.

    Args:
        in_features: Size :math:`n_\text{in}` of each input.
        out_features: Size :math:`n_\text{out}` of each output.
        bias: Whether to learn the bias :math:`b`.
        dtype: The dtype of the parameters (default: the default dtype).

    Attributes:
        weight: :math:`W`, shape ``(out_features, in_features)``.
        bias: :math:`b`, shape ``(out_features,)``, or ``None``.

    Shapes:
        input ``(..., in_features)`` to output ``(..., out_features)``.

    The initialization is PyTorch's: :math:`W` from
    ``kaiming_uniform_(a=sqrt(5))``, which is uniform on
    :math:`[-1/\sqrt{n_\text{in}}, 1/\sqrt{n_\text{in}}]`, and :math:`b`
    uniform on the same interval :cite:p:`he2015delving`.

    Example:
        >>> from glassnn import Tensor, nn
        >>> layer = nn.Linear(3, 2)
        >>> layer(Tensor([[1.0, 2.0, 3.0]])).shape
        (1, 2)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create and initialize the parameters."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        xp = backend.xp
        self.weight = Parameter(xp.zeros((out_features, in_features)), dtype=dtype)
        self.bias = Parameter(xp.zeros(out_features), dtype=dtype) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw new initial values (from the global generator)."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Apply the affine map (see :func:`glassnn.functional.linear`)."""
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        """Show the sizes and whether there is a bias."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
