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
        parametrization: ``"standard"`` (the default, as in PyTorch) or
            ``"ntk"``, see below.

    Attributes:
        weight: :math:`W`, shape ``(out_features, in_features)``.
        bias: :math:`b`, shape ``(out_features,)``, or ``None``.

    Shapes:
        input: ``(..., in_features)``.
        output: ``(..., out_features)``.

    The initialization is PyTorch's: :math:`W` from
    ``kaiming_uniform_(a=sqrt(5))``, which is uniform on
    :math:`[-1/\sqrt{n_\text{in}}, 1/\sqrt{n_\text{in}}]`, and :math:`b`
    uniform on the same interval :cite:p:`he2015delving`.

    **NTK parametrization.** With ``parametrization="ntk"`` the layer computes

    .. math:: y = \frac{1}{\sqrt{n_\text{in}}}\, x W^\top + b,
        \qquad W_{ij}, b_i \sim \mathcal N(0, 1),

    :cite:p:`jacot2018neural,lee2019wide` (with
    :math:`\sigma_w = \sigma_b = 1`). The forward pass at initialization
    has the same distribution as with weights
    :math:`\mathcal N(0, 1/n_\text{in})`, but the gradient
    :math:`\bar W = n_\text{in}^{-1/2}\, \bar Y^\top X` is smaller by the
    factor :math:`1/\sqrt{n_\text{in}}`. With this scaling, and a fixed
    learning rate, the neural tangent kernel has a finite limit as the
    width grows, and training of a very wide network stays close to its
    linearization (book chapter 8).

    Note:
        Differences from PyTorch: ``torch.nn.Linear`` has no
        ``parametrization`` argument.

    Raises:
        ValueError: For an unknown ``parametrization``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> layer = nn.Linear(3, 2)
        >>> layer(Tensor([[1.0, 2.0, 3.0]])).shape
        (1, 2)
        >>> nn.Linear(3, 2, parametrization="ntk")
        Linear(in_features=3, out_features=2, bias=True, parametrization='ntk')
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: npt.DTypeLike | None = None,
        parametrization: str = "standard",
    ) -> None:
        """Create and initialize the parameters."""
        super().__init__()
        if parametrization not in ("standard", "ntk"):
            raise ValueError(
                f"parametrization must be 'standard' or 'ntk', got {parametrization!r}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.parametrization = parametrization
        xp = backend.xp
        self.weight = Parameter(xp.zeros((out_features, in_features)), dtype=dtype)
        self.bias = Parameter(xp.zeros(out_features), dtype=dtype) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw new initial values (from the global generator)."""
        if self.parametrization == "ntk":
            init.normal_(self.weight)
            if self.bias is not None:
                init.normal_(self.bias)
            return
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Apply the affine map (see :func:`glassnn.functional.linear`)."""
        if self.parametrization == "ntk":
            scaled_weight = self.weight / math.sqrt(self.in_features)
            return F.linear(input, scaled_weight, self.bias)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        """Show the sizes, the bias, and a non-standard parametrization."""
        text = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
        if self.parametrization != "standard":
            text += f", parametrization={self.parametrization!r}"
        return text
