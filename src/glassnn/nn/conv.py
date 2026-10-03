"""Convolution layers: ``Conv1d`` and ``Conv2d``.

Book chapter: :book:`Convolutions for sequences <chapters/09-convolutions.html>`.
"""

import math
from typing import Any

import numpy.typing as npt

from glassnn import backend
from glassnn import functional as F
from glassnn.nn import init
from glassnn.nn.module import Module
from glassnn.nn.parameter import Parameter
from glassnn.tensor import Tensor


class _ConvNd(Module):
    """Parameters, initialization and printing shared by Conv1d and Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...] | str,
        dilation: tuple[int, ...],
        bias: bool,
        dtype: npt.DTypeLike | None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        xp = backend.xp
        shape = (out_channels, in_channels, *kernel_size)
        self.weight = Parameter(xp.zeros(shape), dtype=dtype)
        self.bias = Parameter(xp.zeros(out_channels), dtype=dtype) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """PyTorch's default: as for ``Linear``, with fan-in C_in * prod(kernel)."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * math.prod(self.kernel_size)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        """Show the channels and the settings."""
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding!r}, dilation={self.dilation}, "
            f"bias={self.bias is not None}"
        )


def _ntuple(value: Any, n: int) -> Any:
    """``3`` -> ``(3,) * n``; tuples and strings are kept."""
    return (value,) * n if isinstance(value, int) else value


class Conv1d(_ConvNd):
    r"""1-D convolution layer over sequences.

    .. math:: y_{n,o,i} = b_o + \sum_{c} \sum_{k} w_{o,c,k}\, x_{n,c,\, s i + d k}

    See :func:`glassnn.functional.conv1d` for the details and the gradient.
    For DNA one-hot encoded as ``(N, 4, L)``, each output channel is a
    learned position weight matrix scanned along the sequence
    :cite:p:`alipanahi2015predicting`.

    Args:
        in_channels: :math:`C_\text{in}`.
        out_channels: :math:`C_\text{out}`, the number of filters.
        kernel_size: :math:`K`.
        stride: :math:`s`.
        padding: An ``int``, ``"valid"`` or ``"same"`` (zeros).
        dilation: :math:`d`.
        bias: Whether to learn :math:`b`.
        dtype: The dtype of the parameters.

    Attributes:
        weight: Shape ``(C_out, C_in, K)``.
        bias: Shape ``(C_out,)``, or ``None``.

    Shapes:
        input: ``(N, C_in, L)``.
        output: ``(N, C_out, L_out)``.

    The initialization is PyTorch's: uniform on
    :math:`[-1/\sqrt{C_\text{in} K}, 1/\sqrt{C_\text{in} K}]` for the weight
    and the bias.

    Note:
        Differences from PyTorch: no ``groups`` and no ``padding_mode``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> layer = nn.Conv1d(4, 8, kernel_size=5, padding="same")
        >>> layer(Tensor([[[0.0] * 20] * 4])).shape
        (1, 8, 20)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        bias: bool = True,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create and initialize the parameters."""
        super().__init__(
            in_channels,
            out_channels,
            _ntuple(kernel_size, 1),
            _ntuple(stride, 1),
            _ntuple(padding, 1),
            _ntuple(dilation, 1),
            bias,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        """Convolve (see :func:`glassnn.functional.conv1d`)."""
        padding = self.padding if isinstance(self.padding, str) else self.padding[0]
        return F.conv1d(
            input, self.weight, self.bias, self.stride[0], padding, self.dilation[0]
        )


class Conv2d(_ConvNd):
    r"""2-D convolution layer over images.

    See :func:`glassnn.functional.conv2d`. ``kernel_size``, ``stride``,
    ``padding`` and ``dilation`` take an ``int`` or a pair.

    Args:
        in_channels: :math:`C_\text{in}`.
        out_channels: :math:`C_\text{out}`.
        kernel_size: :math:`(K_H, K_W)`.
        stride: :math:`(s_1, s_2)`.
        padding: An ``int``, a pair, ``"valid"`` or ``"same"`` (zeros).
        dilation: :math:`(d_1, d_2)`.
        bias: Whether to learn :math:`b`.
        dtype: The dtype of the parameters.

    Attributes:
        weight: Shape ``(C_out, C_in, K_H, K_W)``.
        bias: Shape ``(C_out,)``, or ``None``.

    Shapes:
        input: ``(N, C_in, H, W)``.
        output: ``(N, C_out, H_out, W_out)``.

    Note:
        Differences from PyTorch: no ``groups`` and no ``padding_mode``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> layer = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)
        >>> layer(Tensor([[[[0.0] * 8] * 8]])).shape
        (1, 2, 4, 4)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create and initialize the parameters."""
        super().__init__(
            in_channels,
            out_channels,
            tuple(_ntuple(kernel_size, 2)),
            tuple(_ntuple(stride, 2)),
            _ntuple(padding, 2),
            tuple(_ntuple(dilation, 2)),
            bias,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        """Convolve (see :func:`glassnn.functional.conv2d`)."""
        return F.conv2d(
            input, self.weight, self.bias, self.stride, self.padding, self.dilation
        )
