"""Max and average pooling layers (1-D and 2-D), and ``Flatten``.

Book chapter: :book:`Convolutions for sequences <chapters/09-convolutions.html>`.
"""

import math

from glassnn import functional as F
from glassnn.nn.module import Module
from glassnn.tensor import Tensor


class MaxPool1d(Module):
    r"""Maximum over sliding windows; see :func:`glassnn.functional.max_pool1d`.

    With ``kernel_size`` equal to the sequence length this is *global max
    pooling*: one value per channel, the best match of a filter anywhere in
    the sequence.

    Args:
        kernel_size: The window length.
        stride: The step (default: ``kernel_size``).
        padding: :math:`-\infty` added at both ends.

    Shapes:
        input: ``(N, C, L)``.
        output: ``(N, C, L_out)``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.MaxPool1d(2)(Tensor([[[1.0, 5.0, 2.0, 0.0]]]))
        Tensor([[[5., 2.]]])
    """

    def __init__(
        self, kernel_size: int, stride: int | None = None, padding: int = 0
    ) -> None:
        """Store the settings."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        """Pool."""
        return F.max_pool1d(input, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        """Show the settings."""
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class AvgPool1d(MaxPool1d):
    """Mean over sliding windows; see :func:`glassnn.functional.avg_pool1d`.

    Args:
        kernel_size: The window length.
        stride: The step (default: ``kernel_size``).
        padding: Zeros added at both ends (they count in the mean).

    Shapes:
        input: ``(N, C, L)``.
        output: ``(N, C, L_out)``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.AvgPool1d(2)(Tensor([[[1.0, 3.0, 2.0, 4.0]]]))
        Tensor([[[2., 3.]]])
    """

    def forward(self, input: Tensor) -> Tensor:
        """Pool."""
        return F.avg_pool1d(input, self.kernel_size, self.stride, self.padding)


class MaxPool2d(MaxPool1d):
    r"""Maximum over sliding windows of an image.

    See :func:`glassnn.functional.max_pool2d`.

    Args:
        kernel_size: The window size, an ``int`` or a pair.
        stride: The step, an ``int`` or a pair (default: ``kernel_size``).
        padding: :math:`-\infty` added on all sides, an ``int`` or a pair.

    Shapes:
        input: ``(N, C, H, W)``.
        output: ``(N, C, H_out, W_out)``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.MaxPool2d(2)(Tensor([[[[1.0, 2.0], [4.0, 3.0]]]]))
        Tensor([[[[4.]]]])
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        """Store the settings."""
        # MaxPool1d stores the three settings; here they may also be pairs.
        super().__init__(kernel_size, stride, padding)  # type: ignore[arg-type]

    def forward(self, input: Tensor) -> Tensor:
        """Pool."""
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding)


class AvgPool2d(MaxPool2d):
    """Mean over sliding windows of an image.

    See :func:`glassnn.functional.avg_pool2d`.

    Args:
        kernel_size: The window size, an ``int`` or a pair.
        stride: The step, an ``int`` or a pair (default: ``kernel_size``).
        padding: Zeros added on all sides (they count in the mean).

    Shapes:
        input: ``(N, C, H, W)``.
        output: ``(N, C, H_out, W_out)``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.AvgPool2d(2)(Tensor([[[[1.0, 2.0], [4.0, 3.0]]]]))
        Tensor([[[[2.5]]]])
    """

    def forward(self, input: Tensor) -> Tensor:
        """Pool."""
        return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding)


class Flatten(Module):
    """Merge the dimensions ``start_dim`` to ``end_dim`` into one.

    The usual bridge from convolutional feature maps ``(N, C, L)`` to a
    ``Linear`` layer: ``Flatten()`` gives ``(N, C * L)``. A reshape, so the
    gradient is the incoming gradient reshaped back.

    Args:
        start_dim: The first dimension to merge.
        end_dim: The last dimension to merge (negative counts from the end).

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.Flatten()(Tensor([[[1.0, 2.0], [3.0, 4.0]]])).shape
        (1, 4)
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """Store the range of dimensions."""
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """Reshape."""
        shape = input.shape
        start = self.start_dim % len(shape)
        end = self.end_dim % len(shape)
        merged = math.prod(shape[start : end + 1])
        return input.reshape(*shape[:start], merged, *shape[end + 1 :])

    def extra_repr(self) -> str:
        """Show the range of dimensions."""
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
