"""Normalization layers: ``LayerNorm`` and ``BatchNorm1d``.

Book chapter: :book:`Regularization <chapters/06-regularization.html>`.
"""

import numpy.typing as npt

from glassnn import backend
from glassnn import functional as F
from glassnn.nn.module import Module
from glassnn.nn.parameter import Parameter
from glassnn.tensor import Tensor


class LayerNorm(Module):
    r"""Layer normalization over the trailing dimensions of each sample.

    .. math:: y = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \odot \gamma + \beta

    with the mean :math:`\mu` and the biased variance :math:`\sigma^2` of
    each sample over the last ``len(normalized_shape)`` dimensions
    :cite:p:`ba2016layer`; see :func:`glassnn.functional.layer_norm`.

    Args:
        normalized_shape: The trailing shape to normalize over (an ``int``
            means one dimension).
        eps: :math:`\varepsilon`, added to the variance.
        elementwise_affine: Whether to learn :math:`\gamma` (initialized to
            ones) and :math:`\beta` (zeros).
        bias: Whether to learn :math:`\beta` when ``elementwise_affine``.
        dtype: The dtype of the parameters.

    Attributes:
        weight: :math:`\gamma`, shape ``normalized_shape``, or ``None``.
        bias: :math:`\beta`, shape ``normalized_shape``, or ``None``.

    Shapes:
        input: ``(..., *normalized_shape)``.
        output: the same shape.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.LayerNorm(2)(Tensor([[1.0, 3.0], [-4.0, 0.0]]))
        Tensor([[-0.999995 ,  0.999995 ],
                [-0.9999988,  0.9999988]], op='add')
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create the parameters."""
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        xp = backend.xp
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = Parameter(xp.ones(self.normalized_shape), dtype=dtype)
            if bias:
                self.bias = Parameter(xp.zeros(self.normalized_shape), dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        """Normalize (see :func:`glassnn.functional.layer_norm`)."""
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        """Show the shape and the settings."""
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class BatchNorm1d(Module):
    r"""Batch normalization of each channel over the batch.

    In training mode, every channel is standardized with the mean and the
    variance of the current batch, and the running statistics (buffers
    ``running_mean`` and ``running_var``) are updated; in evaluation mode
    the running statistics are used instead, so the output of a sample no
    longer depends on the rest of the batch :cite:p:`ioffe2015batch`. See
    :func:`glassnn.functional.batch_norm` for the formulas.

    Args:
        num_features: The number of channels :math:`C`.
        eps: :math:`\varepsilon`, added to the variance.
        momentum: :math:`\rho` of the running statistics.
        affine: Whether to learn :math:`\gamma` (ones) and :math:`\beta`
            (zeros).
        track_running_stats: Whether to keep running statistics. If
            ``False``, batch statistics are used in both modes.
        dtype: The dtype of the parameters and buffers.

    Attributes:
        weight: :math:`\gamma`, shape ``(C,)``, or ``None``.
        bias: :math:`\beta`, shape ``(C,)``, or ``None``.
        running_mean: Buffer of shape ``(C,)`` (zeros at first), or ``None``.
        running_var: Buffer of shape ``(C,)`` (ones at first), or ``None``.

    Shapes:
        input: ``(N, C)`` or ``(N, C, L)``.
        output: the same shape.

    Note:
        Differences from PyTorch: no ``num_batches_tracked`` buffer and no
        ``momentum=None`` (cumulative average).

    Example:
        >>> from glassnn import Tensor, nn
        >>> layer = nn.BatchNorm1d(1)
        >>> layer(Tensor([[1.0], [3.0]]))
        Tensor([[-0.999995],
                [ 0.999995]], op='add')
        >>> layer.running_mean, layer.running_var
        (Tensor([0.2]), Tensor([1.1]))
    """

    running_mean: Tensor | None
    running_var: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create the parameters and the running statistics."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        xp = backend.xp
        self.weight = None
        self.bias = None
        if affine:
            self.weight = Parameter(xp.ones(num_features), dtype=dtype)
            self.bias = Parameter(xp.zeros(num_features), dtype=dtype)
        if track_running_stats:
            self.register_buffer(
                "running_mean", Tensor(xp.zeros(num_features), dtype=dtype)
            )
            self.register_buffer(
                "running_var", Tensor(xp.ones(num_features), dtype=dtype)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def forward(self, input: Tensor) -> Tensor:
        """Normalize (see :func:`glassnn.functional.batch_norm`)."""
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        """Show the number of channels and the settings."""
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )
