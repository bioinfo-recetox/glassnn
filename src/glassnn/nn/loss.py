"""Loss functions as modules (each calls :mod:`glassnn.functional`)."""

from typing import Any

from glassnn import functional as F
from glassnn.nn.module import Module
from glassnn.tensor import Tensor


class MSELoss(Module):
    """Mean squared error; see :func:`glassnn.functional.mse_loss`."""

    def __init__(self, reduction: str = "mean") -> None:
        """Store the reduction (``"mean"``, ``"sum"`` or ``"none"``)."""
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Any) -> Tensor:
        """Compute the loss."""
        return F.mse_loss(input, target, reduction=self.reduction)


class CrossEntropyLoss(Module):
    """Cross-entropy of logits; see :func:`glassnn.functional.cross_entropy`."""

    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0) -> None:
        """Store the reduction and the label smoothing."""
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Any) -> Tensor:
        """Compute the loss."""
        return F.cross_entropy(
            input,
            target,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class BCEWithLogitsLoss(Module):
    """Binary cross-entropy of logits.

    See :func:`glassnn.functional.binary_cross_entropy_with_logits`.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Store the reduction."""
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Any) -> Tensor:
        """Compute the loss."""
        return F.binary_cross_entropy_with_logits(
            input, target, reduction=self.reduction
        )
