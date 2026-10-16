"""Loss functions as modules (each calls :mod:`glassnn.functional`).

Book chapter: :book:`Backpropagation in a multilayer perceptron
<chapters/03-mlp.html>`.
"""

from typing import Any

from glassnn import functional as F
from glassnn.nn.module import Module
from glassnn.tensor import Tensor


class MSELoss(Module):
    r"""Mean squared error, :math:`\ell = \frac1n \sum_i (a_i - y_i)^2`.

    See :func:`glassnn.functional.mse_loss`.

    Args:
        reduction: ``"mean"`` (the formula), ``"sum"`` (without
            :math:`\frac1n`) or ``"none"`` (the squared errors).

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.MSELoss()(Tensor([1.0, 3.0]), Tensor([0.0, 0.0]))
        Tensor(5.)
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Store the reduction (``"mean"``, ``"sum"`` or ``"none"``)."""
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Any) -> Tensor:
        """Compute the loss."""
        return F.mse_loss(input, target, reduction=self.reduction)


class CrossEntropyLoss(Module):
    r"""Cross-entropy of logits.

    .. math:: \ell_i = -\log \operatorname{softmax}(a_i)_{y_i}

    With label smoothing :math:`\varepsilon` the target puts
    :math:`1 - \varepsilon + \varepsilon / C` on the true class and
    :math:`\varepsilon / C` on each of the :math:`C - 1` others
    :cite:p:`szegedy2016rethinking`. See
    :func:`glassnn.functional.cross_entropy`.

    Args:
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` over the examples.
        label_smoothing: :math:`\varepsilon \in [0, 1]`.

    Example:
        >>> from glassnn import Tensor, nn
        >>> loss = nn.CrossEntropyLoss()(Tensor([[0.0, 0.0]]), Tensor([1]))
        >>> round(loss.item(), 4)
        0.6931
    """

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
    r"""Binary cross-entropy of logits, computed without overflow.

    .. math:: \ell_i = -y_i \log \sigma(a_i) - (1 - y_i) \log (1 - \sigma(a_i))

    See :func:`glassnn.functional.binary_cross_entropy_with_logits`.

    Args:
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> loss = nn.BCEWithLogitsLoss()(Tensor([0.0]), Tensor([1.0]))
        >>> round(loss.item(), 4)
        0.6931
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
