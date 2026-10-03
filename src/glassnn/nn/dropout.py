"""The ``Dropout`` layer.

Book chapter: :book:`Regularization <chapters/06-regularization.html>`.
"""

from glassnn import functional as F
from glassnn.nn.module import Module
from glassnn.tensor import Tensor


class Dropout(Module):
    r"""Inverted dropout in training, the identity in evaluation.

    Each call in training mode zeros every element with probability ``p``
    and multiplies the others by :math:`1 / (1 - p)`
    :cite:p:`srivastava2014dropout`; see :func:`glassnn.functional.dropout`.
    The masks are drawn from the global generator
    (:func:`glassnn.backend.manual_seed`).

    Args:
        p: The probability of zeroing an element.

    Shapes:
        input: any shape.
        output: the same shape.

    Note:
        Differences from PyTorch: there is no ``inplace`` argument.

    Example:
        >>> from glassnn import Tensor, nn
        >>> layer = nn.Dropout(p=0.5)
        >>> layer.eval()
        Dropout(p=0.5)
        >>> layer(Tensor([1.0, 2.0]))
        Tensor([1., 2.])
    """

    def __init__(self, p: float = 0.5) -> None:
        """Store the probability; it is checked when the layer is called."""
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        """Apply dropout if the module is in training mode."""
        return F.dropout(input, self.p, self.training)

    def extra_repr(self) -> str:
        """Show the probability."""
        return f"p={self.p}"
