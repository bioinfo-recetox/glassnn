"""Gradient clipping."""

import math
from collections.abc import Iterable

from glassnn import backend
from glassnn.tensor import Tensor


def clip_grad_norm_(parameters: Tensor | Iterable[Tensor], max_norm: float) -> Tensor:
    r"""Rescale gradients so that their total Euclidean norm is at most ``max_norm``.

    With all gradients stacked into one vector :math:`g`,

    .. math:: g \leftarrow g \cdot \min\left(1,
        \frac{\text{max\_norm}}{\lVert g \rVert_2 + 10^{-6}}\right).

    The direction of the update is kept; only very large steps (e.g. at the
    start of training) are shortened.

    Args:
        parameters: One tensor or an iterable of tensors; those without a
            gradient are ignored.
        max_norm: The largest allowed norm.

    Returns:
        The total norm before clipping, as a scalar tensor.

    Note:
        Differences from PyTorch: it lives in ``glassnn.optim``
        (PyTorch: ``torch.nn.utils.clip_grad_norm_``), and only the
        Euclidean norm is supported.

    Example:
        >>> from glassnn import Tensor, nn, optim
        >>> p = nn.Parameter([0.0, 0.0])
        >>> p.grad = Tensor([3.0, 4.0])
        >>> optim.clip_grad_norm_(p, max_norm=1.0)
        Tensor(5.)
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    pairs = [(p, p.grad) for p in parameters if p.grad is not None]
    total = math.sqrt(sum(float((grad.data**2).sum()) for _, grad in pairs))
    scale = min(1.0, max_norm / (total + 1e-6))
    if scale < 1.0:
        for p, grad in pairs:
            p.grad = Tensor(grad.data * scale, dtype=grad.dtype)
    return Tensor(backend.xp.asarray(total))
