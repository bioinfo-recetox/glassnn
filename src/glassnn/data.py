"""Mini-batch loading (``DataLoader``) and ``one_hot``.

Book chapter: :book:`Optimization <chapters/05-optimization.html>`.
"""

import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from glassnn import backend
from glassnn.tensor import Tensor


class DataLoader:
    """Iterate over mini-batches of arrays held in memory.

    Each pass (``for xb, yb in loader``) is one epoch. With ``shuffle=True``
    the samples are visited in a new random order in every epoch.

    Args:
        X: Inputs, shape ``(N, ...)`` (an array or a Tensor).
        y: Targets, shape ``(N, ...)``, or ``None`` to iterate over ``X``
            only (e.g. for an autoencoder).
        batch_size: Number of samples per batch.
        shuffle: Whether to visit the samples in random order.
        drop_last: Whether to skip a last batch smaller than ``batch_size``.
        generator: A ``numpy.random.Generator`` for shuffling; ``None`` uses
            the global generator (:func:`glassnn.backend.manual_seed`).

    Yields:
        ``(X_batch, y_batch)`` as Tensors, or ``X_batch`` if ``y`` is
        ``None``. Values follow the dtype rule of
        :class:`~glassnn.tensor.Tensor` (float inputs get the default dtype,
        integer labels stay integer).

    Raises:
        ValueError: If ``X`` and ``y`` have different lengths (the message
            names both) or ``batch_size < 1``.

    Note:
        Differences from PyTorch: ``torch.utils.data.DataLoader`` takes a
        ``Dataset`` object; GlassNN takes the arrays directly, which is all
        the course examples need.

    Example:
        >>> import numpy as np
        >>> from glassnn.data import DataLoader
        >>> loader = DataLoader(np.zeros((10, 3)), np.arange(10), batch_size=4)
        >>> [tuple(xb.shape) for xb, yb in loader]
        [(4, 3), (4, 3), (2, 3)]
    """

    def __init__(
        self,
        X: Any,
        y: Any = None,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: np.random.Generator | None = None,
    ) -> None:
        """Store the data (converted once) and the settings."""
        self.X = Tensor(X).data
        self.y = None if y is None else Tensor(y).data
        if self.y is not None and len(self.y) != len(self.X):
            raise ValueError(
                f"X and y must have the same length, got {len(self.X)} and "
                f"{len(self.y)}."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __len__(self) -> int:
        """The number of batches per epoch."""
        if self.drop_last:
            return len(self.X) // self.batch_size
        return math.ceil(len(self.X) / self.batch_size)

    def __iter__(self) -> Iterator[Any]:
        """Yield the batches of one epoch."""
        n = len(self.X)
        if self.shuffle:
            generator = self.generator or backend.get_generator()
            order = generator.permutation(n)
        else:
            order = backend.xp.arange(n)
        for batch in range(len(self)):
            index = order[batch * self.batch_size : (batch + 1) * self.batch_size]
            xb = Tensor(self.X[index], dtype=self.X.dtype)
            if self.y is None:
                yield xb
            else:
                yield xb, Tensor(self.y[index], dtype=self.y.dtype)


def one_hot(tensor: Any, num_classes: int = -1) -> Tensor:
    """Encode integer labels as one-hot rows.

    Args:
        tensor: Integer labels of shape ``(N,)`` (any shape works; a last
            dimension is added).
        num_classes: The number of classes :math:`C`; ``-1`` uses
            ``max(label) + 1``.

    Returns:
        An int64 tensor of shape ``(..., C)`` with a 1 at each label.

    Raises:
        TypeError: If the labels are not integers.
        ValueError: If a label is not smaller than ``num_classes``.

    Note:
        Differences from PyTorch: it lives in ``glassnn.data``
        (PyTorch: ``torch.nn.functional.one_hot``).

    Example:
        >>> from glassnn import Tensor
        >>> from glassnn.data import one_hot
        >>> one_hot(Tensor([0, 2]))
        Tensor([[1, 0, 0],
                [0, 0, 1]])
    """
    labels = tensor.data if isinstance(tensor, Tensor) else backend.asarray(tensor)
    if labels.dtype.kind not in "iu":
        raise TypeError(f"one_hot needs integer labels, got dtype {labels.dtype}.")
    largest = int(labels.max()) if labels.size else -1
    if num_classes == -1:
        num_classes = largest + 1
    if largest >= num_classes:
        raise ValueError(f"Label {largest} does not fit num_classes={num_classes}.")
    eye = backend.xp.eye(num_classes, dtype=backend.xp.int64)
    return Tensor(eye[labels])
