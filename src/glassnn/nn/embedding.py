"""``Embedding`` and the sinusoidal ``PositionalEncoding``.

Book chapter: :book:`Attention <chapters/10-attention.html>`.
"""

import math

import numpy.typing as npt

from glassnn import backend
from glassnn import functional as F
from glassnn.nn import init
from glassnn.nn.module import Module
from glassnn.nn.parameter import Parameter
from glassnn.tensor import Tensor


class Embedding(Module):
    r"""A lookup table of learned vectors, one per token.

    Token :math:`v` is mapped to row :math:`W_v` of the weight; see
    :func:`glassnn.functional.embedding`. Equivalent to a ``Linear`` layer
    without bias applied to one-hot vectors, but without building them.

    Args:
        num_embeddings: The vocabulary size :math:`V`.
        embedding_dim: The vector size :math:`D`.
        padding_idx: A token whose vector is zero at initialization and
            receives no gradient (e.g. padding), or ``None``.
        dtype: The dtype of the weight.

    Attributes:
        weight: Shape ``(V, D)``, initialized from :math:`\mathcal N(0, 1)`
            as in PyTorch.

    Shapes:
        input: integer indices, any shape ``(*)``.
        output: ``(*, D)``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.Embedding(10, 3)(Tensor([[1, 2, 2]])).shape
        (1, 3, 3)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create and initialize the weight."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        shape = (num_embeddings, embedding_dim)
        self.weight = Parameter(backend.xp.zeros(shape), dtype=dtype)
        init.normal_(self.weight)
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0.0  # initialization, like nn.init

    def forward(self, input: Tensor) -> Tensor:
        """Look up the vectors."""
        return F.embedding(input, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        """Show the sizes and the padding index."""
        text = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            text += f", padding_idx={self.padding_idx}"
        return text


class PositionalEncoding(Module):
    r"""Add the sinusoidal positional encoding of :cite:t:`vaswani2017attention`.

    Attention treats its input as a set: permuting the positions permutes the
    output. To make the order visible, position :math:`p` gets the vector

    .. math:: \mathrm{PE}_{p, 2i} = \sin\frac{p}{10000^{2i/d}}, \qquad
        \mathrm{PE}_{p, 2i+1} = \cos\frac{p}{10000^{2i/d}},

    which is added to the input embedding. The frequencies form a geometric
    series, and :math:`\mathrm{PE}_{p+k}` is a fixed linear function
    (a rotation in every pair of coordinates) of :math:`\mathrm{PE}_p`, so
    relative positions are easy to read. The table is a buffer, not a
    parameter.

    Args:
        d_model: The embedding size :math:`d` (even).
        max_len: The longest sequence supported.
        dropout: Dropout probability after the addition.
        dtype: The dtype of the table.

    Shapes:
        input: ``(N, L, d)``.
        output: ``(N, L, d)``.

    Raises:
        ValueError: If ``d_model`` is odd, or (when called) if the sequence
            is longer than ``max_len``.

    Note:
        Differences from PyTorch: PyTorch has no such module (its tutorials
        define one); here it is batch-first and its dropout defaults to 0.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.PositionalEncoding(4, max_len=8).pe.shape
        (8, 4)
    """

    pe: Tensor | None

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Compute the table."""
        super().__init__()
        if d_model % 2:
            raise ValueError(f"d_model must be even, got {d_model}.")
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        xp = backend.xp
        position = xp.arange(max_len)[:, None]
        frequency = xp.exp(-math.log(10000.0) * xp.arange(0, d_model, 2) / d_model)
        table = xp.zeros((max_len, d_model))
        table[:, 0::2] = xp.sin(position * frequency)
        table[:, 1::2] = xp.cos(position * frequency)
        self.register_buffer("pe", Tensor(table, dtype=dtype))

    def forward(self, input: Tensor) -> Tensor:
        """Add the encoding of the first ``L`` positions."""
        length = input.shape[1]
        if length > self.max_len:
            raise ValueError(
                f"The input has {length} positions, more than max_len {self.max_len}."
            )
        assert self.pe is not None
        out = input + self.pe[:length]
        return F.dropout(out, self.dropout, self.training)

    def extra_repr(self) -> str:
        """Show the settings."""
        return f"d_model={self.d_model}, max_len={self.max_len}, dropout={self.dropout}"
