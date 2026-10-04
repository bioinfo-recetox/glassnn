"""``MultiheadAttention``.

Book chapter: :book:`Attention <chapters/10-attention.html>`.
"""

import math

import numpy.typing as npt

from glassnn import functional as F
from glassnn.functional import _attention
from glassnn.nn import init
from glassnn.nn.linear import Linear
from glassnn.nn.module import Module
from glassnn.tensor import Tensor


class MultiheadAttention(Module):
    r"""Multi-head attention :cite:p:`vaswani2017attention`.

    The queries, keys and values are projected to :math:`h` heads of size
    :math:`d = E / h`; each head attends separately with
    :func:`glassnn.functional.scaled_dot_product_attention`; the heads are
    concatenated and projected back:

    .. math:: \mathrm{head}_j = \operatorname{softmax}\Bigl(
        \frac{Q W^Q_j (K W^K_j)^\top}{\sqrt d} + M\Bigr)\, V W^V_j, \qquad
        Y = [\mathrm{head}_1, \dots, \mathrm{head}_h]\, W^O .

    Each head can attend to different positions for different reasons; the
    total cost is that of one head of size :math:`E`.

    Args:
        embed_dim: :math:`E`.
        num_heads: :math:`h`; must divide :math:`E`.
        dropout: Dropout probability on the attention weights (training
            mode only).
        bias: Whether the four projections have biases.
        batch_first: Must be ``True`` (inputs ``(N, L, E)``).
        dtype: The dtype of the parameters.

    Attributes:
        q_proj, k_proj, v_proj: ``Linear(E, E)``, the stacked
            :math:`W^Q_j, W^K_j, W^V_j` of all heads.
        out_proj: ``Linear(E, E)``, :math:`W^O`.

    Shapes:
        query: ``(N, L, E)``; key and value: ``(N, S, E)``.
        output: ``(N, L, E)``; weights: ``(N, L, S)`` (averaged over heads)
        or ``(N, h, L, S)``.

    The initialization follows PyTorch: the input projections are uniform
    on :math:`\pm\sqrt{6 / 4E}` (Xavier uniform of PyTorch's packed
    :math:`(3E, E)` matrix), all biases are zero, and ``out_proj.weight`` is
    initialized like a ``Linear`` layer.

    Note:
        Differences from PyTorch: inputs are batch-first only; the input
        projections are three ``Linear`` modules (PyTorch packs them into
        one ``in_proj_weight`` of shape ``(3E, E)``, so the ``state_dict``
        names differ); no ``kdim``, ``vdim``, ``add_bias_kv`` or
        ``add_zero_attn``; ``is_causal=True`` without ``attn_mask`` builds
        the causal mask (PyTorch requires the mask).

    Example:
        >>> from glassnn import Tensor, nn
        >>> attention = nn.MultiheadAttention(embed_dim=8, num_heads=2)
        >>> x = Tensor([[[0.1] * 8] * 5])  # (N=1, L=5, E=8)
        >>> output, weights = attention(x, x, x)
        >>> output.shape, weights.shape
        ((1, 5, 8), (1, 5, 5))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create the four projections."""
        super().__init__()
        if not batch_first:
            raise NotImplementedError(
                "GlassNN attention takes inputs of shape (N, L, E) only; "
                "use batch_first=True."
            )
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound = math.sqrt(6 / (4 * self.embed_dim))
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            init.uniform_(layer.weight, -bound, bound)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        r"""Attend from ``query`` to ``key``/``value``.

        Args:
            query: ``(N, L, E)``.
            key: ``(N, S, E)``.
            value: ``(N, S, E)``.
            key_padding_mask: ``(N, S)``; ``True`` (or :math:`-\infty`)
                marks keys to ignore, e.g. padding.
            need_weights: Whether to return the attention weights.
            attn_mask: ``(L, S)`` or ``(N * h, L, S)``; ``True`` (or
                :math:`-\infty`) where a query may not attend to a key.
            average_attn_weights: Average the returned weights over heads.
            is_causal: Hide the keys after each query (builds the causal
                mask if ``attn_mask`` is ``None``).

        Returns:
            ``(output, weights)``; ``weights`` is ``None`` if not
            ``need_weights``.
        """
        n, length, _ = query.shape
        source_length = key.shape[1]
        q = self._split_heads(self.q_proj(query))  # (N, h, L, d)
        k = self._split_heads(self.k_proj(key))  # (N, h, S, d)
        v = self._split_heads(self.v_proj(value))
        mask = self._combine_masks(
            key_padding_mask, attn_mask, is_causal, n, length, source_length, q.dtype
        )
        dropout_p = self.dropout if self.training else 0.0
        heads, weights = _attention(q, k, v, mask, dropout_p, False, None)
        # (N, h, L, d) -> (N, L, h, d) -> (N, L, E): concatenate the heads.
        merged = heads.transpose(1, 2).reshape(n, length, self.embed_dim)
        output = self.out_proj(merged)
        if not need_weights:
            return output, None
        return output, weights.mean(dim=1) if average_attn_weights else weights

    def _split_heads(self, x: Tensor) -> Tensor:
        """(N, L, E) -> (N, h, L, d)."""
        n, length, _ = x.shape
        return x.reshape(n, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_masks(
        self,
        key_padding_mask: Tensor | None,
        attn_mask: Tensor | None,
        is_causal: bool,
        n: int,
        length: int,
        source_length: int,
        dtype: npt.DTypeLike,
    ) -> Tensor | None:
        """One additive mask, broadcastable to (N, h, L, S), or None."""
        if attn_mask is None and is_causal:
            attn_mask = F.causal_mask(length, source_length)
        total = None
        if attn_mask is not None:
            total = F.additive_mask(attn_mask, dtype)
            if total.ndim == 3:  # (N * h, L, S) -> (N, h, L, S)
                total = total.reshape(n, self.num_heads, length, source_length)
        if key_padding_mask is not None:
            padding = F.additive_mask(key_padding_mask, dtype)
            padding = padding.reshape(n, 1, 1, source_length)
            total = padding if total is None else total + padding
        return total

    def extra_repr(self) -> str:
        """Show the sizes."""
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"
