"""``TransformerEncoderLayer`` and ``TransformerEncoder``.

Book chapter: :book:`Attention <chapters/10-attention.html>`.
"""

import copy

import numpy.typing as npt

from glassnn import functional as F
from glassnn.nn.attention import MultiheadAttention
from glassnn.nn.dropout import Dropout
from glassnn.nn.linear import Linear
from glassnn.nn.module import Module, ModuleList
from glassnn.nn.normalization import LayerNorm
from glassnn.tensor import Tensor


class TransformerEncoderLayer(Module):
    r"""One block of the transformer encoder: self-attention and a feed-forward net.

    With :math:`\mathrm{SA}` multi-head self-attention, :math:`\mathrm{FF}(x)
    = W_2\, \phi(W_1 x + b_1) + b_2` applied at every position, and
    :math:`\mathrm{LN}` layer normalization, the pre-LN block (the default)
    computes

    .. math:: x \leftarrow x + \mathrm{SA}(\mathrm{LN}_1(x)), \qquad
        x \leftarrow x + \mathrm{FF}(\mathrm{LN}_2(x)),

    and the post-LN block of :cite:t:`vaswani2017attention`
    (``norm_first=False``)

    .. math:: x \leftarrow \mathrm{LN}_1(x + \mathrm{SA}(x)), \qquad
        x \leftarrow \mathrm{LN}_2(x + \mathrm{FF}(x)).

    In the pre-LN form, the residual path from input to output contains no
    normalization, which makes deep stacks easier to train. Dropout is
    applied to the attention weights, inside the feed-forward net, and to
    each branch before the residual addition.

    Args:
        d_model: The embedding size :math:`E`.
        nhead: The number of attention heads.
        dim_feedforward: The hidden size of the feed-forward net.
        dropout: The dropout probability.
        activation: :math:`\phi`, ``"gelu"`` (default) or ``"relu"``.
        layer_norm_eps: :math:`\varepsilon` of the layer normalizations.
        batch_first: Must be ``True``.
        norm_first: Pre-LN (``True``, default) or post-LN.
        bias: Whether the linear layers and normalizations have biases.
        dtype: The dtype of the parameters.

    Shapes:
        src: ``(N, L, E)``.
        output: ``(N, L, E)``.

    Note:
        Differences from PyTorch: inputs are batch-first only; the defaults
        are ``norm_first=True`` and ``activation="gelu"`` (PyTorch:
        ``False`` and ``"relu"``); ``activation`` must be a string.

    Example:
        >>> from glassnn import Tensor, nn
        >>> layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
        >>> layer(Tensor([[[0.1] * 8] * 5])).shape
        (1, 5, 8)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Create the attention, the feed-forward net and the normalizations."""
        super().__init__()
        if not batch_first:
            raise NotImplementedError(
                "GlassNN transformers take inputs of shape (N, L, E) only; "
                "use batch_first=True."
            )
        if activation not in ("relu", "gelu"):
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {activation!r}."
            )
        self.activation = activation
        self.norm_first = norm_first
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, dtype=dtype
        )
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, dtype=dtype)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, dtype=dtype)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Apply the block; the masks are those of :class:`MultiheadAttention`."""
        x = src
        if self.norm_first:
            x = x + self._attention_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal
            )
            x = x + self._feedforward_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._attention_block(x, src_mask, src_key_padding_mask, is_causal)
            )
            x = self.norm2(x + self._feedforward_block(x))
        return x

    def _attention_block(
        self,
        x: Tensor,
        mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        out, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=mask,
            is_causal=is_causal,
        )
        return self.dropout1(out)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        phi = F.gelu if self.activation == "gelu" else F.relu
        return self.dropout2(self.linear2(self.dropout(phi(self.linear1(x)))))

    def extra_repr(self) -> str:
        """Show the variant."""
        return f"norm_first={self.norm_first}, activation={self.activation!r}"


class TransformerEncoder(Module):
    """A stack of ``num_layers`` copies of an encoder layer.

    Each layer is a deep copy of ``encoder_layer``, so all layers start with
    the *same* parameter values (as in PyTorch) but are trained separately.

    Args:
        encoder_layer: A :class:`TransformerEncoderLayer` to copy.
        num_layers: The number of layers.
        norm: An optional final normalization (a pre-LN stack usually ends
            with ``LayerNorm(d_model)``).

    Attributes:
        layers: A :class:`~glassnn.nn.ModuleList` of the layers.
        norm: The final normalization, or ``None``.

    Shapes:
        src: ``(N, L, E)``.
        output: ``(N, L, E)``.

    Note:
        Differences from PyTorch: no ``enable_nested_tensor`` or
        ``mask_check``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16)
        >>> encoder = nn.TransformerEncoder(layer, num_layers=2)
        >>> encoder(Tensor([[[0.1] * 8] * 5])).shape
        (1, 5, 8)
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Module | None = None,
    ) -> None:
        """Copy the layer ``num_layers`` times."""
        super().__init__()
        self.layers = ModuleList(
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        )
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Apply the layers in order, then the final normalization."""
        x = src
        for layer in self.layers:
            x = layer(
                x,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x
