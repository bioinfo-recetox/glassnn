"""Tests of embeddings, attention and the transformer encoder."""

import math

import numpy as np
import pytest

import glassnn.functional as F
from glassnn import Tensor, backend, manual_seed, nn
from glassnn.gradcheck import gradcheck


def leaf(array):
    return Tensor(array, requires_grad=True)


def with_parameters(module, call):
    """``call`` as a function of an input and of the module's parameters.

    gradcheck passes perturbed *copies* of its inputs; this loads them into
    the module for the duration of the call, so that finite differences
    see the change.
    """
    parameters = list(module.parameters())

    def fn(x, *values):
        saved = [p.data for p in parameters]
        for p, value in zip(parameters, values, strict=True):
            p.data = value.data
        try:
            return call(x)
        finally:
            for p, data in zip(parameters, saved, strict=True):
                p.data = data

    return fn, parameters


def softmax(a):
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention_reference(q, k, v, additive=0.0, scale=None):
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    weights = softmax(q @ np.swapaxes(k, -1, -2) * scale + additive)
    return weights @ v, weights


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------


def test_embedding_looks_up_rows_and_adds_gradients_of_repeats(rng):
    weight = leaf(rng.normal(size=(5, 3)))
    index = np.array([[0, 2, 2], [4, 0, 2]])
    out = F.embedding(Tensor(index), weight)
    np.testing.assert_array_equal(out.data, weight.data[index])
    out.sum().backward()
    np.testing.assert_array_equal(weight.grad.data[:, 0], [2, 0, 3, 0, 1])
    assert gradcheck(lambda w: F.embedding(Tensor(index), w), [weight])


def test_embedding_padding_index_gets_no_gradient(rng):
    weight = leaf(rng.normal(size=(4, 2)))
    F.embedding(Tensor([1, 0, 0, 3]), weight, padding_idx=0).sum().backward()
    np.testing.assert_array_equal(weight.grad.data, [[0, 0], [1, 1], [0, 0], [1, 1]])


def test_embedding_errors():
    weight = Tensor(np.zeros((3, 2)))
    with pytest.raises(TypeError, match="integer"):
        F.embedding(Tensor([0.0, 1.0]), weight)
    with pytest.raises(IndexError, match=r"\[0, 3\).*5"):
        F.embedding(Tensor([0, 5]), weight)


def test_embedding_module():
    manual_seed(0)
    layer = nn.Embedding(1000, 20, padding_idx=1)
    assert layer.weight.shape == (1000, 20)
    np.testing.assert_array_equal(layer.weight.data[1], 0.0)
    assert layer.weight.data.std() == pytest.approx(1.0, rel=0.02)  # N(0, 1)
    assert layer(Tensor([[3, 1]])).shape == (1, 2, 20)
    assert repr(layer) == "Embedding(1000, 20, padding_idx=1)"
    assert repr(nn.Embedding(3, 2)) == "Embedding(3, 2)"


# --------------------------------------------------------------------------
# scaled_dot_product_attention
# --------------------------------------------------------------------------


def test_attention_matches_the_formula_and_gradcheck(rng):
    q = leaf(rng.normal(size=(2, 3, 4, 5)))  # (N, heads, L, E)
    k = leaf(rng.normal(size=(2, 3, 6, 5)))  # S = 6
    v = leaf(rng.normal(size=(2, 3, 6, 7)))
    out = F.scaled_dot_product_attention(q, k, v)
    expected, _ = attention_reference(q.data, k.data, v.data)
    np.testing.assert_allclose(out.data, expected, rtol=1e-12)
    assert gradcheck(F.scaled_dot_product_attention, [q, k, v])


def test_attention_boolean_mask_true_means_masked(rng):
    q, k, v = (leaf(rng.normal(size=(2, 4, 3))) for _ in range(3))
    mask = np.array(
        [[False, True, True, False]] * 2 + [[False, False, True, True]] * 2
    )[:, :4]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=Tensor(mask))
    expected, _ = attention_reference(
        q.data, k.data, v.data, np.where(mask, -np.inf, 0)
    )
    np.testing.assert_allclose(out.data, expected, rtol=1e-12)
    assert gradcheck(
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, Tensor(mask)), [q, k, v]
    )


def test_attention_float_mask_scale_and_causal(rng):
    q, k, v = (leaf(rng.normal(size=(5, 3))) for _ in range(3))
    bias = rng.normal(size=(5, 5))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=Tensor(bias), scale=0.3)
    expected, _ = attention_reference(q.data, k.data, v.data, bias, scale=0.3)
    np.testing.assert_allclose(out.data, expected, rtol=1e-12)
    causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    expected, _ = attention_reference(
        q.data, k.data, v.data, np.where(upper, -np.inf, 0)
    )
    np.testing.assert_allclose(causal.data, expected, rtol=1e-12)
    with pytest.raises(ValueError, match="either"):
        F.scaled_dot_product_attention(q, k, v, attn_mask=Tensor(bias), is_causal=True)


def test_attention_dropout_on_the_weights(rng):
    q, k, v = (Tensor(rng.normal(size=(8, 8))) for _ in range(3))
    manual_seed(0)
    dropped = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5).data
    plain = F.scaled_dot_product_attention(q, k, v).data
    assert not np.allclose(dropped, plain)
    manual_seed(0)
    np.testing.assert_array_equal(
        F.scaled_dot_product_attention(q, k, v, dropout_p=0.5).data, dropped
    )


def test_attention_shape_errors():
    with pytest.raises(ValueError, match=r"\(2, 3\).*\(2, 4\)"):
        F.scaled_dot_product_attention(
            Tensor(np.zeros((2, 3))), Tensor(np.zeros((2, 4))), Tensor(np.zeros((2, 4)))
        )
    with pytest.raises(ValueError, match=r"\(3, 2\).*\(4, 2\)"):
        F.scaled_dot_product_attention(
            Tensor(np.zeros((2, 2))), Tensor(np.zeros((3, 2))), Tensor(np.zeros((4, 2)))
        )


# --------------------------------------------------------------------------
# ModuleList
# --------------------------------------------------------------------------


def test_module_list_registers_its_modules():
    layers = nn.ModuleList([nn.Linear(2, 3), nn.ReLU()])
    layers.append(nn.Linear(3, 1))
    assert len(layers) == 3
    assert isinstance(layers[0], nn.Linear) and isinstance(layers[-1], nn.Linear)
    assert [type(m).__name__ for m in layers] == ["Linear", "ReLU", "Linear"]
    assert [n for n, _ in layers.named_parameters()] == [
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
    ]
    layers.extend([nn.Tanh()])
    assert len(layers) == 4 and len(nn.ModuleList()) == 0
    with pytest.raises(NotImplementedError, match="ModuleList"):
        layers(Tensor([1.0]))


# --------------------------------------------------------------------------
# PositionalEncoding
# --------------------------------------------------------------------------


def test_positional_encoding_adds_sines_and_cosines():
    layer = nn.PositionalEncoding(d_model=6, max_len=10)
    pe = layer.pe.data
    assert pe.shape == (10, 6)
    position, i = 3, 1
    angle = position / 10000 ** (2 * i / 6)
    assert pe[position, 2 * i] == pytest.approx(math.sin(angle))
    assert pe[position, 2 * i + 1] == pytest.approx(math.cos(angle))
    x = Tensor(np.ones((2, 4, 6)))
    np.testing.assert_allclose(layer(x).data, np.broadcast_to(1 + pe[:4], (2, 4, 6)))
    assert "pe" in layer.state_dict()
    assert not list(layer.parameters())
    assert repr(layer) == "PositionalEncoding(d_model=6, max_len=10, dropout=0.0)"


def test_positional_encoding_errors():
    with pytest.raises(ValueError, match="even"):
        nn.PositionalEncoding(d_model=5)
    with pytest.raises(ValueError, match=r"12.*max_len 10"):
        nn.PositionalEncoding(4, max_len=10)(Tensor(np.zeros((1, 12, 4))))


# --------------------------------------------------------------------------
# MultiheadAttention
# --------------------------------------------------------------------------


def mha_reference(mha, query, key, value, additive=0.0):
    """Per-head attention written out with NumPy; additive mask (N, L, S)."""

    def project(layer, x):
        return x @ layer.weight.data.T + layer.bias.data

    n, length, e = query.shape
    h = mha.num_heads
    d = e // h

    def split(x):
        return x.reshape(n, x.shape[1], h, d).transpose(0, 2, 1, 3)

    q = split(project(mha.q_proj, query))
    k = split(project(mha.k_proj, key))
    v = split(project(mha.v_proj, value))
    out, weights = attention_reference(q, k, v, np.asarray(additive)[:, None])
    out = out.transpose(0, 2, 1, 3).reshape(n, length, e)
    return project(mha.out_proj, out), weights


def test_multihead_attention_matches_the_reference(rng):
    mha = nn.MultiheadAttention(embed_dim=6, num_heads=3)
    for p in [mha.q_proj.bias, mha.k_proj.bias, mha.v_proj.bias, mha.out_proj.bias]:
        p.data = rng.normal(size=p.shape)
    query = rng.normal(size=(2, 4, 6))
    memory = rng.normal(size=(2, 5, 6))
    out, weights = mha(Tensor(query), Tensor(memory), Tensor(memory))
    expected, per_head = mha_reference(mha, query, memory, memory, np.zeros((2, 4, 5)))
    np.testing.assert_allclose(out.data, expected, rtol=1e-10)
    assert weights.shape == (2, 4, 5)
    np.testing.assert_allclose(weights.data, per_head.mean(axis=1), rtol=1e-10)
    _, all_heads = mha(
        Tensor(query), Tensor(memory), Tensor(memory), average_attn_weights=False
    )
    np.testing.assert_allclose(all_heads.data, per_head, rtol=1e-10)
    out, none = mha(Tensor(query), Tensor(memory), Tensor(memory), need_weights=False)
    assert none is None


def test_multihead_attention_masks(rng):
    mha = nn.MultiheadAttention(4, 2)
    x = rng.normal(size=(2, 5, 4))
    padding = np.array([[False] * 5, [False, False, False, True, True]])
    causal = np.triu(np.ones((5, 5), dtype=bool), k=1)
    out, weights = mha(
        Tensor(x),
        Tensor(x),
        Tensor(x),
        key_padding_mask=Tensor(padding),
        attn_mask=Tensor(causal),
    )
    additive = np.where(padding[:, None, :] | causal[None], -np.inf, 0.0)
    expected, _ = mha_reference(mha, x, x, x, additive)
    np.testing.assert_allclose(out.data, expected, rtol=1e-10)
    np.testing.assert_array_equal(weights.data[1, :, 3:], 0.0)  # padded keys
    np.testing.assert_array_equal(weights.data[0][causal], 0.0)  # future keys
    via_flag, _ = mha(
        Tensor(x), Tensor(x), Tensor(x), is_causal=True, attn_mask=Tensor(causal)
    )
    np.testing.assert_allclose(via_flag.data[0], expected[0], rtol=1e-10)


def test_multihead_attention_gradcheck(rng):
    mha = nn.MultiheadAttention(4, 2)
    x = leaf(rng.normal(size=(2, 3, 4)))
    mask = Tensor(np.array([[False, False, True], [False, False, False]]))
    fn, params = with_parameters(mha, lambda x: mha(x, x, x, key_padding_mask=mask)[0])
    assert gradcheck(fn, [x, *params])


def test_multihead_attention_settings_and_errors():
    mha = nn.MultiheadAttention(8, 2, bias=False)
    assert [n for n, _ in mha.named_parameters()] == [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
    ]
    assert repr(mha).startswith("MultiheadAttention(")
    with pytest.raises(ValueError, match="divisible"):
        nn.MultiheadAttention(6, 4)
    with pytest.raises(NotImplementedError, match="batch_first"):
        nn.MultiheadAttention(4, 2, batch_first=False)


def test_multihead_attention_initialization():
    # PyTorch's packed in_proj_weight (3E, E) is Xavier uniform: bound sqrt(6 / 4E).
    manual_seed(0)
    mha = nn.MultiheadAttention(64, 4)
    bound = math.sqrt(6 / (4 * 64))
    for layer in [mha.q_proj, mha.k_proj, mha.v_proj]:
        assert np.abs(layer.weight.data).max() <= bound
        assert layer.weight.data.std() == pytest.approx(bound / math.sqrt(3), rel=0.05)
        np.testing.assert_array_equal(layer.bias.data, 0.0)
    np.testing.assert_array_equal(mha.out_proj.bias.data, 0.0)


def test_multihead_attention_dropout_only_in_training(rng):
    mha = nn.MultiheadAttention(4, 2, dropout=0.5)
    x = Tensor(rng.normal(size=(1, 6, 4)))
    mha.eval()
    first = mha(x, x, x)[0].data
    np.testing.assert_array_equal(mha(x, x, x)[0].data, first)
    mha.train()
    assert not np.allclose(mha(x, x, x)[0].data, first)


# --------------------------------------------------------------------------
# TransformerEncoderLayer and TransformerEncoder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_encoder_layer_gradcheck(rng, norm_first, activation):
    layer = nn.TransformerEncoderLayer(
        4,
        2,
        dim_feedforward=6,
        dropout=0.0,
        activation=activation,
        norm_first=norm_first,
    )
    x = leaf(rng.normal(size=(2, 3, 4)))
    fn, params = with_parameters(layer, layer)
    assert gradcheck(fn, [x, *params])


def test_encoder_layer_structure_and_defaults():
    layer = nn.TransformerEncoderLayer(8, 2)
    assert layer.norm_first and layer.activation == "gelu"
    assert layer.linear1.weight.shape == (2048, 8)
    names = {n.split(".")[0] for n, _ in layer.named_parameters()}
    assert names == {"self_attn", "linear1", "linear2", "norm1", "norm2"}
    with pytest.raises(ValueError, match="'relu' or 'gelu'"):
        nn.TransformerEncoderLayer(8, 2, activation="tanh")
    with pytest.raises(NotImplementedError, match="batch_first"):
        nn.TransformerEncoderLayer(8, 2, batch_first=False)


def test_causal_mask_hides_the_future(rng):
    manual_seed(0)
    layer = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0)
    x = rng.normal(size=(1, 6, 4))
    changed = x.copy()
    changed[0, 4:] += 10.0  # change the last two positions only
    out = layer(Tensor(x), is_causal=True).data
    out_changed = layer(Tensor(changed), is_causal=True).data
    np.testing.assert_allclose(out[0, :4], out_changed[0, :4], rtol=1e-12)
    assert not np.allclose(out[0, 4:], out_changed[0, 4:])


def test_transformer_encoder_copies_the_layer(rng):
    manual_seed(0)
    layer = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0)
    encoder = nn.TransformerEncoder(layer, num_layers=3, norm=nn.LayerNorm(4))
    assert len(encoder.layers) == 3
    first, second = encoder.layers[0], encoder.layers[1]
    assert first is not second and first.linear1.weight is not second.linear1.weight
    np.testing.assert_array_equal(first.linear1.weight.data, second.linear1.weight.data)
    x = Tensor(rng.normal(size=(2, 5, 4)))
    expected = x
    for block in encoder.layers:
        expected = block(expected)
    expected = encoder.norm(expected)
    np.testing.assert_allclose(encoder(x).data, expected.data)
    count = sum(p.data.size for p in layer.parameters())
    assert sum(p.data.size for p in encoder.parameters()) == 3 * count + 8


def test_transformer_encoder_passes_masks(rng):
    manual_seed(0)
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0), 2
    )
    x = rng.normal(size=(1, 5, 4))
    padded = np.concatenate([x, rng.normal(size=(1, 2, 4))], axis=1)
    mask = Tensor(np.array([[False] * 5 + [True] * 2]))
    out = encoder(Tensor(padded), src_key_padding_mask=mask).data
    np.testing.assert_allclose(out[:, :5], encoder(Tensor(x)).data, rtol=1e-10)


def test_float32_attention_stays_float32(rng):
    backend.set_default_dtype("float32")
    layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16)
    out = layer(Tensor(rng.normal(size=(2, 5, 8))))
    assert out.dtype == np.float32


def test_multihead_attention_mask_per_batch_and_head(rng):
    mha = nn.MultiheadAttention(4, 2)
    x = Tensor(rng.normal(size=(3, 5, 4)))
    mask = rng.uniform(size=(5, 5)) < 0.3
    mask[:, 0] = False
    shared, _ = mha(x, x, x, attn_mask=Tensor(mask))
    per_head = np.broadcast_to(mask, (3 * 2, 5, 5))  # (N * h, L, S)
    repeated, _ = mha(x, x, x, attn_mask=Tensor(per_head))
    np.testing.assert_allclose(repeated.data, shared.data)


def test_extra_reprs():
    assert nn.MultiheadAttention(8, 2).extra_repr() == "embed_dim=8, num_heads=2"
    layer = nn.TransformerEncoderLayer(8, 2, activation="relu", norm_first=False)
    assert layer.extra_repr() == "norm_first=False, activation='relu'"
