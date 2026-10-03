"""Tests of dropout and normalization: functions and modules."""

import numpy as np
import pytest

import glassnn.functional as F
from glassnn import Tensor, backend, manual_seed, nn
from glassnn.gradcheck import gradcheck


def leaf(array):
    return Tensor(array, requires_grad=True)


# --------------------------------------------------------------------------
# Dropout
# --------------------------------------------------------------------------


def test_dropout_is_the_identity_in_evaluation_or_with_p_zero(rng):
    x = Tensor(rng.normal(size=(3, 4)))
    assert F.dropout(x, p=0.5, training=False) is x
    assert F.dropout(x, p=0.0) is x


def test_dropout_zeros_a_fraction_p_and_rescales_the_rest():
    x = Tensor(np.ones((200, 500)))
    out = F.dropout(x, p=0.3, generator=np.random.default_rng(0)).data
    assert (out == 0).mean() == pytest.approx(0.3, abs=0.01)
    np.testing.assert_allclose(out[out != 0], 1 / 0.7)
    assert out.mean() == pytest.approx(1.0, abs=0.01)  # inverted dropout


def test_dropout_gradient_uses_the_same_mask(rng):
    x = leaf(rng.normal(size=(4, 5)))
    assert gradcheck(
        lambda t: F.dropout(t, p=0.4, generator=np.random.default_rng(3)), [x]
    )
    out = F.dropout(x, p=0.4, generator=np.random.default_rng(3))
    x.grad = None
    out.sum().backward()
    np.testing.assert_allclose(x.grad.data, (out.data != 0) / 0.6)


def test_dropout_with_p_one_gives_zeros(rng):
    x = leaf(rng.normal(size=(2, 3)))
    out = F.dropout(x, p=1.0)
    np.testing.assert_array_equal(out.data, 0.0)
    out.sum().backward()
    np.testing.assert_array_equal(x.grad.data, 0.0)


def test_dropout_rejects_p_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\].*1.5"):
        F.dropout(Tensor([1.0]), p=1.5)


def test_dropout_draws_from_the_global_generator(rng):
    x = Tensor(rng.normal(size=(10, 10)))
    manual_seed(7)
    first = F.dropout(x).data
    manual_seed(7)
    np.testing.assert_array_equal(F.dropout(x).data, first)


def test_dropout_keeps_float32():
    x = Tensor(np.ones(10), dtype="float32")
    assert F.dropout(x, p=0.5).dtype == np.float32


def test_dropout_handles_an_empty_batch():
    assert F.dropout(Tensor(np.zeros((0, 3))), p=0.5).shape == (0, 3)


def test_dropout_module_follows_training_mode(rng):
    layer = nn.Dropout(p=0.5)
    x = Tensor(rng.normal(size=(20, 20)))
    assert (layer(x).data == 0).any()
    layer.eval()
    np.testing.assert_array_equal(layer(x).data, x.data)
    assert repr(layer) == "Dropout(p=0.5)"
    assert not list(layer.parameters())


# --------------------------------------------------------------------------
# Layer normalization
# --------------------------------------------------------------------------


def layer_norm_reference(x, k, weight=None, bias=None, eps=1e-5):
    axes = tuple(range(x.ndim - k, x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


@pytest.mark.parametrize("shape", [(4, 6), (2, 3, 5)])
def test_layer_norm_forward_and_gradient(rng, shape):
    x = leaf(rng.normal(1.0, 2.0, size=shape))
    w = leaf(rng.normal(size=shape[-1:]))
    b = leaf(rng.normal(size=shape[-1:]))
    out = F.layer_norm(x, shape[-1:], w, b)
    np.testing.assert_allclose(
        out.data, layer_norm_reference(x.data, 1, w.data, b.data), rtol=1e-12
    )
    assert gradcheck(lambda x, w, b: F.layer_norm(x, shape[-1:], w, b), [x, w, b])


def test_layer_norm_over_two_dimensions(rng):
    x = leaf(rng.normal(size=(3, 4, 5)))
    out = F.layer_norm(x, (4, 5))
    np.testing.assert_allclose(out.data, layer_norm_reference(x.data, 2), rtol=1e-12)
    assert gradcheck(lambda t: F.layer_norm(t, (4, 5)), [x])


def test_layer_norm_ignores_a_large_offset(rng):
    x = rng.normal(size=(3, 8))
    shifted = F.layer_norm(Tensor(x + 1e6), (8,)).data
    np.testing.assert_allclose(shifted, F.layer_norm(Tensor(x), (8,)).data, atol=1e-6)


def test_layer_norm_of_a_constant_row_is_zero_and_has_finite_gradient():
    x = leaf(np.full((2, 4), 3.0))
    out = F.layer_norm(x, (4,))
    np.testing.assert_array_equal(out.data, 0.0)
    (out * Tensor(np.arange(8.0).reshape(2, 4))).sum().backward()
    assert np.isfinite(x.grad.data).all()


def test_layer_norm_handles_an_empty_batch(rng):
    x = leaf(np.zeros((0, 4)))
    out = F.layer_norm(x, (4,), leaf(np.ones(4)), leaf(np.zeros(4)))
    assert out.shape == (0, 4)
    out.sum().backward()
    assert x.grad.shape == (0, 4)


def test_layer_norm_checks_the_normalized_shape():
    with pytest.raises(ValueError, match=r"\(3,\).*\(2, 4\)"):
        F.layer_norm(Tensor(np.zeros((2, 4))), (3,))


def test_layer_norm_module(rng):
    layer = nn.LayerNorm(5)
    assert layer.normalized_shape == (5,)
    np.testing.assert_array_equal(layer.weight.data, np.ones(5))
    np.testing.assert_array_equal(layer.bias.data, np.zeros(5))
    x = Tensor(rng.normal(size=(3, 5)))
    expected = F.layer_norm(x, (5,), layer.weight, layer.bias).data
    np.testing.assert_array_equal(layer(x).data, expected)
    assert repr(layer) == "LayerNorm((5,), eps=1e-05, elementwise_affine=True)"


def test_layer_norm_module_without_affine_or_bias():
    assert not list(nn.LayerNorm((2, 3), elementwise_affine=False).parameters())
    names = [n for n, _ in nn.LayerNorm(3, bias=False).named_parameters()]
    assert names == ["weight"]
    assert nn.LayerNorm(3, dtype="float32").weight.dtype == np.float32


# --------------------------------------------------------------------------
# Batch normalization
# --------------------------------------------------------------------------


def batch_norm_reference(x, weight, bias, eps=1e-5):
    axes = (0,) if x.ndim == 2 else (0, 2)
    shape = (-1,) if x.ndim == 2 else (-1, 1)
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    return out * weight.reshape(shape) + bias.reshape(shape)


@pytest.mark.parametrize("shape", [(6, 3), (4, 3, 5)])
def test_batch_norm_training_forward_and_gradient(rng, shape):
    x = leaf(rng.normal(2.0, 3.0, size=shape))
    w = leaf(rng.normal(size=3))
    b = leaf(rng.normal(size=3))
    out = F.batch_norm(x, None, None, w, b, training=True)
    np.testing.assert_allclose(
        out.data, batch_norm_reference(x.data, w.data, b.data), rtol=1e-12
    )
    assert gradcheck(
        lambda x, w, b: F.batch_norm(x, None, None, w, b, training=True), [x, w, b]
    )


@pytest.mark.parametrize("shape", [(6, 3), (4, 3, 5)])
def test_batch_norm_updates_the_running_statistics(rng, shape):
    x = rng.normal(2.0, 3.0, size=shape)
    running_mean = Tensor(np.zeros(3))
    running_var = Tensor(np.ones(3))
    F.batch_norm(Tensor(x), running_mean, running_var, training=True, momentum=0.2)
    axes = (0,) if x.ndim == 2 else (0, 2)
    unbiased = x.var(axis=axes, ddof=1)
    np.testing.assert_allclose(running_mean.data, 0.2 * x.mean(axis=axes))
    np.testing.assert_allclose(running_var.data, 0.8 + 0.2 * unbiased)


def test_batch_norm_in_evaluation_uses_the_running_statistics(rng):
    x = leaf(rng.normal(size=(5, 3)))
    mean, var = Tensor([0.5, -1.0, 2.0]), Tensor([1.0, 4.0, 0.25])
    w, b = leaf(rng.normal(size=3)), leaf(rng.normal(size=3))
    out = F.batch_norm(x, mean, var, w, b, training=False)
    expected = (x.data - mean.data) / np.sqrt(var.data + 1e-5) * w.data + b.data
    np.testing.assert_allclose(out.data, expected, rtol=1e-12)
    np.testing.assert_array_equal(mean.data, [0.5, -1.0, 2.0])  # unchanged
    assert gradcheck(
        lambda x, w, b: F.batch_norm(x, mean, var, w, b, training=False), [x, w, b]
    )


def test_batch_norm_without_running_statistics_uses_the_batch(rng):
    x = Tensor(rng.normal(size=(5, 3)))
    np.testing.assert_allclose(
        F.batch_norm(x, None, None, training=False).data,
        F.batch_norm(x, None, None, training=True).data,
    )


def test_batch_norm_needs_more_than_one_value_per_channel():
    with pytest.raises(ValueError, match="more than one value per channel"):
        F.batch_norm(Tensor(np.zeros((1, 3))), None, None, training=True)


def test_batch_norm_checks_the_input_shape():
    with pytest.raises(ValueError, match=r"\(N, C\) or \(N, C, L\).*\(2, 3, 4, 5\)"):
        F.batch_norm(Tensor(np.zeros((2, 3, 4, 5))), None, None, training=True)
    with pytest.raises(ValueError, match=r"4 channels.*\(3,\)"):
        F.batch_norm(Tensor(np.zeros((2, 4))), Tensor(np.zeros(3)), None)


def test_batch_norm_ignores_a_large_offset(rng):
    x = rng.normal(size=(16, 4))
    shifted = F.batch_norm(Tensor(x + 1e6), None, None, training=True).data
    expected = F.batch_norm(Tensor(x), None, None, training=True).data
    np.testing.assert_allclose(shifted, expected, atol=1e-6)


def test_batch_norm_module_state_and_modes(rng):
    layer = nn.BatchNorm1d(3, momentum=0.5)
    assert [n for n, _ in layer.named_parameters()] == ["weight", "bias"]
    assert list(layer.state_dict()) == ["weight", "bias", "running_mean", "running_var"]
    x = Tensor(rng.normal(4.0, 2.0, size=(50, 3)))
    layer(x)
    np.testing.assert_allclose(layer.running_mean.data, 0.5 * x.data.mean(axis=0))
    trained_mean = layer.running_mean.data.copy()
    layer.eval()
    out = layer(x)
    np.testing.assert_array_equal(layer.running_mean.data, trained_mean)
    expected = (x.data - layer.running_mean.data) / np.sqrt(
        layer.running_var.data + 1e-5
    )
    np.testing.assert_allclose(out.data, expected)


def test_batch_norm_module_without_running_statistics_or_affine(rng):
    layer = nn.BatchNorm1d(3, affine=False, track_running_stats=False)
    assert not list(layer.parameters()) and not list(layer.buffers())
    layer.eval()
    x = Tensor(rng.normal(size=(6, 3)))
    expected = batch_norm_reference(x.data, np.ones(3), np.zeros(3))
    np.testing.assert_allclose(layer(x).data, expected, rtol=1e-12)


def test_batch_norm_module_repr_and_dtype():
    layer = nn.BatchNorm1d(4, dtype="float32")
    assert repr(layer) == (
        "BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)"
    )
    assert layer.running_var.dtype == np.float32
    assert layer.weight.dtype == np.float32


def test_batch_norm_statistics_survive_state_dict(rng):
    manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU())
    for _ in range(3):
        model(Tensor(rng.normal(size=(8, 2))))
    model.eval()
    x = Tensor(rng.normal(size=(5, 2)))
    expected = model(x).data
    manual_seed(1)
    copy = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU())
    copy.load_state_dict(model.state_dict())
    copy.eval()
    np.testing.assert_array_equal(copy(x).data, expected)


def test_batch_norm_keeps_float32(rng):
    backend.set_default_dtype("float32")
    layer = nn.BatchNorm1d(3)
    out = layer(Tensor(rng.normal(size=(4, 3))))
    assert out.dtype == np.float32
    assert layer.running_mean.dtype == np.float32
    assert layer.running_var.dtype == np.float32
