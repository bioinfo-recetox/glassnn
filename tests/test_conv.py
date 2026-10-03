"""Tests of convolution and pooling: functions against slow loops, and modules."""

import itertools
import math

import numpy as np
import pytest

import glassnn.functional as F
from glassnn import Tensor, backend, nn
from glassnn.gradcheck import gradcheck


def leaf(array):
    return Tensor(array, requires_grad=True)


# --------------------------------------------------------------------------
# Slow, obviously correct references (NumPy, explicit loops)
# --------------------------------------------------------------------------


def conv_loop(x, w, b, stride, pads, dilation):
    """N-d cross-correlation; ``pads`` is one (left, right) pair per dimension."""
    xp = np.pad(x, [(0, 0), (0, 0), *pads])
    spatial = xp.shape[2:]
    kernel = w.shape[2:]
    out_shape = [
        (size - d * (k - 1) - 1) // s + 1
        for size, k, s, d in zip(spatial, kernel, stride, dilation, strict=True)
    ]
    out = np.zeros((x.shape[0], w.shape[0], *out_shape))
    for n, o in itertools.product(range(x.shape[0]), range(w.shape[0])):
        for pos in itertools.product(*[range(m) for m in out_shape]):
            total = 0.0 if b is None else b[o]
            for c in range(x.shape[1]):
                for off in itertools.product(*[range(k) for k in kernel]):
                    idx = tuple(
                        p * s + q * d
                        for p, q, s, d in zip(pos, off, stride, dilation, strict=True)
                    )
                    total += xp[(n, c, *idx)] * w[(o, c, *off)]
            out[(n, o, *pos)] = total
    return out


def pool_loop(x, kernel, stride, padding, mode):
    fill = -np.inf if mode == "max" else 0.0
    xp = np.pad(x, [(0, 0), (0, 0), *[(p, p) for p in padding]], constant_values=fill)
    out_shape = [
        (size - k) // s + 1
        for size, k, s in zip(xp.shape[2:], kernel, stride, strict=True)
    ]
    out = np.zeros((*x.shape[:2], *out_shape))
    for n, c in itertools.product(range(x.shape[0]), range(x.shape[1])):
        for pos in itertools.product(*[range(m) for m in out_shape]):
            window = xp[
                (
                    n,
                    c,
                    *[
                        slice(p * s, p * s + k)
                        for p, s, k in zip(pos, stride, kernel, strict=True)
                    ],
                )
            ]
            out[(n, c, *pos)] = window.max() if mode == "max" else window.mean()
    return out


# --------------------------------------------------------------------------
# conv1d
# --------------------------------------------------------------------------

CONV1D_CASES = [
    # (length, kernel, stride, padding, dilation, expected (left, right) padding)
    (9, 3, 1, 0, 1, (0, 0)),
    (9, 3, 2, 1, 1, (1, 1)),
    (10, 4, 3, 2, 1, (2, 2)),
    (11, 3, 1, 0, 2, (0, 0)),
    (11, 3, 2, 2, 3, (2, 2)),
    (8, 3, 1, "same", 1, (1, 1)),
    (8, 4, 1, "same", 1, (1, 2)),  # odd total padding: the extra one on the right
    (8, 3, 1, "same", 2, (2, 2)),
    (8, 3, 1, "valid", 1, (0, 0)),
    (5, 5, 1, 0, 1, (0, 0)),  # kernel as long as the input: one output
]


@pytest.mark.parametrize(
    ("length", "kernel", "stride", "padding", "dilation", "pads"), CONV1D_CASES
)
def test_conv1d_matches_the_loop_and_gradcheck(
    rng, length, kernel, stride, padding, dilation, pads
):
    x = leaf(rng.normal(size=(2, 3, length)))
    w = leaf(rng.normal(size=(4, 3, kernel)))
    b = leaf(rng.normal(size=4))
    out = F.conv1d(x, w, b, stride=stride, padding=padding, dilation=dilation)
    expected = conv_loop(x.data, w.data, b.data, (stride,), [pads], (dilation,))
    np.testing.assert_allclose(out.data, expected, rtol=1e-12, atol=1e-12)
    assert gradcheck(
        lambda x, w, b: F.conv1d(x, w, b, stride, padding, dilation), [x, w, b]
    )


def test_conv1d_without_bias_and_with_an_empty_batch(rng):
    w = leaf(rng.normal(size=(2, 3, 3)))
    x = leaf(rng.normal(size=(2, 3, 7)))
    np.testing.assert_allclose(
        F.conv1d(x, w).data, conv_loop(x.data, w.data, None, (1,), [(0, 0)], (1,))
    )
    empty = leaf(np.zeros((0, 3, 7)))
    out = F.conv1d(empty, w)
    assert out.shape == (0, 2, 5)
    out.sum().backward()
    assert empty.grad.shape == (0, 3, 7)
    np.testing.assert_array_equal(w.grad.data, 0.0)


def test_conv1d_errors_name_the_shapes():
    x = Tensor(np.zeros((1, 3, 4)))
    with pytest.raises(ValueError, match=r"\(1, 3, 4\).*\(2, 5, 3\)"):
        F.conv1d(x, Tensor(np.zeros((2, 5, 3))))
    with pytest.raises(ValueError, match=r"\(N, C, L\).*\(3, 4\)"):
        F.conv1d(Tensor(np.zeros((3, 4))), Tensor(np.zeros((2, 3, 3))))
    with pytest.raises(ValueError, match=r"too short.*\(1, 3, 4\).*\(2, 3, 5\)"):
        F.conv1d(x, Tensor(np.zeros((2, 3, 5))))
    with pytest.raises(ValueError, match=r"'same'.*stride 1"):
        F.conv1d(x, Tensor(np.zeros((2, 3, 3))), stride=2, padding="same")
    with pytest.raises(ValueError, match=r"bias.*\(2,\).*\(3,\)"):
        F.conv1d(x, Tensor(np.zeros((2, 3, 3))), Tensor(np.zeros(3)))
    with pytest.raises(ValueError, match="padding"):
        F.conv1d(x, Tensor(np.zeros((2, 3, 3))), padding="full")


def test_conv1d_keeps_float32(rng):
    x = Tensor(rng.normal(size=(1, 2, 6)), dtype="float32")
    w = Tensor(rng.normal(size=(3, 2, 3)), dtype="float32")
    assert F.conv1d(x, w, padding="same").dtype == np.float32


# --------------------------------------------------------------------------
# conv2d
# --------------------------------------------------------------------------

CONV2D_CASES = [
    # (height, width, kernel, stride, padding, dilation, (left, right) per dim)
    ((6, 7), (3, 3), (1, 1), 0, (1, 1), [(0, 0), (0, 0)]),
    ((6, 7), (3, 2), (2, 1), (1, 2), (1, 1), [(1, 1), (2, 2)]),
    ((7, 7), (3, 3), (1, 2), 1, (2, 1), [(1, 1), (1, 1)]),
    ((5, 6), (2, 3), 1, "same", 1, [(0, 1), (1, 1)]),
    ((4, 4), (4, 4), 1, 0, 1, [(0, 0), (0, 0)]),
]


@pytest.mark.parametrize(
    ("size", "kernel", "stride", "padding", "dilation", "pads"), CONV2D_CASES
)
def test_conv2d_matches_the_loop_and_gradcheck(
    rng, size, kernel, stride, padding, dilation, pads
):
    x = leaf(rng.normal(size=(2, 2, *size)))
    w = leaf(rng.normal(size=(3, 2, *kernel)))
    b = leaf(rng.normal(size=3))
    out = F.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation)

    def pair(v):
        return (v, v) if isinstance(v, int) else v

    expected = conv_loop(x.data, w.data, b.data, pair(stride), pads, pair(dilation))
    np.testing.assert_allclose(out.data, expected, rtol=1e-12, atol=1e-12)
    assert gradcheck(
        lambda x, w, b: F.conv2d(x, w, b, stride, padding, dilation), [x, w, b]
    )


def test_conv2d_checks_the_input_shape():
    with pytest.raises(ValueError, match=r"\(N, C, H, W\).*\(2, 3, 4\)"):
        F.conv2d(Tensor(np.zeros((2, 3, 4))), Tensor(np.zeros((1, 3, 2, 2))))


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------

POOL_CASES = [
    # (spatial size, kernel, stride, padding)
    ((9,), 3, None, 0),
    ((10,), 3, 2, 1),
    ((8,), 4, 1, 2),
    ((7,), 7, None, 0),
    ((6, 7), 2, None, 0),
    ((6, 7), (3, 2), (2, 1), (1, 1)),
]


@pytest.mark.parametrize("mode", ["max", "avg"])
@pytest.mark.parametrize(("size", "kernel", "stride", "padding"), POOL_CASES)
def test_pooling_matches_the_loop_and_gradcheck(
    rng, mode, size, kernel, stride, padding
):
    x = leaf(rng.normal(size=(2, 3, *size)))  # distinct values: no ties
    dims = len(size)
    function = getattr(F, f"{mode}_pool{dims}d")
    out = function(x, kernel, stride, padding)

    def as_tuple(v):
        return (v,) * dims if isinstance(v, int) else tuple(v)

    expected = pool_loop(
        x.data,
        as_tuple(kernel),
        as_tuple(kernel if stride is None else stride),
        as_tuple(padding),
        mode,
    )
    np.testing.assert_allclose(out.data, expected, rtol=1e-12)
    assert gradcheck(lambda t: function(t, kernel, stride, padding), [x])


def test_max_pool_sends_the_gradient_to_the_first_maximum():
    x = leaf([[[1.0, 3.0, 3.0, 0.0]]])
    out = F.max_pool1d(x, kernel_size=4)
    out.backward(np.ones((1, 1, 1)))
    np.testing.assert_array_equal(x.grad.data, [[[0.0, 1.0, 0.0, 0.0]]])


def test_overlapping_max_pool_adds_gradients():
    x = leaf([[[0.0, 5.0, 1.0]]])
    F.max_pool1d(x, kernel_size=2, stride=1).sum().backward()
    np.testing.assert_array_equal(x.grad.data, [[[0.0, 2.0, 0.0]]])


def test_pooling_errors():
    x = Tensor(np.zeros((1, 1, 4)))
    with pytest.raises(ValueError, match="half the kernel size"):
        F.max_pool1d(x, kernel_size=2, padding=2)
    with pytest.raises(ValueError, match=r"too short.*\(1, 1, 4\)"):
        F.avg_pool1d(x, kernel_size=5)
    with pytest.raises(ValueError, match=r"\(N, C, L\)"):
        F.max_pool1d(Tensor(np.zeros((1, 4))), kernel_size=2)


# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------


def test_conv1d_module(rng):
    layer = nn.Conv1d(4, 6, kernel_size=5, padding="same", dilation=2)
    assert layer.weight.shape == (6, 4, 5)
    assert layer.bias.shape == (6,)
    x = Tensor(rng.normal(size=(3, 4, 20)))
    expected = F.conv1d(x, layer.weight, layer.bias, 1, "same", 2)
    np.testing.assert_array_equal(layer(x).data, expected.data)
    assert layer(x).shape == (3, 6, 20)
    assert repr(layer) == (
        "Conv1d(4, 6, kernel_size=(5,), stride=(1,), padding='same', "
        "dilation=(2,), bias=True)"
    )


def test_conv_modules_use_the_default_initialization_of_pytorch():
    backend.manual_seed(0)
    layer = nn.Conv2d(8, 300, kernel_size=(3, 5))
    bound = 1 / math.sqrt(8 * 3 * 5)
    assert np.abs(layer.weight.data).max() <= bound
    assert np.abs(layer.bias.data).max() <= bound
    assert layer.weight.data.std() == pytest.approx(bound / math.sqrt(3), rel=0.02)


def test_conv2d_module_without_bias_and_dtype(rng):
    layer = nn.Conv2d(2, 3, 3, stride=2, padding=1, bias=False, dtype="float32")
    assert layer.bias is None
    assert layer.weight.dtype == np.float32
    assert layer.kernel_size == (3, 3) and layer.stride == (2, 2)
    out = layer(Tensor(rng.normal(size=(1, 2, 7, 7)), dtype="float32"))
    assert out.shape == (1, 3, 4, 4)


def test_pooling_modules_and_flatten(rng):
    x = Tensor(rng.normal(size=(2, 3, 10)))
    np.testing.assert_array_equal(
        nn.MaxPool1d(3, stride=2)(x).data, F.max_pool1d(x, 3, 2).data
    )
    np.testing.assert_array_equal(nn.AvgPool1d(2)(x).data, F.avg_pool1d(x, 2).data)
    assert repr(nn.MaxPool1d(3)) == "MaxPool1d(kernel_size=3, stride=3, padding=0)"
    assert (
        repr(nn.AvgPool1d(2, 1, 1)) == "AvgPool1d(kernel_size=2, stride=1, padding=1)"
    )
    assert nn.Flatten()(Tensor(np.zeros((2, 3, 4, 5)))).shape == (2, 60)
    assert nn.Flatten(start_dim=2)(Tensor(np.zeros((2, 3, 4, 5)))).shape == (2, 3, 20)
    assert nn.Flatten(0, 1)(Tensor(np.zeros((2, 3, 4)))).shape == (6, 4)
    assert repr(nn.Flatten()) == "Flatten(start_dim=1, end_dim=-1)"


def test_flatten_gradient(rng):
    x = leaf(rng.normal(size=(2, 3, 4)))
    assert gradcheck(nn.Flatten(), [x])
