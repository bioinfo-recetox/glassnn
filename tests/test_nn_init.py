"""Tests of glassnn.nn.init."""

import math

import numpy as np
import pytest

from glassnn import backend, nn
from glassnn.nn import init

SHAPE = (500, 400)  # fan_out = 500, fan_in = 400


@pytest.mark.parametrize(
    ("nonlinearity", "param", "expected"),
    [
        ("linear", None, 1.0),
        ("sigmoid", None, 1.0),
        ("tanh", None, 5 / 3),
        ("relu", None, math.sqrt(2)),
        ("leaky_relu", None, math.sqrt(2 / (1 + 0.01**2))),
        ("leaky_relu", 0.2, math.sqrt(2 / (1 + 0.2**2))),
        ("selu", None, 3 / 4),
    ],
)
def test_calculate_gain(nonlinearity, param, expected):
    assert init.calculate_gain(nonlinearity, param) == pytest.approx(expected)


def test_calculate_gain_rejects_unknown_names():
    with pytest.raises(ValueError, match="'swish'"):
        init.calculate_gain("swish")


def test_fans_of_matrices_and_higher_tensors():
    assert init._calculate_fan_in_and_fan_out(nn.Parameter(np.zeros((5, 3)))) == (3, 5)
    weight = nn.Parameter(np.zeros((8, 4, 3)))  # e.g. a Conv1d kernel
    assert init._calculate_fan_in_and_fan_out(weight) == (12, 24)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        init._calculate_fan_in_and_fan_out(nn.Parameter(np.zeros(3)))


def check_uniform(tensor, bound):
    assert np.abs(tensor.data).max() <= bound
    assert tensor.data.std() == pytest.approx(bound / math.sqrt(3), rel=0.02)


def check_normal(tensor, std):
    assert tensor.data.mean() == pytest.approx(0.0, abs=0.01 * std)
    assert tensor.data.std() == pytest.approx(std, rel=0.02)


def test_xavier_uniform_and_normal():
    backend.manual_seed(0)
    w = nn.Parameter(np.zeros(SHAPE))
    assert init.xavier_uniform_(w, gain=2.0) is w
    check_uniform(w, 2.0 * math.sqrt(6 / (400 + 500)))
    init.xavier_normal_(w)
    check_normal(w, math.sqrt(2 / (400 + 500)))


@pytest.mark.parametrize(("mode", "fan"), [("fan_in", 400), ("fan_out", 500)])
def test_kaiming_uniform_and_normal(mode, fan):
    backend.manual_seed(0)
    w = nn.Parameter(np.zeros(SHAPE))
    init.kaiming_uniform_(w, mode=mode, nonlinearity="relu")
    check_uniform(w, math.sqrt(2) * math.sqrt(3 / fan))
    init.kaiming_normal_(w, mode=mode, nonlinearity="relu")
    check_normal(w, math.sqrt(2) / math.sqrt(fan))


def test_kaiming_uniform_with_a_sqrt5_is_the_linear_default():
    backend.manual_seed(0)
    w = nn.Parameter(np.zeros(SHAPE))
    init.kaiming_uniform_(w, a=math.sqrt(5))
    check_uniform(w, 1 / math.sqrt(400))


def test_kaiming_rejects_unknown_mode():
    with pytest.raises(ValueError, match="'fan_in' or 'fan_out'"):
        init.kaiming_normal_(nn.Parameter(np.zeros((2, 2))), mode="fan_avg")


def test_uniform_normal_zeros_ones():
    backend.manual_seed(0)
    w = nn.Parameter(np.zeros(SHAPE))
    init.uniform_(w, -0.5, 1.5)
    assert w.data.min() >= -0.5 and w.data.max() <= 1.5
    init.normal_(w, mean=3.0, std=0.5)
    assert w.data.mean() == pytest.approx(3.0, abs=0.01)
    assert w.data.std() == pytest.approx(0.5, rel=0.02)
    np.testing.assert_array_equal(init.zeros_(w).data, 0.0)
    np.testing.assert_array_equal(init.ones_(w).data, 1.0)


def test_init_keeps_dtype_and_requires_grad():
    w = nn.Parameter(np.zeros((3, 3)), dtype="float32")
    init.xavier_normal_(w)
    assert w.dtype == np.float32
    assert w.requires_grad and w.is_leaf


def test_explicit_generator_is_used_instead_of_the_global_one():
    w1 = nn.Parameter(np.zeros((4, 4)))
    w2 = nn.Parameter(np.zeros((4, 4)))
    init.normal_(w1, generator=np.random.default_rng(42))
    backend.manual_seed(0)
    init.normal_(w2, generator=np.random.default_rng(42))
    np.testing.assert_array_equal(w1.data, w2.data)
