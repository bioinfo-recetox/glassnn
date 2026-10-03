"""Tests of Linear, the activation modules and the loss modules."""

import math

import numpy as np
import pytest

import glassnn.functional as F
from glassnn import Tensor, backend, nn


def test_linear_shapes_and_forward(rng):
    layer = nn.Linear(3, 5)
    assert layer.weight.shape == (5, 3)
    assert layer.bias.shape == (5,)
    x = Tensor(rng.normal(size=(4, 3)))
    expected = x.data @ layer.weight.data.T + layer.bias.data
    np.testing.assert_allclose(layer(x).data, expected)


def test_linear_without_bias(rng):
    layer = nn.Linear(3, 2, bias=False)
    assert layer.bias is None
    assert [n for n, _ in layer.named_parameters()] == ["weight"]
    x = Tensor(rng.normal(size=(4, 3)))
    np.testing.assert_allclose(layer(x).data, x.data @ layer.weight.data.T)


def test_linear_gradients_match_the_closed_form(rng):
    # With Y = X W^T + b and L = sum(Y * G): dW = G^T X, db = sum_n G_n, dX = G W.
    layer = nn.Linear(3, 2)
    x = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    g = rng.normal(size=(4, 2))
    (layer(x) * g).sum().backward()
    np.testing.assert_allclose(layer.weight.grad.data, g.T @ x.data)
    np.testing.assert_allclose(layer.bias.grad.data, g.sum(axis=0))
    np.testing.assert_allclose(x.grad.data, g @ layer.weight.data)


def test_linear_default_initialization_matches_pytorch_bounds():
    backend.manual_seed(0)
    fan_in = 400
    layer = nn.Linear(fan_in, 300)
    bound = 1 / math.sqrt(fan_in)
    assert np.abs(layer.weight.data).max() <= bound
    assert np.abs(layer.bias.data).max() <= bound
    # Uniform on [-bound, bound] has standard deviation bound / sqrt(3).
    assert layer.weight.data.std() == pytest.approx(bound / math.sqrt(3), rel=0.02)


def test_linear_dtype():
    assert nn.Linear(2, 2, dtype="float32").weight.dtype == np.float32
    assert nn.Linear(2, 2).weight.dtype == np.float64  # the test default


def test_linear_initialization_is_reproducible():
    backend.manual_seed(5)
    first = nn.Linear(3, 3).weight.data
    backend.manual_seed(5)
    np.testing.assert_array_equal(nn.Linear(3, 3).weight.data, first)


ACTIVATION_MODULES = [
    (nn.ReLU(), F.relu),
    (nn.LeakyReLU(0.2), lambda t: F.leaky_relu(t, 0.2)),
    (nn.GELU(), F.gelu),
    (nn.GELU(approximate="tanh"), lambda t: F.gelu(t, approximate="tanh")),
    (nn.Tanh(), F.tanh),
    (nn.Sigmoid(), F.sigmoid),
    (nn.Softplus(beta=2.0, threshold=10.0), lambda t: F.softplus(t, 2.0, 10.0)),
    (nn.Softmax(dim=1), lambda t: F.softmax(t, dim=1)),
    (nn.LogSoftmax(dim=0), lambda t: F.log_softmax(t, dim=0)),
]


@pytest.mark.parametrize(("module", "function"), ACTIVATION_MODULES)
def test_activation_modules_call_the_functions(rng, module, function):
    x = Tensor(rng.normal(size=(3, 4)))
    np.testing.assert_array_equal(module(x).data, function(x).data)
    assert not list(module.parameters())


def test_activation_module_reprs():
    assert repr(nn.LeakyReLU(0.2)) == "LeakyReLU(negative_slope=0.2)"
    assert repr(nn.Softmax(dim=1)) == "Softmax(dim=1)"
    assert repr(nn.GELU()) == "GELU(approximate='none')"
    assert repr(nn.Softplus()) == "Softplus(beta=1.0, threshold=20.0)"
    assert repr(nn.LogSoftmax(dim=-1)) == "LogSoftmax(dim=-1)"


def test_loss_modules_call_the_functions(rng):
    logits = Tensor(rng.normal(size=(4, 3)))
    labels = Tensor(rng.integers(0, 3, size=4))
    targets = Tensor(rng.uniform(size=(4, 3)))
    pairs = [
        (nn.MSELoss(), F.mse_loss(logits, targets)),
        (nn.MSELoss(reduction="sum"), F.mse_loss(logits, targets, reduction="sum")),
        (
            nn.BCEWithLogitsLoss(),
            F.binary_cross_entropy_with_logits(logits, targets),
        ),
    ]
    for module, expected in pairs:
        np.testing.assert_array_equal(module(logits, targets).data, expected.data)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    np.testing.assert_array_equal(
        ce(logits, labels).data,
        F.cross_entropy(logits, labels, label_smoothing=0.1).data,
    )


# --------------------------------------------------------------------------
# Linear with the NTK parametrization
# --------------------------------------------------------------------------


def test_ntk_linear_initializes_weight_and_bias_standard_normal():
    backend.manual_seed(0)
    layer = nn.Linear(300, 400, parametrization="ntk")
    assert layer.weight.data.mean() == pytest.approx(0.0, abs=0.01)
    assert layer.weight.data.std() == pytest.approx(1.0, rel=0.01)
    assert layer.bias.data.std() == pytest.approx(1.0, rel=0.1)


def test_ntk_linear_scales_the_preactivation(rng):
    layer = nn.Linear(4, 3, parametrization="ntk")
    x = Tensor(rng.normal(size=(5, 4)))
    expected = x.data @ layer.weight.data.T / 2.0 + layer.bias.data
    np.testing.assert_allclose(layer(x).data, expected)


def test_ntk_linear_gradients_carry_the_scale(rng):
    # With Y = X W^T / sqrt(n) + b and L = sum(Y * G): dW = G^T X / sqrt(n).
    layer = nn.Linear(4, 2, parametrization="ntk")
    x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    g = rng.normal(size=(3, 2))
    (layer(x) * g).sum().backward()
    np.testing.assert_allclose(layer.weight.grad.data, g.T @ x.data / 2.0)
    np.testing.assert_allclose(layer.bias.grad.data, g.sum(axis=0))
    np.testing.assert_allclose(x.grad.data, g @ layer.weight.data / 2.0)


def test_ntk_preactivation_scale_does_not_depend_on_width(rng):
    # Var(z_i) = |x|^2 / n + 1 = 2 for inputs with unit second moment.
    backend.manual_seed(1)
    x = Tensor(rng.normal(size=(2000, 50)))
    for width in (50, 5000):
        hidden = nn.Linear(50, width, parametrization="ntk")
        h = F.relu(hidden(x))
        out = nn.Linear(width, 1, parametrization="ntk")(h)
        assert hidden(x).data.std() == pytest.approx(np.sqrt(2.0), rel=0.15)
        assert 0.5 < out.data.std() < 3.0


def test_linear_rejects_an_unknown_parametrization():
    with pytest.raises(ValueError, match=r"'standard' or 'ntk'.*'mup'"):
        nn.Linear(2, 2, parametrization="mup")


def test_ntk_linear_repr():
    layer = nn.Linear(2, 3, parametrization="ntk")
    assert repr(layer) == (
        "Linear(in_features=2, out_features=3, bias=True, parametrization='ntk')"
    )
    assert repr(nn.Linear(2, 3)) == "Linear(in_features=2, out_features=3, bias=True)"


def test_ntk_linear_without_bias(rng):
    layer = nn.Linear(4, 3, bias=False, parametrization="ntk")
    assert layer.bias is None
    x = Tensor(rng.normal(size=(2, 4)))
    np.testing.assert_allclose(layer(x).data, x.data @ layer.weight.data.T / 2.0)
