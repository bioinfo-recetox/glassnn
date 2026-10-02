"""Tests of glassnn.functional: activations, softmax family, losses, linear."""

import numpy as np
import pytest
from scipy import special

import glassnn.functional as F
from glassnn import Tensor
from glassnn.gradcheck import gradcheck

# Overflow, division by zero and invalid values are errors; underflow to 0 is fine.
NO_OVERFLOW = {"over": "raise", "divide": "raise", "invalid": "raise"}


def leaf(rng, *shape, low=-3.0, high=3.0):
    return Tensor(rng.uniform(low, high, size=shape), requires_grad=True)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------


def gelu_tanh_reference(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


ACTIVATIONS = {
    "relu": (F.relu, lambda x: np.maximum(x, 0)),
    "leaky_relu": (F.leaky_relu, lambda x: np.where(x > 0, x, 0.01 * x)),
    "leaky_relu_0.2": (
        lambda t: F.leaky_relu(t, negative_slope=0.2),
        lambda x: np.where(x > 0, x, 0.2 * x),
    ),
    "gelu": (F.gelu, lambda x: 0.5 * x * (1 + special.erf(x / np.sqrt(2)))),
    "gelu_tanh": (lambda t: F.gelu(t, approximate="tanh"), gelu_tanh_reference),
    "tanh": (F.tanh, np.tanh),
    "sigmoid": (F.sigmoid, logistic),
    "softplus": (F.softplus, lambda x: np.log1p(np.exp(x))),
    "softplus_beta2": (
        lambda t: F.softplus(t, beta=2.0),
        lambda x: np.log1p(np.exp(2 * x)) / 2,
    ),
}


@pytest.mark.parametrize("name", list(ACTIVATIONS))
def test_activation_forward_and_gradient(rng, name):
    fn, reference = ACTIVATIONS[name]
    # Values away from 0, where relu and leaky_relu have a kink.
    x = leaf(rng, 4, 5)
    x = Tensor(np.where(np.abs(x.data) < 0.1, 0.5, x.data), requires_grad=True)
    np.testing.assert_allclose(fn(x).data, reference(x.data), rtol=1e-12)
    assert gradcheck(fn, [x])


def test_relu_and_leaky_relu_derivative_at_zero_is_the_left_one():
    x = Tensor([0.0], requires_grad=True)
    F.relu(x).sum().backward()
    assert x.grad.item() == 0.0
    x.grad = None
    F.leaky_relu(x, 0.1).sum().backward()
    assert x.grad.item() == pytest.approx(0.1)


def test_gelu_rejects_unknown_approximation():
    with pytest.raises(ValueError, match="'none' or 'tanh'"):
        F.gelu(Tensor([1.0]), approximate="fast")


@pytest.mark.parametrize("name", ["sigmoid", "softplus", "tanh", "gelu", "gelu_tanh"])
def test_activations_are_finite_for_extreme_inputs(name):
    fn, _ = ACTIVATIONS[name]
    x = Tensor([-1000.0, -50.0, 0.0, 50.0, 1000.0], requires_grad=True)
    with np.errstate(**NO_OVERFLOW):
        out = fn(x)
        out.sum().backward()
    assert np.all(np.isfinite(out.data))
    assert np.all(np.isfinite(x.grad.data))


def test_sigmoid_and_softplus_limits():
    x = Tensor([-1000.0, 1000.0])
    np.testing.assert_allclose(F.sigmoid(x).data, [0.0, 1.0])
    np.testing.assert_allclose(F.softplus(x).data, [0.0, 1000.0])


def test_softplus_above_the_threshold_is_the_identity():
    x = Tensor([25.0, 30.0], requires_grad=True)
    out = F.softplus(x, threshold=20.0)
    np.testing.assert_array_equal(out.data, x.data)
    out.sum().backward()
    np.testing.assert_array_equal(x.grad.data, [1.0, 1.0])


# --------------------------------------------------------------------------
# softmax, log_softmax, logsumexp
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax(rng, dim):
    x = leaf(rng, 3, 4)
    expected = special.softmax(x.data, axis=dim)
    np.testing.assert_allclose(F.softmax(x, dim=dim).data, expected, rtol=1e-12)
    assert gradcheck(lambda t: F.softmax(t, dim=dim), [x])


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_log_softmax(rng, dim):
    x = leaf(rng, 3, 4)
    expected = special.log_softmax(x.data, axis=dim)
    np.testing.assert_allclose(F.log_softmax(x, dim=dim).data, expected, rtol=1e-12)
    assert gradcheck(lambda t: F.log_softmax(t, dim=dim), [x])


@pytest.mark.parametrize(("dim", "keepdim"), [(0, False), (1, False), (-1, True)])
def test_logsumexp(rng, dim, keepdim):
    x = leaf(rng, 3, 4)
    expected = special.logsumexp(x.data, axis=dim, keepdims=keepdim)
    out = F.logsumexp(x, dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(out.data, expected, rtol=1e-12)
    assert gradcheck(lambda t: F.logsumexp(t, dim=dim, keepdim=keepdim), [x])


def test_softmax_family_with_huge_logits():
    x = Tensor(
        [[1000.0, 0.0, -1000.0], [-1000.0, -1000.0, -1000.0]], requires_grad=True
    )
    with np.errstate(**NO_OVERFLOW):
        probabilities = F.softmax(x, dim=1)
        log_probabilities = F.log_softmax(x, dim=1)
        total = F.logsumexp(x, dim=1)
        (probabilities.sum() + log_probabilities.sum() + total.sum()).backward()
    np.testing.assert_allclose(probabilities.data[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(probabilities.data[1], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(log_probabilities.data[0], [0.0, -1000.0, -2000.0])
    np.testing.assert_allclose(total.data, [1000.0, -1000.0 + np.log(3)])
    assert np.all(np.isfinite(x.grad.data))


# --------------------------------------------------------------------------
# linear
# --------------------------------------------------------------------------


@pytest.mark.parametrize("with_bias", [True, False])
def test_linear(rng, with_bias):
    x = leaf(rng, 5, 3)
    w = leaf(rng, 4, 3)
    b = leaf(rng, 4) if with_bias else None
    expected = x.data @ w.data.T + (b.data if with_bias else 0)
    np.testing.assert_allclose(F.linear(x, w, b).data, expected)
    inputs = [x, w, b] if with_bias else [x, w]
    assert gradcheck(lambda *t: F.linear(*t), inputs)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def cross_entropy_reference(logits, target, smoothing=0.0):
    log_p = special.log_softmax(logits, axis=1)
    nll = -log_p[np.arange(len(target)), target]
    return (1 - smoothing) * nll + smoothing * (-log_p.mean(axis=1))


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy(rng, smoothing, reduction):
    logits = leaf(rng, 6, 4)
    target = Tensor(rng.integers(0, 4, size=6))
    per_sample = cross_entropy_reference(logits.data, target.data, smoothing)
    expected = {"mean": per_sample.mean(), "sum": per_sample.sum(), "none": per_sample}
    out = F.cross_entropy(
        logits, target, reduction=reduction, label_smoothing=smoothing
    )
    np.testing.assert_allclose(out.data, expected[reduction], rtol=1e-12)
    assert gradcheck(
        lambda t: F.cross_entropy(
            t, target, reduction=reduction, label_smoothing=smoothing
        ),
        [logits],
    )


def test_cross_entropy_accepts_a_numpy_target(rng):
    logits = leaf(rng, 3, 2)
    out = F.cross_entropy(logits, np.array([0, 1, 1]))
    np.testing.assert_allclose(
        out.item(), cross_entropy_reference(logits.data, [0, 1, 1]).mean()
    )


def test_cross_entropy_with_huge_logits_is_finite():
    logits = Tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
    with np.errstate(**NO_OVERFLOW):
        loss = F.cross_entropy(logits, Tensor([1, 1]))
        loss.backward()
    assert loss.item() == pytest.approx(1000.0)
    assert np.all(np.isfinite(logits.grad.data))


def test_cross_entropy_shape_errors_name_both_shapes():
    with pytest.raises(ValueError, match=r"\(3, 2\).*\(4,\)"):
        F.cross_entropy(Tensor(np.zeros((3, 2))), Tensor([0, 1, 1, 0]))
    with pytest.raises(ValueError, match=r"\(3,\)"):
        F.cross_entropy(Tensor(np.zeros(3)), Tensor([0]))


def test_cross_entropy_needs_integer_targets():
    with pytest.raises(TypeError, match="integer"):
        F.cross_entropy(Tensor(np.zeros((2, 2))), Tensor([0.0, 1.0]))


def bce_reference(x, t):
    p = logistic(x)
    return -(t * np.log(p) + (1 - t) * np.log(1 - p))


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_binary_cross_entropy_with_logits(rng, reduction):
    x = leaf(rng, 5, 2)
    t = Tensor(rng.uniform(0, 1, size=(5, 2)), requires_grad=True)
    per_element = bce_reference(x.data, t.data)
    expected = {
        "mean": per_element.mean(),
        "sum": per_element.sum(),
        "none": per_element,
    }
    out = F.binary_cross_entropy_with_logits(x, t, reduction=reduction)
    np.testing.assert_allclose(out.data, expected[reduction], rtol=1e-12)
    assert gradcheck(
        lambda a, b: F.binary_cross_entropy_with_logits(a, b, reduction=reduction),
        [x, t],
    )


def test_binary_cross_entropy_with_huge_logits_is_exact_and_finite():
    x = Tensor([1000.0, -1000.0, 1000.0], requires_grad=True)
    t = Tensor([1.0, 0.0, 0.0])
    with np.errstate(**NO_OVERFLOW):
        loss = F.binary_cross_entropy_with_logits(x, t, reduction="none")
        loss.sum().backward()
    np.testing.assert_allclose(loss.data, [0.0, 0.0, 1000.0])
    np.testing.assert_allclose(x.grad.data, [0.0, 0.0, 1.0])


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_loss(rng, reduction):
    a = leaf(rng, 4, 3)
    b = leaf(rng, 4, 3)
    per_element = (a.data - b.data) ** 2
    expected = {
        "mean": per_element.mean(),
        "sum": per_element.sum(),
        "none": per_element,
    }
    np.testing.assert_allclose(
        F.mse_loss(a, b, reduction=reduction).data, expected[reduction]
    )
    assert gradcheck(lambda s, t: F.mse_loss(s, t, reduction=reduction), [a, b])


@pytest.mark.parametrize(
    "loss",
    [F.mse_loss, F.binary_cross_entropy_with_logits],
)
def test_elementwise_losses_need_equal_shapes(loss):
    with pytest.raises(ValueError, match=r"\(3, 1\).*\(3,\)"):
        loss(Tensor(np.zeros((3, 1))), Tensor(np.zeros(3)))


def test_unknown_reduction_is_rejected():
    with pytest.raises(ValueError, match="'mean', 'sum' or 'none'"):
        F.mse_loss(Tensor([1.0]), Tensor([1.0]), reduction="average")
