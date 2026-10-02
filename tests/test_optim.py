"""Tests of optimizers, learning-rate schedulers and gradient clipping."""

import math

import numpy as np
import pytest

from glassnn import Tensor, nn, optim


def param_with_grads(rng, shape=(3, 2)):
    """A parameter and a function that sets a fresh random gradient."""
    p = nn.Parameter(rng.normal(size=shape))

    def set_grad():
        g = rng.normal(size=shape)
        p.grad = Tensor(g)
        return g

    return p, set_grad


# --------------------------------------------------------------------------
# SGD
# --------------------------------------------------------------------------


def test_sgd_plain_step(rng):
    p, set_grad = param_with_grads(rng)
    before = p.data.copy()
    g = set_grad()
    optim.SGD([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, before - 0.1 * g)


@pytest.mark.parametrize("nesterov", [False, True])
def test_sgd_momentum_and_weight_decay(rng, nesterov):
    p, set_grad = param_with_grads(rng)
    opt = optim.SGD([p], lr=0.1, momentum=0.9, nesterov=nesterov, weight_decay=0.01)
    x = p.data.copy()
    buf = None
    for _ in range(3):
        g = set_grad() + 0.01 * x
        buf = g if buf is None else 0.9 * buf + g
        step = g + 0.9 * buf if nesterov else buf
        x = x - 0.1 * step
        opt.step()
        np.testing.assert_allclose(p.data, x)


def test_sgd_nesterov_needs_momentum(rng):
    p, _ = param_with_grads(rng)
    with pytest.raises(ValueError, match="momentum"):
        optim.SGD([p], lr=0.1, nesterov=True)


# --------------------------------------------------------------------------
# Adam, AdamW, RMSprop
# --------------------------------------------------------------------------


def adam_reference(x, grads, lr, betas, eps, weight_decay, decoupled):
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trajectory = []
    for t, g in enumerate(grads, start=1):
        if decoupled:
            x = x * (1 - lr * weight_decay)
        else:
            g = g + weight_decay * x
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g**2
        m_hat = m / (1 - betas[0] ** t)
        v_hat = v / (1 - betas[1] ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(x)
    return trajectory


@pytest.mark.parametrize("cls", [optim.Adam, optim.AdamW])
def test_adam_and_adamw_follow_the_reference(rng, cls):
    p, set_grad = param_with_grads(rng)
    opt = cls([p], lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.1)
    x0 = p.data.copy()
    grads = []
    for _ in range(4):
        grads.append(set_grad())
        opt.step()
    expected = adam_reference(
        x0, grads, 0.01, (0.8, 0.95), 1e-6, 0.1, decoupled=cls is optim.AdamW
    )[-1]
    np.testing.assert_allclose(p.data, expected, rtol=1e-12)


def test_adam_first_step_has_size_lr(rng):
    # With m_hat = g and v_hat = g^2, the first step is lr * g / (|g| + eps).
    p, set_grad = param_with_grads(rng)
    before = p.data.copy()
    set_grad()
    optim.Adam([p], lr=0.01, eps=0.0).step()
    np.testing.assert_allclose(np.abs(p.data - before), 0.01)


def test_defaults_match_pytorch(rng):
    p, _ = param_with_grads(rng)
    adam = optim.Adam([p])
    assert (adam.lr, adam.betas, adam.eps, adam.weight_decay) == (
        1e-3,
        (0.9, 0.999),
        1e-8,
        0.0,
    )
    assert optim.AdamW([p]).weight_decay == 1e-2
    rmsprop = optim.RMSprop([p])
    assert (rmsprop.lr, rmsprop.alpha, rmsprop.eps) == (1e-2, 0.99, 1e-8)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_rmsprop_follows_the_reference(rng, momentum):
    p, set_grad = param_with_grads(rng)
    opt = optim.RMSprop(
        [p], lr=0.01, alpha=0.9, eps=1e-6, weight_decay=0.1, momentum=momentum
    )
    x = p.data.copy()
    v = np.zeros_like(x)
    buf = np.zeros_like(x)
    for _ in range(3):
        g = set_grad() + 0.1 * x
        v = 0.9 * v + 0.1 * g**2
        update = g / (np.sqrt(v) + 1e-6)
        if momentum:
            buf = momentum * buf + update
            update = buf
        x = x - 0.01 * update
        opt.step()
        np.testing.assert_allclose(p.data, x, rtol=1e-12)


# --------------------------------------------------------------------------
# Common behaviour
# --------------------------------------------------------------------------

ALL_OPTIMIZERS = [
    lambda ps: optim.SGD(ps, lr=0.1, momentum=0.9),
    lambda ps: optim.Adam(ps, lr=0.1),
    lambda ps: optim.AdamW(ps, lr=0.1),
    lambda ps: optim.RMSprop(ps, lr=0.01),
]


@pytest.mark.parametrize("make", ALL_OPTIMIZERS)
def test_parameters_without_gradient_are_skipped(rng, make):
    p, set_grad = param_with_grads(rng)
    q, _ = param_with_grads(rng)
    q_before = q.data.copy()
    set_grad()
    make([p, q]).step()
    np.testing.assert_array_equal(q.data, q_before)


@pytest.mark.parametrize("make", ALL_OPTIMIZERS)
def test_optimizers_minimize_a_quadratic(make):
    target = Tensor([1.0, -2.0, 3.0])
    p = nn.Parameter([0.0, 0.0, 0.0])
    opt = make([p])
    for _ in range(500):
        opt.zero_grad()
        ((p - target) ** 2).sum().backward()
        opt.step()
    np.testing.assert_allclose(p.data, target.data, atol=0.05)


@pytest.mark.parametrize("make", ALL_OPTIMIZERS)
def test_step_keeps_parameters_as_float_leaves(rng, make):
    p = nn.Parameter(rng.normal(size=3), dtype="float32")
    p.grad = Tensor(rng.normal(size=3), dtype="float32")
    make([p]).step()
    assert p.dtype == np.float32
    assert p.is_leaf and p.requires_grad


def test_zero_grad_resets_gradients(rng):
    p, set_grad = param_with_grads(rng)
    set_grad()
    opt = optim.SGD([p], lr=0.1)
    opt.zero_grad()
    assert p.grad is None


def test_optimizer_accepts_a_generator_of_parameters():
    model = nn.Linear(2, 3)
    opt = optim.SGD(model.parameters(), lr=0.1)
    assert len(opt.params) == 2


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda ps: optim.SGD(ps, lr=-1.0), "lr"),
        (lambda ps: optim.SGD(ps, lr=0.1, momentum=-0.5), "momentum"),
        (lambda ps: optim.Adam(ps, betas=(1.0, 0.999)), "betas"),
        (lambda ps: optim.Adam(ps, eps=-1.0), "eps"),
        (lambda ps: optim.SGD(ps, lr=0.1, weight_decay=-1.0), "weight_decay"),
        (lambda ps: optim.RMSprop(ps, alpha=1.5), "alpha"),
        (lambda ps: optim.SGD([], lr=0.1), "no parameters"),
    ],
)
def test_invalid_arguments_are_rejected(rng, make, message):
    p, _ = param_with_grads(rng)
    with pytest.raises(ValueError, match=message):
        make([p] if message != "no parameters" else [])


# --------------------------------------------------------------------------
# Schedulers
# --------------------------------------------------------------------------


def learning_rates(scheduler, optimizer, steps):
    rates = [optimizer.lr]
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.lr)
    return rates


@pytest.fixture
def sgd(rng):
    p, set_grad = param_with_grads(rng)
    set_grad()
    return optim.SGD([p], lr=1.0)


def test_step_lr(sgd):
    scheduler = optim.StepLR(sgd, step_size=2, gamma=0.5)
    rates = learning_rates(scheduler, sgd, 5)
    np.testing.assert_allclose(rates, [1.0, 1.0, 0.5, 0.5, 0.25, 0.25])
    assert scheduler.get_last_lr() == pytest.approx(0.25)


def test_cosine_annealing_lr(sgd):
    scheduler = optim.CosineAnnealingLR(sgd, T_max=4, eta_min=0.1)
    rates = learning_rates(scheduler, sgd, 4)
    expected = [0.1 + 0.9 * (1 + math.cos(math.pi * t / 4)) / 2 for t in range(5)]
    np.testing.assert_allclose(rates, expected)
    assert rates[-1] == pytest.approx(0.1)


def test_linear_warmup(sgd):
    scheduler = optim.LinearWarmup(sgd, warmup_steps=4, start_factor=0.2)
    rates = learning_rates(scheduler, sgd, 6)
    np.testing.assert_allclose(rates, [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])


def test_linear_warmup_rejects_invalid_arguments(sgd):
    with pytest.raises(ValueError, match="warmup_steps"):
        optim.LinearWarmup(sgd, warmup_steps=0)
    with pytest.raises(ValueError, match="start_factor"):
        optim.LinearWarmup(sgd, warmup_steps=3, start_factor=0.0)


# --------------------------------------------------------------------------
# Gradient clipping
# --------------------------------------------------------------------------


def test_clip_grad_norm_scales_down_large_gradients():
    p = nn.Parameter([0.0, 0.0])
    q = nn.Parameter([0.0])
    p.grad = Tensor([3.0, 0.0])
    q.grad = Tensor([4.0])
    total = optim.clip_grad_norm_([p, q], max_norm=1.0)
    assert total.item() == pytest.approx(5.0)
    new_norm = np.sqrt((p.grad.data**2).sum() + (q.grad.data**2).sum())
    assert new_norm == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(p.grad.data / q.grad.data[0], [0.75, 0.0])


def test_clip_grad_norm_leaves_small_gradients_alone():
    p = nn.Parameter([0.0, 0.0])
    p.grad = Tensor([0.3, 0.4])
    total = optim.clip_grad_norm_(p, max_norm=1.0)
    assert total.item() == pytest.approx(0.5)
    np.testing.assert_array_equal(p.grad.data, [0.3, 0.4])


def test_clip_grad_norm_ignores_parameters_without_gradient():
    p = nn.Parameter([0.0])
    assert optim.clip_grad_norm_([p], max_norm=1.0).item() == 0.0


def test_base_classes_leave_the_update_rule_to_subclasses(rng):
    p, _ = param_with_grads(rng)
    with pytest.raises(NotImplementedError):
        optim.Optimizer([p], lr=0.1).step()
    with pytest.raises(NotImplementedError):
        optim.LRScheduler(optim.SGD([p], lr=0.1))
