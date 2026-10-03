"""Tests of glassnn.analysis: count_parameters, activation_stats, empirical_ntk."""

import numpy as np
import pytest

import glassnn.functional as F
from glassnn import Tensor, manual_seed, nn
from glassnn.analysis import activation_stats, count_parameters, empirical_ntk


def small_mlp(width=5, outputs=1):
    return nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, outputs))


# --------------------------------------------------------------------------
# count_parameters
# --------------------------------------------------------------------------


def test_count_parameters():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2, bias=False))
    assert count_parameters(model) == 3 * 4 + 4 + 4 * 2
    shared = nn.Linear(2, 2)
    assert count_parameters(nn.Sequential(shared, shared)) == 6


# --------------------------------------------------------------------------
# activation_stats
# --------------------------------------------------------------------------


def test_activation_stats_records_every_linear_output(rng):
    model = nn.Sequential(
        nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 4), nn.Sequential(nn.Linear(4, 2))
    )
    x = Tensor(rng.normal(size=(50, 3)))
    stats = activation_stats(model, x)
    assert list(stats) == ["0", "2", "3.0"]
    z0 = model[0](x).data
    z2 = model[2](F.relu(model[0](x))).data
    z3 = model[3](model[2](F.relu(model[0](x)))).data
    for name, z in [("0", z0), ("2", z2), ("3.0", z3)]:
        assert stats[name].mean == pytest.approx(z.mean())
        assert stats[name].std == pytest.approx(z.std())


def test_activation_stats_accepts_arrays_and_leaves_gradients_alone(rng):
    model = small_mlp()
    assert list(activation_stats(model, rng.normal(size=(4, 3)))) == ["0", "2"]
    assert all(p.grad is None for p in model.parameters())


def test_activation_stats_needs_a_sequential(rng):
    with pytest.raises(TypeError, match=r"Sequential.*Linear"):
        activation_stats(nn.Linear(3, 2), rng.normal(size=(4, 3)))


# --------------------------------------------------------------------------
# empirical_ntk
# --------------------------------------------------------------------------


def numerical_jacobian(model, X, eps=1e-6):
    """Jacobian of the scalar outputs with respect to all parameters."""
    params = list(model.parameters())
    columns = []
    for p in params:
        for i in range(p.data.size):
            original = p.data.flat[i]
            p.data.flat[i] = original + eps
            plus = model(Tensor(X)).data.reshape(-1)
            p.data.flat[i] = original - eps
            minus = model(Tensor(X)).data.reshape(-1)
            p.data.flat[i] = original
            columns.append((plus - minus) / (2 * eps))
    return np.stack(columns, axis=1)


def test_ntk_of_a_linear_model_is_the_linear_kernel_plus_one(rng):
    # f(x) = w.x + b: the gradient with respect to (w, b) is (x, 1).
    X1, X2 = rng.normal(size=(4, 3)), rng.normal(size=(2, 3))
    K = empirical_ntk(nn.Linear(3, 1), X1, X2)
    assert isinstance(K, Tensor) and K.shape == (4, 2)
    np.testing.assert_allclose(K.data, X1 @ X2.T + 1)


def test_ntk_equals_the_product_of_jacobians(rng):
    manual_seed(0)
    model = small_mlp()
    X1, X2 = rng.normal(size=(5, 3)), rng.normal(size=(3, 3))
    J1, J2 = numerical_jacobian(model, X1), numerical_jacobian(model, X2)
    np.testing.assert_allclose(empirical_ntk(model, X1, X2).data, J1 @ J2.T, rtol=1e-7)


def test_ntk_on_one_set_is_symmetric_and_positive_semidefinite(rng):
    manual_seed(1)
    model = small_mlp(width=8)
    X = rng.normal(size=(6, 3))
    K = empirical_ntk(model, X).data
    np.testing.assert_allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    np.testing.assert_allclose(K, empirical_ntk(model, X, X).data)


def test_ntk_selects_one_output(rng):
    manual_seed(2)
    model = small_mlp(outputs=3)
    X = rng.normal(size=(4, 3))
    selected = nn.Sequential(model, nn.Linear(3, 1, bias=False))
    selected[1].weight.data = np.array([[0.0, 1.0, 0.0]])
    selected[1].weight.requires_grad = False
    np.testing.assert_allclose(
        empirical_ntk(model, X, output_index=1).data, empirical_ntk(selected, X).data
    )
    with pytest.raises(ValueError, match=r"3 outputs.*output_index"):
        empirical_ntk(model, X)


def test_ntk_keeps_existing_gradients_and_skips_frozen_parameters(rng):
    model = nn.Linear(3, 1)
    model(Tensor(rng.normal(size=(2, 3)))).sum().backward()
    saved = model.weight.grad.data.copy()
    model.bias.requires_grad = False
    X = rng.normal(size=(4, 3))
    K = empirical_ntk(model, X)
    np.testing.assert_allclose(K.data, X @ X.T)  # no bias column
    np.testing.assert_array_equal(model.weight.grad.data, saved)
    assert model.bias.grad is not None


def test_ntk_parametrization_concentrates_with_width(rng):
    # At initialization, the NTK of an NTK-parametrized network has a
    # deterministic limit, so its spread over seeds shrinks with the width.
    X = rng.normal(size=(4, 3))

    def spread(width):
        kernels = []
        for seed in range(8):
            manual_seed(seed)
            model = nn.Sequential(
                nn.Linear(3, width, parametrization="ntk"),
                nn.ReLU(),
                nn.Linear(width, 1, parametrization="ntk"),
            )
            kernels.append(empirical_ntk(model, X).data)
        kernels = np.stack(kernels)
        return (kernels.std(axis=0) / np.abs(kernels.mean(axis=0))).mean()

    assert spread(1000) < 0.25 * spread(20)


def test_ntk_of_an_empty_set_and_of_unused_parameters(rng):
    class WithUnused(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 1)
            self.unused = nn.Parameter(np.ones(2))

        def forward(self, x):
            return self.layer(x)

    X = rng.normal(size=(3, 3))
    model = WithUnused()
    np.testing.assert_allclose(empirical_ntk(model, X).data, X @ X.T + 1)
    assert empirical_ntk(model, np.zeros((0, 3)), X).shape == (0, 3)
