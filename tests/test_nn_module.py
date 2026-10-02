"""Tests of Module, Parameter and Sequential."""

import numpy as np
import pytest

from glassnn import Tensor, nn


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(3, 4)
        self.scale = nn.Parameter(np.ones(4))
        self.second = nn.Linear(4, 2)
        self.note = "not a parameter"

    def forward(self, x):
        return self.second(self.first(x) * self.scale)


def test_parameter_is_a_tensor_that_requires_grad():
    p = nn.Parameter([1.0, 2.0])
    assert isinstance(p, Tensor)
    assert p.requires_grad
    assert repr(p) == "Parameter([1., 2.], requires_grad=True)"


def test_operations_on_parameters_return_plain_tensors():
    p = nn.Parameter([1.0, 2.0])
    assert type(p * 2) is Tensor


def test_parameters_and_modules_are_registered_by_assignment():
    model = TwoLayer()
    names = [name for name, _ in model.named_parameters()]
    assert names == [
        "scale",
        "first.weight",
        "first.bias",
        "second.weight",
        "second.bias",
    ]
    assert len(list(model.parameters())) == 5
    assert [type(m).__name__ for m in model.modules()] == [
        "TwoLayer",
        "Linear",
        "Linear",
    ]


def test_nested_names_use_dots():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)), nn.Linear(2, 1))
    assert [n for n, _ in model.named_parameters()] == [
        "0.0.weight",
        "0.0.bias",
        "1.weight",
        "1.bias",
    ]


def test_reassigning_an_attribute_unregisters_it():
    model = TwoLayer()
    model.scale = None
    assert "scale" not in dict(model.named_parameters())


def test_shared_parameters_are_listed_once():
    shared = nn.Linear(2, 2)
    model = nn.Sequential(shared, shared)
    assert len(list(model.parameters())) == 2


def test_forgetting_super_init_gives_a_clear_error():
    class Broken(nn.Module):
        def __init__(self):
            self.layer = nn.Linear(2, 2)

    with pytest.raises(AttributeError, match=r"super\(\).__init__\(\)"):
        Broken()


def test_call_runs_forward(rng):
    model = TwoLayer()
    x = Tensor(rng.normal(size=(5, 3)))
    np.testing.assert_array_equal(model(x).data, model.forward(x).data)
    assert model(x).shape == (5, 2)


def test_module_without_forward_raises():
    with pytest.raises(NotImplementedError, match="forward"):
        nn.Module()(Tensor([1.0]))


def test_train_and_eval_propagate_to_submodules():
    model = TwoLayer()
    assert model.eval() is model
    assert not any(m.training for m in model.modules())
    model.train()
    assert all(m.training for m in model.modules())


def test_zero_grad_sets_gradients_to_none(rng):
    model = TwoLayer()
    model(Tensor(rng.normal(size=(2, 3)))).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_state_dict_is_a_copy_and_load_state_dict_restores_it(rng):
    model = TwoLayer()
    saved = model.state_dict()
    assert list(saved) == [n for n, _ in model.named_parameters()]
    original = model.first.weight.data.copy()
    model.first.weight.data = model.first.weight.data + 1.0
    np.testing.assert_array_equal(saved["first.weight"].data, original)
    model.load_state_dict(saved)
    np.testing.assert_array_equal(model.first.weight.data, original)
    assert model.first.weight.requires_grad


def test_load_state_dict_reports_missing_and_unexpected_keys():
    model = TwoLayer()
    state = model.state_dict()
    del state["scale"]
    state["extra"] = Tensor([1.0])
    with pytest.raises(KeyError, match=r"missing.*'scale'.*unexpected.*'extra'"):
        model.load_state_dict(state)
    model.load_state_dict(state, strict=False)


def test_load_state_dict_reports_shape_mismatch():
    model = TwoLayer()
    state = model.state_dict()
    state["second.bias"] = Tensor(np.zeros(3))
    with pytest.raises(ValueError, match=r"second.bias.*\(2,\).*\(3,\)"):
        model.load_state_dict(state)


def test_load_state_dict_accepts_numpy_arrays_and_keeps_the_dtype():
    model = nn.Linear(2, 1, dtype="float32")
    model.load_state_dict({"weight": np.ones((1, 2)), "bias": np.zeros(1)})
    assert model.weight.dtype == np.float32
    np.testing.assert_array_equal(model.weight.data, [[1.0, 1.0]])


def test_apply_visits_children_before_parents():
    visited = []
    model = nn.Sequential(nn.Linear(1, 1), nn.Sequential(nn.Tanh()))
    assert model.apply(lambda m: visited.append(type(m).__name__)) is model
    assert visited == ["Linear", "Tanh", "Sequential", "Sequential"]


def test_sequential_indexing_and_forward(rng):
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    assert len(model) == 3
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[-1], nn.Linear)
    assert isinstance(model[:2], nn.Sequential) and len(model[:2]) == 2
    assert [type(m).__name__ for m in model] == ["Linear", "ReLU", "Linear"]
    x = Tensor(rng.normal(size=(5, 3)))
    expected = model[2](model[1](model[0](x)))
    np.testing.assert_array_equal(model(x).data, expected.data)


def test_repr_shows_the_structure():
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1, bias=False))
    assert repr(model) == (
        "Sequential(\n"
        "  (0): Linear(in_features=2, out_features=3, bias=True)\n"
        "  (1): Tanh()\n"
        "  (2): Linear(in_features=3, out_features=1, bias=False)\n"
        ")"
    )
