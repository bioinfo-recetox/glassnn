"""Tests of the Tensor class: forward values, gradients and the backward engine."""

import numpy as np
import pytest

from glassnn import Tensor, backend, no_grad
from glassnn.gradcheck import gradcheck
from glassnn.tensor import _unbroadcast


def leaf(rng, *shape, low=-2.0, high=2.0):
    """A float64 leaf tensor with uniform random values that requires grad."""
    return Tensor(rng.uniform(low, high, size=shape), requires_grad=True)


# Pairs of shapes that broadcast against each other.
BROADCAST_PAIRS = [
    ((3, 4), (3, 4)),
    ((3, 4), (4,)),
    ((4,), (3, 4)),
    ((3, 1), (1, 4)),
    ((2, 3, 4), (3, 1)),
    ((), (3, 4)),
    ((3, 4), ()),
    ((0, 4), (4,)),
]


# --------------------------------------------------------------------------
# Creation
# --------------------------------------------------------------------------


def test_floating_input_gets_the_default_dtype():
    backend.set_default_dtype("float32")
    assert Tensor([1.0, 2.0]).dtype == np.float32
    assert Tensor(np.ones(3)).dtype == np.float32
    assert Tensor(2.5).dtype == np.float32


def test_integer_input_keeps_its_dtype_and_explicit_dtype_wins():
    assert Tensor([1, 2]).dtype.kind == "i"
    assert Tensor([1, 2], dtype="float64").dtype == np.float64


def test_attributes_of_a_new_tensor(rng):
    x = Tensor(rng.normal(size=(2, 3)))
    assert x.shape == (2, 3)
    assert x.ndim == 2
    assert len(x) == 2
    assert x.grad is None
    assert x.requires_grad is False
    assert x.is_leaf


def test_tensor_from_tensor_copies_the_values_not_the_graph(rng):
    x = leaf(rng, 3)
    y = Tensor(x * 2)
    assert y.is_leaf and not y.requires_grad
    np.testing.assert_array_equal(y.data, 2 * x.data)


def test_integer_tensors_cannot_require_grad():
    with pytest.raises(TypeError, match="floating"):
        Tensor([1, 2], requires_grad=True)


def test_repr_shows_values_and_grad_information(rng):
    x = Tensor([1.0, 2.0], requires_grad=True)
    assert repr(x).startswith("Tensor([1., 2.]")
    assert "requires_grad=True" in repr(x)
    assert "op='mul'" in repr(x * 2)
    assert repr(Tensor([1.0])) == "Tensor([1.])"


# --------------------------------------------------------------------------
# _unbroadcast
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grad_shape", "target_shape"),
    [
        ((3, 4), (3, 4)),
        ((3, 4), (4,)),
        ((3, 4), (1, 4)),
        ((2, 3, 4), (3, 1)),
        ((3, 4), ()),
        ((0, 4), (4,)),
    ],
)
def test_unbroadcast_sums_over_broadcast_axes(rng, grad_shape, target_shape):
    grad = rng.normal(size=grad_shape)
    reduced = _unbroadcast(grad, target_shape)
    assert reduced.shape == target_shape
    # Summing the reduced gradient equals summing the full one.
    np.testing.assert_allclose(reduced.sum(), grad.sum())


def test_unbroadcast_matches_explicit_sums(rng):
    grad = rng.normal(size=(2, 3, 4))
    np.testing.assert_allclose(
        _unbroadcast(grad, (3, 1)), grad.sum(axis=(0, 2)).reshape(3, 1)
    )


# --------------------------------------------------------------------------
# Elementwise binary operations
# --------------------------------------------------------------------------

BINARY_OPS = {
    "add": (lambda a, b: a + b, np.add),
    "sub": (lambda a, b: a - b, np.subtract),
    "mul": (lambda a, b: a * b, np.multiply),
    "div": (lambda a, b: a / b, np.divide),
}


@pytest.mark.parametrize("name", list(BINARY_OPS))
@pytest.mark.parametrize(("shape_a", "shape_b"), BROADCAST_PAIRS)
def test_binary_op_forward_and_gradient(rng, name, shape_a, shape_b):
    op, reference = BINARY_OPS[name]
    a = leaf(rng, *shape_a)
    # Keep divisors away from zero.
    b = leaf(rng, *shape_b, low=0.5, high=2.0)
    np.testing.assert_allclose(op(a, b).data, reference(a.data, b.data))
    assert gradcheck(op, [a, b])


@pytest.mark.parametrize("name", list(BINARY_OPS))
def test_binary_op_with_python_scalars_on_both_sides(rng, name):
    op, reference = BINARY_OPS[name]
    a = leaf(rng, 3, low=0.5, high=2.0)
    np.testing.assert_allclose(op(a, 2.0).data, reference(a.data, 2.0))
    np.testing.assert_allclose(op(2.0, a).data, reference(2.0, a.data))
    assert gradcheck(lambda t: op(t, 2.0), [a])
    assert gradcheck(lambda t: op(2.0, t), [a])


def test_numpy_array_on_the_left_gives_a_tensor(rng):
    a = leaf(rng, 3)
    out = np.ones(3) + a
    assert isinstance(out, Tensor)
    assert out.requires_grad


def test_python_scalars_keep_the_tensor_dtype():
    x = Tensor([1.0, 2.0], dtype="float32")
    assert (x * 2.5).dtype == np.float32
    assert (3 - x).dtype == np.float32


def test_negation(rng):
    a = leaf(rng, 3, 4)
    np.testing.assert_array_equal((-a).data, -a.data)
    assert gradcheck(lambda t: -t, [a])


@pytest.mark.parametrize("exponent", [2, 3.0, 0.5, -1.5])
def test_power_with_scalar_exponent(rng, exponent):
    a = leaf(rng, 3, 4, low=0.5, high=2.0)
    np.testing.assert_allclose((a**exponent).data, a.data**exponent)
    assert gradcheck(lambda t: t**exponent, [a])


@pytest.mark.parametrize(("shape_a", "shape_b"), BROADCAST_PAIRS)
def test_power_with_tensor_exponent(rng, shape_a, shape_b):
    a = leaf(rng, *shape_a, low=0.5, high=2.0)
    b = leaf(rng, *shape_b)
    np.testing.assert_allclose((a**b).data, a.data**b.data)
    assert gradcheck(lambda s, t: s**t, [a, b])


def test_scalar_to_the_power_of_a_tensor(rng):
    b = leaf(rng, 4)
    np.testing.assert_allclose((2.0**b).data, 2.0**b.data)
    assert gradcheck(lambda t: 2.0**t, [b])


def test_square_of_a_negative_base_has_a_finite_gradient():
    a = Tensor([-2.0, 0.0, 3.0], requires_grad=True)
    (a**2).sum().backward()
    np.testing.assert_allclose(a.grad.data, [-4.0, 0.0, 6.0])


# --------------------------------------------------------------------------
# Elementwise functions
# --------------------------------------------------------------------------


def test_exp(rng):
    a = leaf(rng, 3, 4)
    np.testing.assert_allclose(a.exp().data, np.exp(a.data))
    assert gradcheck(lambda t: t.exp(), [a])


def test_log(rng):
    a = leaf(rng, 3, 4, low=0.1, high=3.0)
    np.testing.assert_allclose(a.log().data, np.log(a.data))
    assert gradcheck(lambda t: t.log(), [a])


def test_exp_and_log_with_large_and_small_values():
    a = Tensor([-700.0, -50.0, 0.0, 50.0, 700.0], requires_grad=True)
    out = a.exp()
    assert np.all(np.isfinite(out.data))
    out.sum().backward()
    np.testing.assert_allclose(a.grad.data, np.exp(a.data))
    b = Tensor([1e-300, 1e-8, 1.0, 1e8, 1e300], requires_grad=True)
    b.log().sum().backward()
    np.testing.assert_allclose(b.grad.data, 1.0 / b.data)


# --------------------------------------------------------------------------
# Matrix multiplication
# --------------------------------------------------------------------------

MATMUL_SHAPES = [
    ((3, 4), (4, 5)),
    ((4,), (4,)),
    ((4,), (4, 5)),
    ((3, 4), (4,)),
    ((2, 3, 4), (4, 5)),
    ((2, 3, 4), (2, 4, 5)),
    ((1, 3, 4), (2, 4, 5)),
    ((3, 4), (2, 4, 5)),
    ((2, 3, 4), (4,)),
    ((0, 4), (4, 5)),
]


@pytest.mark.parametrize(("shape_a", "shape_b"), MATMUL_SHAPES)
def test_matmul_forward_and_gradient(rng, shape_a, shape_b):
    a = leaf(rng, *shape_a)
    b = leaf(rng, *shape_b)
    np.testing.assert_allclose((a @ b).data, a.data @ b.data)
    assert gradcheck(lambda s, t: s @ t, [a, b])


def test_matmul_with_a_numpy_array_on_either_side(rng):
    a = leaf(rng, 3, 4)
    m = rng.normal(size=(4, 2))
    np.testing.assert_allclose((a @ m).data, a.data @ m)
    np.testing.assert_allclose((m.T @ a.T).data, m.T @ a.data.T)


@pytest.mark.parametrize(
    ("shape_a", "shape_b"), [((3, 4), (5, 2)), ((4,), (5,)), ((2, 3), (2,))]
)
def test_matmul_shape_mismatch_names_both_shapes(shape_a, shape_b):
    with pytest.raises(ValueError) as error:
        Tensor(np.ones(shape_a)) @ Tensor(np.ones(shape_b))
    assert str(shape_a) in str(error.value)
    assert str(shape_b) in str(error.value)


def test_matmul_rejects_scalars():
    with pytest.raises(ValueError, match="at least one dimension"):
        Tensor(2.0) @ Tensor(np.ones(3))


# --------------------------------------------------------------------------
# Reductions
# --------------------------------------------------------------------------

REDUCTION_ARGS = [
    {},
    {"dim": 0},
    {"dim": -1},
    {"dim": 1, "keepdim": True},
    {"dim": (0, 2)},
    {"dim": (0, 2), "keepdim": True},
]


@pytest.mark.parametrize("kwargs", REDUCTION_ARGS)
def test_sum(rng, kwargs):
    a = leaf(rng, 2, 3, 4)
    axis = kwargs.get("dim")
    expected = a.data.sum(axis=axis, keepdims=kwargs.get("keepdim", False))
    np.testing.assert_allclose(a.sum(**kwargs).data, expected)
    assert gradcheck(lambda t: t.sum(**kwargs), [a])


@pytest.mark.parametrize("kwargs", REDUCTION_ARGS)
def test_mean(rng, kwargs):
    a = leaf(rng, 2, 3, 4)
    axis = kwargs.get("dim")
    expected = a.data.mean(axis=axis, keepdims=kwargs.get("keepdim", False))
    np.testing.assert_allclose(a.mean(**kwargs).data, expected)
    assert gradcheck(lambda t: t.mean(**kwargs), [a])


def test_sum_over_an_empty_batch(rng):
    a = leaf(rng, 0, 3)
    out = a.sum(dim=0)
    np.testing.assert_array_equal(out.data, np.zeros(3))
    out.sum().backward()
    assert a.grad.shape == (0, 3)


# --------------------------------------------------------------------------
# Shape operations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(6, 2), (12,), (3, -1), (2, 2, 3)])
def test_reshape(rng, shape):
    a = leaf(rng, 3, 4)
    np.testing.assert_array_equal(a.reshape(shape).data, a.data.reshape(shape))
    np.testing.assert_array_equal(a.reshape(*shape).data, a.data.reshape(shape))
    assert gradcheck(lambda t: t.reshape(shape), [a])


@pytest.mark.parametrize(("dim0", "dim1"), [(0, 1), (0, 2), (-1, -2)])
def test_transpose(rng, dim0, dim1):
    a = leaf(rng, 2, 3, 4)
    expected = np.swapaxes(a.data, dim0, dim1)
    np.testing.assert_array_equal(a.transpose(dim0, dim1).data, expected)
    assert gradcheck(lambda t: t.transpose(dim0, dim1), [a])


@pytest.mark.parametrize("dims", [(2, 0, 1), (1, 2, 0), (0, 1, 2)])
def test_permute(rng, dims):
    a = leaf(rng, 2, 3, 4)
    np.testing.assert_array_equal(a.permute(*dims).data, a.data.transpose(dims))
    np.testing.assert_array_equal(a.permute(dims).data, a.data.transpose(dims))
    assert gradcheck(lambda t: t.permute(*dims), [a])


@pytest.mark.parametrize("shape", [(), (3,), (3, 4)])
def test_T_on_up_to_two_dimensions(rng, shape):
    a = leaf(rng, *shape)
    np.testing.assert_array_equal(a.T.data, a.data.T)
    assert gradcheck(lambda t: t.T, [a])


def test_T_rejects_more_than_two_dimensions():
    with pytest.raises(ValueError, match="permute"):
        _ = Tensor(np.ones((2, 3, 4))).T


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

INDICES = [
    1,
    -1,
    slice(1, 3),
    (slice(None), 2),
    (Ellipsis, slice(None, None, 2)),
    np.array([0, 2, 2, 0]),  # repeated indices: gradients must add up
    (np.array([0, 1, 2]), np.array([3, 3, 0])),
    np.array([True, False, True]),
    (None, 1),
]


@pytest.mark.parametrize("index", INDICES)
def test_indexing(rng, index):
    a = leaf(rng, 3, 4)
    np.testing.assert_array_equal(a[index].data, a.data[index])
    assert gradcheck(lambda t: t[index], [a])


def test_indexing_with_an_integer_tensor(rng):
    a = leaf(rng, 3, 4)
    idx = Tensor([2, 0, 2])
    np.testing.assert_array_equal(a[idx].data, a.data[[2, 0, 2]])
    a[idx].sum().backward()
    np.testing.assert_array_equal(a.grad.data[:, 0], [1.0, 0.0, 2.0])


# --------------------------------------------------------------------------
# No in-place operations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    ["x += 1", "x -= 1", "x *= 2", "x /= 2", "x **= 2", "x @= y", "x[0] = 1.0"],
)
def test_in_place_operations_raise(statement):
    x = Tensor(np.ones((2, 2)))
    y = Tensor(np.ones((2, 2)))
    with pytest.raises(NotImplementedError, match="in-place"):
        exec(statement, {}, {"x": x, "y": y})


# --------------------------------------------------------------------------
# The backward pass
# --------------------------------------------------------------------------


def test_backward_of_a_small_expression():
    # f(x, y) = x * y + x**2, df/dx = y + 2x, df/dy = x
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(4.0, requires_grad=True)
    (x * y + x**2).backward()
    assert x.grad.item() == pytest.approx(10.0)
    assert y.grad.item() == pytest.approx(3.0)


def test_grad_is_a_tensor_without_grad_and_with_the_leaf_dtype(rng):
    x = Tensor([1.0, 2.0], requires_grad=True, dtype="float32")
    (x * 3).sum().backward()
    assert isinstance(x.grad, Tensor)
    assert not x.grad.requires_grad
    assert x.grad.dtype == np.float32


def test_gradients_accumulate_across_backward_calls():
    x = Tensor(2.0, requires_grad=True)
    (x * 3).backward()
    (x * 3).backward()
    assert x.grad.item() == pytest.approx(6.0)


def test_the_graph_is_kept_after_backward():
    x = Tensor(2.0, requires_grad=True)
    y = x * x
    y.backward()
    y.backward()
    assert x.grad.item() == pytest.approx(8.0)


def test_a_node_used_twice_receives_both_contributions():
    # The "diamond": y = a * a with a = 2x, so dy/dx = 8x.
    x = Tensor(1.5, requires_grad=True)
    a = 2 * x
    (a * a).backward()
    assert x.grad.item() == pytest.approx(12.0)


def test_backward_through_a_long_chain_does_not_recurse():
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(10_000):
        y = y * 1.0
    y.backward()
    assert x.grad.item() == pytest.approx(1.0)


def test_only_leaves_keep_their_gradient_unless_retain_grad(rng):
    x = leaf(rng, 3)
    hidden = x * 2
    kept = x * 3
    kept.retain_grad()
    (hidden + kept).sum().backward()
    assert hidden.grad is None
    np.testing.assert_array_equal(kept.grad.data, np.ones(3))
    np.testing.assert_array_equal(x.grad.data, np.full(3, 5.0))


def test_retain_grad_needs_requires_grad():
    with pytest.raises(RuntimeError, match="requires_grad"):
        Tensor([1.0]).retain_grad()


def test_backward_of_a_non_scalar_needs_a_gradient(rng):
    x = leaf(rng, 3)
    with pytest.raises(RuntimeError, match=r"\(3,\)"):
        (x * 2).backward()


def test_backward_with_an_explicit_gradient(rng):
    x = leaf(rng, 3)
    (x * 2).backward(np.array([1.0, 0.0, -1.0]))
    np.testing.assert_array_equal(x.grad.data, [2.0, 0.0, -2.0])
    x.grad = None
    (x * 2).backward(Tensor([1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(x.grad.data, [2.0, 2.0, 2.0])


def test_backward_gradient_must_match_the_shape(rng):
    x = leaf(rng, 3)
    with pytest.raises(ValueError, match=r"\(2,\).*\(3,\)"):
        (x * 2).backward(np.ones(2))


def test_backward_on_a_tensor_without_grad_raises():
    with pytest.raises(RuntimeError, match="requires_grad"):
        Tensor(1.0).backward()


def test_requires_grad_propagates_only_from_inputs_that_require_it(rng):
    x = leaf(rng, 3)
    c = Tensor(rng.normal(size=3))
    assert (x * c).requires_grad
    assert not (c * c).requires_grad
    (x * c).sum().backward()
    assert c.grad is None


# --------------------------------------------------------------------------
# no_grad, detach, item, numpy
# --------------------------------------------------------------------------


def test_no_grad_as_a_context_manager(rng):
    x = leaf(rng, 3)
    with no_grad():
        y = x * 2
    assert not y.requires_grad
    assert y.is_leaf
    assert (x * 2).requires_grad


def test_no_grad_as_a_decorator(rng):
    @no_grad()
    def double(t):
        return t * 2

    assert not double(leaf(rng, 3)).requires_grad


def test_no_grad_nests_and_restores_after_an_exception(rng):
    x = leaf(rng, 3)
    with pytest.raises(ZeroDivisionError), no_grad():
        with no_grad():
            pass
        assert not (x * 2).requires_grad
        _ = 1 / 0
    assert (x * 2).requires_grad


def test_detach_shares_values_but_not_the_graph(rng):
    x = leaf(rng, 3)
    d = (x * 2).detach()
    assert d.is_leaf and not d.requires_grad
    np.testing.assert_array_equal(d.data, 2 * x.data)


def test_item_and_numpy():
    x = Tensor([[1.5]])
    assert x.item() == 1.5
    array = Tensor([1.0, 2.0]).numpy()
    assert isinstance(array, np.ndarray)


def test_numpy_returns_a_copy():
    x = Tensor([1.0, 2.0])
    array = x.numpy()
    array[0] = 100.0
    assert x.data[0] == 1.0
