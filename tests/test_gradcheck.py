import numpy as np
import pytest

from glassnn import Tensor
from glassnn.gradcheck import GradcheckError, gradcheck
from glassnn.tensor import _result


def wrong_square(t):
    """t**2 with a deliberately wrong backward (3t instead of 2t)."""
    return _result(t.data**2, (t,), lambda grad: (grad * 3 * t.data,), "wrong_square")


def test_a_correct_gradient_passes(rng):
    x = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    assert gradcheck(lambda t: (t * t).sum(), [x]) is True


def test_a_wrong_gradient_is_detected_and_located(rng):
    x = Tensor(rng.normal(size=4), requires_grad=True)
    with pytest.raises(GradcheckError, match=r"input 0.*element \(\d,\)"):
        gradcheck(wrong_square, [x])


def test_non_scalar_outputs_are_checked(rng):
    x = Tensor(rng.normal(size=(2, 3)), requires_grad=True)
    y = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    assert gradcheck(lambda a, b: a @ b, [x, y])


def test_inputs_without_requires_grad_are_not_checked(rng):
    x = Tensor(rng.normal(size=3), requires_grad=True)
    c = Tensor(rng.normal(size=3))
    assert gradcheck(lambda a, b: a * b, [x, c])
    assert c.grad is None


def test_float32_inputs_are_rejected():
    x = Tensor([1.0, 2.0], requires_grad=True, dtype="float32")
    with pytest.raises(TypeError, match="float64"):
        gradcheck(lambda t: t * 2, [x])


def test_an_input_that_does_not_reach_the_output_has_zero_gradient(rng):
    x = Tensor(rng.normal(size=3), requires_grad=True)
    unused = Tensor(rng.normal(size=2), requires_grad=True)
    assert gradcheck(lambda a, b: a * 2, [x, unused])


def test_existing_gradients_are_cleared(rng):
    x = Tensor(rng.normal(size=3), requires_grad=True)
    x.grad = Tensor(np.full(3, 100.0))
    assert gradcheck(lambda t: t * 2, [x])
