r"""Gradient verification by central finite differences.

:func:`gradcheck` compares the gradient computed by :meth:`Tensor.backward`
with a numerical estimate. For a function :math:`f` and a small step
:math:`\varepsilon` the *central difference*

.. math::

    \frac{\partial f}{\partial x_i} \approx
    \frac{f(x + \varepsilon e_i) - f(x - \varepsilon e_i)}{2 \varepsilon}

has error :math:`O(\varepsilon^2)`, against :math:`O(\varepsilon)` for the
one-sided difference :cite:p:`griewank2008evaluating`. Rounding error grows
like :math:`1/\varepsilon`, so the check needs float64.

Book chapter: :book:`Automatic differentiation <chapters/02-autodiff.html>`.
"""

from collections.abc import Callable, Sequence

import numpy as np

from glassnn import backend
from glassnn.tensor import Tensor, no_grad


class GradcheckError(AssertionError):
    """Raised when an analytical gradient differs from the numerical one."""


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> bool:
    r"""Check the gradients of ``fn`` at ``inputs`` against finite differences.

    A non-scalar output :math:`y` is reduced to the scalar
    :math:`L = \sum_k w_k y_k` with fixed weights :math:`w_k = \cos k`, which
    have mixed signs and distinct values, so that errors in different
    elements cannot cancel. The check therefore verifies the
    vector-Jacobian product :math:`w^\top J`, which is exactly what
    :meth:`Tensor.backward` computes. An element passes if
    :math:`|g_{\text{backward}} - g_{\text{numerical}}| \le
    \text{atol} + \text{rtol}\, |g_{\text{numerical}}|`.

    Args:
        fn: A function of the tensors in ``inputs`` that returns a Tensor.
        inputs: float64 tensors. Those with ``requires_grad=True`` are
            checked; their ``.grad`` is overwritten.
        eps: The finite-difference step :math:`\varepsilon`.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        ``True`` if every checked gradient matches.

    Raises:
        TypeError: If a checked input is not float64.
        GradcheckError: At the first mismatch, naming the input, the element,
            and both values.

    Example:
        >>> from glassnn import Tensor
        >>> from glassnn.gradcheck import gradcheck
        >>> x = Tensor([0.5, 1.5], requires_grad=True, dtype="float64")
        >>> gradcheck(lambda t: (t * t.exp()).sum(), [x])
        True
    """
    checked = [i for i, x in enumerate(inputs) if x.requires_grad]
    for i in checked:
        if inputs[i].dtype != np.float64:
            raise TypeError(
                f"gradcheck needs float64 inputs; input {i} has dtype "
                f"{inputs[i].dtype}. Use Tensor(..., dtype='float64')."
            )

    weights = _weights(fn(*inputs).shape)
    for i in checked:
        inputs[i].grad = None
    _weighted_sum(fn(*inputs), weights).backward()

    for i in checked:
        analytic = _gradient_or_zeros(inputs[i])
        numeric = _numerical_gradient(fn, inputs, i, weights, eps)
        _compare(analytic, numeric, i, atol, rtol)
    return True


def _weights(shape: tuple[int, ...]) -> Tensor:
    """Fixed weights cos(1), cos(2), ... with the given shape."""
    size = int(np.prod(shape))
    values = np.cos(np.arange(1, size + 1, dtype=np.float64)).reshape(shape)
    return Tensor(values, dtype="float64")


def _weighted_sum(output: Tensor, weights: Tensor) -> Tensor:
    return (output * weights).sum()


def _gradient_or_zeros(x: Tensor) -> np.ndarray:
    """The gradient of ``x``; zero if ``x`` does not influence the output."""
    if x.grad is None:
        return np.zeros(x.shape)
    return backend.to_numpy(x.grad.data)


def _numerical_gradient(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    index: int,
    weights: Tensor,
    eps: float,
) -> np.ndarray:
    """Central differences of the weighted output with respect to one input."""
    x = backend.to_numpy(inputs[index].data)
    gradient = np.zeros(x.shape)
    for position in np.ndindex(x.shape):
        shifted = []
        for sign in (+1.0, -1.0):
            values = x.copy()
            values[position] += sign * eps
            perturbed = list(inputs)
            perturbed[index] = Tensor(values, dtype="float64")
            with no_grad():
                shifted.append(_weighted_sum(fn(*perturbed), weights).item())
        gradient[position] = (shifted[0] - shifted[1]) / (2 * eps)
    return gradient


def _compare(
    analytic: np.ndarray, numeric: np.ndarray, index: int, atol: float, rtol: float
) -> None:
    """Raise GradcheckError at the worst element if the gradients differ."""
    excess = np.abs(analytic - numeric) - (atol + rtol * np.abs(numeric))
    if excess.size == 0 or np.all(excess <= 0):
        return
    worst = np.unravel_index(np.argmax(excess), excess.shape)
    raise GradcheckError(
        f"Gradient mismatch for input {index} at element {tuple(map(int, worst))}: "
        f"backward gives {float(analytic[worst]):.10g}, finite differences give "
        f"{float(numeric[worst]):.10g}."
    )
