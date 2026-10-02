r"""Stateless operations: activations, softmax and friends, losses, ``linear``.

Import it as ``import glassnn.functional as F``, the same way as
``torch.nn.functional``. Every function takes and returns tensors. Functions
with a derivative that is simpler (or more stable) than the derivative of
their parts have their own backward function, written in the docstring with
the adjoint notation of :mod:`glassnn.tensor`; the others are compositions of
tensor operations and need no backward of their own.

Book chapter: :book:`Backpropagation in a multilayer perceptron
<chapters/03-mlp.html>`.
"""

import math
from typing import Any

from glassnn import backend
from glassnn.tensor import Tensor, _result

__all__ = [
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "gelu",
    "leaky_relu",
    "linear",
    "log_softmax",
    "logsumexp",
    "mse_loss",
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "tanh",
]


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------


def relu(input: Tensor) -> Tensor:
    r"""Rectified linear unit, elementwise.

    .. math:: z = \max(a, 0), \qquad \bar a = \bar z \cdot [a > 0]

    The derivative at :math:`a = 0` is taken as 0, as in PyTorch.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.relu(Tensor([-1.0, 2.0]))
        Tensor([0., 2.])
    """
    a = input
    positive = a.data > 0
    out = backend.xp.maximum(a.data, 0)
    return _result(out, (a,), lambda grad: (grad * positive,), "relu")


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    r"""Leaky rectified linear unit, elementwise.

    .. math:: z = \begin{cases} a & a > 0 \\ \alpha a & a \le 0 \end{cases},
        \qquad \bar a = \bar z \cdot \begin{cases} 1 & a > 0 \\ \alpha & a \le 0
        \end{cases}

    Args:
        input: Any shape.
        negative_slope: The slope :math:`\alpha` for negative inputs.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.leaky_relu(Tensor([-2.0, 2.0]), negative_slope=0.5)
        Tensor([-1.,  2.])
    """
    a = input
    slope = backend.xp.where(a.data > 0, 1.0, negative_slope).astype(a.dtype)
    return _result(a.data * slope, (a,), lambda grad: (grad * slope,), "leaky_relu")


def tanh(input: Tensor) -> Tensor:
    r"""Hyperbolic tangent, elementwise.

    .. math:: z = \tanh a, \qquad \bar a = \bar z \odot (1 - z^2)
    """
    out = backend.xp.tanh(input.data)
    return _result(out, (input,), lambda grad: (grad * (1 - out**2),), "tanh")


def _logistic(x: Any) -> Any:
    """The logistic function on an array, without overflow.

    ``exp(-|x|)`` is at most 1, so neither branch can overflow:
    for x >= 0, 1 / (1 + e^{-x}); for x < 0, e^{x} / (1 + e^{x}).
    """
    e = backend.xp.exp(-backend.xp.abs(x))
    return backend.xp.where(x >= 0, 1 / (1 + e), e / (1 + e))


def sigmoid(input: Tensor) -> Tensor:
    r"""Logistic sigmoid, elementwise, computed without overflow.

    .. math:: z = \sigma(a) = \frac{1}{1 + e^{-a}},
        \qquad \bar a = \bar z \odot z \odot (1 - z)

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.sigmoid(Tensor([0.0]))
        Tensor([0.5])
    """
    out = _logistic(input.data)
    return _result(out, (input,), lambda grad: (grad * out * (1 - out),), "sigmoid")


def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    r"""Softplus, a smooth approximation of ReLU, elementwise.

    .. math:: z = \frac{1}{\beta} \ln\left(1 + e^{\beta a}\right),
        \qquad \bar a = \bar z \odot \sigma(\beta a)

    Where :math:`\beta a > \text{threshold}` the function is replaced by the
    identity (as in PyTorch); the difference is below :math:`e^{-20}`, and
    :math:`e^{\beta a}` is never computed for large arguments.

    Args:
        input: Any shape.
        beta: The sharpness :math:`\beta`.
        threshold: Above this value of :math:`\beta a` the output is ``a``.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.softplus(Tensor([0.0, 100.0]))
        Tensor([  0.6931472, 100.       ])
    """
    xp = backend.xp
    a = input
    scaled = beta * a.data
    linear = scaled > threshold
    smooth = xp.log1p(xp.exp(xp.minimum(scaled, threshold))) / beta
    out = xp.where(linear, a.data, smooth).astype(a.dtype)
    slope = xp.where(linear, 1.0, _logistic(scaled)).astype(a.dtype)
    return _result(out, (a,), lambda grad: (grad * slope,), "softplus")


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    r"""Gaussian error linear unit, elementwise :cite:p:`hendrycks2016gaussian`.

    With :math:`\Phi` and :math:`\varphi` the standard normal distribution
    function and density, the exact GELU is

    .. math:: z = a\, \Phi(a), \qquad \bar a = \bar z \odot
        \bigl(\Phi(a) + a\, \varphi(a)\bigr),

    with :math:`\Phi(a) = \tfrac12 (1 + \operatorname{erf}(a / \sqrt 2))`.
    The tanh approximation replaces :math:`\Phi(a)` by
    :math:`\tfrac12 (1 + \tanh u)` with
    :math:`u = \sqrt{2/\pi}\, (a + 0.044715\, a^3)`.

    Args:
        input: Any shape.
        approximate: ``"none"`` (exact) or ``"tanh"``.

    Raises:
        ValueError: For another value of ``approximate``.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.gelu(Tensor([0.0]))
        Tensor([0.])
    """
    if approximate == "none":
        return _gelu_exact(input)
    if approximate == "tanh":
        return _gelu_tanh(input)
    raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}.")


def _gelu_exact(a: Tensor) -> Tensor:
    xp = backend.xp
    cdf = 0.5 * (1 + backend.erf(a.data / math.sqrt(2)))
    pdf = xp.exp(-0.5 * a.data**2) / math.sqrt(2 * math.pi)
    slope = cdf + a.data * pdf
    return _result(a.data * cdf, (a,), lambda grad: (grad * slope,), "gelu")


def _gelu_tanh(a: Tensor) -> Tensor:
    k = math.sqrt(2 / math.pi)
    t = backend.xp.tanh(k * (a.data + 0.044715 * a.data**3))
    out = 0.5 * a.data * (1 + t)
    slope = 0.5 * (1 + t) + 0.5 * a.data * (1 - t**2) * k * (
        1 + 3 * 0.044715 * a.data**2
    )
    return _result(out, (a,), lambda grad: (grad * slope,), "gelu_tanh")


# ----------------------------------------------------------------------
# softmax, log_softmax, logsumexp
# ----------------------------------------------------------------------


def logsumexp(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    r"""Logarithm of the sum of exponentials along ``dim``, without overflow.

    With :math:`m = \max_j a_j`,

    .. math:: z = \ln \sum_j e^{a_j} = m + \ln \sum_j e^{a_j - m},
        \qquad \bar a_j = \bar z\, e^{a_j - z} = \bar z \operatorname{softmax}(a)_j .

    Subtracting :math:`m` changes nothing mathematically, but makes the
    largest exponent :math:`e^0 = 1`, so nothing overflows.

    Args:
        input: Any shape.
        dim: The dimension to reduce.
        keepdim: Whether to keep ``dim`` with size 1.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> round(F.logsumexp(Tensor([[1000.0, 1000.0]]), dim=1).item(), 2)
        1000.69
    """
    xp = backend.xp
    a = input
    m = a.data.max(axis=dim, keepdims=True)
    out_kept = m + xp.log(xp.exp(a.data - m).sum(axis=dim, keepdims=True))
    out = out_kept if keepdim else xp.squeeze(out_kept, axis=dim)

    def backward(grad):
        if not keepdim:
            grad = xp.expand_dims(grad, dim)
        return (grad * xp.exp(a.data - out_kept),)

    return _result(out, (a,), backward, "logsumexp")


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Softmax along ``dim``: positive values that sum to 1.

    .. math:: s_j = \frac{e^{a_j - m}}{\sum_k e^{a_k - m}},
        \qquad \bar a_j = s_j \Bigl(\bar s_j - \sum_k \bar s_k s_k\Bigr)

    with :math:`m = \max_k a_k` for stability (it cancels).

    Args:
        input: Any shape.
        dim: The dimension along which the values sum to 1.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.softmax(Tensor([[0.0, 0.0]]), dim=1)
        Tensor([[0.5, 0.5]])
    """
    xp = backend.xp
    a = input
    e = xp.exp(a.data - a.data.max(axis=dim, keepdims=True))
    s = e / e.sum(axis=dim, keepdims=True)

    def backward(grad):
        return (s * (grad - (grad * s).sum(axis=dim, keepdims=True)),)

    return _result(s, (a,), backward, "softmax")


def log_softmax(input: Tensor, dim: int) -> Tensor:
    r"""Logarithm of the softmax along ``dim``, computed without overflow.

    .. math:: z_j = a_j - \operatorname{logsumexp}(a),
        \qquad \bar a_j = \bar z_j - \operatorname{softmax}(a)_j \sum_k \bar z_k

    Computing ``log(softmax(a))`` instead would give :math:`\ln 0 = -\infty`
    for very negative logits.

    Args:
        input: Any shape.
        dim: The dimension of the classes.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.log_softmax(Tensor([[0.0, -1000.0]]), dim=1)
        Tensor([[    0., -1000.]])
    """
    xp = backend.xp
    a = input
    m = a.data.max(axis=dim, keepdims=True)
    lse = m + xp.log(xp.exp(a.data - m).sum(axis=dim, keepdims=True))
    out = a.data - lse

    def backward(grad):
        return (grad - xp.exp(out) * grad.sum(axis=dim, keepdims=True),)

    return _result(out, (a,), backward, "log_softmax")


# ----------------------------------------------------------------------
# Linear layer
# ----------------------------------------------------------------------


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    r"""Affine map :math:`y = x W^\top + b`.

    A composition of ``@``, ``.T`` and ``+``, so its gradient comes from
    those operations: :math:`\bar W = \bar Y^\top X`,
    :math:`\bar X = \bar Y W`, :math:`\bar b = \sum_n \bar Y_n`.

    Args:
        input: Shape ``(..., in_features)``.
        weight: Shape ``(out_features, in_features)``.
        bias: Shape ``(out_features,)``, or ``None``.

    Returns:
        Shape ``(..., out_features)``.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.linear(Tensor([[1.0, 2.0]]), Tensor([[1.0, 1.0]]), Tensor([0.5]))
        Tensor([[3.5]])
    """
    out = input @ weight.T
    if bias is not None:
        out = out + bias
    return out


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------


def cross_entropy(
    input: Tensor,
    target: Any,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""Cross-entropy between logits and integer class labels.

    For logits :math:`a_n \in \mathbb{R}^C` and label :math:`y_n`, with
    :math:`\ell_{n} = \operatorname{log\_softmax}(a_n)`,

    .. math:: L_n = -(1 - \varepsilon)\, \ell_{n, y_n}
        - \frac{\varepsilon}{C} \sum_{c=1}^{C} \ell_{n, c},

    where :math:`\varepsilon` is the label smoothing
    :cite:p:`szegedy2016rethinking`. Built from :func:`log_softmax` and
    indexing, so the gradient
    :math:`\partial L_n / \partial a_n = \operatorname{softmax}(a_n) - q_n`
    (:math:`q_n` the smoothed one-hot target) comes for free.

    Args:
        input: Logits of shape ``(N, C)``.
        target: Integer labels in ``[0, C)`` of shape ``(N,)`` (a Tensor or
            an array).
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` (per-sample losses).
        label_smoothing: :math:`\varepsilon \in [0, 1]`.

    Returns:
        A scalar, or shape ``(N,)`` for ``reduction="none"``.

    Raises:
        ValueError: If the shapes do not fit (the message names both).
        TypeError: If ``target`` is not integer.

    Note:
        Differences from PyTorch: only ``(N, C)`` logits and integer targets
        are supported (no class probabilities, ``weight`` or
        ``ignore_index``).

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> round(F.cross_entropy(Tensor([[0.0, 0.0]]), Tensor([1])).item(), 4)
        0.6931
    """
    target = _to_tensor(target)
    if input.ndim != 2:
        raise ValueError(
            f"cross_entropy needs logits of shape (N, C), got {input.shape}."
        )
    if target.shape != input.shape[:1]:
        raise ValueError(
            f"Logits of shape {input.shape} need targets of shape "
            f"{input.shape[:1]}, got targets of shape {target.shape}."
        )
    if target.dtype.kind not in "iu":
        raise TypeError(
            f"cross_entropy needs integer class labels, got dtype {target.dtype}."
        )
    log_p = log_softmax(input, dim=1)
    rows = backend.xp.arange(input.shape[0])
    loss = -log_p[rows, target.data]
    if label_smoothing > 0:
        loss = (1 - label_smoothing) * loss - label_smoothing * log_p.mean(dim=1)
    return _reduce(loss, reduction)


def binary_cross_entropy_with_logits(
    input: Tensor, target: Any, reduction: str = "mean"
) -> Tensor:
    r"""Binary cross-entropy of logits, in a form that cannot overflow.

    With :math:`p = \sigma(a)`,
    :math:`L = -t \ln p - (1 - t) \ln (1 - p)` is rewritten as

    .. math:: L = \max(a, 0) - a t + \ln\left(1 + e^{-|a|}\right),
        \qquad \bar a = \bar L \odot (\sigma(a) - t),
        \qquad \bar t = -\bar L \odot a .

    Args:
        input: Logits, any shape.
        target: Targets in :math:`[0, 1]`, the same shape.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Raises:
        ValueError: If the shapes differ (the message names both).

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> loss = F.binary_cross_entropy_with_logits(Tensor([0.0]), Tensor([1.0]))
        >>> round(loss.item(), 4)
        0.6931
    """
    xp = backend.xp
    a, t = input, _to_tensor(target)
    _check_same_shape(a, t, "binary_cross_entropy_with_logits")
    out = xp.maximum(a.data, 0) - a.data * t.data + xp.log1p(xp.exp(-xp.abs(a.data)))

    def backward(grad):
        return grad * (_logistic(a.data) - t.data), -grad * a.data

    loss = _result(out, (a, t), backward, "binary_cross_entropy_with_logits")
    return _reduce(loss, reduction)


def mse_loss(input: Tensor, target: Any, reduction: str = "mean") -> Tensor:
    r"""Mean squared error.

    .. math:: L = \frac{1}{n} \sum_i (a_i - t_i)^2

    Args:
        input: Any shape.
        target: The same shape.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Raises:
        ValueError: If the shapes differ (the message names both).

    Note:
        Differences from PyTorch: PyTorch broadcasts different shapes with a
        warning; GlassNN raises, because broadcasting here is almost always
        a bug (e.g. ``(N, 1)`` against ``(N,)``).

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.mse_loss(Tensor([1.0, 3.0]), Tensor([0.0, 0.0]))
        Tensor(5.)
    """
    target = _to_tensor(target)
    _check_same_shape(input, target, "mse_loss")
    return _reduce((input - target) ** 2, reduction)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_tensor(value: Any) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


def _check_same_shape(input: Tensor, target: Tensor, name: str) -> None:
    if input.shape != target.shape:
        raise ValueError(
            f"{name} needs input and target of the same shape, got "
            f"{input.shape} and {target.shape}."
        )


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}.")
