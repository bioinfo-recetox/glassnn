r"""Stateless operations: activations, losses, normalization, convolution, pooling.

Import it as ``import glassnn.functional as F``, the same way as
``torch.nn.functional``. Every function takes and returns tensors. Functions
with a derivative that is simpler (or more stable) than the derivative of
their parts have their own backward function, written in the docstring with
the adjoint notation of :mod:`glassnn.tensor`; the others are compositions of
tensor operations and need no backward of their own.

Book chapters: :book:`Backpropagation in a multilayer perceptron
<chapters/03-mlp.html>`; dropout and normalization:
:book:`Regularization <chapters/06-regularization.html>`; convolution and
pooling: :book:`Convolutions for sequences <chapters/09-convolutions.html>`.
"""

import itertools
import math
from typing import Any

from glassnn import backend
from glassnn.tensor import Tensor, _result

__all__ = [
    "avg_pool1d",
    "avg_pool2d",
    "batch_norm",
    "binary_cross_entropy_with_logits",
    "conv1d",
    "conv2d",
    "cross_entropy",
    "dropout",
    "gelu",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_softmax",
    "logsumexp",
    "max_pool1d",
    "max_pool2d",
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
# Dropout and normalization
# ----------------------------------------------------------------------


def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, generator: Any = None
) -> Tensor:
    r"""Inverted dropout: zero each element with probability ``p``.

    With a random mask :math:`m_i \sim \text{Bernoulli}(1 - p)`, drawn anew
    at every call,

    .. math:: z = \frac{m \odot a}{1 - p}, \qquad
        \bar a = \frac{m \odot \bar z}{1 - p} .

    Dividing by :math:`1 - p` keeps :math:`\mathbb E[z] = a`, so nothing has
    to be rescaled at evaluation time, when dropout is the identity
    :cite:p:`srivastava2014dropout`.

    Args:
        input: Any shape.
        p: The probability of zeroing an element, in :math:`[0, 1]`.
        training: If ``False`` (evaluation), ``input`` is returned unchanged.
        generator: The random generator; ``None`` uses the global one (see
            :func:`glassnn.backend.manual_seed`).

    Returns:
        The same shape and dtype as ``input``.

    Raises:
        ValueError: If ``p`` is not in :math:`[0, 1]`.

    Note:
        Differences from PyTorch: there is no ``inplace`` argument; the
        ``generator`` argument does not exist in ``torch.nn.functional``.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor, manual_seed
        >>> _ = manual_seed(0)
        >>> F.dropout(Tensor([1.0, 1.0, 1.0, 1.0]), p=0.5)
        Tensor([2., 0., 0., 0.])
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The dropout probability must be in [0, 1], got {p}.")
    if not training or p == 0.0:
        return input
    if generator is None:
        generator = backend.get_generator()
    keep = generator.random(input.shape) >= p
    scale = 0.0 if p == 1.0 else 1.0 / (1.0 - p)
    mask = (keep * scale).astype(input.dtype)
    return _result(input.data * mask, (input,), lambda grad: (grad * mask,), "dropout")


def _standardize(input: Tensor, axes: tuple[int, ...], eps: float) -> Tensor:
    r"""Subtract the mean and divide by the standard deviation over ``axes``.

    With :math:`\mu` and :math:`\sigma^2` the mean and the (biased) variance
    of the :math:`m` elements that share a mean,

    .. math:: \hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}},
        \qquad \bar x = \frac{1}{\sqrt{\sigma^2 + \varepsilon}} \Bigl(\bar{\hat x}
        - \operatorname{mean}(\bar{\hat x})
        - \hat x \operatorname{mean}(\bar{\hat x} \odot \hat x)\Bigr),

    with the means taken over ``axes``. The two subtracted terms remove the
    parts of :math:`\bar{\hat x}` that would change :math:`\mu` and
    :math:`\sigma`, which are both functions of :math:`x`
    :cite:p:`ioffe2015batch`.

    The mean is subtracted before squaring (two passes), so a large common
    offset does not destroy the variance by cancellation.
    """
    xp = backend.xp
    x = input.data
    mean = x.mean(axis=axes, keepdims=True)
    centered = x - mean
    var = (centered**2).mean(axis=axes, keepdims=True)
    inv_std = 1 / xp.sqrt(var + eps)
    x_hat = centered * inv_std

    def backward(grad):
        grad_mean = grad.mean(axis=axes, keepdims=True)
        projection = (grad * x_hat).mean(axis=axes, keepdims=True)
        return (inv_std * (grad - grad_mean - x_hat * projection),)

    return _result(x_hat, (input,), backward, "standardize")


def layer_norm(
    input: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Layer normalization: standardize each sample over its last dimensions.

    .. math:: y = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \odot \gamma + \beta,

    where :math:`\mu` and :math:`\sigma^2` (biased) are computed over the
    last ``len(normalized_shape)`` dimensions of every sample separately
    :cite:p:`ba2016layer`. The statistics do not depend on the other
    samples of the batch, so training and evaluation behave the same. The
    gradient of the standardization is written in the docstring of
    ``_standardize`` (see the source); :math:`\gamma, \beta` act by ``*``
    and ``+``.

    Args:
        input: Shape ``(..., *normalized_shape)``.
        normalized_shape: The trailing shape over which to normalize.
        weight: :math:`\gamma`, shape ``normalized_shape``, or ``None``.
        bias: :math:`\beta`, shape ``normalized_shape``, or ``None``.
        eps: :math:`\varepsilon`, added to the variance.

    Returns:
        The same shape as ``input``.

    Raises:
        ValueError: If the trailing shape of ``input`` is not
            ``normalized_shape`` (the message names both).

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.layer_norm(Tensor([[1.0, 3.0]]), (2,))
        Tensor([[-0.999995,  0.999995]])
    """
    normalized_shape = tuple(normalized_shape)
    k = len(normalized_shape)
    if input.shape[input.ndim - k :] != normalized_shape:
        raise ValueError(
            f"layer_norm over the trailing shape {normalized_shape} needs an "
            f"input that ends with it, got shape {input.shape}."
        )
    out = _standardize(input, tuple(range(input.ndim - k, input.ndim)), eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def batch_norm(
    input: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Batch normalization: standardize each channel over the batch.

    For input of shape ``(N, C)`` or ``(N, C, L)``, channel :math:`c` is
    standardized with the mean :math:`\mu_c` and the (biased) variance
    :math:`\sigma_c^2` over all its :math:`m = N` (or :math:`N L`) values,

    .. math:: y = \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}}\, \gamma_c
        + \beta_c ,

    :cite:p:`ioffe2015batch`. In training, the batch statistics are used
    (and depend on the other samples, so the gradient flows through them;
    see ``_standardize``), and the running statistics are updated as

    .. math:: \hat\mu \leftarrow (1 - \rho)\, \hat\mu + \rho\, \mu_c,
        \qquad \hat\sigma^2 \leftarrow (1 - \rho)\, \hat\sigma^2
        + \rho\, \frac{m}{m - 1}\, \sigma_c^2

    with :math:`\rho` the ``momentum`` (the running variance is unbiased).
    In evaluation, :math:`\hat\mu, \hat\sigma^2` replace the batch
    statistics, and the layer is a fixed affine map of each sample.

    Args:
        input: Shape ``(N, C)`` or ``(N, C, L)``.
        running_mean: :math:`\hat\mu`, shape ``(C,)``, or ``None``.
        running_var: :math:`\hat\sigma^2`, shape ``(C,)``, or ``None``.
        weight: :math:`\gamma`, shape ``(C,)``, or ``None``.
        bias: :math:`\beta`, shape ``(C,)``, or ``None``.
        training: Use (and update) batch statistics. Without running
            statistics, batch statistics are used in evaluation too.
        momentum: :math:`\rho`.
        eps: :math:`\varepsilon`, added to the variance.

    Returns:
        The same shape as ``input``.

    Raises:
        ValueError: For an input that is not 2-D or 3-D, statistics or
            affine parameters of the wrong shape, or a channel with a single
            value in training.

    Note:
        Like PyTorch, this function updates ``running_mean`` and
        ``running_var`` in place: their ``.data`` is replaced. These buffers
        are not learned, and no gradient flows through them.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.batch_norm(Tensor([[1.0], [3.0]]), None, None, training=True)
        Tensor([[-0.999995],
                [ 0.999995]])
    """
    if input.ndim not in (2, 3):
        raise ValueError(
            f"batch_norm needs an input of shape (N, C) or (N, C, L), "
            f"got {input.shape}."
        )
    channels = input.shape[1]
    for name, tensor in [
        ("running_mean", running_mean),
        ("running_var", running_var),
        ("weight", weight),
        ("bias", bias),
    ]:
        if tensor is not None and tensor.shape != (channels,):
            raise ValueError(
                f"An input with {channels} channels needs {name} of shape "
                f"({channels},), got {tensor.shape}."
            )
    axes = (0,) if input.ndim == 2 else (0, 2)
    per_channel = (channels,) if input.ndim == 2 else (channels, 1)
    use_batch = training or running_mean is None or running_var is None
    if use_batch:
        m = input.data.size // max(channels, 1)
        if m <= 1:
            raise ValueError(
                "batch_norm in training needs more than one value per channel, "
                f"got an input of shape {input.shape}."
            )
        out = _standardize(input, axes, eps)
        if training and running_mean is not None and running_var is not None:
            _update_running_stats(input, axes, m, running_mean, running_var, momentum)
    else:
        assert running_mean is not None and running_var is not None
        mean = running_mean.data.reshape(per_channel)
        inv_std = 1 / backend.xp.sqrt(running_var.data.reshape(per_channel) + eps)
        out = (input - Tensor(mean, dtype=input.dtype)) * Tensor(
            inv_std, dtype=input.dtype
        )
    if weight is not None:
        out = out * weight.reshape(per_channel)
    if bias is not None:
        out = out + bias.reshape(per_channel)
    return out


def _update_running_stats(
    input: Tensor,
    axes: tuple[int, ...],
    m: int,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
) -> None:
    """Move the running statistics towards the statistics of this batch."""
    mean = input.data.mean(axis=axes)
    unbiased_var = input.data.var(axis=axes) * m / (m - 1)
    new_mean = (1 - momentum) * running_mean.data + momentum * mean
    new_var = (1 - momentum) * running_var.data + momentum * unbiased_var
    running_mean.data = new_mean.astype(running_mean.dtype)
    running_var.data = new_var.astype(running_var.dtype)


# ----------------------------------------------------------------------
# Convolution and pooling
# ----------------------------------------------------------------------


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int | str = 0,
    dilation: int = 1,
) -> Tensor:
    r"""1-D convolution (cross-correlation) over sequences.

    For input :math:`x \in \mathbb R^{N \times C_\text{in} \times L}` (after
    zero padding) and weight
    :math:`w \in \mathbb R^{C_\text{out} \times C_\text{in} \times K}`,

    .. math:: y_{n,o,i} = b_o + \sum_{c=1}^{C_\text{in}} \sum_{k=0}^{K-1}
        w_{o,c,k}\, x_{n,c,\, s i + d k},

    with stride :math:`s` and dilation :math:`d`; the output length is
    :math:`L_\text{out} = \lfloor (L - d(K-1) - 1)/s \rfloor + 1`. Like
    every deep-learning library, GlassNN calls this cross-correlation a
    convolution (the kernel is not flipped) :cite:p:`lecun1998gradient`.

    The forward pass gathers all windows :math:`x_{n,c,si+dk}` without
    copying (``sliding_window_view``, "im2col") and contracts them with
    :math:`w` :cite:p:`chellapilla2006high`. The backward pass is

    .. math:: \bar w_{o,c,k} = \sum_{n,i} \bar y_{n,o,i}\, x_{n,c,si+dk},
        \qquad \bar b_o = \sum_{n,i} \bar y_{n,o,i},
        \qquad \bar x_{n,c,\,si+dk} \mathrel{+}= \sum_o \bar y_{n,o,i}\, w_{o,c,k},

    the last one a scatter-add (a "transposed convolution"), done with one
    vectorized slice per kernel offset :math:`k`.

    Args:
        input: Shape ``(N, C_in, L)``.
        weight: Shape ``(C_out, C_in, K)``.
        bias: Shape ``(C_out,)``, or ``None``.
        stride: The step :math:`s` between output positions.
        padding: Zeros added at both ends (an ``int``), ``"valid"`` (none)
            or ``"same"`` (output length equal to input length; stride 1
            only; an odd total padding puts the extra zero on the right).
        dilation: The spacing :math:`d` between kernel elements.

    Returns:
        Shape ``(N, C_out, L_out)``.

    Raises:
        ValueError: For wrong shapes (the message names them), an input too
            short for the kernel, or ``"same"`` with a stride above 1.

    Note:
        Differences from PyTorch: no ``groups`` and only zero padding.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> x = Tensor([[[1.0, 2.0, 3.0, 4.0]]])
        >>> F.conv1d(x, Tensor([[[1.0, -1.0]]]))  # differences of neighbours
        Tensor([[[-1., -1., -1.]]])
    """
    return _conv(input, weight, bias, stride, padding, dilation, 1, "conv1d")


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] | str = 0,
    dilation: int | tuple[int, ...] = 1,
) -> Tensor:
    r"""2-D convolution (cross-correlation) over images.

    .. math:: y_{n,o,i,j} = b_o + \sum_{c} \sum_{k,l}
        w_{o,c,k,l}\, x_{n,c,\, s_1 i + d_1 k,\, s_2 j + d_2 l}

    The same computation as :func:`conv1d` with two spatial dimensions; the
    backward pass has the same three formulas, summed over both.

    Args:
        input: Shape ``(N, C_in, H, W)``.
        weight: Shape ``(C_out, C_in, K_H, K_W)``.
        bias: Shape ``(C_out,)``, or ``None``.
        stride: An ``int`` or a pair.
        padding: An ``int``, a pair, ``"valid"`` or ``"same"``.
        dilation: An ``int`` or a pair.

    Returns:
        Shape ``(N, C_out, H_out, W_out)``.

    Raises:
        ValueError: As :func:`conv1d`.

    Note:
        Differences from PyTorch: no ``groups`` and only zero padding.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        >>> F.conv2d(x, Tensor([[[[1.0, 1.0], [1.0, 1.0]]]]))
        Tensor([[[[10.]]]])
    """
    return _conv(input, weight, bias, stride, padding, dilation, 2, "conv2d")


def max_pool1d(
    input: Tensor, kernel_size: int, stride: int | None = None, padding: int = 0
) -> Tensor:
    r"""Maximum over sliding windows of a sequence.

    .. math:: y_{n,c,i} = \max_{0 \le k < K} x_{n,c,\, s i + k},
        \qquad \bar x_{n,c,\, s i + k^\star} \mathrel{+}= \bar y_{n,c,i},

    where :math:`k^\star` is the position of the maximum (the first one if
    there are ties, as in PyTorch); the other elements of the window get no
    gradient. Padding adds :math:`-\infty`.

    Args:
        input: Shape ``(N, C, L)``.
        kernel_size: The window length :math:`K`.
        stride: The step :math:`s` (default: ``kernel_size``).
        padding: :math:`-\infty` added at both ends, at most ``K // 2``.

    Returns:
        Shape ``(N, C, L_out)`` with
        :math:`L_\text{out} = \lfloor (L + 2p - K)/s \rfloor + 1`.

    Raises:
        ValueError: For a wrong shape, too much padding, or a too short input.

    Note:
        Differences from PyTorch: no ``dilation``, ``ceil_mode`` or
        ``return_indices``.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.max_pool1d(Tensor([[[1.0, 5.0, 2.0, 0.0]]]), kernel_size=2)
        Tensor([[[5., 2.]]])
    """
    return _pool(input, kernel_size, stride, padding, 1, "max", "max_pool1d")


def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    r"""Maximum over sliding windows of an image; see :func:`max_pool1d`.

    Args:
        input: Shape ``(N, C, H, W)``.
        kernel_size: An ``int`` or a pair.
        stride: An ``int`` or a pair (default: ``kernel_size``).
        padding: An ``int`` or a pair.

    Returns:
        Shape ``(N, C, H_out, W_out)``.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.max_pool2d(Tensor([[[[1.0, 2.0], [4.0, 3.0]]]]), kernel_size=2)
        Tensor([[[[4.]]]])
    """
    return _pool(input, kernel_size, stride, padding, 2, "max", "max_pool2d")


def avg_pool1d(
    input: Tensor, kernel_size: int, stride: int | None = None, padding: int = 0
) -> Tensor:
    r"""Mean over sliding windows of a sequence.

    .. math:: y_{n,c,i} = \frac1K \sum_{k=0}^{K-1} x_{n,c,\, s i + k},
        \qquad \bar x_{n,c,\, s i + k} \mathrel{+}= \frac{\bar y_{n,c,i}}{K}.

    Padding adds zeros, which count in the mean (PyTorch's default
    ``count_include_pad=True``).

    Args:
        input: Shape ``(N, C, L)``.
        kernel_size: The window length :math:`K`.
        stride: The step :math:`s` (default: ``kernel_size``).
        padding: Zeros added at both ends, at most ``K // 2``.

    Returns:
        Shape ``(N, C, L_out)``.

    Note:
        Differences from PyTorch: no ``ceil_mode`` or
        ``count_include_pad=False``.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.avg_pool1d(Tensor([[[1.0, 3.0, 2.0, 4.0]]]), kernel_size=2)
        Tensor([[[2., 3.]]])
    """
    return _pool(input, kernel_size, stride, padding, 1, "avg", "avg_pool1d")


def avg_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    r"""Mean over sliding windows of an image; see :func:`avg_pool1d`.

    Args:
        input: Shape ``(N, C, H, W)``.
        kernel_size: An ``int`` or a pair.
        stride: An ``int`` or a pair (default: ``kernel_size``).
        padding: An ``int`` or a pair.

    Returns:
        Shape ``(N, C, H_out, W_out)``.

    Example:
        >>> import glassnn.functional as F
        >>> from glassnn import Tensor
        >>> F.avg_pool2d(Tensor([[[[1.0, 2.0], [4.0, 3.0]]]]), kernel_size=2)
        Tensor([[[[2.5]]]])
    """
    return _pool(input, kernel_size, stride, padding, 2, "avg", "avg_pool2d")


# The helpers below work for any number n of spatial dimensions. An input has
# shape (N, C, *spatial); a "window view" has shape (N, C, *out, *kernel).

_SPATIAL_NAMES = {1: "(N, C, L)", 2: "(N, C, H, W)"}


def _ntuple(value: int | tuple[int, ...], n: int) -> tuple[int, ...]:
    """``3`` -> ``(3,) * n``; a tuple is returned as it is."""
    return (value,) * n if isinstance(value, int) else tuple(value)


def _check_input(input: Tensor, n: int, name: str) -> None:
    if input.ndim != n + 2:
        raise ValueError(
            f"{name} needs an input of shape {_SPATIAL_NAMES[n]}, got {input.shape}."
        )


def _conv_padding(
    padding: Any, extent: tuple[int, ...], stride: tuple[int, ...], n: int
) -> list[tuple[int, int]]:
    """The (left, right) zero padding of each spatial dimension."""
    if padding == "valid":
        return [(0, 0)] * n
    if padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("padding='same' needs stride 1.")
        # The kernel covers `extent` positions; extent - 1 zeros keep the length.
        return [((e - 1) // 2, e - 1 - (e - 1) // 2) for e in extent]
    if isinstance(padding, str):
        raise ValueError(
            f"padding must be an int, a tuple, 'valid' or 'same', got {padding!r}."
        )
    return [(p, p) for p in _ntuple(padding, n)]


def _windows(
    padded: Any,
    extent: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
) -> Any:
    """View of all windows, shape (N, C, *out, *kernel), without copying."""
    n = len(extent)
    view = backend.xp.lib.stride_tricks.sliding_window_view(
        padded, extent, axis=tuple(range(2, 2 + n))
    )
    keep = tuple(slice(None, None, s) for s in stride)
    spaced = tuple(slice(None, None, d) for d in dilation)
    return view[(slice(None), slice(None), *keep, *spaced)]


def _offset_slices(
    offset: tuple[int, ...],
    out_shape: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[slice, ...]:
    """Positions s * i + d * k of the input, for all output positions i."""
    return tuple(
        slice(d * k, d * k + s * (m - 1) + 1, s)
        for k, m, s, d in zip(offset, out_shape, stride, dilation, strict=True)
    )


def _check_length(
    padded_shape: tuple[int, ...], extent: tuple[int, ...], input: Tensor, other: Any
) -> None:
    if any(size < e for size, e in zip(padded_shape, extent, strict=True)):
        raise ValueError(
            f"The input is too short for the kernel: input of shape {input.shape}, "
            f"{other}."
        )


def _conv(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Any,
    padding: Any,
    dilation: Any,
    n: int,
    name: str,
) -> Tensor:
    """Convolution with n spatial dimensions; see conv1d for the formulas."""
    xp = backend.xp
    _check_input(input, n, name)
    if weight.ndim != n + 2 or weight.shape[1] != input.shape[1]:
        raise ValueError(
            f"{name}: an input of shape {input.shape} needs a weight of shape "
            f"(C_out, {input.shape[1]}, ...) with {n} kernel dimensions, "
            f"got {weight.shape}."
        )
    if bias is not None and bias.shape != weight.shape[:1]:
        raise ValueError(
            f"{name}: a weight of shape {weight.shape} needs a bias of shape "
            f"{weight.shape[:1]}, got {bias.shape}."
        )
    kernel = weight.shape[2:]
    stride, dilation = _ntuple(stride, n), _ntuple(dilation, n)
    extent = tuple(d * (k - 1) + 1 for k, d in zip(kernel, dilation, strict=True))
    pads = _conv_padding(padding, extent, stride, n)
    padded = xp.pad(input.data, [(0, 0), (0, 0), *pads])
    _check_length(padded.shape[2:], extent, input, f"weight of shape {weight.shape}")
    windows = _windows(padded, extent, stride, dilation)
    out_shape = windows.shape[2 : 2 + n]

    # Contract channels and kernel offsets: (N, *out, C_out) -> (N, C_out, *out).
    window_axes = [1, *range(2 + n, 2 + 2 * n)]
    weight_axes = list(range(1, 2 + n))
    out = xp.moveaxis(
        xp.tensordot(windows, weight.data, (window_axes, weight_axes)), -1, 1
    )
    if bias is not None:
        out = out + bias.data.reshape(-1, *([1] * n))

    def backward(grad):
        batch_and_out = [0, *range(2, 2 + n)]
        grad_weight = xp.tensordot(grad, windows, (batch_and_out, batch_and_out))
        grad_padded = xp.zeros(padded.shape, dtype=padded.dtype)
        for offset in itertools.product(*[range(k) for k in kernel]):
            w_k = weight.data[(slice(None), slice(None), *offset)]  # (C_out, C_in)
            contribution = xp.moveaxis(xp.tensordot(grad, w_k, ([1], [0])), -1, 1)
            where = _offset_slices(offset, out_shape, stride, dilation)
            grad_padded[(slice(None), slice(None), *where)] += contribution
        crop = tuple(
            slice(left, left + size)
            for (left, _), size in zip(pads, input.shape[2:], strict=True)
        )
        grad_input = grad_padded[(slice(None), slice(None), *crop)]
        if bias is None:
            return grad_input, grad_weight
        return grad_input, grad_weight, grad.sum(axis=tuple(batch_and_out))

    parents = (input, weight) if bias is None else (input, weight, bias)
    return _result(out, parents, backward, name)


def _pool(
    input: Tensor,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    n: int,
    mode: str,
    name: str,
) -> Tensor:
    """Max or average pooling with n spatial dimensions."""
    xp = backend.xp
    _check_input(input, n, name)
    kernel = _ntuple(kernel_size, n)
    stride = kernel if stride is None else _ntuple(stride, n)
    pads = _ntuple(padding, n)
    if any(2 * p > k for p, k in zip(pads, kernel, strict=True)):
        raise ValueError(
            f"{name}: the padding {pads} must be at most half the kernel size {kernel}."
        )
    fill = -xp.inf if mode == "max" else 0.0
    padded = xp.pad(
        input.data, [(0, 0), (0, 0), *[(p, p) for p in pads]], constant_values=fill
    )
    _check_length(padded.shape[2:], kernel, input, f"kernel size {kernel}")
    ones = (1,) * n
    windows = _windows(padded, kernel, stride, ones)
    out_shape = windows.shape[2 : 2 + n]
    flat = windows.reshape(*windows.shape[: 2 + n], -1)  # one axis for the window
    if mode == "max":
        argmax = flat.argmax(axis=-1)  # the first maximum
        out = xp.take_along_axis(flat, argmax[..., None], axis=-1)[..., 0]
    else:
        out = flat.mean(axis=-1)

    def backward(grad):
        grad_padded = xp.zeros(padded.shape, dtype=padded.dtype)
        offsets = itertools.product(*[range(k) for k in kernel])
        for index, offset in enumerate(offsets):
            share = grad * (argmax == index) if mode == "max" else grad / flat.shape[-1]
            where = _offset_slices(offset, out_shape, stride, ones)
            grad_padded[(slice(None), slice(None), *where)] += share
        crop = tuple(
            slice(p, p + size) for p, size in zip(pads, input.shape[2:], strict=True)
        )
        return (grad_padded[(slice(None), slice(None), *crop)],)

    return _result(out, (input,), backward, name)


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
