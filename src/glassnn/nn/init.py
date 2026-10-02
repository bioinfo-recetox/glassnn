r"""Weight initialization.

Each function overwrites the values of a tensor (``tensor.data``) and returns
the tensor; the trailing underscore marks this, as in PyTorch. Together with
optimizer steps and :meth:`~glassnn.nn.Module.load_state_dict`, these are
the only places where GlassNN changes values in place.

Initialization keeps the variance of the signal roughly constant from layer
to layer. For a layer with :math:`n_\text{in}` inputs and
:math:`n_\text{out}` outputs:

- Xavier/Glorot :cite:p:`glorot2010understanding`:
  :math:`\operatorname{Var}(w) = g^2\, \frac{2}{n_\text{in} + n_\text{out}}`,
  a compromise between the forward and the backward pass.
- Kaiming/He :cite:p:`he2015delving`:
  :math:`\operatorname{Var}(w) = g^2 / n`, with :math:`n = n_\text{in}`
  (forward) or :math:`n_\text{out}` (backward), and the gain
  :math:`g = \sqrt 2` compensating ReLU, which zeroes half of the signal.

A uniform distribution on :math:`[-b, b]` has variance :math:`b^2 / 3`, so
:math:`b = \sqrt 3 \cdot \text{std}`.

Every function takes an optional ``generator``; by default it uses the
global generator of :func:`glassnn.backend.manual_seed`.

Book chapter: :book:`Initialization and the flow of signals
<chapters/04-initialization.html>`.
"""

import math
from typing import Any

from glassnn import backend
from glassnn.tensor import Tensor

_GAINS = {
    "linear": 1.0,
    "conv1d": 1.0,
    "conv2d": 1.0,
    "sigmoid": 1.0,
    "tanh": 5.0 / 3,
    "relu": math.sqrt(2.0),
    "selu": 3.0 / 4,
}


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    r"""The recommended gain :math:`g` for a nonlinearity.

    ``"linear"``, ``"conv1d"``, ``"conv2d"``, ``"sigmoid"``: 1;
    ``"tanh"``: 5/3; ``"relu"``: :math:`\sqrt 2`; ``"selu"``: 3/4;
    ``"leaky_relu"`` with slope :math:`\alpha` (``param``, default 0.01):
    :math:`\sqrt{2 / (1 + \alpha^2)}`. The values are PyTorch's.

    Raises:
        ValueError: For an unknown nonlinearity.

    Example:
        >>> from glassnn.nn import init
        >>> round(init.calculate_gain("relu"), 4)
        1.4142
    """
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1 + slope**2))
    if nonlinearity not in _GAINS:
        raise ValueError(
            f"Unknown nonlinearity {nonlinearity!r}; expected one of "
            f"{sorted([*_GAINS, 'leaky_relu'])}."
        )
    return _GAINS[nonlinearity]


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Fan-in and fan-out of a weight of shape ``(out, in, *kernel)``."""
    if tensor.ndim < 2:
        raise ValueError(
            "Fan-in and fan-out need a tensor with at least 2 dimensions, "
            f"got shape {tensor.shape}."
        )
    receptive_field = math.prod(tensor.shape[2:])
    return tensor.shape[1] * receptive_field, tensor.shape[0] * receptive_field


def _generator(generator: Any) -> Any:
    return backend.get_generator() if generator is None else generator


def _fill(tensor: Tensor, values: Any) -> Tensor:
    tensor.data = backend.xp.asarray(values, dtype=tensor.dtype)
    return tensor


def uniform_(
    tensor: Tensor, a: float = 0.0, b: float = 1.0, generator: Any = None
) -> Tensor:
    """Fill with values drawn uniformly from :math:`[a, b)`."""
    return _fill(tensor, _generator(generator).uniform(a, b, size=tensor.shape))


def normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, generator: Any = None
) -> Tensor:
    r"""Fill with values drawn from :math:`\mathcal N(\text{mean}, \text{std}^2)`."""
    return _fill(tensor, _generator(generator).normal(mean, std, size=tensor.shape))


def zeros_(tensor: Tensor) -> Tensor:
    """Fill with zeros."""
    return _fill(tensor, backend.xp.zeros(tensor.shape))


def ones_(tensor: Tensor) -> Tensor:
    """Fill with ones."""
    return _fill(tensor, backend.xp.ones(tensor.shape))


def xavier_uniform_(tensor: Tensor, gain: float = 1.0, generator: Any = None) -> Tensor:
    r"""Xavier/Glorot uniform :cite:p:`glorot2010understanding`.

    .. math:: w \sim U[-b, b], \qquad
        b = g \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}

    Example:
        >>> from glassnn import nn
        >>> from glassnn.nn import init
        >>> w = init.xavier_uniform_(nn.Parameter([[0.0] * 3] * 2))
        >>> bool(abs(w.data).max() <= (6 / 5) ** 0.5)
        True
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound, generator)


def xavier_normal_(tensor: Tensor, gain: float = 1.0, generator: Any = None) -> Tensor:
    r"""Xavier/Glorot normal :cite:p:`glorot2010understanding`.

    .. math:: w \sim \mathcal N(0, \sigma^2), \qquad
        \sigma = g \sqrt{\frac{2}{n_\text{in} + n_\text{out}}}
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std, generator)


def _kaiming_std(tensor: Tensor, a: float, mode: str, nonlinearity: str) -> float:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}.")
    fan = fan_in if mode == "fan_in" else fan_out
    return calculate_gain(nonlinearity, a) / math.sqrt(fan)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Any = None,
) -> Tensor:
    r"""Kaiming/He uniform :cite:p:`he2015delving`.

    .. math:: w \sim U[-b, b], \qquad b = g \sqrt{\frac{3}{n}}

    with :math:`n` the fan-in or fan-out (``mode``) and :math:`g` the gain of
    ``nonlinearity`` (``a`` is the negative slope for ``"leaky_relu"``).

    Note:
        ``kaiming_uniform_(w, a=math.sqrt(5))`` gives :math:`g = \sqrt{1/3}`
        and hence :math:`b = 1 / \sqrt{n_\text{in}}`: the default of
        :class:`~glassnn.nn.Linear`, as in PyTorch.

    Raises:
        ValueError: For an unknown ``mode`` or ``nonlinearity``.
    """
    bound = math.sqrt(3.0) * _kaiming_std(tensor, a, mode, nonlinearity)
    return uniform_(tensor, -bound, bound, generator)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Any = None,
) -> Tensor:
    r"""Kaiming/He normal :cite:p:`he2015delving`.

    .. math:: w \sim \mathcal N(0, \sigma^2), \qquad \sigma = \frac{g}{\sqrt n}

    Raises:
        ValueError: For an unknown ``mode`` or ``nonlinearity``.
    """
    std = _kaiming_std(tensor, a, mode, nonlinearity)
    return normal_(tensor, 0.0, std, generator)
