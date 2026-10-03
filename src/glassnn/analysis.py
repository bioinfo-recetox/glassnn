r"""Analysis tools: parameter counts, activation statistics, empirical NTK.

Book chapters: :book:`Initialization <chapters/04-initialization.html>`
(activation statistics) and :book:`The neural tangent kernel
<chapters/08-ntk.html>`.
"""

from typing import Any, NamedTuple

from glassnn import backend
from glassnn.nn.linear import Linear
from glassnn.nn.module import Module, Sequential
from glassnn.nn.parameter import Parameter
from glassnn.tensor import Tensor, no_grad

__all__ = ["LayerStats", "activation_stats", "count_parameters", "empirical_ntk"]

# The layers whose outputs (pre-activations) activation_stats records.
_RECORDED_LAYERS: tuple[type[Module], ...] = (Linear,)


def count_parameters(model: Module) -> int:
    """The number of scalar parameters of a model.

    A parameter shared by several layers is counted once.

    Args:
        model: Any module.

    Returns:
        The sum of the sizes of all parameters.

    Example:
        >>> from glassnn import nn
        >>> from glassnn.analysis import count_parameters
        >>> count_parameters(nn.Linear(3, 2))
        8
    """
    return sum(int(parameter.data.size) for parameter in model.parameters())


class LayerStats(NamedTuple):
    """Mean and standard deviation of the output of one layer."""

    mean: float
    std: float


def activation_stats(model: Sequential, X: Any) -> dict[str, LayerStats]:
    r"""Mean and standard deviation of the pre-activations of every layer.

    Runs ``model`` on ``X`` (without recording a graph) and, for every
    :class:`~glassnn.nn.Linear` inside it, records the mean and the standard
    deviation of its output :math:`Z_l = H_{l-1} W_l^\top + b_l` over all
    samples and units. A well-initialized deep network keeps
    :math:`\operatorname{std}(Z_l)` roughly constant across layers
    :cite:p:`glorot2010understanding,he2015delving`.

    Args:
        model: A :class:`~glassnn.nn.Sequential`, possibly nested. The model
            is run in its current mode; call ``model.eval()`` first if it
            contains dropout or batch normalization.
        X: The inputs, a Tensor or an array of shape ``(N, ...)``.

    Returns:
        A dictionary from the dotted name of each linear layer (as in
        :meth:`~glassnn.nn.Module.named_parameters`, e.g. ``"0"`` or
        ``"3.0"``) to its :class:`LayerStats`, in the order of the forward
        pass.

    Raises:
        TypeError: If ``model`` is not a ``Sequential``.

    Example:
        >>> from glassnn import manual_seed, nn
        >>> from glassnn.analysis import activation_stats
        >>> _ = manual_seed(0)
        >>> model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
        >>> list(activation_stats(model, [[1.0, 2.0], [3.0, 4.0]]))
        ['0', '2']
    """
    if not isinstance(model, Sequential):
        raise TypeError(
            "activation_stats walks the layers of a Sequential and records the "
            f"outputs of its Linear layers; got a {type(model).__name__}."
        )
    stats: dict[str, LayerStats] = {}
    x = X if isinstance(X, Tensor) else Tensor(X)
    with no_grad():
        _run_and_record(model, x, "", stats)
    return stats


def _run_and_record(
    model: Sequential, x: Tensor, prefix: str, stats: dict[str, LayerStats]
) -> Tensor:
    """Apply the layers of ``model`` one by one, recording selected outputs."""
    for name, layer in model._modules.items():
        if isinstance(layer, Sequential):
            x = _run_and_record(layer, x, prefix + name + ".", stats)
            continue
        x = layer(x)
        if isinstance(layer, _RECORDED_LAYERS):
            values = backend.to_numpy(x.data)
            stats[prefix + name] = LayerStats(float(values.mean()), float(values.std()))
    return x


def empirical_ntk(
    model: Module, X1: Any, X2: Any = None, output_index: int | None = None
) -> Tensor:
    r"""The empirical neural tangent kernel of a model at its current parameters.

    For a network :math:`f(x; \theta)` with one output per example, the
    neural tangent kernel :cite:p:`jacot2018neural` is

    .. math:: \Theta(x, x') = \nabla_\theta f(x; \theta)^\top
        \nabla_\theta f(x'; \theta) .

    With the Jacobian :math:`J(X) \in \mathbb R^{n \times P}` whose row
    :math:`i` is :math:`\nabla_\theta f(x_i)^\top`, the kernel matrix is
    :math:`K = J(X_1)\, J(X_2)^\top`. Each row of :math:`J` is the gradient
    of a scalar, so it costs one backward pass: :math:`n_1 + n_2` passes,
    and memory :math:`O((n_1 + n_2) P)` for the two Jacobians.

    Under gradient flow on the squared loss, the outputs on the training
    set evolve as :math:`\dot f = -K (f - y)`; for very wide networks
    :math:`K` hardly changes during training ("lazy training"), and the
    network behaves like its linearization around the initial parameters
    :cite:p:`lee2019wide,chizat2019lazy`.

    Args:
        model: A module whose output for an input of shape ``(1, ...)`` has
            one element, or several of which ``output_index`` selects one.
            The model is run on one example at a time, in its current mode
            (call ``model.eval()`` first if it contains dropout or batch
            normalization).
        X1: The first inputs, shape ``(n1, ...)`` (Tensor or array).
        X2: The second inputs, shape ``(n2, ...)``; ``None`` means ``X1``
            (and computes the Jacobian once).
        output_index: The output whose kernel is computed, for models with
            several outputs per example (the flat index into the output of
            one example).

    Returns:
        :math:`K`, a Tensor of shape ``(n1, n2)`` that does not require
        gradients. Only parameters with ``requires_grad=True`` take part;
        their ``.grad`` is left as it was.

    Raises:
        ValueError: If the model has several outputs per example and
            ``output_index`` is ``None``.

    Note:
        PyTorch has no such function (``torch.func.jacrev`` plus a matrix
        product computes the same).

    Example:
        >>> from glassnn import nn
        >>> from glassnn.analysis import empirical_ntk
        >>> model = nn.Linear(2, 1)  # gradient (x, 1): K = x . x' + 1
        >>> empirical_ntk(model, [[1.0, 0.0], [0.0, 2.0]])
        Tensor([[2., 1.],
                [1., 5.]])
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    saved_grads = [p.grad for p in parameters]
    try:
        J1 = _jacobian(model, parameters, X1, output_index)
        J2 = J1 if X2 is None else _jacobian(model, parameters, X2, output_index)
    finally:
        for parameter, grad in zip(parameters, saved_grads, strict=True):
            parameter.grad = grad
    return Tensor(J1 @ J2.T, dtype=J1.dtype)


def _jacobian(
    model: Module, parameters: list[Parameter], X: Any, output_index: int | None
) -> Any:
    """One row per example: the gradient of its output, all parameters flattened."""
    x = X if isinstance(X, Tensor) else Tensor(X)
    rows = []
    for i in range(len(x)):
        output = model(x[i : i + 1]).reshape(-1)
        if output_index is None:
            if output.shape[0] != 1:
                raise ValueError(
                    f"The model has {output.shape[0]} outputs per example; "
                    "choose one with output_index."
                )
            output_index = 0
        for parameter in parameters:
            parameter.grad = None
        output[output_index].backward()
        rows.append(
            backend.xp.concatenate([_flat_grad(parameter) for parameter in parameters])
        )
    if not rows:
        size = sum(int(p.data.size) for p in parameters)
        return backend.xp.zeros((0, size), dtype=backend.get_default_dtype())
    return backend.xp.stack(rows)


def _flat_grad(parameter: Parameter) -> Any:
    """The gradient as a vector; zeros if the output does not depend on it."""
    if parameter.grad is None:
        return backend.xp.zeros(parameter.data.size, dtype=parameter.dtype)
    return parameter.grad.data.reshape(-1)
