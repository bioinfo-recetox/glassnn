"""Stochastic gradient descent, with momentum and Nesterov momentum."""

from collections.abc import Iterable

from glassnn.optim.optimizer import Optimizer, _check_non_negative
from glassnn.tensor import Tensor


class SGD(Optimizer):
    r"""Stochastic gradient descent :cite:p:`sutskever2013importance`.

    With gradient :math:`g_t` (plus :math:`\lambda \theta_{t-1}` for weight
    decay :math:`\lambda`), momentum :math:`\mu` and learning rate
    :math:`\eta`:

    .. math::

        b_t = \mu\, b_{t-1} + g_t \quad (b_1 = g_1), \qquad
        \theta_t = \theta_{t-1} - \eta\, d_t,

    with :math:`d_t = b_t` (heavy ball) or :math:`d_t = g_t + \mu b_t`
    (Nesterov). Without momentum, :math:`d_t = g_t`. These are PyTorch's
    formulas (with ``dampening=0``).

    Args:
        params: The parameters to optimize.
        lr: The learning rate :math:`\eta`.
        momentum: :math:`\mu \ge 0`.
        nesterov: Use Nesterov momentum (needs ``momentum > 0``).
        weight_decay: :math:`\lambda \ge 0` (L2 penalty added to the gradient).

    Example:
        >>> from glassnn import Tensor, nn, optim
        >>> p = nn.Parameter([1.0])
        >>> p.grad = Tensor([2.0])
        >>> optim.SGD([p], lr=0.1).step()
        >>> p
        Parameter([0.8], requires_grad=True)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        """Check the settings and create the momentum buffers."""
        super().__init__(params, lr)
        _check_non_negative("momentum", momentum)
        _check_non_negative("weight_decay", weight_decay)
        if nesterov and momentum == 0:
            raise ValueError("Nesterov momentum needs momentum > 0.")
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.buffers = [None] * len(self.params)

    def step(self) -> None:
        """Update every parameter that has a gradient."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                b = self.buffers[i]
                b = g if b is None else self.momentum * b + g
                self.buffers[i] = b
                g = g + self.momentum * b if self.nesterov else b
            p.data = p.data - self.lr * g
