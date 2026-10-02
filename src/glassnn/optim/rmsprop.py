"""RMSprop."""

from collections.abc import Iterable

from glassnn import backend
from glassnn.optim.optimizer import Optimizer, _check_fraction, _check_non_negative
from glassnn.tensor import Tensor


class RMSprop(Optimizer):
    r"""RMSprop :cite:p:`hinton2012lecture`.

    Divides the gradient by a running root mean square of recent gradients:

    .. math::

        v_t = \alpha v_{t-1} + (1 - \alpha)\, g_t^2, \qquad
        d_t = \frac{g_t}{\sqrt{v_t} + \epsilon}, \qquad
        \theta_t = \theta_{t-1} - \eta\, d_t .

    With momentum :math:`\mu > 0`, :math:`b_t = \mu b_{t-1} + d_t` replaces
    :math:`d_t`. Weight decay :math:`\lambda` adds :math:`\lambda \theta` to
    the gradient. These are PyTorch's formulas (without ``centered``).

    Args:
        params: The parameters to optimize.
        lr: The learning rate :math:`\eta`.
        alpha: The smoothing constant :math:`\alpha \in [0, 1)`.
        eps: :math:`\epsilon \ge 0`.
        weight_decay: :math:`\lambda \ge 0`.
        momentum: :math:`\mu \ge 0`.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        """Check the settings and create the running averages."""
        super().__init__(params, lr)
        _check_fraction("alpha", alpha)
        _check_non_negative("eps", eps)
        _check_non_negative("weight_decay", weight_decay)
        _check_non_negative("momentum", momentum)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.v = [backend.xp.zeros_like(p.data) for p in self.params]
        self.buffers = [backend.xp.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Update every parameter that has a gradient."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * g**2
            d = g / (backend.xp.sqrt(self.v[i]) + self.eps)
            if self.momentum:
                self.buffers[i] = self.momentum * self.buffers[i] + d
                d = self.buffers[i]
            p.data = p.data - self.lr * d
