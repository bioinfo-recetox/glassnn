"""Adam and AdamW."""

from collections.abc import Iterable

from glassnn import backend
from glassnn.optim.optimizer import Optimizer, _check_fraction, _check_non_negative
from glassnn.tensor import Tensor


class Adam(Optimizer):
    r"""Adam :cite:p:`kingma2015adam`, with L2 weight decay added to the gradient.

    With gradient :math:`g_t` (plus :math:`\lambda \theta_{t-1}` if
    ``weight_decay`` :math:`\lambda > 0`):

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad
        v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2, \\
        \hat m_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
        \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad
        \theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} .

    :math:`m` and :math:`v` are running means of the gradient and of its
    square; dividing by :math:`1 - \beta^t` removes their bias towards the
    initial value 0. Each coordinate's step is about :math:`\eta` in size,
    whatever the scale of its gradient.

    Args:
        params: The parameters to optimize.
        lr: The learning rate :math:`\eta`.
        betas: :math:`(\beta_1, \beta_2)`, each in :math:`[0, 1)`.
        eps: :math:`\epsilon \ge 0`, which avoids division by zero.
        weight_decay: :math:`\lambda \ge 0`; see :class:`AdamW` for why the
            decoupled form is usually better.

    Example:
        >>> from glassnn import Tensor, nn, optim
        >>> p = nn.Parameter([1.0])
        >>> p.grad = Tensor([5.0])
        >>> optim.Adam([p], lr=0.1).step()
        >>> round(p.item(), 4)
        0.9
    """

    #: ``False``: weight decay is added to the gradient (L2 regularization).
    decoupled_weight_decay = False

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Check the settings and create the running means."""
        super().__init__(params, lr)
        _check_fraction("betas[0]", betas[0])
        _check_fraction("betas[1]", betas[1])
        _check_non_negative("eps", eps)
        _check_non_negative("weight_decay", weight_decay)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [backend.xp.zeros_like(p.data) for p in self.params]
        self.v = [backend.xp.zeros_like(p.data) for p in self.params]
        self.steps = [0] * len(self.params)

    def step(self) -> None:
        """Update every parameter that has a gradient."""
        beta1, beta2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay and self.decoupled_weight_decay:
                p.data = p.data * (1 - self.lr * self.weight_decay)
            elif self.weight_decay:
                g = g + self.weight_decay * p.data
            self.steps[i] += 1
            t = self.steps[i]
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g**2
            m_hat = self.m[i] / (1 - beta1**t)
            v_hat = self.v[i] / (1 - beta2**t)
            p.data = p.data - self.lr * m_hat / (backend.xp.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    r"""Adam with decoupled weight decay :cite:p:`loshchilov2019decoupled`.

    Before the Adam step, the parameters shrink:
    :math:`\theta \leftarrow (1 - \eta \lambda)\, \theta`. In :class:`Adam`
    the decay term :math:`\lambda \theta` is added to the gradient and is
    then divided by :math:`\sqrt{\hat v}`, so parameters with large
    gradients are barely regularized; the decoupled form shrinks all
    parameters at the same rate.

    Args:
        params: The parameters to optimize.
        lr: The learning rate :math:`\eta`.
        betas: :math:`(\beta_1, \beta_2)`.
        eps: :math:`\epsilon`.
        weight_decay: :math:`\lambda` (default 0.01, as in PyTorch).
    """

    decoupled_weight_decay = True

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        """See :class:`Adam`; only the default weight decay differs."""
        super().__init__(params, lr, betas, eps, weight_decay)
