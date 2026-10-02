r"""Learning-rate schedules.

A scheduler changes ``optimizer.lr`` once per call of :meth:`step`, usually
once per epoch, after ``optimizer.step()``:

.. code-block:: python

    for epoch in range(epochs):
        train_one_epoch(model, optimizer)
        scheduler.step()

Each scheduler computes the rate in closed form from the initial rate
:math:`\eta_0` (``optimizer.lr`` when the scheduler is created) and the
number of calls :math:`t`.
"""

import math

from glassnn.optim.optimizer import Optimizer


class LRScheduler:
    """Base class: counts the steps and sets ``optimizer.lr``.

    Args:
        optimizer: The optimizer whose ``lr`` is changed.
    """

    def __init__(self, optimizer: Optimizer) -> None:
        """Remember the initial rate and set the rate for step 0."""
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = 0
        optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        """The learning rate after ``self.last_epoch`` steps."""
        raise NotImplementedError

    def step(self) -> None:
        """Advance by one step and update ``optimizer.lr``."""
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_last_lr(self) -> float:
        """The current learning rate."""
        return self.optimizer.lr


class StepLR(LRScheduler):
    r"""Multiply the rate by ``gamma`` every ``step_size`` steps.

    .. math:: \eta_t = \eta_0\, \gamma^{\lfloor t / s \rfloor}

    Example:
        >>> from glassnn import nn, optim
        >>> opt = optim.SGD([nn.Parameter([0.0])], lr=1.0)
        >>> scheduler = optim.StepLR(opt, step_size=2, gamma=0.1)
        >>> rates = []
        >>> for _ in range(4):
        ...     rates.append(opt.lr)
        ...     scheduler.step()
        >>> [round(r, 4) for r in rates]
        [1.0, 1.0, 0.1, 0.1]
    """

    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1
    ) -> None:
        """Store the period ``step_size`` and the factor ``gamma``."""
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self) -> float:
        """See the class docstring."""
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class CosineAnnealingLR(LRScheduler):
    r"""Cosine decay from :math:`\eta_0` to ``eta_min`` in ``T_max`` steps.

    .. math:: \eta_t = \eta_\min + \frac{\eta_0 - \eta_\min}{2}
        \left(1 + \cos \frac{\pi t}{T_\max}\right)

    :cite:p:`loshchilov2017sgdr` (without the restarts).
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        """Store the period ``T_max`` and the final rate ``eta_min``."""
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self) -> float:
        """See the class docstring."""
        cosine = math.cos(math.pi * self.last_epoch / self.T_max)
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + cosine) / 2


class LinearWarmup(LRScheduler):
    r"""Increase the rate linearly during the first ``warmup_steps`` steps.

    .. math:: \eta_t = \eta_0 \left(f_0 + (1 - f_0)
        \frac{\min(t, T)}{T}\right)

    with :math:`f_0` = ``start_factor`` and :math:`T` = ``warmup_steps``.

    Note:
        Differences from PyTorch: the same schedule is
        ``torch.optim.lr_scheduler.LinearLR(optimizer, start_factor,
        end_factor=1.0, total_iters=warmup_steps)``.

    Raises:
        ValueError: If ``warmup_steps < 1`` or ``start_factor`` is not in
            :math:`(0, 1]`.
    """

    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, start_factor: float = 1 / 3
    ) -> None:
        """Store the length of the warmup and the starting factor."""
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be at least 1, got {warmup_steps}.")
        if not 0 < start_factor <= 1:
            raise ValueError(f"start_factor must be in (0, 1], got {start_factor}.")
        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        super().__init__(optimizer)

    def get_lr(self) -> float:
        """See the class docstring."""
        progress = min(self.last_epoch, self.warmup_steps) / self.warmup_steps
        return self.base_lr * (self.start_factor + (1 - self.start_factor) * progress)
