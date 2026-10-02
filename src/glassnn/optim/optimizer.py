"""The base class of all optimizers."""

from collections.abc import Iterable

from glassnn.tensor import Tensor


class Optimizer:
    """Base class: holds the parameters and the learning rate.

    A subclass implements :meth:`step`, which reads ``p.grad`` of every
    parameter ``p`` and replaces ``p.data`` by the updated values. Parameters
    whose ``grad`` is ``None`` are skipped.

    Args:
        params: The parameters to optimize, e.g. ``model.parameters()``.
        lr: The learning rate; schedulers change ``optimizer.lr``.

    Raises:
        ValueError: If there are no parameters or ``lr`` is negative.

    Note:
        Differences from PyTorch: there is one parameter group, and the
        learning rate is the attribute ``lr`` (PyTorch:
        ``optimizer.param_groups[0]["lr"]``).
    """

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        """Store the parameters and check the learning rate."""
        self.params = list(params)
        if not self.params:
            raise ValueError("The optimizer got no parameters.")
        _check_non_negative("lr", lr)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset the gradients of all parameters to ``None``."""
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        """Update the parameters once; every subclass defines it."""
        raise NotImplementedError


def _check_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def _check_fraction(name: str, value: float) -> None:
    if not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")
