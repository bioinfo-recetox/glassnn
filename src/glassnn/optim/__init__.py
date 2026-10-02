"""Optimizers, learning-rate schedulers and gradient clipping.

The names follow ``torch.optim``: ``optim.SGD``, ``optim.Adam``,
``optim.AdamW``, ``optim.RMSprop``, and the schedulers ``optim.StepLR``,
``optim.CosineAnnealingLR`` and ``optim.LinearWarmup`` (PyTorch keeps
schedulers in ``torch.optim.lr_scheduler``; they are also importable from
``glassnn.optim.lr_scheduler``).

Book chapter: :book:`Optimization <chapters/05-optimization.html>`.
"""

from glassnn.optim.adam import Adam, AdamW
from glassnn.optim.clip_grad import clip_grad_norm_
from glassnn.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmup,
    LRScheduler,
    StepLR,
)
from glassnn.optim.optimizer import Optimizer
from glassnn.optim.rmsprop import RMSprop
from glassnn.optim.sgd import SGD

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "CosineAnnealingLR",
    "LRScheduler",
    "LinearWarmup",
    "Optimizer",
    "RMSprop",
    "StepLR",
    "clip_grad_norm_",
]
