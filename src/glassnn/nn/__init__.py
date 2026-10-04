"""Modules, parameters, layers, initializers and losses.

The names follow ``torch.nn``: ``from glassnn import nn`` and then
``nn.Linear``, ``nn.Sequential``, ``nn.init.kaiming_uniform_``...
"""

from glassnn.nn import init
from glassnn.nn.activation import (
    GELU,
    LeakyReLU,
    LogSoftmax,
    ReLU,
    Sigmoid,
    Softmax,
    Softplus,
    Tanh,
)
from glassnn.nn.conv import Conv1d, Conv2d
from glassnn.nn.dropout import Dropout
from glassnn.nn.linear import Linear
from glassnn.nn.loss import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from glassnn.nn.module import Module, Sequential
from glassnn.nn.normalization import BatchNorm1d, LayerNorm
from glassnn.nn.parameter import Parameter
from glassnn.nn.pooling import AvgPool1d, AvgPool2d, Flatten, MaxPool1d, MaxPool2d

__all__ = [
    "GELU",
    "AvgPool1d",
    "AvgPool2d",
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "Conv1d",
    "Conv2d",
    "CrossEntropyLoss",
    "Dropout",
    "Flatten",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LogSoftmax",
    "MSELoss",
    "MaxPool1d",
    "MaxPool2d",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "Tanh",
    "init",
]
