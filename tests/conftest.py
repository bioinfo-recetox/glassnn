"""Shared fixtures: every test runs in float64 and restores the default dtype."""

import numpy as np
import pytest

from glassnn import backend


@pytest.fixture(autouse=True)
def float64_default():
    previous = backend.get_default_dtype()
    backend.set_default_dtype("float64")
    yield
    backend.set_default_dtype(previous)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
