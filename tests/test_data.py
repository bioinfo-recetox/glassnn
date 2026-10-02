"""Tests of DataLoader and one_hot."""

import numpy as np
import pytest

from glassnn import Tensor, backend
from glassnn.data import DataLoader, one_hot


@pytest.fixture
def xy():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10)
    return X, y


def test_batches_in_order_without_shuffle(xy):
    X, y = xy
    loader = DataLoader(X, y, batch_size=4)
    batches = list(loader)
    assert len(loader) == len(batches) == 3
    assert [len(yb) for _, yb in batches] == [4, 4, 2]
    xb, yb = batches[0]
    assert isinstance(xb, Tensor) and isinstance(yb, Tensor)
    np.testing.assert_array_equal(xb.data, X[:4])
    np.testing.assert_array_equal(yb.data, y[:4])


def test_dtypes_follow_the_tensor_rule(xy):
    X, y = xy
    backend.set_default_dtype("float32")
    xb, yb = next(iter(DataLoader(X, y, batch_size=4)))
    assert xb.dtype == np.float32
    assert yb.dtype.kind == "i"


def test_drop_last(xy):
    X, y = xy
    loader = DataLoader(X, y, batch_size=4, drop_last=True)
    assert len(loader) == 2
    assert [len(yb) for _, yb in loader] == [4, 4]


def test_shuffle_visits_every_sample_once_per_epoch(xy):
    X, y = xy
    loader = DataLoader(
        X, y, batch_size=3, shuffle=True, generator=np.random.default_rng(0)
    )
    first = np.concatenate([yb.data for _, yb in loader])
    second = np.concatenate([yb.data for _, yb in loader])
    assert sorted(first) == list(range(10))
    assert not np.array_equal(first, second)  # a new order each epoch
    for xb, yb in loader:
        np.testing.assert_array_equal(xb.data, X[yb.data])


def test_shuffle_is_reproducible_with_the_global_seed(xy):
    X, y = xy

    def order():
        backend.manual_seed(3)
        return np.concatenate([yb.data for _, yb in DataLoader(X, y, 4, shuffle=True)])

    np.testing.assert_array_equal(order(), order())


def test_inputs_only(xy):
    X, _ = xy
    batches = list(DataLoader(Tensor(X), batch_size=5))
    assert len(batches) == 2
    assert isinstance(batches[0], Tensor)
    np.testing.assert_array_equal(batches[1].data, X[5:])


def test_length_mismatch_names_both_lengths(xy):
    X, y = xy
    with pytest.raises(ValueError, match=r"10.*9"):
        DataLoader(X, y[:9])


def test_batch_size_must_be_positive(xy):
    X, y = xy
    with pytest.raises(ValueError, match="batch_size"):
        DataLoader(X, y, batch_size=0)


def test_one_hot():
    out = one_hot(Tensor([0, 2, 1]))
    np.testing.assert_array_equal(out.data, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert out.dtype == np.int64
    assert one_hot(np.array([1]), num_classes=4).shape == (1, 4)


def test_one_hot_errors():
    with pytest.raises(TypeError, match="integer"):
        one_hot(Tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="num_classes"):
        one_hot(Tensor([0, 5]), num_classes=3)
