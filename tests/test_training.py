"""Behavioural tests: small networks learn, and training is reproducible."""

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import glassnn.functional as F
from glassnn import Tensor, backend, manual_seed, nn, no_grad, optim
from glassnn.data import DataLoader


def accuracy(logits, labels):
    return float((logits.data.argmax(axis=1) == labels).mean())


def test_mlp_learns_xor():
    manual_seed(0)
    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = Tensor([[0.0], [1.0], [1.0], [0.0]])
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(2000):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X), y)
        loss.backward()
        optimizer.step()
    predictions = model(X).data > 0
    np.testing.assert_array_equal(predictions, y.data == 1)
    assert loss.item() < 0.01


def train_moons(seed, epochs):
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    loader = DataLoader(X_train, y_train, batch_size=32, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    with no_grad():
        test_accuracy = accuracy(model(Tensor(X_test)), y_test)
    return model, test_accuracy


def test_mlp_learns_two_moons():
    _, test_accuracy = train_moons(seed=0, epochs=60)
    assert test_accuracy >= 0.95


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_training_is_bit_for_bit_reproducible(dtype):
    backend.set_default_dtype(dtype)
    first, _ = train_moons(seed=1, epochs=3)
    second, _ = train_moons(seed=1, epochs=3)
    for (name, p), (_, q) in zip(
        first.named_parameters(), second.named_parameters(), strict=True
    ):
        assert p.data.tobytes() == q.data.tobytes(), name


def test_float32_training_stays_in_float32():
    backend.set_default_dtype("float32")
    model, _ = train_moons(seed=2, epochs=1)
    for name, p in model.named_parameters():
        assert p.dtype == np.float32, name
        assert p.grad.dtype == np.float32, name
