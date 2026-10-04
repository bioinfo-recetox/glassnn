"""The expected findings of the example notebooks 01 to 06, at a small scale.

Each test reproduces the experiment of one notebook in ``examples/`` with
the same recipe (data, model, optimizer) but fewer repetitions, and asserts
the finding that the notebook and the book state. The notebooks themselves
are executed by a separate CI job (``examples/_execute.py``).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC

import glassnn.functional as F
from glassnn import Tensor, backend, manual_seed, nn, no_grad, optim
from glassnn.analysis import activation_stats, empirical_ntk
from glassnn.data import DataLoader
from glassnn.gradcheck import gradcheck
from glassnn.nn import init

# --------------------------------------------------------------------------
# 01_autodiff: backpropagation by hand agrees with Tensor.backward
# --------------------------------------------------------------------------


def test_01_manual_backprop_of_a_two_layer_mlp_matches_autodiff(rng):
    X = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 1))
    W1 = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    b1 = Tensor(rng.normal(size=4), requires_grad=True)
    W2 = Tensor(rng.normal(size=(1, 4)), requires_grad=True)
    b2 = Tensor(rng.normal(size=1), requires_grad=True)

    def loss_fn(W1, b1, W2, b2):
        h = F.tanh(F.linear(Tensor(X), W1, b1))
        return F.mse_loss(F.linear(h, W2, b2), Tensor(y))

    loss_fn(W1, b1, W2, b2).backward()
    # The backward pass written out with the adjoint notation of chapter 3.
    z1 = X @ W1.data.T + b1.data
    h = np.tanh(z1)
    out = h @ W2.data.T + b2.data
    out_bar = 2 * (out - y) / out.size
    h_bar = out_bar @ W2.data
    z1_bar = h_bar * (1 - h**2)
    np.testing.assert_allclose(W2.grad.data, out_bar.T @ h)
    np.testing.assert_allclose(b2.grad.data, out_bar.sum(axis=0))
    np.testing.assert_allclose(W1.grad.data, z1_bar.T @ X)
    np.testing.assert_allclose(b1.grad.data, z1_bar.sum(axis=0))
    assert gradcheck(loss_fn, [W1, b1, W2, b2])


# --------------------------------------------------------------------------
# 02_init_and_depth: only Kaiming keeps the scale of a deep ReLU network
# --------------------------------------------------------------------------

INITIALIZERS = {
    "small": lambda w: init.normal_(w, std=0.01),
    "default": lambda w: init.kaiming_uniform_(w, a=np.sqrt(5)),
    "xavier": init.xavier_normal_,
    "kaiming": lambda w: init.kaiming_normal_(w, nonlinearity="relu"),
    "large": lambda w: init.normal_(w, std=0.1),
}


def std_ratio_last_to_first(initializer, width=256, depth=20):
    manual_seed(0)
    layers = []
    for _ in range(depth):
        layer = nn.Linear(width, width)
        initializer(layer.weight)
        init.zeros_(layer.bias)
        layers += [layer, nn.ReLU()]
    x = np.random.default_rng(0).normal(size=(256, width))
    stats = list(activation_stats(nn.Sequential(*layers), x).values())
    return stats[-1].std / stats[0].std


def test_02_signal_scale_through_depth():
    ratios = {name: std_ratio_last_to_first(f) for name, f in INITIALIZERS.items()}
    # Per layer, std is multiplied by sqrt(n * Var(w) / 2): 1 only for Kaiming.
    assert 0.5 < ratios["kaiming"] < 2.0
    assert ratios["xavier"] < 1e-2  # (1 / sqrt 2)^19
    assert ratios["default"] < 1e-5  # (1 / sqrt 6)^19
    assert ratios["small"] < 1e-15
    assert ratios["large"] > 5.0


# --------------------------------------------------------------------------
# 03_double_descent: random ReLU features with the minimum-norm readout
# --------------------------------------------------------------------------


def sparse_regression(n, seed, d=20, k=5, noise=0.1):
    rng = np.random.default_rng(seed)
    beta = np.zeros(d)
    beta[:k] = 1 / np.sqrt(k)
    X = rng.normal(size=(n, d))
    return X, X @ beta + noise * rng.normal(size=n)


def random_features_errors(p, seed, n_train=100):
    X_train, y_train = sparse_regression(n_train, seed=0)
    X_test, y_test = sparse_regression(2000, seed=1)
    manual_seed(seed)
    features = nn.Sequential(nn.Linear(X_train.shape[1], p), nn.ReLU())
    with no_grad():
        phi_train = features(Tensor(X_train)).data
        phi_test = features(Tensor(X_test)).data
    theta = np.linalg.pinv(phi_train) @ y_train  # minimum-norm least squares
    train_error = np.mean((phi_train @ theta - y_train) ** 2)
    test_error = np.mean((phi_test @ theta - y_test) ** 2)
    return train_error, test_error


def test_03_double_descent_peaks_at_the_interpolation_threshold():
    widths = [10, 25, 50, 100, 200, 400, 3000]
    errors = {
        p: np.median([random_features_errors(p, seed) for seed in range(5)], axis=0)
        for p in widths
    }
    train = {p: e[0] for p, e in errors.items()}
    test = {p: e[1] for p, e in errors.items()}
    assert all(train[p] < 1e-12 for p in widths if p >= 100)  # interpolation
    assert test[100] > 10 * test[50] and test[100] > 10 * test[200]
    best_underparametrized = min(test[p] for p in widths if p < 100)
    assert test[3000] < 0.5 * best_underparametrized


# --------------------------------------------------------------------------
# 04_ntk_lazy_training: the wider the network, the less its NTK changes
# --------------------------------------------------------------------------


def ntk_and_parameter_change(width, seed, steps=500, lr=0.5):
    rng = np.random.default_rng(0)
    X = np.hstack([rng.uniform(-1, 1, size=(20, 1)), np.full((20, 1), 0.5)])
    y = np.sin(3 * X[:, :1])
    manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, width, parametrization="ntk"),
        nn.ReLU(),
        nn.Linear(width, 1, parametrization="ntk"),
    )
    K0 = empirical_ntk(model, X).data
    theta0 = np.concatenate([p.data.ravel() for p in model.parameters()])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        (0.5 * F.mse_loss(model(Tensor(X)), Tensor(y))).backward()
        optimizer.step()
    K1 = empirical_ntk(model, X).data
    theta1 = np.concatenate([p.data.ravel() for p in model.parameters()])
    with no_grad():
        train_error = F.mse_loss(model(Tensor(X)), Tensor(y)).item()
    return (
        np.linalg.norm(K1 - K0) / np.linalg.norm(K0),
        np.linalg.norm(theta1 - theta0) / np.linalg.norm(theta0),
        train_error,
    )


def test_04_lazy_training_in_wide_networks():
    narrow = np.median([ntk_and_parameter_change(10, s) for s in range(3)], axis=0)
    wide = np.median([ntk_and_parameter_change(1000, s) for s in range(3)], axis=0)
    assert narrow[0] > 0.1  # a narrow network changes its kernel
    assert wide[0] < 0.15 * narrow[0]
    assert wide[1] < 0.25 * narrow[1]
    assert wide[2] < 0.1  # ... and yet the wide network fits the data


# --------------------------------------------------------------------------
# 05_memorizing_noise: random labels are memorized; regularization helps
# --------------------------------------------------------------------------


class Memorization:
    """The data and the network of notebook 05."""

    d, n = 20, 200

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.w_star = self.rng.normal(size=self.d)
        self.X_train, self.y_train = self.sample(self.n)
        self.X_test, self.y_test = self.sample(2000)
        self.y_random = self.rng.integers(0, 2, size=self.n)

    def sample(self, n):
        X = self.rng.normal(size=(n, self.d))
        return X, (X @ self.w_star > 0).astype(np.int64)

    @staticmethod
    def accuracy(model, X, y):
        model.eval()
        with no_grad():
            value = float((model(Tensor(X)).data.argmax(axis=1) == y).mean())
        model.train()
        return value

    def train(self, labels, epochs, weight_decay=0.0, X_val=None, y_val=None):
        """Train; return (train accuracy, validation accuracy) per epoch."""
        X_val = self.X_test if X_val is None else X_val
        y_val = self.y_test if y_val is None else y_val
        manual_seed(0)
        model = nn.Sequential(
            nn.Linear(self.d, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        loader = DataLoader(self.X_train, labels, batch_size=32, shuffle=True)
        history = []
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                optimizer.step()
            history.append(
                (
                    self.accuracy(model, self.X_train, labels),
                    self.accuracy(model, X_val, y_val),
                )
            )
        return history


@pytest.fixture
def memorization():
    backend.set_default_dtype("float32")
    return Memorization()


def first_perfect_epoch(history):
    return next(epoch for epoch, (train, _) in enumerate(history) if train == 1.0)


def test_05_random_labels_are_memorized_but_later(memorization):
    true = memorization.train(memorization.y_train, epochs=40)
    random = memorization.train(memorization.y_random, epochs=40)
    assert random[-1][0] == 1.0  # every random label is fitted
    assert abs(random[-1][1] - 0.5) < 0.05  # test accuracy at chance
    assert true[-1][1] > 0.85
    assert first_perfect_epoch(random) > 2 * first_perfect_epoch(true)


def test_05_strong_weight_decay_prevents_memorization_but_keeps_the_signal(
    memorization,
):
    random = memorization.train(memorization.y_random, epochs=100, weight_decay=50.0)
    true = memorization.train(memorization.y_train, epochs=100, weight_decay=50.0)
    assert random[-1][0] < 0.6
    assert true[-1][0] > 0.9 and true[-1][1] > 0.85


def test_05_early_stopping_beats_training_to_zero_error_on_noisy_labels(
    memorization,
):
    flipped = memorization.rng.uniform(size=memorization.n) < 0.3
    noisy = np.where(flipped, 1 - memorization.y_train, memorization.y_train)
    X_val, y_val = memorization.sample(1000)
    history = memorization.train(noisy, epochs=100, X_val=X_val, y_val=y_val)
    validation = [v for _, v in history]
    best = int(np.argmax(validation))
    assert history[-1][0] == 1.0  # the noise is memorized
    assert history[best][0] < 1.0  # ... but not yet at the best epoch
    assert validation[best] > validation[-1] + 0.015


# --------------------------------------------------------------------------
# 06_cnn_dna_motif: a 1-D CNN finds a planted motif
# --------------------------------------------------------------------------

_PATH = Path(__file__).resolve().parents[1] / "examples" / "dna_motifs.py"
_spec = importlib.util.spec_from_file_location("dna_motifs", _PATH)
dna_motifs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dna_motifs)


def motif_data(mutation_rate):
    sequences, labels, positions = dna_motifs.make_motif_dataset(
        2000, mutation_rate=mutation_rate, seed=1
    )
    return sequences[:1000], sequences[1000:], labels[:1000], labels[1000:], positions


def train_motif_cnn(train_sequences, train_labels, seed=0):
    """Conv1d (16 filters of width 8) -> ReLU -> global max pool -> Linear."""
    X = dna_motifs.one_hot(train_sequences)
    manual_seed(seed)
    model = nn.Sequential(
        nn.Conv1d(4, 16, kernel_size=8),
        nn.ReLU(),
        nn.MaxPool1d(X.shape[2] - 8 + 1),
        nn.Flatten(),
        nn.Linear(16, 2),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    for _ in range(30):
        for xb, yb in DataLoader(X, train_labels, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()
    return model


def cnn_accuracy(model, sequences, labels):
    with no_grad():
        logits = model(Tensor(dna_motifs.one_hot(sequences))).data
    return float((logits.argmax(axis=1) == labels).mean())


@pytest.fixture
def float32():
    backend.set_default_dtype("float32")


def test_06_cnn_detects_and_locates_the_exact_motif(float32):
    train_s, test_s, train_y, test_y, positions = motif_data(0.0)
    model = train_motif_cnn(train_s, train_y)
    assert cnn_accuracy(model, test_s, test_y) > 0.95
    # The filter that votes most for "motif" fires at a fixed offset from it.
    votes = model[4].weight.data[1] - model[4].weight.data[0]
    best = int(np.argmax(votes))
    positives = test_y == 1
    with no_grad():
        responses = model[0](Tensor(dna_motifs.one_hot(test_s))).data[positives, best]
    offsets = responses.argmax(axis=1) - positions[1000:][positives]
    values, counts = np.unique(offsets, return_counts=True)
    assert counts.max() / counts.sum() > 0.9
    assert abs(values[counts.argmax()]) < 8  # the window overlaps the motif


def best_svm_accuracy(train_features, train_y, test_features, test_y, Cs):
    return max(
        SVC(kernel="linear", C=C)
        .fit(train_features, train_y)
        .score(test_features, test_y)
        for C in Cs
    )


def test_06_cnn_beats_the_string_kernels_on_mutated_motifs(float32):
    train_s, test_s, train_y, test_y, _ = motif_data(0.15)
    cnn = cnn_accuracy(train_motif_cnn(train_s, train_y), test_s, test_y)
    spectrum = max(
        best_svm_accuracy(
            dna_motifs.kmer_counts(train_s, k),
            train_y,
            dna_motifs.kmer_counts(test_s, k),
            test_y,
            Cs=(0.001, 0.01, 0.1),
        )
        for k in (3, 4, 5)  # k = 6 is slower and not better here
    )
    mismatch = best_svm_accuracy(
        dna_motifs.mismatch_counts(train_s, 6, 1),
        train_y,
        dna_motifs.mismatch_counts(test_s, 6, 1),
        test_y,
        Cs=(0.001, 0.01),
    )
    # Allowing one mismatch per 6-mer closes part of the gap, not all of it.
    assert spectrum < mismatch
    assert cnn > mismatch + 0.02


# --------------------------------------------------------------------------
# 07_attention_dna_motif: a transformer encoder on the same motif data
# --------------------------------------------------------------------------


class TokenTransformer(nn.Module):
    """Tokens (with a CLS token first) -> embedding + position -> encoder."""

    def __init__(self, vocabulary, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(vocabulary, d_model)
        self.position = nn.PositionalEncoding(d_model, max_len=100)
        layer = nn.TransformerEncoderLayer(d_model, 2, 64, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(d_model))
        self.head = nn.Linear(d_model, 2)

    def forward(self, tokens):
        return self.head(self.encoder(self.position(self.embed(tokens)))[:, 0])

    def cls_attention(self, tokens):
        """Attention weights of the CLS query in the (only) layer."""
        layer = self.encoder.layers[0]
        with no_grad():
            h = layer.norm1(self.position(self.embed(tokens)))
            _, weights = layer.self_attn(h, h, h)
        return weights.data[:, 0]


class HybridTransformer(nn.Module):
    """Conv1d stem -> position -> encoder -> mean over positions."""

    def __init__(self, d_model=32):
        super().__init__()
        self.stem = nn.Conv1d(4, d_model, 5, padding="same")
        self.position = nn.PositionalEncoding(d_model, max_len=100)
        layer = nn.TransformerEncoderLayer(d_model, 2, 64, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(d_model))
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.position(self.stem(x).transpose(1, 2))  # (N, L, d)
        return self.head(self.encoder(h).mean(dim=1))


def with_cls(tokens, cls_id):
    return np.hstack([np.full((len(tokens), 1), cls_id), tokens])


def fit(model, X, y, epochs, lr=3e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in DataLoader(X, y, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()
    return model


def accuracy_on(model, X, y):
    with no_grad():
        return float((model(Tensor(X)).data.argmax(axis=1) == y).mean())


def test_07_single_base_tokens_fail_and_3mer_tokens_work(float32):
    train_s, test_s, train_y, test_y, positions = motif_data(0.0)
    single = [with_cls(dna_motifs.kmer_tokens(s, 1), 4) for s in (train_s, test_s)]
    manual_seed(0)
    model = fit(TokenTransformer(5), single[0], train_y, epochs=10)
    assert accuracy_on(model, single[1], test_y) < 0.6

    triple = [with_cls(dna_motifs.kmer_tokens(s, 3), 64) for s in (train_s, test_s)]
    manual_seed(0)
    model = fit(TokenTransformer(65), triple[0], train_y, epochs=10)
    assert accuracy_on(model, triple[1], test_y) > 0.8
    # The CLS token attends to the six 3-mers inside the motif.
    positives = np.flatnonzero(test_y == 1)
    weights = model.cls_attention(Tensor(triple[1][positives]))
    starts = positions[1000:][positives] + 1  # +1: the CLS token comes first
    on_motif = [w[p : p + 6].sum() for w, p in zip(weights, starts, strict=True)]
    assert np.mean(on_motif) > 3 * 6 / weights.shape[1]


def test_07_a_convolutional_stem_reaches_the_cnn(float32):
    train_s, test_s, train_y, test_y, _ = motif_data(0.0)
    manual_seed(0)
    model = fit(HybridTransformer(), dna_motifs.one_hot(train_s), train_y, epochs=10)
    assert accuracy_on(model, dna_motifs.one_hot(test_s), test_y) > 0.95


# --------------------------------------------------------------------------
# 08_autoencoder_expression: a 2-D autoencoder against PCA and kernel PCA
# --------------------------------------------------------------------------

_PATH = Path(__file__).resolve().parents[1] / "examples" / "expression.py"
_spec = importlib.util.spec_from_file_location("expression", _PATH)
expression = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(expression)


def standardized_expression():
    X_train, _, _ = expression.make_expression_data(1000, seed=0)
    X_test, _, _ = expression.make_expression_data(1000, seed=10)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def mean_squared_error(a, b):
    return float(((a - b) ** 2).mean())


def test_08_a_2d_autoencoder_beats_2d_pca_and_kernel_pca(float32):
    X_train, X_test = standardized_expression()
    manual_seed(0)
    encoder = nn.Sequential(nn.Linear(200, 64), nn.GELU(), nn.Linear(64, 2))
    decoder = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 200))
    model = nn.Sequential(encoder, decoder)
    with no_grad():
        initial = mean_squared_error(X_test, model(Tensor(X_test)).data)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(60):
        for xb, _ in DataLoader(
            X_train, np.zeros(len(X_train)), batch_size=64, shuffle=True
        ):
            optimizer.zero_grad()
            F.mse_loss(model(xb), xb).backward()
            optimizer.step()
    with no_grad():
        autoencoder = mean_squared_error(X_test, model(Tensor(X_test)).data)

    def pca_error(k):
        pca = PCA(k).fit(X_train)
        return mean_squared_error(X_test, pca.inverse_transform(pca.transform(X_test)))

    kernel_pca = min(
        mean_squared_error(X_test, kp.inverse_transform(kp.transform(X_test)))
        for gamma in (0.001, 0.003, 0.01)
        for kp in [
            KernelPCA(
                2, kernel="rbf", gamma=gamma, fit_inverse_transform=True, alpha=0.1
            ).fit(X_train)
        ]
    )
    assert autoencoder < initial / 3  # training reduces the reconstruction error
    assert autoencoder < 0.8 * pca_error(2)
    assert autoencoder < pca_error(3)  # as good as PCA with more components
    # The approximate pre-image of kernel PCA reconstructs worse than PCA.
    assert kernel_pca > pca_error(2)
