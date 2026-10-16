"""Backend parity: NumPy and CuPy give the same forward and backward results.

Each check builds the same computation twice, once per backend, from the same
NumPy inputs and the same seed, and compares every returned array on the host.
Random numbers come from the NumPy generator on both backends (PLAN, decision
41), so models, dropout masks and shuffling are identical too.

Run with ``uv run pytest -m gpu`` on a machine with CuPy and a CUDA GPU.
"""

import numpy as np
import pytest

import glassnn.functional as F
from glassnn import Tensor, analysis, backend, manual_seed, nn, optim
from glassnn.data import DataLoader
from glassnn.gradcheck import gradcheck

pytestmark = pytest.mark.gpu

# Differences come only from the order of floating-point additions.
FLOAT64 = {"rtol": 1e-10, "atol": 1e-12}
FLOAT32 = {"rtol": 1e-4, "atol": 1e-5}


@pytest.fixture(autouse=True)
def cupy():
    """Skip without CuPy; always return to the NumPy backend afterwards."""
    module = pytest.importorskip("cupy", reason="the CuPy backend needs CuPy")
    yield module
    backend.set_backend("numpy")


def run_on(name, build):
    """Run ``build()`` with backend ``name``; return its arrays on the host."""
    backend.set_backend(name)
    try:
        return [backend.to_numpy(array) for array in build()]
    finally:
        backend.set_backend("numpy")


def assert_parity(build, tolerance=FLOAT64):
    expected = run_on("numpy", build)
    actual = run_on("cupy", build)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, **tolerance)


def forward_backward(fn, *arrays):
    """The output of ``fn`` and the gradients of all its (float) inputs."""
    inputs = [Tensor(array, requires_grad=True) for array in arrays]
    output = fn(*inputs)
    weights = np.cos(np.arange(1, output.data.size + 1)).reshape(output.shape)
    (output * Tensor(weights)).sum().backward()
    return [output.data] + [x.grad.data for x in inputs]


def random_arrays(*shapes):
    rng = np.random.default_rng(0)
    return [rng.normal(size=shape) for shape in shapes]


# ----------------------------------------------------------------------
# The backend itself
# ----------------------------------------------------------------------


def test_set_backend_cupy_switches_the_array_module(cupy):
    backend.set_backend("cupy")
    assert backend.get_backend() == "cupy"
    assert backend.xp is cupy
    x = Tensor([1.0, 2.0])
    assert isinstance(x.data, cupy.ndarray)
    assert x.dtype == backend.get_default_dtype()


def test_tensors_on_the_gpu_print_and_convert(cupy):
    backend.set_backend("cupy")
    x = Tensor([1.0, 2.0], requires_grad=True)
    assert repr(x) == "Tensor([1., 2.], requires_grad=True)"
    assert x.sum().item() == 3.0
    assert isinstance(x.numpy(), np.ndarray)


def test_to_numpy_copies_a_cupy_array_to_the_host(cupy):
    out = backend.to_numpy(cupy.arange(3.0))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [0.0, 1.0, 2.0])


def test_erf_runs_on_the_gpu(cupy):
    from scipy import special

    backend.set_backend("cupy")
    x = np.linspace(-4, 4, 17)
    out = backend.erf(cupy.asarray(x))
    assert isinstance(out, cupy.ndarray)
    np.testing.assert_allclose(out.get(), special.erf(x), rtol=1e-14)


def test_from_dlpack_shares_gpu_memory(cupy):
    backend.set_backend("cupy")
    x = cupy.arange(4.0)
    y = backend.from_dlpack(x)
    assert isinstance(y, cupy.ndarray)
    assert y.data.ptr == x.data.ptr


def test_to_dlpack_round_trip_on_the_gpu(cupy):
    backend.set_backend("cupy")
    x = cupy.arange(4.0)
    y = cupy.from_dlpack(backend.to_dlpack(x))
    assert y.data.ptr == x.data.ptr


def test_a_gpu_tensor_is_not_silently_used_by_the_numpy_backend(cupy):
    backend.set_backend("cupy")
    x = Tensor([1.0, 2.0])
    backend.set_backend("numpy")
    with pytest.raises(TypeError, match="to_numpy"):
        Tensor(x)


def test_manual_seed_gives_the_same_weights_on_both_backends():
    def build():
        manual_seed(0)
        return [p.data for p in nn.Linear(5, 3).parameters()]

    assert_parity(build, {"rtol": 0, "atol": 0})


def test_gradcheck_runs_on_the_gpu():
    backend.set_backend("cupy")
    (x,) = random_arrays((3, 4))
    assert gradcheck(lambda t: F.gelu(t).sum(), [Tensor(x, requires_grad=True)])


# ----------------------------------------------------------------------
# Tensor operations and functional ops
# ----------------------------------------------------------------------

OPS = {
    "add_broadcast": (lambda a, b: a + b, [(3, 4), (4,)]),
    "sub_mul_div": (lambda a, b: (a - b) * a / (b * b + 1), [(3, 4), (3, 4)]),
    "pow_neg": (lambda a: -(a**3), [(5,)]),
    "matmul": (lambda a, b: a @ b, [(2, 3, 4), (4, 5)]),
    "exp_log": (lambda a: (a * a + 1).log() + a.exp(), [(4, 3)]),
    "sum_mean": (lambda a: a.sum(dim=0) + a.mean(dim=0), [(4, 3)]),
    "reshape_permute": (lambda a: a.reshape(4, 6).T.permute(1, 0), [(2, 3, 4)]),
    "index_repeated": (lambda a: a[np.array([0, 2, 0, 1])], [(3, 4)]),
    "slice": (lambda a: a[1:, ::2], [(3, 4)]),
    "relu": (F.relu, [(4, 5)]),
    "leaky_relu": (lambda a: F.leaky_relu(a, 0.1), [(4, 5)]),
    "gelu": (F.gelu, [(4, 5)]),
    "gelu_tanh": (lambda a: F.gelu(a, approximate="tanh"), [(4, 5)]),
    "tanh": (F.tanh, [(4, 5)]),
    "sigmoid": (F.sigmoid, [(4, 5)]),
    "softplus": (lambda a: F.softplus(a, beta=2.0), [(4, 5)]),
    "softmax": (lambda a: F.softmax(a, dim=1), [(4, 5)]),
    "log_softmax": (lambda a: F.log_softmax(a, dim=0), [(4, 5)]),
    "logsumexp": (lambda a: F.logsumexp(a * 100, dim=1), [(4, 5)]),
    "linear": (F.linear, [(4, 3), (2, 3), (2,)]),
    "layer_norm": (lambda a, w, b: F.layer_norm(a, (5,), w, b), [(4, 5), (5,), (5,)]),
    "batch_norm": (
        lambda a, w, b: F.batch_norm(a, None, None, w, b, training=True),
        [(6, 3), (3,), (3,)],
    ),
    "conv1d": (
        lambda x, w, b: F.conv1d(x, w, b, stride=2, padding=1, dilation=2),
        [(2, 3, 11), (4, 3, 3), (4,)],
    ),
    "conv2d": (
        lambda x, w, b: F.conv2d(x, w, b, padding="same"),
        [(2, 3, 6, 5), (4, 3, 3, 3), (4,)],
    ),
    "max_pool1d": (lambda x: F.max_pool1d(x, 3, stride=2, padding=1), [(2, 3, 9)]),
    "avg_pool1d": (lambda x: F.avg_pool1d(x, 2), [(2, 3, 9)]),
    "max_pool2d": (lambda x: F.max_pool2d(x, 2), [(2, 3, 6, 6)]),
    "avg_pool2d": (lambda x: F.avg_pool2d(x, (2, 3), padding=1), [(2, 3, 6, 6)]),
    "embedding": (
        lambda w: F.embedding(np.array([[1, 0, 1], [4, 1, 3]]), w, padding_idx=0),
        [(5, 3)],
    ),
    "attention_boolean_mask": (
        lambda q, k, v: F.scaled_dot_product_attention(
            q, k, v, attn_mask=Tensor(np.triu(np.ones((4, 6), dtype=bool), k=3))
        ),
        [(2, 4, 3), (2, 6, 3), (2, 6, 5)],
    ),
    "attention_causal": (
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        [(2, 4, 3), (2, 4, 3), (2, 4, 3)],
    ),
    "cross_entropy": (
        lambda a: F.cross_entropy(a, np.array([0, 3, 1, 4]), label_smoothing=0.1),
        [(4, 5)],
    ),
    "bce_with_logits": (
        lambda a: F.binary_cross_entropy_with_logits(
            a * 30, np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        ),
        [(3, 2)],
    ),
    "mse_loss": (lambda a, b: F.mse_loss(a, b, reduction="sum"), [(3, 2), (3, 2)]),
}


@pytest.mark.parametrize("name", OPS)
def test_op_parity_float64(name):
    fn, shapes = OPS[name]
    assert_parity(lambda: forward_backward(fn, *random_arrays(*shapes)))


@pytest.mark.parametrize("name", ["matmul", "gelu", "conv1d", "cross_entropy"])
def test_op_parity_float32(name):
    backend.set_default_dtype("float32")
    fn, shapes = OPS[name]
    assert_parity(lambda: forward_backward(fn, *random_arrays(*shapes)), FLOAT32)


def test_dropout_masks_agree():
    def build():
        manual_seed(3)
        return forward_backward(lambda a: F.dropout(a, p=0.4), *random_arrays((6, 7)))

    assert_parity(build)


# ----------------------------------------------------------------------
# Layers: same seed, same weights, same outputs and gradients
# ----------------------------------------------------------------------

LAYERS = {
    "linear_ntk": (lambda: nn.Linear(4, 3, parametrization="ntk"), (5, 4)),
    "conv1d": (lambda: nn.Conv1d(4, 6, 5, padding="same"), (3, 4, 20)),
    "conv2d": (lambda: nn.Conv2d(2, 3, 3, stride=2), (2, 2, 7, 7)),
    "batch_norm": (lambda: nn.BatchNorm1d(4), (8, 4)),
    "layer_norm": (lambda: nn.LayerNorm(6), (2, 3, 6)),
    "mlp_dropout": (
        lambda: nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 2)
        ),
        (8, 4),
    ),
    "cnn": (
        lambda: nn.Sequential(
            nn.Conv1d(4, 8, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(8 * 8, 2),
        ),
        (3, 4, 20),
    ),
    "multihead_attention": (lambda: _SelfAttention(8, 2), (2, 5, 8)),
    "transformer_encoder": (
        lambda: nn.Sequential(
            nn.PositionalEncoding(8, dropout=0.1),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.1),
                num_layers=2,
                norm=nn.LayerNorm(8),
            ),
        ),
        (2, 6, 8),
    ),
}


class _SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        output, weights = self.attention(x, x, x)
        return output + weights.sum()


def layer_forward_backward(make_layer, input_shape):
    manual_seed(0)
    layer = make_layer()
    (x,) = random_arrays(input_shape)
    out, x_grad = forward_backward(layer, x)
    return [out, x_grad] + [p.grad.data for p in layer.parameters()]


@pytest.mark.parametrize("name", LAYERS)
def test_layer_parity(name):
    make_layer, input_shape = LAYERS[name]
    assert_parity(lambda: layer_forward_backward(make_layer, input_shape))


def test_embedding_layer_parity():
    def build():
        manual_seed(0)
        layer = nn.Embedding(6, 4, padding_idx=2)
        out = layer(np.array([[0, 2, 5], [5, 5, 1]]))
        out.sum().backward()
        return [out.data, layer.weight.grad.data]

    assert_parity(build)


def test_batch_norm_running_statistics_agree():
    def build():
        manual_seed(0)
        layer = nn.BatchNorm1d(3)
        for x in random_arrays((8, 3), (8, 3)):
            layer(Tensor(x))
        layer.eval()
        return [layer.running_mean.data, layer.running_var.data, layer(Tensor(x)).data]

    assert_parity(build)


# ----------------------------------------------------------------------
# Training and analysis
# ----------------------------------------------------------------------

OPTIMIZERS = {
    "sgd_momentum": lambda p: optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True),
    "adam": lambda p: optim.Adam(p, lr=0.01, weight_decay=1e-3),
    "adamw": lambda p: optim.AdamW(p, lr=0.01, weight_decay=1e-2),
    "rmsprop": lambda p: optim.RMSprop(p, lr=0.01),
}


def train_mlp(make_optimizer, steps=20):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 3))
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)
    manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 2))
    optimizer = make_optimizer(model.parameters())
    scheduler = optim.CosineAnnealingLR(optimizer, T_max=steps)
    losses = []
    loader = DataLoader(X, y, batch_size=16, shuffle=True)
    while len(losses) < steps:
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optim.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.data)
    return [backend.xp.stack(losses)] + [p.data for p in model.parameters()]


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_training_parity_float64(name):
    assert_parity(lambda: train_mlp(OPTIMIZERS[name]))


def test_training_parity_float32():
    backend.set_default_dtype("float32")
    assert_parity(lambda: train_mlp(OPTIMIZERS["adam"]), FLOAT32)


def test_empirical_ntk_and_activation_stats_agree():
    def build():
        manual_seed(0)
        model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
        X1, X2 = random_arrays((4, 3), (2, 3))
        ntk = analysis.empirical_ntk(model, X1, X2)
        stats = analysis.activation_stats(model, X1)
        means = [s.mean for s in stats.values()] + [s.std for s in stats.values()]
        return [ntk.data, np.array(means)]

    assert_parity(build)
