"""Cross-checks against PyTorch (marker ``torch``; needs torch installed).

Same weights, same inputs, float64 everywhere. Unless stated otherwise,
outputs and gradients must agree to ``RTOL = 1e-10`` and ``ATOL = 1e-12``.
"""

import math

import numpy as np
import pytest

import glassnn.functional as F
from glassnn import Tensor, manual_seed, nn, optim
from glassnn.nn import init

torch = pytest.importorskip("torch", reason="PyTorch is not installed")
TF = torch.nn.functional

pytestmark = pytest.mark.torch

RTOL = 1e-10
ATOL = 1e-12


def close(ours, theirs, rtol=RTOL, atol=ATOL):
    expected = theirs.detach().numpy() if torch.is_tensor(theirs) else theirs
    np.testing.assert_allclose(np.asarray(ours), expected, rtol=rtol, atol=atol)


def both(array, requires_grad=True):
    """The same values as a GlassNN tensor and a PyTorch tensor."""
    return (
        Tensor(array, requires_grad=requires_grad, dtype="float64"),
        torch.tensor(array, dtype=torch.float64, requires_grad=requires_grad),
    )


# --------------------------------------------------------------------------
# Layers and functions: forward values and input gradients
# --------------------------------------------------------------------------


def test_linear(rng):
    layer = nn.Linear(5, 3)
    reference = torch.nn.Linear(5, 3, dtype=torch.float64)
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(layer.weight.data))
        reference.bias.copy_(torch.from_numpy(layer.bias.data))
    x, xt = both(rng.normal(size=(4, 5)))
    upstream = rng.normal(size=(4, 3))
    out, out_t = layer(x), reference(xt)
    close(out.data, out_t)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(layer.weight.grad.data, reference.weight.grad)
    close(layer.bias.grad.data, reference.bias.grad)
    close(x.grad.data, xt.grad)


FUNCTIONS = {
    "relu": (F.relu, TF.relu),
    "leaky_relu": (lambda t: F.leaky_relu(t, 0.2), lambda t: TF.leaky_relu(t, 0.2)),
    "gelu": (F.gelu, TF.gelu),
    "gelu_tanh": (
        lambda t: F.gelu(t, approximate="tanh"),
        lambda t: TF.gelu(t, approximate="tanh"),
    ),
    "tanh": (F.tanh, torch.tanh),
    "sigmoid": (F.sigmoid, torch.sigmoid),
    "softplus": (F.softplus, TF.softplus),
    "softplus_beta": (
        lambda t: F.softplus(t, beta=2.0, threshold=5.0),
        lambda t: TF.softplus(t, beta=2.0, threshold=5.0),
    ),
    "softmax": (lambda t: F.softmax(t, dim=1), lambda t: TF.softmax(t, dim=1)),
    "log_softmax": (
        lambda t: F.log_softmax(t, dim=0),
        lambda t: TF.log_softmax(t, dim=0),
    ),
    "logsumexp": (
        lambda t: F.logsumexp(t, dim=1, keepdim=True),
        lambda t: torch.logsumexp(t, dim=1, keepdim=True),
    ),
}


@pytest.mark.parametrize("name", list(FUNCTIONS))
@pytest.mark.parametrize("scale", [1.0, 50.0])
def test_functions(rng, name, scale):
    ours, theirs = FUNCTIONS[name]
    x, xt = both(scale * rng.normal(size=(4, 6)))
    out, out_t = ours(x), theirs(xt)
    close(out.data, out_t)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(x.grad.data, xt.grad)


@pytest.mark.parametrize("smoothing", [0.0, 0.2])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_cross_entropy(rng, smoothing, reduction, scale):
    x, xt = both(scale * rng.normal(size=(7, 5)))
    labels = rng.integers(0, 5, size=7)
    out = F.cross_entropy(x, Tensor(labels), reduction, smoothing)
    out_t = TF.cross_entropy(
        xt, torch.from_numpy(labels), reduction=reduction, label_smoothing=smoothing
    )
    close(out.data, out_t)
    out.sum().backward()
    out_t.sum().backward()
    close(x.grad.data, xt.grad)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_binary_cross_entropy_with_logits(rng, reduction, scale):
    x, xt = both(scale * rng.normal(size=(6, 2)))
    t, tt = both(rng.uniform(size=(6, 2)))
    out = F.binary_cross_entropy_with_logits(x, t, reduction)
    out_t = TF.binary_cross_entropy_with_logits(xt, tt, reduction=reduction)
    close(out.data, out_t)
    out.sum().backward()
    out_t.sum().backward()
    close(x.grad.data, xt.grad)
    close(t.grad.data, tt.grad)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_loss(rng, reduction):
    a, at = both(rng.normal(size=(5, 3)))
    b, bt = both(rng.normal(size=(5, 3)))
    out = F.mse_loss(a, b, reduction)
    out_t = TF.mse_loss(at, bt, reduction=reduction)
    close(out.data, out_t)
    out.sum().backward()
    out_t.sum().backward()
    close(a.grad.data, at.grad)
    close(b.grad.data, bt.grad)


# --------------------------------------------------------------------------
# Initialization formulas
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nonlinearity", "param"),
    [
        ("linear", None),
        ("tanh", None),
        ("relu", None),
        ("leaky_relu", 0.3),
        ("leaky_relu", math.sqrt(5)),
        ("selu", None),
        ("sigmoid", None),
    ],
)
def test_calculate_gain(nonlinearity, param):
    assert init.calculate_gain(nonlinearity, param) == pytest.approx(
        torch.nn.init.calculate_gain(nonlinearity, param), rel=1e-15
    )


@pytest.mark.parametrize("shape", [(5, 3), (8, 4, 3), (6, 2, 3, 3)])
def test_fans(shape):
    ours = init._calculate_fan_in_and_fan_out(nn.Parameter(np.zeros(shape)))
    theirs = torch.nn.init._calculate_fan_in_and_fan_out(torch.zeros(shape))
    assert ours == theirs


# --------------------------------------------------------------------------
# Optimizers and schedulers: the same gradient sequence, five steps
# --------------------------------------------------------------------------

OPTIMIZERS = {
    "sgd": (
        lambda ps: optim.SGD(ps, lr=0.1),
        lambda ps: torch.optim.SGD(ps, lr=0.1),
    ),
    "sgd_momentum_wd": (
        lambda ps: optim.SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.01),
        lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.01),
    ),
    "sgd_nesterov": (
        lambda ps: optim.SGD(ps, lr=0.05, momentum=0.8, nesterov=True),
        lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.8, nesterov=True),
    ),
    "adam": (
        lambda ps: optim.Adam(ps, lr=0.01, betas=(0.8, 0.95), weight_decay=0.1),
        lambda ps: torch.optim.Adam(ps, lr=0.01, betas=(0.8, 0.95), weight_decay=0.1),
    ),
    "adamw": (
        lambda ps: optim.AdamW(ps, lr=0.01, weight_decay=0.1),
        lambda ps: torch.optim.AdamW(ps, lr=0.01, weight_decay=0.1),
    ),
    "rmsprop": (
        lambda ps: optim.RMSprop(ps, lr=0.01, alpha=0.9, weight_decay=0.05),
        lambda ps: torch.optim.RMSprop(ps, lr=0.01, alpha=0.9, weight_decay=0.05),
    ),
    "rmsprop_momentum": (
        lambda ps: optim.RMSprop(ps, lr=0.01, momentum=0.9),
        lambda ps: torch.optim.RMSprop(ps, lr=0.01, momentum=0.9),
    ),
}


@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_optimizer_steps(rng, name):
    make_ours, make_theirs = OPTIMIZERS[name]
    start = [rng.normal(size=(3, 4)), rng.normal(size=4)]
    ours = [nn.Parameter(a) for a in start]
    theirs = [torch.nn.Parameter(torch.tensor(a)) for a in start]
    opt, opt_t = make_ours(ours), make_theirs(theirs)
    for _ in range(5):
        for p, pt in zip(ours, theirs, strict=True):
            g = rng.normal(size=p.shape)
            p.grad = Tensor(g)
            pt.grad = torch.tensor(g)
        opt.step()
        opt_t.step()
        for p, pt in zip(ours, theirs, strict=True):
            close(p.data, pt)


SCHEDULERS = {
    "step": (
        lambda o: optim.StepLR(o, step_size=3, gamma=0.5),
        lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=3, gamma=0.5),
    ),
    "cosine": (
        lambda o: optim.CosineAnnealingLR(o, T_max=10, eta_min=0.01),
        lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=10, eta_min=0.01),
    ),
    "warmup": (
        lambda o: optim.LinearWarmup(o, warmup_steps=4, start_factor=0.25),
        lambda o: torch.optim.lr_scheduler.LinearLR(
            o, start_factor=0.25, end_factor=1.0, total_iters=4
        ),
    ),
}


@pytest.mark.parametrize("name", list(SCHEDULERS))
def test_schedulers(name):
    make_ours, make_theirs = SCHEDULERS[name]
    p, pt = nn.Parameter([0.0]), torch.nn.Parameter(torch.zeros(1))
    p.grad, pt.grad = Tensor([0.0]), torch.zeros(1)
    opt, opt_t = optim.SGD([p], lr=0.5), torch.optim.SGD([pt], lr=0.5)
    scheduler, scheduler_t = make_ours(opt), make_theirs(opt_t)
    for _ in range(10):
        assert opt.lr == pytest.approx(opt_t.param_groups[0]["lr"], rel=1e-12)
        opt.step()
        opt_t.step()
        scheduler.step()
        scheduler_t.step()


@pytest.mark.parametrize("max_norm", [0.5, 100.0])
def test_clip_grad_norm(rng, max_norm):
    grads = [rng.normal(size=(3, 2)), rng.normal(size=5)]
    ours = [nn.Parameter(np.zeros(g.shape)) for g in grads]
    theirs = [
        torch.nn.Parameter(torch.zeros(g.shape, dtype=torch.float64)) for g in grads
    ]
    for p, pt, g in zip(ours, theirs, grads, strict=True):
        p.grad, pt.grad = Tensor(g), torch.tensor(g)
    total = optim.clip_grad_norm_(ours, max_norm)
    total_t = torch.nn.utils.clip_grad_norm_(theirs, max_norm)
    close(total.data, total_t)
    for p, pt in zip(ours, theirs, strict=True):
        close(p.grad.data, pt.grad)


def test_one_training_step_of_an_mlp(rng):
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 4))
    reference = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4)
    ).double()
    reference.load_state_dict(
        {name: torch.from_numpy(t.data) for name, t in model.state_dict().items()}
    )
    x = rng.normal(size=(16, 3))
    labels = rng.integers(0, 4, size=16)
    opt = optim.Adam(model.parameters(), lr=0.01)
    opt_t = torch.optim.Adam(reference.parameters(), lr=0.01)
    for _ in range(3):
        opt.zero_grad()
        opt_t.zero_grad()
        F.cross_entropy(model(Tensor(x)), Tensor(labels)).backward()
        TF.cross_entropy(
            reference(torch.from_numpy(x)), torch.from_numpy(labels)
        ).backward()
        opt.step()
        opt_t.step()
    for (name, p), (name_t, pt) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == name_t
        close(p.data, pt)


# --------------------------------------------------------------------------
# Normalization and dropout (milestone M3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(6, 4), (3, 2, 4), (2, 3, 2, 4)])
def test_layer_norm(rng, shape):
    layer = nn.LayerNorm(shape[-1])
    reference = torch.nn.LayerNorm(shape[-1], dtype=torch.float64)
    layer.weight.data = rng.normal(size=shape[-1])
    layer.bias.data = rng.normal(size=shape[-1])
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(layer.weight.data))
        reference.bias.copy_(torch.from_numpy(layer.bias.data))
    x, xt = both(rng.normal(3.0, 2.0, size=shape))
    upstream = rng.normal(size=shape)
    out, out_t = layer(x), reference(xt)
    close(out.data, out_t)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(x.grad.data, xt.grad)
    close(layer.weight.grad.data, reference.weight.grad)
    close(layer.bias.grad.data, reference.bias.grad)


def test_layer_norm_over_two_dimensions(rng):
    x, xt = both(rng.normal(size=(3, 4, 5)))
    out, out_t = F.layer_norm(x, (4, 5)), TF.layer_norm(xt, (4, 5))
    close(out.data, out_t)
    out.sum().backward()
    (out_t * 1.0).sum().backward()
    close(x.grad.data, xt.grad, atol=1e-10)


@pytest.mark.parametrize("shape", [(8, 3), (4, 3, 5)])
def test_batch_norm_training_steps_and_evaluation(rng, shape):
    """Three training steps (outputs, gradients, running statistics), then eval."""
    layer = nn.BatchNorm1d(3, momentum=0.3)
    reference = torch.nn.BatchNorm1d(3, momentum=0.3, dtype=torch.float64)
    layer.weight.data = rng.normal(size=3)
    layer.bias.data = rng.normal(size=3)
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(layer.weight.data))
        reference.bias.copy_(torch.from_numpy(layer.bias.data))
    for _ in range(3):
        x, xt = both(rng.normal(1.0, 2.0, size=shape))
        upstream = rng.normal(size=shape)
        out, out_t = layer(x), reference(xt)
        close(out.data, out_t)
        (out * Tensor(upstream)).sum().backward()
        (out_t * torch.from_numpy(upstream)).sum().backward()
        close(x.grad.data, xt.grad)
        close(layer.running_mean.data, reference.running_mean)
        close(layer.running_var.data, reference.running_var)
    close(layer.weight.grad.data, reference.weight.grad)
    close(layer.bias.grad.data, reference.bias.grad)
    layer.eval()
    reference.eval()
    x, xt = both(rng.normal(size=shape))
    out, out_t = layer(x), reference(xt)
    close(out.data, out_t)
    out.sum().backward()
    out_t.sum().backward()
    close(x.grad.data, xt.grad)


def test_batch_norm_state_dict_names_match_pytorch():
    ours = set(nn.BatchNorm1d(3).state_dict())
    theirs = set(torch.nn.BatchNorm1d(3).state_dict())
    assert theirs - ours == {"num_batches_tracked"}  # documented difference
    assert ours <= theirs


def test_dropout_scaling_matches_pytorch(rng):
    # The masks come from different generators; compare what survives.
    x = rng.uniform(1.0, 2.0, size=(400, 500))
    ours = F.dropout(Tensor(x), p=0.25).data
    theirs = TF.dropout(torch.from_numpy(x), p=0.25).numpy()
    for out in (ours, theirs):
        kept = out != 0
        np.testing.assert_allclose(out[kept], x[kept] / 0.75, rtol=1e-12)
        assert kept.mean() == pytest.approx(0.75, abs=0.005)
    np.testing.assert_array_equal(
        F.dropout(Tensor(x), p=0.25, training=False).data,
        TF.dropout(torch.from_numpy(x), p=0.25, training=False).numpy(),
    )


# --------------------------------------------------------------------------
# Convolution and pooling (milestone M4)
# --------------------------------------------------------------------------

CONV_SETTINGS = [
    # (dims, input spatial size, kernel, stride, padding, dilation)
    (1, (20,), 5, 1, 0, 1),
    (1, (20,), 4, 1, "same", 1),
    (1, (21,), 3, 2, 2, 3),
    (2, (9, 10), 3, 1, 1, 1),
    (2, (9, 10), (3, 2), (2, 1), "valid", (1, 2)),
    (2, (8, 8), (2, 3), 1, "same", 1),
]


@pytest.mark.filterwarnings("ignore:Using padding='same'")  # PyTorch performance note
@pytest.mark.parametrize(
    ("dims", "size", "kernel", "stride", "padding", "dilation"), CONV_SETTINGS
)
def test_conv_layers(rng, dims, size, kernel, stride, padding, dilation):
    ours = getattr(nn, f"Conv{dims}d")(3, 4, kernel, stride, padding, dilation)
    reference = getattr(torch.nn, f"Conv{dims}d")(
        3, 4, kernel, stride, padding, dilation, dtype=torch.float64
    )
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(ours.weight.data))
        reference.bias.copy_(torch.from_numpy(ours.bias.data))
    x, xt = both(rng.normal(size=(2, 3, *size)))
    out, out_t = ours(x), reference(xt)
    close(out.data, out_t)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(x.grad.data, xt.grad)
    close(ours.weight.grad.data, reference.weight.grad)
    close(ours.bias.grad.data, reference.bias.grad)


POOL_SETTINGS = [
    # (dims, input spatial size, kernel, stride, padding)
    (1, (12,), 3, None, 0),
    (1, (13,), 4, 2, 2),
    (1, (12,), 12, None, 0),
    (2, (8, 9), 2, None, 0),
    (2, (8, 9), (3, 2), (2, 1), 1),
]


@pytest.mark.parametrize("mode", ["max", "avg"])
@pytest.mark.parametrize(("dims", "size", "kernel", "stride", "padding"), POOL_SETTINGS)
def test_pooling(rng, mode, dims, size, kernel, stride, padding):
    name = f"{mode}_pool{dims}d"
    x, xt = both(rng.normal(size=(2, 3, *size)))
    out = getattr(F, name)(x, kernel, stride, padding)
    out_t = getattr(TF, name)(xt, kernel, stride, padding)
    close(out.data, out_t)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(x.grad.data, xt.grad)


def test_conv_default_initialization_matches_pytorch_bounds():
    # Same distribution (not the same numbers): both are uniform on
    # [-b, b] with b = 1 / sqrt(C_in * K), whose standard deviation is
    # b / sqrt(3). Seeded, with tolerances of several standard errors.
    manual_seed(0)
    torch.manual_seed(0)
    ours = nn.Conv1d(4, 500, 7)
    theirs = torch.nn.Conv1d(4, 500, 7)
    bound = 1 / math.sqrt(4 * 7)
    pairs = [
        (ours.weight.data, theirs.weight, 0.03),  # 14000 values
        (ours.bias.data, theirs.bias, 0.15),  # 500 values
    ]
    for p, q, rel in pairs:
        q = q.detach().numpy()
        for values in (p, q):
            assert np.abs(values).max() <= bound
            assert values.std() == pytest.approx(bound / math.sqrt(3), rel=rel)


def test_flatten(rng):
    x, xt = both(rng.normal(size=(2, 3, 4, 5)))
    close(nn.Flatten()(x).data, torch.nn.Flatten()(xt))
    close(nn.Flatten(0, 2)(x).data, torch.nn.Flatten(0, 2)(xt))


@pytest.mark.parametrize("mode", ["Max", "Avg"])
def test_2d_pooling_modules(rng, mode):
    ours = getattr(nn, f"{mode}Pool2d")((3, 2), stride=(2, 1), padding=1)
    theirs = getattr(torch.nn, f"{mode}Pool2d")((3, 2), stride=(2, 1), padding=1)
    x, xt = both(rng.normal(size=(2, 3, 7, 8)))
    out, out_t = ours(x), theirs(xt)
    close(out.data, out_t)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(x.grad.data, xt.grad)


# --------------------------------------------------------------------------
# Embedding, attention and the transformer encoder (milestone M5)
# --------------------------------------------------------------------------


def test_embedding(rng):
    ours = nn.Embedding(6, 3, padding_idx=2)
    theirs = torch.nn.Embedding(6, 3, padding_idx=2, dtype=torch.float64)
    with torch.no_grad():
        theirs.weight.copy_(torch.from_numpy(ours.weight.data))
    index = rng.integers(0, 6, size=(4, 5))
    out, out_t = ours(Tensor(index)), theirs(torch.from_numpy(index))
    close(out.data, out_t)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(ours.weight.grad.data, theirs.weight.grad)


@pytest.mark.parametrize("mask_kind", ["none", "bool", "float", "causal"])
def test_scaled_dot_product_attention(rng, mask_kind):
    q, qt = both(rng.normal(size=(2, 3, 4, 5)))
    k, kt = both(rng.normal(size=(2, 3, 6, 5)))
    v, vt = both(rng.normal(size=(2, 3, 6, 7)))
    kwargs, kwargs_t = {}, {}
    if mask_kind == "bool":
        masked = rng.uniform(size=(4, 6)) < 0.3
        masked[:, 0] = False  # every query sees at least one key
        kwargs = {"attn_mask": Tensor(masked)}
        kwargs_t = {"attn_mask": torch.from_numpy(~masked)}  # PyTorch: True = attend
    elif mask_kind == "float":
        bias = rng.normal(size=(4, 6))
        kwargs = {"attn_mask": Tensor(bias)}
        kwargs_t = {"attn_mask": torch.from_numpy(bias)}
    elif mask_kind == "causal":
        q, qt = both(rng.normal(size=(2, 3, 6, 5)))
        kwargs = kwargs_t = {"is_causal": True}
    out = F.scaled_dot_product_attention(q, k, v, **kwargs)
    out_t = TF.scaled_dot_product_attention(qt, kt, vt, **kwargs_t)
    close(out.data, out_t)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    for ours, theirs in [(q, qt), (k, kt), (v, vt)]:
        close(ours.grad.data, theirs.grad)


def copy_attention(ours, theirs):
    """Copy GlassNN's q/k/v/out projections into PyTorch's packed parameters."""
    with torch.no_grad():
        theirs.in_proj_weight.copy_(
            torch.from_numpy(
                np.concatenate(
                    [
                        ours.q_proj.weight.data,
                        ours.k_proj.weight.data,
                        ours.v_proj.weight.data,
                    ]
                )
            )
        )
        theirs.in_proj_bias.copy_(
            torch.from_numpy(
                np.concatenate(
                    [
                        ours.q_proj.bias.data,
                        ours.k_proj.bias.data,
                        ours.v_proj.bias.data,
                    ]
                )
            )
        )
        theirs.out_proj.weight.copy_(torch.from_numpy(ours.out_proj.weight.data))
        theirs.out_proj.bias.copy_(torch.from_numpy(ours.out_proj.bias.data))


def attention_gradients(module):
    return [
        module.q_proj.weight,
        module.k_proj.weight,
        module.v_proj.weight,
        module.out_proj.weight,
    ]


@pytest.mark.parametrize("masks", ["none", "padding", "attn", "both"])
@pytest.mark.parametrize("average", [True, False])
def test_multihead_attention(rng, masks, average):
    ours = nn.MultiheadAttention(8, 2)
    theirs = torch.nn.MultiheadAttention(8, 2, batch_first=True, dtype=torch.float64)
    for layer in [ours.q_proj, ours.k_proj, ours.v_proj, ours.out_proj]:
        layer.bias.data = rng.normal(size=8)
    copy_attention(ours, theirs)
    x, xt = both(rng.normal(size=(3, 5, 8)))
    memory, memory_t = both(rng.normal(size=(3, 6, 8)))
    kwargs = {}
    if masks in ("padding", "both"):
        padding = np.zeros((3, 6), dtype=bool)
        padding[1, 4:] = True
        kwargs["key_padding_mask"] = padding
    if masks in ("attn", "both"):
        hidden = rng.uniform(size=(5, 6)) < 0.3
        hidden[:, 0] = False
        kwargs["attn_mask"] = hidden
    out, weights = ours(
        x,
        memory,
        memory,
        average_attn_weights=average,
        **{k: Tensor(v) for k, v in kwargs.items()},
    )
    out_t, weights_t = theirs(
        xt,
        memory_t,
        memory_t,
        average_attn_weights=average,
        **{k: torch.from_numpy(v) for k, v in kwargs.items()},
    )
    close(out.data, out_t)
    close(weights.data, weights_t)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(x.grad.data, xt.grad)
    close(memory.grad.data, memory_t.grad)
    in_grad = np.concatenate([p.grad.data for p in attention_gradients(ours)[:3]])
    close(in_grad, theirs.in_proj_weight.grad)
    close(ours.out_proj.weight.grad.data, theirs.out_proj.weight.grad)


def copy_encoder_layer(ours, theirs):
    copy_attention(ours.self_attn, theirs.self_attn)
    with torch.no_grad():
        for name in ["linear1", "linear2", "norm1", "norm2"]:
            for attribute in ["weight", "bias"]:
                getattr(getattr(theirs, name), attribute).copy_(
                    torch.from_numpy(getattr(getattr(ours, name), attribute).data)
                )


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_transformer_encoder_layer(rng, norm_first, activation):
    ours = nn.TransformerEncoderLayer(
        8,
        2,
        dim_feedforward=16,
        dropout=0.0,
        activation=activation,
        norm_first=norm_first,
    )
    theirs = torch.nn.TransformerEncoderLayer(
        8,
        2,
        dim_feedforward=16,
        dropout=0.0,
        activation=activation,
        norm_first=norm_first,
        batch_first=True,
        dtype=torch.float64,
    )
    copy_encoder_layer(ours, theirs)
    x, xt = both(rng.normal(size=(3, 5, 8)))
    padding = np.zeros((3, 5), dtype=bool)
    padding[2, 3:] = True
    out = ours(x, src_key_padding_mask=Tensor(padding))
    out_t = theirs(xt, src_key_padding_mask=torch.from_numpy(padding))
    close(out.data, out_t)
    upstream = rng.normal(size=out.shape)
    (out * Tensor(upstream)).sum().backward()
    (out_t * torch.from_numpy(upstream)).sum().backward()
    close(x.grad.data, xt.grad)
    close(ours.linear1.weight.grad.data, theirs.linear1.weight.grad)
    close(ours.norm2.weight.grad.data, theirs.norm2.weight.grad)


def test_transformer_encoder_with_causal_mask(rng):
    layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0)
    ours = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(8))
    torch_layer = torch.nn.TransformerEncoderLayer(
        8,
        2,
        dim_feedforward=16,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        batch_first=True,
        dtype=torch.float64,
    )
    theirs = torch.nn.TransformerEncoder(
        torch_layer,
        2,
        norm=torch.nn.LayerNorm(8, dtype=torch.float64),
        enable_nested_tensor=False,
    )
    for i in range(2):
        ours.layers[i].linear1.weight.data = rng.normal(size=(16, 8)) * 0.3
        copy_encoder_layer(ours.layers[i], theirs.layers[i])
    x, xt = both(rng.normal(size=(2, 6, 8)))
    causal = np.triu(np.ones((6, 6), dtype=bool), k=1)
    out = ours(x, mask=Tensor(causal), is_causal=True)
    out_t = theirs(xt, mask=torch.from_numpy(causal), is_causal=True)
    close(out.data, out_t)
    out.sum().backward()
    out_t.sum().backward()
    close(x.grad.data, xt.grad)
