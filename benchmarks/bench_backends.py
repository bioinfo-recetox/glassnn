"""Time one training step of three small models on the NumPy and CuPy backends.

The models are the smoke benchmarks of PLAN.md, section 8: an MLP with two
hidden layers of width 64, a two-layer Conv1d model on one-hot DNA of length
60, and a one-layer transformer encoder of dimension 32. Each is timed at
batch 32 (lecture size) and at a large batch, in float32. A step is forward,
backward and an Adam update; ``loss.item()`` waits for the GPU to finish.

This is a smoke test, not a CI gate. Run it on a machine with a GPU:

    uv run python benchmarks/bench_backends.py
    uv run python benchmarks/bench_backends.py \
        --json book/assets/benchmark_backends.json

Without CuPy only the NumPy rows are produced. The JSON file feeds the table
in book chapter 13, together with the hardware and the library versions.
"""

import argparse
import datetime
import json
import platform
import time

import numpy as np
import scipy

import glassnn
import glassnn.functional as F
from glassnn import Tensor, backend, manual_seed, nn, optim

BATCH_SIZES = (32, 1024)


class TransformerClassifier(nn.Module):
    """Embedding, one pre-LN encoder layer, mean over positions, linear head."""

    def __init__(self, vocabulary: int = 4, d_model: int = 32, length: int = 60):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, d_model)
        self.position = nn.PositionalEncoding(d_model, max_len=length)
        self.encoder = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=64, dropout=0.0
        )
        self.head = nn.Linear(d_model, 2)

    def forward(self, tokens: Tensor) -> Tensor:
        h = self.encoder(self.position(self.embedding(tokens)))
        return self.head(h.mean(dim=1))


def make_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
    )


def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(4, 16, 9, padding="same"),
        nn.ReLU(),
        nn.Conv1d(16, 16, 9, padding="same"),
        nn.ReLU(),
        nn.MaxPool1d(60),
        nn.Flatten(),
        nn.Linear(16, 2),
    )


def make_inputs(model: str, batch: int, rng: np.random.Generator) -> np.ndarray:
    if model == "MLP":
        return rng.normal(size=(batch, 20))
    bases = rng.integers(0, 4, size=(batch, 60))
    if model == "Transformer":
        return bases
    return np.eye(4)[bases].transpose(0, 2, 1)  # one-hot, shape (N, 4, 60)


MODELS = {
    "MLP": make_mlp,
    "Conv1d": make_cnn,
    "Transformer": TransformerClassifier,
}


def time_steps(model_name: str, batch: int, steps: int) -> float:
    """Median time of one training step on the active backend, in ms."""
    manual_seed(0)
    model = MODELS[model_name]()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    X = make_inputs(model_name, batch, rng)
    x = Tensor(X) if model_name != "Transformer" else X
    y = Tensor(rng.integers(0, 2, size=batch))
    times = []
    for _ in range(steps + 3):  # the first steps compile GPU kernels
        start = time.perf_counter()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        loss.item()
        times.append(time.perf_counter() - start)
    return 1000 * float(np.median(times[3:]))


def available_backends() -> list[str]:
    try:
        backend.set_backend("cupy")
    except ImportError:
        return ["numpy"]
    backend.set_backend("numpy")
    return ["numpy", "cupy"]


def cpu_name() -> str:
    """The CPU model on Linux, or what the platform module knows elsewhere."""
    try:
        with open("/proc/cpuinfo") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def describe_machine(backends: list[str]) -> dict[str, str]:
    info = {
        "date": datetime.date.today().isoformat(),
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "glassnn": glassnn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if "cupy" in backends:
        import cupy

        device = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
        info["gpu"] = device["name"].decode()
        info["cupy"] = cupy.__version__
        info["cuda_runtime"] = str(cupy.cuda.runtime.runtimeGetVersion())
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()

    backend.set_default_dtype("float32")
    backends = available_backends()
    rows = []
    print(f"{'model':<12}{'batch':>6}" + "".join(f"{b:>12}" for b in backends))
    for model_name in MODELS:
        for batch in BATCH_SIZES:
            row = {"model": model_name, "batch": batch}
            for name in backends:
                backend.set_backend(name)
                row[f"{name}_ms"] = round(time_steps(model_name, batch, args.steps), 3)
            backend.set_backend("numpy")
            rows.append(row)
            times = "".join(f"{row[f'{b}_ms']:>10.2f}ms" for b in backends)
            print(f"{model_name:<12}{batch:>6}{times}")

    if args.json:
        result = {"machine": describe_machine(backends), "rows": rows}
        with open(args.json, "w") as file:
            json.dump(result, file, indent=2)
            file.write("\n")


if __name__ == "__main__":
    main()
