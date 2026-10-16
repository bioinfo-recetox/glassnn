"""The Colab notebooks run and reproduce the findings of their GlassNN
counterparts (examples 02 and 06), with PyTorch.

Marker ``torch``: needs PyTorch, and the ``examples`` extra to execute the
notebooks. The notebooks run in ``examples/``, so notebook B finds
``dna_motifs.py`` there and does not download it.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def printed_output(name):
    """Execute a Colab notebook and return everything it printed."""
    pytest.importorskip("torch", reason="the Colab notebooks need PyTorch")
    nbclient = pytest.importorskip("nbclient", reason="needs the examples extra")
    import nbformat

    notebook = nbformat.read(EXAMPLES / "colab_pytorch" / name, as_version=4)
    nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES)}},
    ).execute()
    return "".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
    )


def test_a_only_kaiming_keeps_the_relu_signal_scale():
    text = printed_output("A_mlp_init.ipynb")
    ratios = {
        label.strip(): float(value)
        for label, value in re.findall(
            r"ReLU  (.+?) std\(Z_20\) / std\(Z_1\) = (\S+)", text
        )
    }
    # The same bounds as the GlassNN example 02 (tests/test_examples.py).
    assert 0.5 < ratios["Kaiming normal (relu)"] < 2.0
    assert ratios["Xavier normal"] < 1e-2
    assert ratios["default of Linear"] < 1e-5
    assert ratios["too small, N(0, 0.01²)"] < 1e-15
    assert ratios["too large, N(0, 0.1²)"] > 5.0


def test_b_the_cnn_finds_the_motif():
    text = printed_output("B_cnn_motif.ipynb")
    accuracy = float(re.search(r"test accuracy: (\S+)", text).group(1))
    consensus = re.search(r"consensus: (\w+)", text).group(1)
    assert accuracy > 0.95
    # The filter may be shifted: it shares at least 6 bases with the motif.
    shared = max(
        sum(a == b for a, b in zip(consensus[s:], "TGACGTCA"[t:], strict=False))
        for s in range(3)
        for t in range(3)
    )
    assert shared >= 6
