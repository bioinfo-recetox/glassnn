"""The Colab notebooks run and reproduce the findings of their GlassNN
counterparts (examples 02 and 06), with PyTorch; notebook C finds that
ESM-2 embeddings beat the amino-acid composition.

Marker ``torch`` (notebooks A and B): needs PyTorch, and the ``examples``
extra to execute the notebooks. Marker ``network`` (notebook C): needs, in
addition, Hugging Face ``transformers`` and internet access to download the
model. The notebooks run in ``examples/``, so notebooks B and C find their
data files there and do not download them. The structure tests of notebook C
need none of this and run with the other tests.
"""

import json
import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
COLAB = EXAMPLES / "colab_pytorch"


def printed_output(name):
    """Execute a Colab notebook and return everything it printed."""
    pytest.importorskip("torch", reason="the Colab notebooks need PyTorch")
    nbclient = pytest.importorskip("nbclient", reason="needs the examples extra")
    import nbformat

    notebook = nbformat.read(COLAB / name, as_version=4)
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


@pytest.mark.torch
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


@pytest.mark.torch
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


@pytest.mark.network
def test_c_transformer_layers_beat_the_composition():
    pytest.importorskip("transformers", reason="notebook C needs transformers")
    text = printed_output("C_protein_embeddings.ipynb")
    baseline = float(re.search(r"composition: CV accuracy (\S+)", text).group(1))
    layers = [float(a) for a in re.findall(r"layer \d: CV accuracy (\S+)", text)]
    assert len(layers) == 7  # the token embedding and 6 transformer layers
    assert 0.6 < baseline < 0.7
    assert abs(layers[0] - baseline) < 0.04  # averaged token embeddings
    assert min(layers[1:]) > baseline + 0.02
    assert max(layers[1:]) > baseline + 0.05


def _code_c():
    cells = json.loads((COLAB / "C_protein_embeddings.ipynb").read_text())["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_colab_notebooks_are_stored_without_outputs():
    for path in sorted(COLAB.glob("*.ipynb")):
        for cell in json.loads(path.read_text())["cells"]:
            assert not cell.get("outputs"), path.name


def test_c_pins_the_model_version():
    code = _code_c()
    assert 'MODEL = "facebook/esm2_t6_8M_UR50D"' in code
    assert re.search(r'^REVISION = "[0-9a-f]{40}"$', code, flags=re.M)
    calls = re.findall(r"from_pretrained\((.*?)\)", code)
    assert len(calls) == 2
    assert all("revision=REVISION" in call for call in calls)


def test_c_downloads_the_data_file_of_the_repository():
    # Decision 51: from main, not from a tag, unlike notebook B.
    url = re.search(r'DATA_URL = \(\s*"(.*?)"\s*"(.*?)"\s*\)', _code_c(), re.S)
    path = "".join(url.groups())
    prefix = "https://raw.githubusercontent.com/bioinfo-recetox/glassnn/main/"
    assert path.startswith(prefix)
    assert (EXAMPLES.parent / path.removeprefix(prefix)).is_file()


def test_c_data_has_250_proteins_per_class():
    lines = (EXAMPLES / "protein_localization.fasta").read_text().splitlines()
    headers, sequences = lines[0::2], lines[1::2]
    assert len(headers) == len(sequences) == 500
    labels = [header.split()[1] for header in headers]
    assert labels.count("nucleus") == labels.count("cytoplasm") == 250
    assert all(re.fullmatch(r">[A-Z0-9]+ (nucleus|cytoplasm)", h) for h in headers)
    assert all(re.fullmatch(r"[A-Z]{50,1000}", s) for s in sequences)
    assert len({h.split()[0] for h in headers}) == 500
