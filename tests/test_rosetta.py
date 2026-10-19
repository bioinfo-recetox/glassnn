"""The Rosetta table (GlassNN against PyTorch) is complete and the same
everywhere it appears (PLAN, decision 43).

The rows live in ``book/assets/rosetta.toml``; book chapter 12 renders them,
and ``examples/colab_pytorch/_sync_rosetta.py`` writes the same table into
the last cell of every Colab notebook.
"""

import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path

import pytest

import glassnn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "book"))

import _rosetta  # noqa: E402

COLAB = REPO_ROOT / "examples" / "colab_pytorch"


def _documented_differences():
    """Objects whose docstring has a 'Differences from PyTorch' note, by name."""
    modules = [glassnn] + [
        importlib.import_module(m.name)
        for m in pkgutil.walk_packages(glassnn.__path__, "glassnn.")
    ]
    found = {}
    for module in modules:
        for name, obj in vars(module).items():
            if name.startswith("_") or not (
                inspect.isfunction(obj) or inspect.isclass(obj)
            ):
                continue
            if obj.__module__ != module.__name__:
                continue
            members = [(f"{module.__name__}.{name}", obj)]
            if inspect.isclass(obj):
                members += [
                    (f"{module.__name__}.{name}.{attr}", value)
                    for attr, value in vars(obj).items()
                    if inspect.isfunction(value) and not attr.startswith("_")
                ]
            for full_name, member in members:
                if "Differences from PyTorch" in (inspect.getdoc(member) or ""):
                    found[full_name] = member
    return found


def test_rows_have_all_fields():
    rows = _rosetta.load_rows()
    assert len(rows) >= 29
    for row in rows:
        assert set(row) == {"group", "glassnn", "pytorch", "difference"}, row
        assert row["glassnn"] and all(isinstance(n, str) for n in row["glassnn"])
        assert row["pytorch"].strip() and row["difference"].strip(), row


def test_every_named_glassnn_object_exists():
    for row in _rosetta.load_rows():
        for name in row["glassnn"]:
            assert _rosetta.resolve(name) is not None, name


def test_every_documented_difference_has_a_row():
    in_table = {
        id(_rosetta.resolve(name))
        for row in _rosetta.load_rows()
        for name in row["glassnn"]
    }
    missing = [
        name
        for name, obj in _documented_differences().items()
        if id(obj) not in in_table
    ]
    assert not missing, sorted(missing)


def test_markdown_table_has_one_line_per_row_plus_header():
    table = _rosetta.markdown_table()
    lines = [line for line in table.splitlines() if line.startswith("|")]
    assert len(lines) == len(_rosetta.load_rows()) + 2


NOTEBOOKS = sorted(COLAB.glob("*.ipynb"))


def test_there_are_three_colab_notebooks():
    assert [path.name for path in NOTEBOOKS] == [
        "A_mlp_init.ipynb",
        "B_cnn_motif.ipynb",
        "C_protein_embeddings.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_each_notebook_ends_with_the_current_table(path):
    last = json.loads(path.read_text())["cells"][-1]
    assert last["cell_type"] == "markdown"
    source = "".join(last["source"])
    assert source == _rosetta.notebook_cell_source(), (
        f"{path.name}: run `uv run python examples/colab_pytorch/_sync_rosetta.py`"
    )
