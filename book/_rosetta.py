"""The Rosetta table: where GlassNN and PyTorch differ.

The rows live in ``book/assets/rosetta.toml`` (one ``[[row]]`` per
difference). Book chapter 12 shows :func:`markdown_table`;
``examples/colab_pytorch/_sync_rosetta.py`` writes
:func:`notebook_cell_source` into the last cell of every Colab notebook,
and ``tests/test_rosetta.py`` checks that the copies are current and that
every "Differences from PyTorch" note in a docstring has a row.
"""

import importlib
import tomllib
from pathlib import Path
from typing import Any

ROSETTA_FILE = Path(__file__).resolve().parent / "assets" / "rosetta.toml"


def load_rows() -> list[dict[str, Any]]:
    """The rows, in file order: ``group``, ``glassnn`` (a list of names
    relative to the package, e.g. ``"nn.Linear"``), ``pytorch`` and
    ``difference`` (Markdown)."""
    with open(ROSETTA_FILE, "rb") as file:
        return tomllib.load(file)["row"]


def resolve(name: str) -> Any:
    """The GlassNN object called ``name`` (e.g. ``"nn.Module.state_dict"``)."""
    parts = ["glassnn", *name.split(".")]
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute, None)
        return obj
    return None


def markdown_table() -> str:
    """The table as Markdown, grouped as in the file."""
    lines = [
        "| Area | GlassNN | PyTorch | Difference |",
        "|---|---|---|---|",
    ]
    for row in load_rows():
        names = ", ".join(f"`{name}`" for name in row["glassnn"])
        cells = [row["group"], names, row["pytorch"], row["difference"]]
        lines.append("| " + " | ".join(_cell(text) for text in cells) + " |")
    return "\n".join(lines)


def notebook_cell_source() -> str:
    """The last cell of every Colab notebook: a heading and the table."""
    return (
        "## Rosetta table: GlassNN and PyTorch\n"
        "\n"
        "Names, arguments and behaviour that differ between the two libraries "
        "(the same table as in chapter 12 of the GlassNN book). Everything not "
        "listed has the same name and meaning.\n"
        "\n" + markdown_table()
    )


def _cell(text: str) -> str:
    """One table cell: on one line, with ``|`` escaped."""
    return " ".join(text.split()).replace("|", "\\|")
