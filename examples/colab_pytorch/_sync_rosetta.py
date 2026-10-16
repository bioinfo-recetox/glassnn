"""Write the Rosetta table into the last cell of every Colab notebook.

Usage: ``uv run python examples/colab_pytorch/_sync_rosetta.py``, after
editing ``book/assets/rosetta.toml``. The table is appended as a new last
cell, or replaces the previous one. ``tests/test_rosetta.py`` fails while a
notebook holds an outdated table.
"""

import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "book"))

import _rosetta  # noqa: E402

HEADING = "## Rosetta table"


def sync(path: Path) -> bool:
    """Update one notebook; return whether it changed."""
    notebook = nbformat.read(path, as_version=4)
    source = _rosetta.notebook_cell_source()
    last = notebook.cells[-1] if notebook.cells else None
    if (
        last is not None
        and last.cell_type == "markdown"
        and last.source.startswith(HEADING)
    ):
        if last.source == source:
            return False
        last.source = source
    else:
        notebook.cells.append(nbformat.v4.new_markdown_cell(source))
    nbformat.write(notebook, path)
    return True


if __name__ == "__main__":
    for path in sorted(HERE.glob("*.ipynb")):
        print(f"{path.name}: {'updated' if sync(path) else 'up to date'}")
