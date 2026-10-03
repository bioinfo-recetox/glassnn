"""Execute every example notebook and fail on the first error.

Usage: ``uv run python examples/_execute.py [notebook ...]``. The notebooks
are run in memory from the ``examples`` directory; the files on disk are not
changed (they are kept without outputs). Used by the ``examples`` CI job.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def execute(path: Path) -> float:
    """Run one notebook; return the time it took in seconds."""
    notebook = nbformat.read(path, as_version=4)
    start = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute()
    return time.perf_counter() - start


def main(names: list[str]) -> None:
    paths = [Path(name) for name in names] or sorted(HERE.glob("[0-9][0-9]_*.ipynb"))
    for path in paths:
        print(f"{path.name}: ", end="", flush=True)
        print(f"ok ({execute(path):.1f} s)")


if __name__ == "__main__":
    main(sys.argv[1:])
