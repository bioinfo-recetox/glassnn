"""Smoke tests for the package skeleton (milestone M0)."""

import importlib
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import glassnn

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_a_nonempty_string():
    assert isinstance(glassnn.__version__, str)
    assert glassnn.__version__


def test_version_looks_like_a_version():
    # hatch-vcs produces e.g. "0.1.0" on a tag or "0.1.dev3+g1234abc" between tags.
    assert re.match(r"^\d+\.\d+", glassnn.__version__)


def test_version_falls_back_when_not_installed(monkeypatch):
    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    try:
        assert importlib.reload(glassnn).__version__ == "0.0.0"
    finally:
        monkeypatch.undo()
        importlib.reload(glassnn)


def test_import_does_not_load_optional_backends():
    code = (
        "import sys, glassnn\n"
        "bad = [m for m in ('cupy', 'torch') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_license_is_mit_with_copyright_line():
    lines = (REPO_ROOT / "LICENSE").read_text().splitlines()
    assert lines[0] == "MIT License"
    copyright_line = next(line for line in lines[1:] if line.strip())
    assert copyright_line == (
        "Copyright (c) 2026 Vlad Popovici, "
        "RECETOX - Faculty of Science, Masaryk University"
    )
