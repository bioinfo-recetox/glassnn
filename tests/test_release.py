"""Release 0.1.0 is consistent: changelog, citation file, metadata, release
notes and every pointer to the tag (milestone M8)."""

import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.1"
TAG = f"v{VERSION}"
REPO_URL = "https://github.com/bioinfo-recetox/glassnn"

CHANGELOG = (ROOT / "CHANGELOG.md").read_text()
RELEASE_HEADING = re.compile(
    rf"^## \[{re.escape(VERSION)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", re.M
)


def _changelog_date():
    match = RELEASE_HEADING.search(CHANGELOG)
    assert match, f"CHANGELOG.md has no heading '## [{VERSION}] - YYYY-MM-DD'"
    return match.group(1)


def _citation():
    yaml = pytest.importorskip("yaml", reason="PyYAML comes with pre-commit (dev)")
    return yaml.safe_load((ROOT / "CITATION.cff").read_text())


def test_changelog_has_an_unreleased_section_above_the_release():
    unreleased = CHANGELOG.index("## [Unreleased]")
    release = RELEASE_HEADING.search(CHANGELOG).start()
    assert unreleased < release


def test_changelog_release_section_starts_with_highlights():
    section = CHANGELOG[RELEASE_HEADING.search(CHANGELOG).end() :]
    assert section.lstrip().startswith("### Highlights")


def test_changelog_has_comparison_links():
    assert f"[Unreleased]: {REPO_URL}/compare/{TAG}...HEAD" in CHANGELOG
    # The first release links to its tag, later ones to the comparison.
    link = re.escape(f"[{VERSION}]: {REPO_URL}/")
    target = rf"(releases/tag/{re.escape(TAG)}|compare/v[\d.]+\.\.\.{re.escape(TAG)})"
    assert re.search(rf"^{link}{target}$", CHANGELOG, flags=re.M)


def test_release_workflow_extracts_non_empty_notes(tmp_path):
    # Run the very script of release.yml on a copy of the changelog.
    yaml = pytest.importorskip("yaml", reason="PyYAML comes with pre-commit (dev)")
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["release"]["steps"]
    script = next(s["run"] for s in steps if s.get("name", "").startswith("Extract"))
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_REF_NAME": TAG},
        check=True,
    )
    notes = (tmp_path / "notes.md").read_text()
    assert notes.lstrip().startswith("### Highlights")
    assert "## [" not in notes  # stops before the next version
    assert "### Highlights" in notes and len(notes.splitlines()) > 3


def test_citation_file_describes_the_release():
    citation = _citation()
    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == VERSION
    assert str(citation["date-released"]) == _changelog_date()
    assert citation["license"] == "MIT"
    assert citation["repository-code"] == REPO_URL
    assert citation["title"] and citation["message"] and citation["abstract"]
    assert citation["authors"][0]["family-names"] == "Popovici"


def test_classifier_is_alpha():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "Development Status :: 3 - Alpha" in project["classifiers"]
    assert not any("Pre-Alpha" in c for c in project["classifiers"])


def test_every_bibliography_entry_is_verified():
    # Comment lines (%) explain the marker; only entries must not carry it.
    bib = (ROOT / "docs" / "references.bib").read_text().splitlines()
    entries = [line for line in bib if not line.lstrip().startswith("%")]
    assert not [line for line in entries if "TODO-verify" in line]


def _tag_pointers():
    """Every 'pinned tag' in the files users copy from: (file, tag)."""
    files = [ROOT / "README.md", *sorted((ROOT / "book" / "chapters").glob("*.qmd"))]
    files += sorted((ROOT / "examples" / "colab_pytorch").glob("*.ipynb"))
    pattern = re.compile(r'glassnn@(v[\d.]+)|GLASSNN_TAG = \\?"(v[\d.]+)')
    return [
        (path.relative_to(ROOT), a or b)
        for path in files
        for a, b in pattern.findall(path.read_text())
    ]


def test_pinned_tags_name_this_release():
    pointers = _tag_pointers()
    assert len(pointers) >= 3  # README, chapter 1, notebook B
    assert {tag for _, tag in pointers} == {TAG}, pointers
