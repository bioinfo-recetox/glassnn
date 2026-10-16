"""Tests of book/_helpers.py, the cross-referencing helpers of the book."""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "book"))

import _helpers  # noqa: E402


def test_show_source_renders_the_real_code_as_a_python_block():
    block = _helpers.show_source("glassnn.tensor._unbroadcast")
    markdown = block._repr_markdown_()
    assert markdown.startswith("```python\ndef _unbroadcast(")
    assert markdown.endswith("```")


def test_show_source_of_a_method():
    markdown = _helpers.show_source("glassnn.Tensor.backward")._repr_markdown_()
    assert "def backward(self" in markdown


@pytest.mark.parametrize(
    ("name", "page", "anchor"),
    [
        ("glassnn.Tensor", "glassnn.tensor", "glassnn.tensor.Tensor"),
        ("glassnn.Tensor.backward", "glassnn.tensor", "glassnn.tensor.Tensor.backward"),
        ("glassnn.Tensor.shape", "glassnn.tensor", "glassnn.tensor.Tensor.shape"),
        ("glassnn.backend", "glassnn.backend", "module-glassnn.backend"),
        (
            "glassnn.gradcheck.gradcheck",
            "glassnn.gradcheck",
            "glassnn.gradcheck.gradcheck",
        ),
    ],
)
def test_api_url_points_to_the_module_page_and_anchor(name, page, anchor):
    assert _helpers.api_url(name) == f"{_helpers.API_URL}{page}.html#{anchor}"


@pytest.mark.parametrize("name", ["glassnn.Tensor", "glassnn.backend"])
def test_every_linked_module_has_an_api_page(name):
    page = _helpers.api_url(name).removeprefix(_helpers.API_URL).split(".html")[0]
    assert (REPO_ROOT / "docs" / f"{page}.rst").exists()


def test_api_link_is_a_markdown_link_with_code_text():
    link = _helpers.api_link("glassnn.Tensor")._repr_markdown_()
    assert link == f"[`glassnn.Tensor`]({_helpers.api_url('glassnn.Tensor')})"
    custom = _helpers.api_link("glassnn.Tensor", "the Tensor class")._repr_markdown_()
    assert custom.startswith("[the Tensor class](")


def test_src_url_with_explicit_lines(monkeypatch):
    monkeypatch.setattr(_helpers, "_git_ref", lambda: "v1.2.3")
    url = _helpers.src_url("src/glassnn/tensor.py", 40, 75)
    assert url == f"{_helpers.REPO_URL}/blob/v1.2.3/src/glassnn/tensor.py#L40-L75"


def test_src_url_of_an_object_finds_its_lines(monkeypatch):
    monkeypatch.setattr(_helpers, "_git_ref", lambda: "main")
    url = _helpers.src_url("glassnn.tensor._unbroadcast")
    path, lines = url.removeprefix(f"{_helpers.REPO_URL}/blob/main/").split("#")
    assert path == "src/glassnn/tensor.py"
    start, end = (int(n) for n in lines.removeprefix("L").split("-L"))
    source = (REPO_ROOT / path).read_text().splitlines()
    assert source[start - 1].startswith("def _unbroadcast(")
    assert start < end


def test_git_ref_is_a_tag_or_main():
    ref = _helpers._git_ref()
    assert ref == "main" or ref.startswith("v")


def test_see_also_lists_api_links():
    markdown = _helpers.see_also("glassnn.Tensor", "glassnn.no_grad")._repr_markdown_()
    assert markdown.startswith("**See also:** [`glassnn.Tensor`](")
    assert "[`glassnn.no_grad`](" in markdown


def test_api_map_inverts_the_book_links_of_the_modules():
    chapters = _helpers.api_map()
    # Every module docstring links to at least one chapter (tests/test_docstrings).
    listed = {module for modules in chapters.values() for module in modules}
    assert "glassnn.tensor" in chapters["chapters/02-autodiff.html"]
    assert "glassnn.backend" in chapters["chapters/13-design-notes.html"]
    assert {"glassnn.optim.adam", "glassnn.nn.linear"} <= listed
    assert list(chapters) == sorted(chapters)


def test_api_map_table_links_every_module():
    table = _helpers.api_map_table().text
    for modules in _helpers.api_map().values():
        for module in modules:
            assert f"[`{module}`]({_helpers.API_URL}{module}.html)" in table


def test_the_reading_list_cites_every_bibliography_entry():
    bib = (REPO_ROOT / "docs" / "references.bib").read_text()
    keys = re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE)
    reading_list = (REPO_ROOT / "book" / "chapters" / "c-reading-list.qmd").read_text()
    missing = [key for key in keys if f"@{key}" not in reading_list]
    assert not missing, missing
