"""Tests of book/_check_api_links.py on a small fake site.

The site has the deployed layout: the book at ``SITE_URL`` and the API
reference at ``SITE_URL/api/``.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "book"))

import _check_api_links as links  # noqa: E402
from _helpers import API_URL, SITE_URL  # noqa: E402


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<html><body>{body}</body></html>")


@pytest.fixture
def site(tmp_path):
    """A consistent site: every link resolves."""
    book, api = tmp_path / "book", tmp_path / "api"
    write(book / "index.html", '<a href="chapters/01.html#goals">1</a>')
    write(
        book / "chapters" / "01.html",
        '<h2 id="goals">Goals</h2>'
        '<a href="../index.html">home</a>'
        '<a href="#goals">here</a>'
        f'<a href="{API_URL}glassnn.tensor.html#glassnn.tensor.Tensor">Tensor</a>'
        '<a href="https://doi.org/10.1038/323533a0">external</a>'
        '<a href="mailto:someone@example.org">mail</a>',
    )
    write(
        api / "glassnn.tensor.html",
        '<dt id="glassnn.tensor.Tensor">Tensor</dt>'
        f'<a href="{SITE_URL}chapters/01.html">book</a>'
        '<a href="index.html">index</a>'
        '<link href="_static/basic.css?v=1234">',
    )
    write(api / "index.html", '<a href="glassnn.tensor.html">tensor</a>')
    (api / "_static").mkdir()
    (api / "_static" / "basic.css").write_text("")
    return book, api


def test_a_consistent_site_has_no_broken_links(site):
    assert links.broken_links(*site) == []


def test_a_missing_api_page_is_reported(site):
    book, api = site
    (api / "glassnn.tensor.html").unlink()
    broken = links.broken_links(book, api)
    assert any("glassnn.tensor.html" in problem for problem in broken)


def test_a_missing_anchor_in_the_api_is_reported(site):
    book, api = site
    write(api / "glassnn.tensor.html", "<p>no anchors</p>")
    broken = links.broken_links(book, api)
    assert broken == [
        f"book: chapters/01.html -> {API_URL}glassnn.tensor.html#glassnn.tensor.Tensor"
    ]


def test_an_api_link_to_a_missing_chapter_is_reported(site):
    book, api = site
    (book / "chapters" / "01.html").unlink()
    write(book / "index.html", "")
    broken = links.broken_links(book, api)
    assert broken == [f"api: glassnn.tensor.html -> {SITE_URL}chapters/01.html"]


def test_a_broken_relative_link_inside_the_book_is_reported(site):
    book, api = site
    write(book / "chapters" / "02.html", '<a href="03.html">next</a>')
    assert links.broken_links(book, api) == ["book: chapters/02.html -> 03.html"]


def test_a_missing_anchor_on_the_same_page_is_reported(site):
    book, api = site
    write(book / "chapters" / "02.html", '<a href="#nowhere">x</a>')
    assert links.broken_links(book, api) == ["book: chapters/02.html -> #nowhere"]


def test_html_escapes_and_percent_encoding_are_decoded(site):
    book, api = site
    write(
        book / "chapters" / "02.html",
        '<h2 id="a b">x</h2><a href="#a%20b">x</a>'
        '<a href="01.html?x=1&amp;y=2#goals">x</a>',
    )
    assert links.broken_links(book, api) == []


def test_a_relative_link_from_the_book_into_api_reaches_the_api_build(site):
    # Deployed, the API reference is the folder api/ of the book's site.
    book, api = site
    write(book / "chapters" / "02.html", '<a href="../api/index.html">API</a>')
    assert links.broken_links(book, api) == []
    (api / "index.html").unlink()
    assert "book: chapters/02.html -> ../api/index.html" in links.broken_links(
        book, api
    )
