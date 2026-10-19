"""Check the links of the rendered book and API reference.

Run after both builds (``sphinx-build docs docs/_build/html`` and
``quarto render book``): ``uv run python book/_check_api_links.py``.

Every link in every HTML page of both sites is followed, if it stays on the
site:

- book -> API reference, and API reference -> book (absolute links to the
  deployed site, which serves the book at ``SITE_URL`` and the API
  reference at ``SITE_URL/api/``);
- links inside the book and inside the API reference (relative links,
  including ``#anchor`` links on the same page).

The target page must exist and, if the link has an anchor, contain an
element with that ``id``. Links to other hosts are not checked here (that
needs the network); run ``uv run sphinx-build -b linkcheck docs
docs/_build/linkcheck`` by hand before a release.

Exits with status 1 and lists the broken links.
"""

import html
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from _helpers import API_URL, SITE_URL

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book" / "_book"
API = ROOT / "docs" / "_build" / "html"

HREF = re.compile(r'href="([^"]*)"')
SKIPPED_SCHEMES = ("mailto:", "javascript:", "data:", "tel:")


def _target(page: Path, href: str, book: Path, api: Path) -> tuple[Path, str] | None:
    """The file and anchor that ``href`` on ``page`` points to.

    Returns ``None`` for links that leave the site.
    """
    href = html.unescape(href)
    if href.startswith(SKIPPED_SCHEMES):
        return None
    address, _, anchor = href.partition("#")
    address = address.partition("?")[0]
    if address.startswith(API_URL):
        path = api / address.removeprefix(API_URL)
    elif address.startswith(SITE_URL):
        path = book / address.removeprefix(SITE_URL)
    elif "://" in address or address.startswith("//"):
        return None
    elif address == "":
        path = page
    else:
        path = page.parent / unquote(address)
    # Deployed, the API reference is the folder api/ of the book's site.
    if path.resolve().is_relative_to(book.resolve() / "api"):
        path = api / path.resolve().relative_to(book.resolve() / "api")
    if path.is_dir() or address.endswith("/"):
        path = path / "index.html"
    return path, unquote(anchor)


def _has_anchor(page_text: str, anchor: str) -> bool:
    return f'id="{html.escape(anchor)}"' in page_text


def broken_links(book: Path = BOOK, api: Path = API) -> list[str]:
    """Every on-site link of both sites whose page or anchor is missing.

    Each entry reads ``"<site>: <page> -> <link>"``.
    """
    texts: dict[Path, str] = {}

    def text(path: Path) -> str:
        if path not in texts:
            texts[path] = path.read_text(errors="replace")
        return texts[path]

    broken = []
    for site, root in (("book", book), ("api", api)):
        for page in sorted(root.rglob("*.html")):
            for href in sorted(set(HREF.findall(text(page)))):
                target = _target(page, href, book, api)
                if target is None:
                    continue
                path, anchor = target
                exists = path.is_file()
                if exists and anchor and path.suffix == ".html":
                    exists = _has_anchor(text(path), anchor)
                if not exists:
                    broken.append(f"{site}: {page.relative_to(root)} -> {href}")
    return broken


if __name__ == "__main__":
    problems = broken_links()
    for problem in problems:
        print("broken:", problem)
    print(f"{len(problems)} broken links")
    sys.exit(1 if problems else 0)
