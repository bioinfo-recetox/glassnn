"""Check that every link from the rendered book into the API reference resolves.

Run after both builds (``sphinx-build docs docs/_build/html`` and
``quarto render book``): ``uv run python book/_check_api_links.py``.
Exits with status 1 and lists the broken links if a page or an anchor is
missing.
"""

import re
import sys
from pathlib import Path

from _helpers import API_URL

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book" / "_book"
API = ROOT / "docs" / "_build" / "html"


def broken_links() -> list[str]:
    broken = []
    pattern = re.compile(r'href="(' + re.escape(API_URL) + r'[^"]*)"')
    for chapter in sorted(BOOK.rglob("*.html")):
        for url in sorted(set(pattern.findall(chapter.read_text()))):
            page, _, anchor = url.removeprefix(API_URL).partition("#")
            target = API / page
            if not target.exists() or (
                anchor and f'id="{anchor}"' not in target.read_text()
            ):
                broken.append(f"{chapter.relative_to(BOOK)}: {url}")
    return broken


if __name__ == "__main__":
    problems = broken_links()
    for problem in problems:
        print("broken:", problem)
    print(f"{len(problems)} broken API links")
    sys.exit(1 if problems else 0)
