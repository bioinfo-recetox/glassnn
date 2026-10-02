"""Helpers for the GlassNN book: source listings and cross-links.

Chapters import these functions in a hidden first cell. They keep the book in
sync with the code: listings come from the installed package, and links are
computed from object names instead of being typed by hand.

- ``show_source(name)``: the real source code of a function, class or method.
- ``api_url(name)`` / ``api_link(name)``: the API reference page of an object.
- ``src_url(...)`` / ``src_link(...)``: the source on GitHub, at the current
  tag (or ``main`` between releases).
- ``see_also(*names)``: a line of API links.

The ``*_link`` functions and ``show_source`` return objects that Jupyter (and
hence Quarto) renders as Markdown.
"""

import importlib
import inspect
import subprocess
from dataclasses import dataclass
from pathlib import Path

SITE_URL = "https://bioinfo-recetox.github.io/glassnn/"
API_URL = SITE_URL + "api/"
REPO_URL = "https://github.com/bioinfo-recetox/glassnn"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Markdown:
    """A piece of Markdown that Jupyter displays as formatted text."""

    text: str

    def _repr_markdown_(self) -> str:
        return self.text


def _resolve(name: str):
    """Import the object with the dotted ``name`` (module, class, method...)."""
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError(f"Cannot resolve {name!r}.")


def _unwrap(obj):
    """The function behind a property, so that it has a module and a name."""
    return obj.fget if isinstance(obj, property) else obj


def _canonical_name(obj) -> tuple[str, str]:
    """Return (module name, full dotted name) where ``obj`` is defined."""
    if inspect.ismodule(obj):
        return obj.__name__, obj.__name__
    return obj.__module__, f"{obj.__module__}.{obj.__qualname__}"


def show_source(name: str) -> Markdown:
    """The source code of ``name`` as a fenced Python block."""
    source = inspect.getsource(_unwrap(_resolve(name)))
    return Markdown(f"```python\n{inspect.cleandoc(source)}\n```")


def api_url(name: str) -> str:
    """The URL of ``name`` in the API reference (one page per module)."""
    obj = _resolve(name)
    module, full_name = _canonical_name(_unwrap(obj))
    if inspect.ismodule(obj):
        return f"{API_URL}{module}.html#module-{module}"
    return f"{API_URL}{module}.html#{full_name}"


def api_link(name: str, text: str | None = None) -> Markdown:
    """A Markdown link to ``name`` in the API reference."""
    label = text if text is not None else f"`{name}`"
    return Markdown(f"[{label}]({api_url(name)})")


def _git_ref() -> str:
    """The tag of the current commit, or ``main`` if it is not tagged."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "main"
    return result.stdout.strip()


def src_url(target: str, start: int | None = None, end: int | None = None) -> str:
    """A permanent GitHub URL of source lines.

    Args:
        target: A repository path (then give ``start`` and ``end``) or the
            dotted name of an object, whose file and lines are found with
            ``inspect``; the latter cannot go out of date.
        start: First line (for a path).
        end: Last line (for a path).
    """
    if start is None:
        obj = _unwrap(_resolve(target))
        path = Path(inspect.getsourcefile(obj)).resolve().relative_to(REPO_ROOT)
        lines, start = inspect.getsourcelines(obj)
        end = start + len(lines) - 1
        target = path.as_posix()
    return f"{REPO_URL}/blob/{_git_ref()}/{target}#L{start}-L{end}"


def src_link(
    target: str, start: int | None = None, end: int | None = None, text: str = "source"
) -> Markdown:
    """A Markdown link to source lines on GitHub (see :func:`src_url`)."""
    return Markdown(f"[{text}]({src_url(target, start, end)})")


def see_also(*names: str) -> Markdown:
    """A "See also" line with API links to ``names``."""
    links = ", ".join(api_link(name).text for name in names)
    return Markdown(f"**See also:** {links}")
