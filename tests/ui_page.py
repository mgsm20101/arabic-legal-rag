"""Shared by the chat page's tests (tests/test_ui_*.py and tests/test_webapp_ui.py); not a test module.

The page is ui/app.html, ui/app.css, and ui/app.js with its five modules under
ui/js/, served from the allowlist in legalrag.webapp. These helpers read those
files as text, pick out what a script imports and what the markup loads, and
copy the scripts to where Node can run them; the tests that need Node skip
without it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The server module needs these; without them the page's tests skip, as the server's own do.
pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

import legalrag.webapp as webapp  # noqa: E402

UI = webapp.UI_DIR

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not on PATH")

TAG = re.compile(r"<[a-zA-Z][^>]*>")

# The scripts the server serves, and nothing else: the entry and its modules.
ALLOWLISTED_JS = tuple(name for name, _content_type in webapp.UI_FILES.values() if name.endswith(".js"))

# A module specifier: `import … from "x"`, `export … from "x"` and a bare `import "x"`, each opening
# its line, so that the word inside a string or a comment does not count; and `import("x")` anywhere.
IMPORT_SPECIFIER = re.compile(
    r"^\s*(?:import|export)\b[^;'\"`]*?\bfrom\s*(['\"])(?P<static>[^'\"]+)\1"
    r"|^\s*import\s*(['\"])(?P<bare>[^'\"]+)\3"
    r"|(?<![\w$.])import\s*\(\s*(['\"`])(?P<dynamic>[^'\"`]+)\5\s*\)",
    re.M,
)

# The attributes whose value is a URL the page may load, run, or send the reader to.
URL_ATTRIBUTES = ("src", "href", "xlink:href", "srcset", "action", "formaction", "poster", "data")
SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*):", re.I)


def read(name: str) -> str:
    return (UI / name).read_text(encoding="utf-8")


def all_js() -> str:
    """Every allowlisted script as one text, so that no scan can miss a module."""
    return "\n;\n".join(read(name) for name in ALLOWLISTED_JS)


def attr(tag: str, name: str) -> str | None:
    """The value of attribute `name` in one start tag, or None."""
    match = re.search(
        rf"\s{re.escape(name)}\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))", tag, re.I
    )
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


def import_specifiers(source: str) -> list[str]:
    """Every module specifier `source` imports or re-exports from, static or dynamic, in order."""
    return [
        match.group("static") or match.group("bare") or match.group("dynamic")
        for match in IMPORT_SPECIFIER.finditer(source)
    ]


def page_references(html: str) -> list[tuple[str, str]]:
    """(element, URL) for every URL attribute of every tag in `html`, #fragments left out."""
    references = []
    for tag in TAG.findall(html):
        element = re.match(r"<([a-zA-Z][\w:-]*)", tag).group(1).lower()
        for name in URL_ATTRIBUTES:
            value = attr(tag, name)
            if value is not None and not value.strip().startswith("#"):
                references.append((element, value.strip()))
    return references


def leaves_the_page(element: str, url: str) -> bool:
    """True for a URL that requests another origin or runs code: http:, https:,
    protocol-relative //, javascript:, and any other scheme, data: among them. A
    relative path is not (it must still be a served file), and neither is a data:
    URL on <link> or <img>, which makes no request and runs nothing. Compared the
    way a browser reads it: tabs and newlines dropped, a backslash taken for a slash."""
    url = re.sub(r"[\t\n\r]", "", url).replace("\\", "/")
    if url.startswith("//"):
        return True
    scheme = SCHEME.match(url)
    if scheme is None:
        return False
    return not (scheme.group(1).lower() == "data" and element in ("link", "img"))


def copy_scripts_for_node(tmp_path: Path) -> Path:
    """The page's scripts, copied under a package.json that has Node read .js as ES modules."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    for name in ALLOWLISTED_JS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(read(name), encoding="utf-8")
    return tmp_path


def run_node(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [NODE, *args], input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=60
    )
