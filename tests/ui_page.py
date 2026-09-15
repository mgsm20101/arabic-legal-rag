"""Shared by the chat page's tests (tests/test_ui_*.py); not a test module.

The page is ui/app.html, ui/app.css, and ui/app.js with its five modules under
ui/js/. These helpers read those files as text, and copy the scripts to where Node
can run them; the tests that need Node skip without it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not on PATH")

TAG = re.compile(r"<[a-zA-Z][^>]*>")

# The scripts the server serves, and nothing else: the entry and its five modules.
ALLOWLISTED_JS = ("app.js", "js/copy.js", "js/api.js", "js/dom.js", "js/documents.js", "js/chat.js")


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
