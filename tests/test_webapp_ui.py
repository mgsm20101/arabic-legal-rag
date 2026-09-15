"""The chat page's files, served from an explicit allowlist — Phase 5.

Every file the page loads is served, and nothing else under ui/ is
reachable, not even by a path that walks out of js/.
"""

import json
import posixpath
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

import legalrag.webapp as webapp  # noqa: E402
from webapp_harness import Harness, assert_error, raw_request  # noqa: E402


@pytest.fixture
def make(tmp_path):
    def build(**options) -> Harness:
        return Harness(tmp_path / "library", **options)
    return build


def test_only_the_allowlisted_ui_files_are_served(make):
    h = make()
    for path, content_type in (("/", "text/html; charset=utf-8"), ("/app.css", "text/css; charset=utf-8"),
                               ("/app.js", "text/javascript; charset=utf-8")):
        response = h.client.get(path)
        assert response.status_code == 200, path
        assert response.headers["content-type"] == content_type
        assert response.content == (webapp.UI_DIR / webapp.UI_FILES[path][0]).read_bytes()

    for path in ("/index.html", "/ui/app.js", "/js/../app.js", "/js/missing.js", "/../README.md",
                 "/app.html", "/js/", "/docs", "/openapi.json"):
        status, _, content = raw_request(h.app, "GET", path)
        assert (status, json.loads(content)["error"]) == (404, "not_found"), path


def test_a_listed_ui_file_that_does_not_exist_yet_is_404(make, tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "app.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    h = make(ui_dir=ui)

    assert h.client.get("/").status_code == 200
    for path in webapp.UI_FILES:
        if path != "/":
            assert_error(h.client.get(path), 404, "not_found")


IMPORT_SPECIFIER = re.compile(r"""\b(?:import|export)\s*(?:[^'"`;]*?\bfrom\s*)?\(?\s*["']([^"']+)["']""")


def test_every_ui_file_the_page_loads_is_served():
    html = (webapp.UI_DIR / "app.html").read_text(encoding="utf-8")
    references = set()
    for tag in re.findall(r"<(?:script|link)\b[^>]*>", html, re.I):
        attribute = "src" if tag[1:].lower().startswith("script") else "href"
        found = re.search(rf"""\s{attribute}\s*=\s*["']([^"']+)["']""", tag, re.I)
        if found:
            references.add(("/", found.group(1)))
    for url, (name, _) in webapp.UI_FILES.items():
        path = webapp.UI_DIR / name
        if name.endswith(".js") and path.exists():
            references.update((url, spec) for spec in IMPORT_SPECIFIER.findall(path.read_text(encoding="utf-8")))

    assert IMPORT_SPECIFIER.findall('import { api } from "./api.js";\nimport "./dom.js";') == ["./api.js", "./dom.js"]
    assert ("/", "app.js") in references
    for base, reference in sorted(references):
        assert not re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", reference, re.I), f"{reference} leaves the origin"
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(base), reference))
        assert resolved in webapp.UI_FILES, f"{base} loads {reference} ({resolved}), which is not served"
