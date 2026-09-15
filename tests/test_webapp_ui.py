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
PAGE_REFERENCE = re.compile(r"<(script|link|img)\b[^>]*>", re.I)
SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*):", re.I)

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8"}


def _page_references(html: str) -> list[tuple[str, str]]:
    """(element, URL) for every <script src>, <link href> and <img src> in `html`."""
    references = []
    for match in PAGE_REFERENCE.finditer(html):
        element = match.group(1).lower()
        attribute = "href" if element == "link" else "src"
        value = re.search(rf"""\s{attribute}\s*=\s*["']([^"']*)["']""", match.group(0), re.I)
        if value:
            references.append((element, value.group(1).strip()))
    return references


def _leaves_the_page(element: str, reference: str) -> bool:
    """True for a URL that requests another origin or runs code: http:, https:,
    protocol-relative //, javascript:, and any other scheme. A relative path is
    not (it must still be a served file), and neither is a data: URL on <link>
    or <img>, which makes no request at all. Compared the way a browser reads
    it: tabs and newlines dropped, a backslash taken for a slash."""
    url = re.sub(r"[\t\n\r]", "", reference).replace(chr(92), "/")
    if url.startswith("//"):
        return True
    scheme = SCHEME.match(url)
    if scheme is None:
        return False
    return not (scheme.group(1).lower() == "data" and element in ("link", "img"))


def test_a_page_reference_may_be_a_data_url_but_never_leave_the_origin_or_run_code():
    fine = '<link rel="icon" href="data:,"><img src="data:image/png;base64,AAAA" alt=""><link rel="stylesheet" href="app.css">'
    assert _page_references(fine) == [("link", "data:,"), ("img", "data:image/png;base64,AAAA"), ("link", "app.css")]
    assert not any(_leaves_the_page(element, reference) for element, reference in _page_references(fine))

    backslashes = chr(92) * 2
    for sample in ('<script type="module" src="https://cdn.example/x.js"></script>',
                   '<link rel="stylesheet" href="http://cdn.example/x.css">',
                   '<script src="//cdn.example/x.js"></script>',
                   f'<script src="{backslashes}cdn.example/x.js"></script>',
                   '<link rel="icon" href="javascript:alert(1)">',
                   '<script src=" JavaScript:alert(1)"></script>',
                   '<script src="data:text/javascript,alert(1)"></script>'):
        [(element, reference)] = _page_references(sample)
        assert _leaves_the_page(element, reference), sample


def test_every_ui_file_the_page_loads_is_served(make):
    # Requested, not only listed: a module served as text/plain never runs in a
    # browser, and a route to a file that is not there is a 404 the page cannot survive.
    h = make()
    for url, (name, _) in webapp.UI_FILES.items():
        response = h.client.get(url)
        assert response.status_code == 200, f"{url} ({name}) answered {response.status_code}"
        assert response.headers["content-type"] == CONTENT_TYPES[Path(name).suffix], url
        assert response.content == (webapp.UI_DIR / name).read_bytes(), url

    html = (webapp.UI_DIR / "app.html").read_text(encoding="utf-8")
    references = {("/", element, reference) for element, reference in _page_references(html)}
    for url, (name, _) in webapp.UI_FILES.items():
        path = webapp.UI_DIR / name
        if name.endswith(".js") and path.exists():
            source = path.read_text(encoding="utf-8")
            references.update((url, "import", spec) for spec in IMPORT_SPECIFIER.findall(source))

    assert IMPORT_SPECIFIER.findall('import { api } from "./api.js";\nimport "./dom.js";') == ["./api.js", "./dom.js"]
    assert ("/", "script", "app.js") in references
    for base, element, reference in sorted(references):
        assert not _leaves_the_page(element, reference), f"{base}: <{element}> {reference} leaves the origin or runs code"
        if SCHEME.match(reference) is None:  # relative: it must be a file this app serves
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(base), reference))
            assert resolved in webapp.UI_FILES, f"{base} loads {reference} ({resolved}), which is not served"
