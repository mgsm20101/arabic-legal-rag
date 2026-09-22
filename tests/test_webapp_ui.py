"""The chat page's files, served from an explicit allowlist — Phase 5.

Every file the page loads is served, and nothing else under ui/ is
reachable, not even by a path that walks out of js/.
"""

import json
import posixpath
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

import legalrag.webapp as webapp  # noqa: E402
from ui_page import SCHEME, import_specifiers, leaves_the_page, page_references  # noqa: E402
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

    # `no_such_endpoint`, not `not_found`: none of these is a document, and the allowlist is
    # what makes them 404 in the first place. The codes are kept apart so a reader of the
    # response can tell "you asked for a path I do not serve" from "that document is gone".
    for path in ("/index.html", "/ui/app.js", "/js/../app.js", "/js/missing.js", "/../README.md",
                 "/app.html", "/js/", "/docs", "/openapi.json"):
        status, _, content = raw_request(h.app, "GET", path)
        assert (status, json.loads(content)["error"]) == (404, "no_such_endpoint"), path


def test_a_listed_ui_file_that_does_not_exist_yet_is_404(make, tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "app.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    h = make(ui_dir=ui)

    assert h.client.get("/").status_code == 200
    for path in webapp.UI_FILES:
        if path != "/":
            assert_error(h.client.get(path), 404, "not_found")


CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8"}


def test_a_page_reference_may_be_a_data_url_but_never_leave_the_origin_or_run_code():
    fine = '<link rel="icon" href="data:,"><img src="data:image/png;base64,AAAA" alt=""><link rel="stylesheet" href="app.css">'
    assert page_references(fine) == [("link", "data:,"), ("img", "data:image/png;base64,AAAA"), ("link", "app.css")]
    assert not any(leaves_the_page(element, reference) for element, reference in page_references(fine))

    backslashes = chr(92) * 2
    for sample in ('<script type="module" src="https://cdn.example/x.js"></script>',
                   '<link rel="stylesheet" href="http://cdn.example/x.css">',
                   '<script src="//cdn.example/x.js"></script>',
                   f'<script src="{backslashes}cdn.example/x.js"></script>',
                   '<link rel="icon" href="javascript:alert(1)">',
                   '<script src=" JavaScript:alert(1)"></script>',
                   '<script src="data:text/javascript,alert(1)"></script>',
                   '<iframe src="data:text/html,%3Cscript%3Ealert(1)%3C/script%3E"></iframe>',
                   '<form action="https://collector.example/q"></form>'):
        [(element, reference)] = page_references(sample)
        assert leaves_the_page(element, reference), sample


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
    references = {("/", element, reference) for element, reference in page_references(html)}
    for url, (name, _) in webapp.UI_FILES.items():
        path = webapp.UI_DIR / name
        if name.endswith(".js") and path.exists():
            source = path.read_text(encoding="utf-8")
            references.update((url, "import", spec) for spec in import_specifiers(source))

    imports = 'import { api } from "./api.js";\nimport "./dom.js";\nconst chat = () => import("./chat.js");'
    assert import_specifiers(imports) == ["./api.js", "./dom.js", "./chat.js"]
    assert ("/", "script", "app.js") in references
    for base, element, reference in sorted(references):
        assert not leaves_the_page(element, reference), f"{base}: <{element}> {reference} leaves the origin or runs code"
        if SCHEME.match(reference) is None:  # relative: it must be a file this app serves
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(base), reference))
            assert resolved in webapp.UI_FILES, f"{base} loads {reference} ({resolved}), which is not served"
