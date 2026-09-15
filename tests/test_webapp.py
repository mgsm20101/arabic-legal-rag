"""The local app's HTTP API — P1 demo layer (ADR-023), Phase 5 task 2.

What must hold whoever calls it: an upload is checked before it touches disk
and its filename never becomes a path; chat shows only what `cite.gate` kept;
only a same-origin page on a loopback Host can change anything; every error
is a stable code with an Arabic sentence, never exception text; and only the
page's own allowlisted files are served. No model and no network: a real
`Library` over `tmp_path` with a keyword encoder, and scripted chat models
behind a real `ClaimsGenerator` and `Pipeline`.
"""

import json
import logging
import posixpath
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

import legalrag.library as library_mod  # noqa: E402
import legalrag.webapp as webapp  # noqa: E402
from legalrag.library import MAX_UPLOAD_BYTES  # noqa: E402
from legalrag.ollama import GeneratorUnavailable  # noqa: E402
from legalrag.webapp import APP_HEADER, RateLimiter  # noqa: E402
from stubs import ANSWERS_YES, claims_json, text_document  # noqa: E402
from webapp_harness import (  # noqa: E402
    HEADERS,
    POLICY,
    QUESTION,
    FakeClock,
    Harness,
    assert_error,
    assert_security_headers,
    raw_request,
    tree,
)


@pytest.fixture
def make(tmp_path):
    def build(**options) -> Harness:
        return Harness(tmp_path / "library", **options)
    return build


# --- upload ---------------------------------------------------------------------


def test_an_upload_answers_201_with_the_document_and_the_list_shows_it(make):
    h = make()
    assert h.client.get("/api/documents").json() == {"documents": []}

    response = h.upload()

    assert response.status_code == 201, response.text
    doc = response.json()
    assert set(doc) == {"doc_id", "title", "kind", "suffix", "pages", "chunks", "size_bytes", "created_at"}
    assert (doc["title"], doc["suffix"], doc["pages"], doc["size_bytes"]) == ("policy.txt", ".txt", 2, len(POLICY))
    assert h.client.get("/api/documents").json() == {"documents": [doc]}


def test_a_file_that_is_not_pdf_or_txt_is_rejected_before_it_touches_disk(make):
    h = make()
    before = tree(h.root)

    for filename in ("notes.docx", "policy.txt.exe", "README"):
        assert_error(h.upload(b"PK\x03\x04 a zip, not a document", filename), 400, "unsupported_file")

    assert tree(h.root) == before
    assert h.library.documents() == []


def test_a_pdf_extension_without_pdf_magic_bytes_is_rejected(make):
    h = make()
    before = tree(h.root)

    assert_error(h.upload("مستند عربي طويل بما يكفي ".encode("utf-8") * 40, "report.pdf"), 400, "unsupported_file")

    assert tree(h.root) == before


def test_the_client_filename_never_becomes_a_path(make, tmp_path):
    h = make()

    response = h.upload(POLICY, "../../escape/evil.txt")

    assert response.status_code == 201, response.text
    doc = response.json()
    assert doc["title"] == "evil.txt"
    assert (h.root / "docs" / doc["doc_id"] / "source.txt").read_bytes() == POLICY
    assert [p.name for p in tmp_path.iterdir()] == ["library"]
    assert not [p for p in tmp_path.rglob("*") if "evil" in p.name or "escape" in p.name]


def test_an_upload_larger_than_the_limit_is_rejected_before_the_body_is_parsed(make, monkeypatch):
    h = make()
    multipart = ("content-type", "multipart/form-data; boundary=b")
    too_long = str(MAX_UPLOAD_BYTES + 1024 * 1024 + 1)

    # raw_request fails the test if anything reads the body
    status, _, content = raw_request(h.app, "POST", "/api/documents",
                                     [(APP_HEADER, "1"), multipart, ("content-length", too_long)])
    assert (status, json.loads(content)["error"]) == (413, "file_too_large")

    status, _, content = raw_request(h.app, "POST", "/api/documents",
                                     [(APP_HEADER, "1"), multipart, ("transfer-encoding", "chunked")])
    assert (status, json.loads(content)["error"]) == (411, "invalid_request")

    # Inside an allowed body, the part itself is read no further than the limit plus one byte.
    monkeypatch.setattr(library_mod, "MAX_UPLOAD_BYTES", 4096)

    def refuse(data, filename):
        raise AssertionError("an oversized part reached Library.add")

    monkeypatch.setattr(h.library, "add", refuse)
    assert_error(h.upload(b"x" * 5000, "big.txt"), 413, "file_too_large")


# --- chat -----------------------------------------------------------------------


def test_chat_returns_only_claims_that_survived_the_gate(make):
    kept = "يصرف بدل الإنترنت الشهري للموظف."
    uncited = "جملة بلا مصدر لا يجوز أن تصل إلى الصفحة."
    invented = "جملة تستشهد بمصدر لم يُعرض على النموذج."
    h = make(relevance=[ANSWERS_YES], claims=[claims_json((kept, [1]), (uncited, []), (invented, [9]))])
    assert h.upload().status_code == 201

    response = h.ask()

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"status", "abstain_reason", "claims", "sources", "dropped", "timings_ms"}
    assert (body["status"], body["abstain_reason"]) == ("partial", None)
    assert body["claims"] == [{"text": kept, "sources": [1]}]
    assert body["dropped"] == {"uncited": 1, "fabricated": 1, "ungrounded": 0}
    assert {"n", "doc_title", "label", "page", "text"} <= set(body["sources"][0])
    assert uncited not in response.text and invented not in response.text


def test_chat_with_no_documents_abstains_without_calling_the_model(make):
    h = make()

    for doc_ids in (None, []):
        response = h.ask(doc_ids=doc_ids)
        assert response.status_code == 200, response.text
        assert (response.json()["status"], response.json()["abstain_reason"]) == ("abstained", "no_sources")

    assert h.relevance.seen == [] and h.claims.seen == []


def test_an_unknown_document_in_the_scope_is_404(make):
    h = make()
    h.upload()

    assert_error(h.ask(doc_ids=["0123456789ab"]), 404, "not_found")
    assert h.relevance.seen == []


def test_an_unreachable_model_server_is_503_and_the_exception_text_never_reaches_the_client(make, caplog):
    h = make(relevance=[GeneratorUnavailable("http://secret-host:9 raw body")])
    h.upload()

    with caplog.at_level(logging.INFO, logger="legalrag.webapp"):
        response = h.ask()

    assert_error(response, 503, "generator_unavailable")
    assert "secret-host" not in response.text and "raw body" not in response.text
    assert "secret-host" in caplog.text, "the cause must still be logged on the server"


def test_an_encoder_that_cannot_load_is_503_for_upload_and_chat(make):
    h = make()
    h.upload()
    h.encoder.broken = True

    for response in (h.upload(text_document("نص جديد عن الإجازة"), "leave.txt"), h.ask()):
        assert_error(response, 503, "encoder_unavailable")
        assert "secret" not in response.text


# --- validation -----------------------------------------------------------------


def test_a_validation_error_does_not_echo_the_submitted_input(make):
    h = make()
    secret = "SUBMITTED-INPUT-7c1f"
    for body in ({"question": 1234567890123}, {"question": QUESTION, "doc_ids": secret},
                 {"question": QUESTION, "doc_ids": [secret]}, {"doc_ids": [secret]}, [secret]):
        response = h.client.post("/api/chat", json=body, headers=HEADERS)
        assert_error(response, 400, "invalid_request")
        assert secret not in response.text and "1234567890123" not in response.text

    broken = h.client.post("/api/chat", content=f'{{"question": "{secret}"'.encode(),
                           headers={**HEADERS, "content-type": "application/json"})
    assert_error(broken, 400, "invalid_request")
    assert secret not in broken.text


def test_invalid_request_blames_the_question_only_when_its_length_is_the_problem(make):
    h = make()
    about_the_question = "السؤال يجب أن يكون بين 3 و500 حرف."
    generic = "طلب غير صالح."

    for question in ("  أب  ", "س" * 501):
        assert assert_error(h.ask(question), 400, "invalid_request")["message_ar"] == about_the_question
    too_many = ["0123456789ab"] * (webapp.MAX_DOC_IDS + 1)
    for doc_ids in (too_many, ["../../etc/passwd"], ["ABCDEF123456"], [7]):
        assert assert_error(h.ask(doc_ids=doc_ids), 400, "invalid_request")["message_ar"] == generic
    not_a_string = h.client.post("/api/chat", json={"question": ["س"] * 5}, headers=HEADERS)
    assert assert_error(not_a_string, 400, "invalid_request")["message_ar"] == generic
    assert h.relevance.seen == []


def test_every_error_code_has_its_arabic_sentence():
    expected = {
        "unsupported_file": "نوع الملف غير مدعوم. ارفع ملف PDF أو TXT.",
        "file_too_large": "حجم الملف أكبر من الحد المسموح (20 ميجابايت).",
        "no_text": "لم يُعثر على نص كافٍ في الملف. إن كان PDF ممسوحاً ضوئياً، "
                   "فالتعرّف الضوئي (OCR) غير مدعوم في هذه النسخة.",
        "not_found": "المستند غير موجود.",
        "invalid_request": "طلب غير صالح.",
        "rate_limited": "طلبات كثيرة في وقت قصير. حاول مرة أخرى بعد قليل.",
        "generator_unavailable": "خادم النموذج المحلي (Ollama) غير متاح. شغّله ثم أعد المحاولة.",
        "encoder_unavailable": "نموذج الاسترجاع غير متاح على الخادم.",
        "forbidden": "طلب مرفوض.",
        "internal": "حدث خطأ غير متوقع.",
    }
    assert webapp.MESSAGES_AR == expected
    assert webapp.QUESTION_MESSAGE_AR == "السؤال يجب أن يكون بين 3 و500 حرف."


# --- who may call it ------------------------------------------------------------


def test_a_post_or_delete_without_the_app_header_is_forbidden(make):
    h = make()
    doc = h.upload().json()

    for headers in ({}, {APP_HEADER: "0"}, {APP_HEADER: "true"}):
        assert_error(h.upload(text_document("مستند آخر عن الإجازة"), "other.txt", headers=headers), 403, "forbidden")
        assert_error(h.ask(headers=headers), 403, "forbidden")
        assert_error(h.client.delete(f"/api/documents/{doc['doc_id']}", headers=headers), 403, "forbidden")

    assert [d.doc_id for d in h.library.documents()] == [doc["doc_id"]]
    assert h.relevance.seen == []
    # a cross-site form post is refused before its body is read
    status, _, content = raw_request(h.app, "POST", "/api/documents",
                                     [("content-type", "multipart/form-data; boundary=b"), ("content-length", "64")])
    assert (status, json.loads(content)["error"]) == (403, "forbidden")


def test_a_foreign_origin_is_forbidden(make):
    h = make()
    for origin in ("http://evil.example", "null", "https://127.0.0.1", "http://127.0.0.1:8001",
                   "http://127.0.0.1.evil.example"):
        assert_error(h.ask(headers={**HEADERS, "origin": origin}), 403, "forbidden")
        assert_error(h.upload(headers={**HEADERS, "origin": origin}), 403, "forbidden")
    assert h.library.documents() == []

    # The client is on port 80, which a browser leaves out of Origin.
    for origin in ("http://127.0.0.1", "http://localhost"):
        assert h.ask(headers={**HEADERS, "origin": origin}).status_code == 200
    on_8765 = {**HEADERS, "host": "localhost:8765"}
    assert h.ask(headers={**on_8765, "origin": "http://127.0.0.1:8765"}).status_code == 200
    assert_error(h.ask(headers={**on_8765, "origin": "http://localhost:8000"}), 403, "forbidden")


def test_a_foreign_host_header_is_rejected(make):
    h = make()
    for host in ("evil.example", "127.0.0.1.evil.example", "evil.example:8000", "[::1]:8000", "localhost.:8000"):
        assert_error(h.client.get("/api/health", headers={"host": host}), 400, "invalid_request")
        assert_error(h.client.get("/", headers={"host": host}), 400, "invalid_request")
        assert_error(h.ask(headers={**HEADERS, "host": host}), 400, "invalid_request")
    for host in ("127.0.0.1:8000", "localhost:8765", "LOCALHOST"):
        assert h.client.get("/api/documents", headers={"host": host}).status_code == 200


def test_delete_is_soft_and_a_malformed_id_is_404(make, monkeypatch):
    h = make()
    doc = h.upload().json()
    doc_dir = h.root / "docs" / doc["doc_id"]

    response = h.client.delete(f"/api/documents/{doc['doc_id']}", headers=HEADERS)

    assert (response.status_code, response.json()) == (200, {"deleted": doc["doc_id"]})
    assert h.client.get("/api/documents").json() == {"documents": []}
    assert json.loads((doc_dir / "meta.json").read_text(encoding="utf-8"))["deleted"] is True
    assert (doc_dir / "source.txt").read_bytes() == POLICY
    assert_error(h.client.delete(f"/api/documents/{doc['doc_id']}", headers=HEADERS), 404, "not_found")
    assert_error(h.client.delete("/api/documents/0123456789ab", headers=HEADERS), 404, "not_found")

    def refuse(doc_id):
        raise AssertionError(f"soft_delete was called with {doc_id!r}")

    monkeypatch.setattr(h.library, "soft_delete", refuse)
    for malformed in ("ABCDEF123456", "abc", "0123456789abc", "0123456789a!", "%2e%2e%2fdocs"):
        assert_error(h.client.delete(f"/api/documents/{malformed}", headers=HEADERS), 404, "not_found")


def test_the_rate_limit_answers_429_with_retry_after(make):
    clock = FakeClock()
    h = make(chat_limiter=RateLimiter(2, 60.0, clock=clock), upload_limiter=RateLimiter(1, 60.0, clock=clock))

    assert h.ask().status_code == 200
    clock.now += 10.0
    assert h.ask().status_code == 200
    clock.now += 0.5
    refused = h.ask()
    assert_error(refused, 429, "rate_limited")
    assert refused.headers["retry-after"] == "50"  # 49.5 s, rounded up
    clock.now = 1060.0
    assert h.ask().status_code == 200

    assert h.upload().status_code == 201
    refused = h.upload(text_document("مستند ثان عن الإجازة"), "leave.txt")
    assert_error(refused, 429, "rate_limited")
    assert refused.headers["retry-after"] == "60"


def test_a_rate_limited_upload_is_refused_before_its_body_is_read(make):
    h = make(upload_limiter=RateLimiter(1, 60.0, clock=FakeClock()))
    upload = [(APP_HEADER, "1"), ("content-type", "multipart/form-data; boundary=b"), ("content-length", "64")]

    first, _, _ = raw_request(h.app, "POST", "/api/documents", upload, body=b"")
    assert first == 400  # an empty multipart body: counted, then refused
    status, headers, content = raw_request(h.app, "POST", "/api/documents", upload)  # body=None: unread
    assert (status, json.loads(content)["error"], headers["retry-after"]) == (429, "rate_limited", "60")


# --- health ---------------------------------------------------------------------


def test_health_never_exposes_the_host_or_exception_text(make):
    def leaky():
        return {"reachable": False, "model": "ollama:gemma3:4b", "gpu_share": None, "version": "0.0-secret",
                "error": "cannot reach Ollama at http://secret-host:9 (raw body)", "host": "http://secret-host:9"}

    def exploding():
        raise GeneratorUnavailable("http://secret-host:9 raw body")

    def up():
        return {"reachable": True, "model": "ollama:gemma3:4b", "gpu_share": 1.0, "version": "0.20.3-secret"}

    cases = (
        (leaky, "degraded", {"reachable": False, "model": "ollama:gemma3:4b", "gpu_share": None}),
        (exploding, "degraded", {"reachable": False, "model": None, "gpu_share": None}),
        (up, "ok", {"reachable": True, "model": "ollama:gemma3:4b", "gpu_share": 1.0}),
    )
    for probe, status, generator in cases:
        h = make(health_probe=probe)
        response = h.client.get("/api/health")
        assert response.status_code == 200, response.text
        assert response.json() == {"status": status, "generator": generator, "documents": 0}
        assert "secret" not in response.text

    h.upload()
    assert h.client.get("/api/health").json()["documents"] == 1


# --- responses, and what is served ----------------------------------------------

def test_every_response_carries_the_security_headers(make):
    h = make(chat_limiter=RateLimiter(1, 60.0, clock=FakeClock()))
    responses = {
        "the page": h.client.get("/"),
        "a 200": h.client.get("/api/documents"),
        "a 400": h.client.post("/api/chat", json={"question": 5}, headers=HEADERS),
        "a 404": h.client.get("/nowhere"),
        "a 403": h.client.post("/api/chat", json={"question": QUESTION}),
        "a foreign host": h.client.get("/api/health", headers={"host": "evil.example"}),
        "a 429": h.ask(),
    }
    statuses = {label: r.status_code for label, r in responses.items()}
    assert statuses == {"the page": 200, "a 200": 200, "a 400": 400, "a 404": 404, "a 403": 403,
                        "a foreign host": 400, "a 429": 429}
    for label, response in responses.items():
        assert_security_headers(response.headers, label)


def test_an_unexpected_error_is_500_with_no_detail(make, monkeypatch, caplog):
    h = make()

    def explode():
        raise RuntimeError("secret detail from /secret/path")

    monkeypatch.setattr(h.library, "documents", explode)
    client = TestClient(h.app, base_url="http://127.0.0.1", raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR):
        response = client.get("/api/documents")

    assert_error(response, 500, "internal")
    assert "secret" not in response.text
    assert_security_headers(response.headers, "a 500")
    assert "secret detail" in caplog.text


def test_a_library_fault_on_this_side_is_500_and_logged_with_its_traceback(make, monkeypatch, caplog):
    h = make()

    class DiskGone(library_mod.LibraryError):
        code = "internal"

    def fail():
        raise DiskGone("the disk under /secret/data went away")

    monkeypatch.setattr(h.library, "documents", fail)

    with caplog.at_level(logging.INFO, logger="legalrag.webapp"):
        response = h.client.get("/api/documents")

    assert_error(response, 500, "internal")
    assert "secret" not in response.text
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert errors and errors[0].exc_info, "a fault on this side is logged as an error, with its traceback"


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
