"""Who may call the local app, and what every answer carries — Phase 5.

Only a same-origin page on a loopback Host can change anything; uploads and
chats are rate-limited per client; every response carries the security
headers; health never shows the host or exception text; and every error is a
stable code with its Arabic sentence, a fault on this side logged with its
traceback. The harness is tests/webapp_harness.py: no model, no network.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

import legalrag.library as library_mod  # noqa: E402
import legalrag.webapp as webapp  # noqa: E402
from legalrag.ollama import GeneratorUnavailable  # noqa: E402
from legalrag.webapp import APP_HEADER, RateLimiter  # noqa: E402
from stubs import text_document  # noqa: E402
from webapp_harness import (  # noqa: E402
    HEADERS,
    QUESTION,
    FakeClock,
    Harness,
    assert_error,
    assert_security_headers,
    raw_request,
)


@pytest.fixture
def make(tmp_path):
    def build(**options) -> Harness:
        return Harness(tmp_path / "library", **options)
    return build


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
