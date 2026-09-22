"""Documents and chat through the local app's HTTP API — P1 demo layer (ADR-023), Phase 5.

An upload is checked before it touches disk and its filename never becomes a
path; a delete is soft; chat shows only what `cite.gate` kept, and says why
when it shows nothing; a model or an encoder that is down is 503 with no
exception text; and a request that does not validate is 400 without echoing
what was sent. No model and no network: `webapp_harness.Harness` is a real
`Library` over `tmp_path` with a keyword encoder, and scripted chat models
behind a real `ClaimsGenerator` and `Pipeline`. Who may call the API is
tests/test_webapp_guard.py; the page's files are tests/test_webapp_ui.py.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

import legalrag.docextract as docextract_mod  # noqa: E402
import legalrag.library as library_mod  # noqa: E402
import legalrag.webapp as webapp  # noqa: E402
from legalrag.claims import ClaimsGenerator  # noqa: E402
from legalrag.library import MAX_UPLOAD_BYTES  # noqa: E402
from legalrag.ollama import GeneratorUnavailable  # noqa: E402
from legalrag.pipeline import Pipeline  # noqa: E402
from legalrag.webapp import APP_HEADER  # noqa: E402
from stubs import ANSWERS_YES, KeywordEncoder, ScriptedChat, claims_json, text_document  # noqa: E402
from webapp_harness import HEADERS, POLICY, QUESTION, Harness, assert_error, raw_request, tree  # noqa: E402


@pytest.fixture
def make(tmp_path):
    def build(**options) -> Harness:
        return Harness(tmp_path / "library", **options)
    return build


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


def test_a_non_deadline_extraction_bug_on_the_second_pass_is_500_never_400(make, monkeypatch):
    """F1 at the HTTP layer: the first extraction pass already proved the upload readable, so a
    bug on the second (Arabic-only) pass is StorageError (`internal`), which _STATUS maps to
    500 — never `unsupported_file`'s 400 — and the raw exception text never reaches the client."""
    statute_like = "مادة 1\nنص المادة الأولى مصحوب بترجمة"
    calls: list[bool] = []

    def flaky(path, keep_latin, max_pages, deadline):
        calls.append(keep_latin)
        if not keep_latin:
            raise RuntimeError("a pdfminer bug on the second pass — must never reach the client")
        return [statute_like], {}

    # T04: extraction now runs through docextract._extract_pages_isolated (a subprocess), so a
    # monkeypatch on the parent process's pdf_text.extract_pages would have no effect on it —
    # see docextract.py's docstring for _extract_pages_isolated.
    monkeypatch.setattr(docextract_mod, "_extract_pages_isolated", flaky)
    h = make()

    response = h.upload(b"%PDF-1.7 bilingual statute", "law.pdf")

    assert_error(response, 500, "internal")
    assert "pdfminer" not in response.text and "RuntimeError" not in response.text
    assert calls == [True, False]
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
    # These reach the route and are refused by `DOC_ID`: a document 404, as before.
    for malformed in ("ABCDEF123456", "abc", "0123456789abc", "0123456789a!"):
        assert_error(h.client.delete(f"/api/documents/{malformed}", headers=HEADERS), 404, "not_found")
    # `%2e%2e%2fdocs` decodes to `../docs`, which is two path segments and so matches no route at
    # all — it never reaches the library. Still a 404 with nothing leaked, which is what this
    # test is really about; only the code differs, and it is the accurate one.
    assert_error(h.client.delete("/api/documents/%2e%2e%2fdocs", headers=HEADERS), 404, "no_such_endpoint")


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


class _CloseableChat(ScriptedChat):
    """An `OllamaChat`-shaped stub (callable, `.calls`) plus a `close()` that
    records whether it ran — enough to verify the app's shutdown hook
    reaches a generator's model without a real Ollama server."""

    def __init__(self, replies=()):
        super().__init__(list(replies))
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_the_apps_shutdown_hook_closes_the_generators_models(tmp_path):
    """T11 finding: `OllamaChat` is never closed anywhere, including at app
    shutdown. Plain `TestClient(app)` (every other test in this file, via
    `Harness`) never sends lifespan events at all — only `with
    TestClient(app) as client:` does, which is why this test builds its own
    app instead of using `Harness`."""
    claims_chat = _CloseableChat([claims_json(("نص كافٍ للإجابة.", [1]))])
    relevance_chat = _CloseableChat([ANSWERS_YES])
    library = library_mod.Library(tmp_path, encoder=KeywordEncoder())
    generator = ClaimsGenerator(model=claims_chat, relevance_model=relevance_chat)
    pipeline = Pipeline(library, generator)
    app = webapp.create_app(library, pipeline)

    with TestClient(app, base_url="http://127.0.0.1"):
        assert claims_chat.closed is False and relevance_chat.closed is False

    assert claims_chat.closed is True and relevance_chat.closed is True


def test_the_apps_shutdown_hook_also_closes_extra_closeables(tmp_path):
    """`build_from_env` passes its health probe's own separate `OllamaChat`
    through `create_app(close_at_shutdown=...)` (webapp.py), because that
    one is not reachable from `pipeline.generator` at all."""
    library = library_mod.Library(tmp_path, encoder=KeywordEncoder())
    pipeline = Pipeline(library, ClaimsGenerator(model=ScriptedChat([])))
    probe_chat = _CloseableChat()
    app = webapp.create_app(library, pipeline, close_at_shutdown=(probe_chat,))

    with TestClient(app, base_url="http://127.0.0.1"):
        assert probe_chat.closed is False

    assert probe_chat.closed is True


def test_the_apps_shutdown_hook_skips_a_model_with_no_close_method(tmp_path):
    """An `hf:` runtime is a plain callable with no `.close` (and no
    `.calls`) at all — the shutdown hook must skip it via `getattr(obj,
    "close", None)`, the same defensive style already used for `.calls`,
    not raise `AttributeError` while the app is stopping."""
    library = library_mod.Library(tmp_path, encoder=KeywordEncoder())

    def plain_hf_style_model(messages):
        return claims_json(("نص كافٍ للإجابة.", [1]))

    pipeline = Pipeline(library, ClaimsGenerator(model=plain_hf_style_model))
    app = webapp.create_app(library, pipeline)

    with TestClient(app, base_url="http://127.0.0.1"):
        pass  # must not raise on shutdown


def test_a_path_that_matches_no_route_does_not_claim_a_document_is_missing(make):
    """Both are 404s, and they had the same sentence: "the document does not exist".

    A person who mistypes the path is then told their document is gone, and goes looking for
    a document that was never the problem — which is exactly what happened to the first caller
    of this API. The route 404 gets its own code and its own sentence; the document 404 keeps
    the one it had, and the test below holds that line so the two cannot merge again.
    """
    h = make()
    h.upload()

    # `/api/ask` is a plausible guess for an endpoint that is really `/api/chat`.
    unknown = h.client.post("/api/ask", json={"question": "؟" * 20}, headers=HEADERS)
    body = assert_error(unknown, 404, "no_such_endpoint")
    assert "المستند" not in body["message_ar"], "a mistyped path still blames the document"

    # The document 404 is unchanged: it comes from DocumentNotFound, never from this handler.
    assert_error(h.ask(doc_ids=["0123456789ab"]), 404, "not_found")


class _SlowWarmLibrary:
    """A library whose encoder takes a while to load, and says when it has."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.warms = 0

    def warm_encoder(self):
        self.warms += 1
        self.started.set()
        self.release.wait(timeout=10)

    def documents(self):
        return []


def test_the_app_warms_the_encoder_at_startup_off_the_request_path(tmp_path):
    """The container's first run answered `/api/chat` with 429 after 244.9s,
    all of it a 1.1 GB encoder download inside that one request. The load has
    to happen somewhere; startup is where, so no user waits for it."""
    library = _SlowWarmLibrary()
    pipeline = Pipeline(library_mod.Library(tmp_path, encoder=KeywordEncoder()),
                        ClaimsGenerator(model=ScriptedChat([])))
    app = webapp.create_app(library, pipeline)

    with TestClient(app, base_url="http://127.0.0.1"):
        assert library.started.wait(timeout=5), "startup must begin warming the encoder"
        library.release.set()

    assert library.warms == 1


def test_warming_does_not_hold_up_the_health_check(tmp_path):
    """It runs in the background for a reason: the container's HEALTHCHECK
    allows 30s of start period and the download takes minutes. A blocking
    warm-up would have the container declared unhealthy while it works."""
    library = _SlowWarmLibrary()
    pipeline = Pipeline(library_mod.Library(tmp_path, encoder=KeywordEncoder()),
                        ClaimsGenerator(model=ScriptedChat([])))
    app = webapp.create_app(library, pipeline)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert library.started.wait(timeout=5)
        # Still loading, deliberately not released: the app must answer anyway.
        assert client.get("/").status_code == 200
        library.release.set()


def test_an_encoder_that_will_not_load_does_not_take_the_app_down_at_startup(tmp_path, caplog):
    """`EncoderUnavailable` at startup must not kill the process. The request
    path already reports it properly, per endpoint, to the caller who asked."""

    class _Broken:
        def warm_encoder(self):
            raise library_mod.EncoderUnavailable("no weights here")

        def documents(self):
            return []

    pipeline = Pipeline(library_mod.Library(tmp_path, encoder=KeywordEncoder()),
                        ClaimsGenerator(model=ScriptedChat([])))
    app = webapp.create_app(_Broken(), pipeline)

    def logged():
        return any("no weights here" in r.getMessage() for r in caplog.records)

    with caplog.at_level(logging.WARNING):
        with TestClient(app, base_url="http://127.0.0.1") as client:
            assert client.get("/").status_code == 200
            # The warning is written from the warm-up thread, so the assertion
            # has to wait for it rather than race it.
            deadline = time.monotonic() + 5
            while not logged() and time.monotonic() < deadline:
                time.sleep(0.01)
    assert logged()
