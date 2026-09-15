"""The test harness for the local app's HTTP API (tests/test_webapp.py).

Like `stubs`, a plain module that a test file imports once it has put `src`
and this directory on `sys.path`. Unlike `stubs`, it imports `legalrag`:
`Harness` is a real `Library` and `Pipeline` behind `create_app`.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi.testclient import TestClient

from legalrag.claims import ClaimsGenerator
from legalrag.library import EncoderUnavailable, Library
from legalrag.pipeline import Pipeline
from legalrag.webapp import APP_HEADER, create_app
from stubs import KeywordEncoder, ScriptedChat, text_document

HEADERS = {APP_HEADER: "1"}
POLICY = text_document("العمل عن بعد يحتاج موافقة المدير", "بدل الإنترنت الشهري للموظف")
QUESTION = "كم بدل الإنترنت الشهري؟"
ARABIC = re.compile(r"[؀-ۿ]")


class SwitchableEncoder(KeywordEncoder):
    """A keyword encoder that can be made to fail the way a missing model does."""

    broken = False

    def encode(self, texts, **kwargs):
        if self.broken:
            raise EncoderUnavailable("sentence-transformers is missing from /secret/site-packages")
        return super().encode(texts, **kwargs)


class Harness:
    """One app over a real Library in `root`, with scripted relevance and claims models."""

    def __init__(self, root: Path, *, relevance=(), claims=(), **app_options):
        self.root = root
        self.encoder = SwitchableEncoder()
        self.library = Library(root, encoder=self.encoder)
        self.relevance = ScriptedChat(list(relevance))
        self.claims = ScriptedChat(list(claims))
        generator = ClaimsGenerator(model=self.claims, relevance_model=self.relevance)
        self.pipeline = Pipeline(self.library, generator)
        self.app = create_app(self.library, self.pipeline, **app_options)
        self.client = TestClient(self.app, base_url="http://127.0.0.1")

    def upload(self, data: bytes = POLICY, filename: str = "policy.txt", headers=HEADERS):
        files = {"file": (filename, data, "application/octet-stream")}
        return self.client.post("/api/documents", files=files, headers=headers)

    def ask(self, question: str = QUESTION, doc_ids=None, headers=HEADERS):
        return self.client.post("/api/chat", json={"question": question, "doc_ids": doc_ids}, headers=headers)


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error", "message_ar"}
    assert body["error"] == code
    assert ARABIC.search(body["message_ar"])
    return body


def tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def raw_request(app, method: str, path: str, headers=(), body: bytes | None = None):
    """Call the ASGI app directly: no client tidies the path first, and with
    `body=None` any attempt to read the request body fails the test."""
    sent = []

    async def receive():
        if body is None:
            raise AssertionError("the request body was read")
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "scheme": "http",
        "method": method, "path": path, "raw_path": path.encode("latin-1"), "query_string": b"",
        "root_path": "", "client": ("127.0.0.1", 50123), "server": ("127.0.0.1", 80),
        "headers": [(b"host", b"127.0.0.1")]
        + [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in headers],
    }
    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    response_headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in start["headers"]}
    content = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], response_headers, content


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


SECURITY_HEADERS = {
    "content-security-policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                               "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
}


def assert_security_headers(headers, label: str) -> None:
    for name, value in SECURITY_HEADERS.items():
        assert headers.get(name) == value, f"{label}: {name}"
