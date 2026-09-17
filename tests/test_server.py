"""The dev benchmark server (tasks.py serve) — Host allowlist and bounded `k`.

`legalrag.server.Handler` is a bare `BaseHTTPRequestHandler` behind a real
socket, unlike `webapp.py`'s ASGI app: there is no in-process ASGI scope to
build, so these tests bind a real `ThreadingHTTPServer` on an OS-assigned
loopback port for the module and speak real HTTP to it with stdlib
`http.client` (no new test dependency — matches this server's own stdlib-only
design, see its module docstring).

Two gaps this closes (F03):
1. No Host-header check at all, unlike `web_guard.ALLOWED_HOSTS` — a page a
   user has open can, after DNS rebinding, reach this loopback-bound server
   with an arbitrary Host header and get an answer.
2. `k` from the query string went straight into `int()` with no try/except
   and no bounds: `k=abc` raised an unhandled 500, `k=-1` or `k=999999999`
   passed straight into the retriever.
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.server import Handler  # noqa: E402


@pytest.fixture(scope="module")
def port():
    """A real ThreadingHTTPServer on an OS-assigned loopback port, for the
    whole module: no corpus is ingested, which _search/_eval_run already
    handle gracefully (they return [] / an error dict, not an exception)."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(bound_port: int, path: str, host: str) -> tuple[int, bytes]:
    """A real GET against the loopback server, with `host` sent verbatim as
    the Host header — this is how a DNS-rebinding page would reach it."""
    conn = http.client.HTTPConnection("127.0.0.1", bound_port, timeout=5)
    try:
        conn.request("GET", path, headers={"Host": host})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def _error_body(body: bytes) -> dict:
    parsed = json.loads(body)
    assert "error" in parsed
    return parsed


def test_a_foreign_host_header_is_rejected(port):
    status, body = _get(port, "/api/status", "evil.example")
    assert status == 400
    _error_body(body)


def test_a_foreign_host_with_a_port_suffix_is_also_rejected(port):
    status, body = _get(port, "/api/status", "evil.example:8000")
    assert status == 400
    _error_body(body)


def test_the_real_bound_loopback_host_with_its_port_succeeds(port):
    status, body = _get(port, "/api/status", f"127.0.0.1:{port}")
    assert status == 200
    assert json.loads(body)["retriever"] in (None, "BM25 (baseline)")


def test_localhost_with_its_port_succeeds_too(port):
    status, body = _get(port, "/api/status", f"localhost:{port}")
    assert status == 200
    assert "corpus" in json.loads(body)


def test_search_with_a_non_numeric_k_is_a_clean_400(port):
    status, body = _get(port, "/api/search?q=x&k=abc", f"127.0.0.1:{port}")
    assert status == 400
    _error_body(body)


def test_search_with_a_negative_k_is_a_clean_400(port):
    status, body = _get(port, "/api/search?q=x&k=-1", f"127.0.0.1:{port}")
    assert status == 400
    _error_body(body)


def test_search_with_k_above_the_declared_upper_bound_is_a_clean_400(port):
    status, body = _get(port, "/api/search?q=x&k=999999", f"127.0.0.1:{port}")
    assert status == 400
    _error_body(body)


def test_eval_with_a_non_numeric_k_is_a_clean_400(port):
    status, body = _get(port, "/api/eval?k=abc", f"127.0.0.1:{port}")
    assert status == 400
    _error_body(body)


def test_eval_with_a_negative_k_is_a_clean_400(port):
    status, body = _get(port, "/api/eval?k=-1", f"127.0.0.1:{port}")
    assert status == 400
    _error_body(body)


def test_a_normal_valid_search_still_returns_200_with_the_expected_shape(port):
    status, body = _get(port, "/api/search?q=x&k=5", f"127.0.0.1:{port}")
    assert status == 200
    assert isinstance(json.loads(body), list)


def test_the_index_page_and_an_unknown_path_are_unchanged_by_the_host_check(port):
    status, body = _get(port, "/", f"127.0.0.1:{port}")
    assert status == 200
    assert b"<html" in body.lower() or b"<!doctype" in body.lower()

    status, body = _get(port, "/nowhere", f"127.0.0.1:{port}")
    assert status == 404
    assert json.loads(body) == {"error": "not found"}
