"""Where the Ollama client connects, and how fast it gives up (Phase 5, task 1).

Two deferred Phase 1 review findings, both about the local app failing
first time in front of a camera:

- `OLLAMA_HOST` is often set system-wide for the Ollama *server*, as a bind
  address (`0.0.0.0`) with no scheme and no port. A client that connects to
  that string verbatim fails, so every host string is normalised first.
- A generation call is allowed 600 s to answer, and so was the connect: an
  unreachable server hung a request for minutes instead of failing in
  seconds. A read timeout is a different failure (the server is there but
  slow) and says so.

No network: `httpx.MockTransport` stands in for the server, and the
environment case runs in a child interpreter, because reloading
`legalrag.ollama` here would swap the exception classes other modules hold.
"""

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag.ollama import (  # noqa: E402
    GeneratorUnavailable,
    health,
    normalize_host,
    ollama_chat,
)

NUM_PREDICT = 16
MESSAGES = [{"role": "user", "content": "x"}]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_host_without_a_scheme_gets_http():
    assert normalize_host("127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert normalize_host("localhost:11434") == "http://localhost:11434"
    assert normalize_host("https://ollama.lan:8443") == "https://ollama.lan:8443"


def test_the_bind_all_address_becomes_loopback():
    assert normalize_host("0.0.0.0") == "http://127.0.0.1:11434"
    assert normalize_host("0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert normalize_host("http://0.0.0.0:8080") == "http://127.0.0.1:8080"
    # the IPv6 bind-all address is the same mistake
    assert normalize_host("[::]:11434") == "http://[::1]:11434"


def test_a_trailing_slash_is_stripped():
    assert normalize_host("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"
    assert normalize_host("http://127.0.0.1:11434//") == "http://127.0.0.1:11434"
    assert normalize_host("http://gpu-box:11434/ollama/") == "http://gpu-box:11434/ollama"


def test_a_missing_port_becomes_the_ollama_default():
    assert normalize_host("localhost") == "http://localhost:11434"
    assert normalize_host("http://localhost") == "http://localhost:11434"
    assert normalize_host("http://localhost/") == "http://localhost:11434"
    assert normalize_host("localhost:") == "http://localhost:11434"
    assert normalize_host("[::1]") == "http://[::1]:11434"


def test_an_empty_host_is_the_local_default():
    assert normalize_host("") == "http://127.0.0.1:11434"
    assert normalize_host("   ") == "http://127.0.0.1:11434"


def test_the_host_from_the_environment_is_normalised_at_import():
    env = {**os.environ, "OLLAMA_HOST": "0.0.0.0", "PYTHONPATH": str(ROOT / "src")}
    child = subprocess.run(
        [sys.executable, "-c", "from legalrag import ollama; print(ollama.OLLAMA_HOST)"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    assert child.stdout.strip() == "http://127.0.0.1:11434"


def test_a_host_passed_as_an_argument_is_normalised():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": "x"}})
        return httpx.Response(200, json={"version": "0.20.3", "models": []})

    chat = ollama_chat("m", host="0.0.0.0/", client=_client(handler), num_predict=NUM_PREDICT)
    assert chat.host == "http://127.0.0.1:11434"
    chat(MESSAGES)
    assert health(host="localhost", client=_client(handler))["reachable"] is True

    assert seen == [
        "http://127.0.0.1:11434/api/chat",
        "http://localhost:11434/api/version",
        "http://localhost:11434/api/ps",
    ]


def test_the_generation_client_gives_up_connecting_within_seconds():
    chat = ollama_chat("m", host="http://127.0.0.1:11434", num_predict=NUM_PREDICT, timeout=600.0)
    client = chat._ensure_client()  # builds the client; sends nothing
    try:
        assert client.timeout.connect == 5.0
        assert client.timeout.read == 600.0
    finally:
        client.close()


def test_a_read_timeout_is_reported_apart_from_an_unreachable_server():
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    def refused(request):
        raise httpx.ConnectError("connection refused", request=request)

    def never_connects(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    messages = {}
    for name, handler in (("slow", slow), ("refused", refused), ("never_connects", never_connects)):
        chat = ollama_chat("m", host="http://127.0.0.1:11434", client=_client(handler),
                           num_predict=NUM_PREDICT)
        with pytest.raises(GeneratorUnavailable) as exc_info:
            chat(MESSAGES)
        messages[name] = str(exc_info.value)

    assert "did not answer" in messages["slow"]
    assert "cannot reach" not in messages["slow"]
    assert "cannot reach" in messages["refused"]
    assert "cannot reach" in messages["never_connects"]
