"""Ollama HTTP client tests — the second generation runtime (ADR-023).

Same rule as `test_generate.py`: no network, no real server. `httpx.MockTransport`
stands in for the Ollama HTTP API, so these tests assert on the exact request
body sent and the exact response fields read, without a server ever running.

Two response fields get their own tests because both are silent failure modes
of the real server, not hypothetical ones: Ollama truncates a prompt longer
than `num_ctx` and still answers instead of erroring, and a generation stopped
by `num_predict` looks identical to one that finished on its own unless
`done_reason` is checked.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.ollama import (  # noqa: E402
    GeneratorUnavailable,
    OllamaChat,
    health,
    ollama_chat,
)


def _chat_body(**overrides) -> dict:
    """A realistic `/api/chat` response, non-streamed."""
    body = {
        "message": {"content": "الجواب"},
        "done_reason": "stop",
        "total_duration": 2_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 300_000_000,
        "eval_count": 20,
        "eval_duration": 1_200_000_000,
    }
    body.update(overrides)
    return body


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_request_is_deterministic_and_not_streamed():
    """B1 is a claim about the system, not one sample of it — a temperature
    or a missing seed would make the citation numbers vary between runs."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["url"] = str(request.url)
        return httpx.Response(200, json=_chat_body())

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    chat = ollama_chat("qwen3:4b", client=_client(handler), num_predict=64, num_ctx=2048)
    chat(messages)

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    body = captured["body"]
    assert body["model"] == "qwen3:4b"
    assert body["messages"] == messages
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert body["options"]["seed"] == 0
    assert body["options"]["num_predict"] == 64
    assert body["options"]["num_ctx"] == 2048


def test_the_reply_text_and_its_timings_are_returned():
    def handler(request):
        return httpx.Response(200, json=_chat_body(message={"content": "  جواب نظيف  "}))

    chat = ollama_chat("qwen2.5:7b-instruct", client=_client(handler))
    text = chat([{"role": "user", "content": "س"}])

    assert text == "جواب نظيف"
    stats = chat.last_stats
    assert stats["total_s"] == 2.0
    assert stats["load_s"] == 0.5
    assert stats["prompt_s"] == 0.3
    assert stats["output_s"] == 1.2
    assert stats["prompt_tokens"] == 10
    assert stats["output_tokens"] == 20


def test_a_prompt_that_fills_the_context_window_is_flagged_not_silently_truncated():
    """Ollama truncates a prompt longer than num_ctx and still answers — a
    full context window is the only visible trace of that happening."""
    def handler_with(count):
        def handler(request):
            return httpx.Response(200, json=_chat_body(prompt_eval_count=count))
        return handler

    full = ollama_chat("m", client=_client(handler_with(100)), num_ctx=100)
    full([{"role": "user", "content": "x"}])
    assert full.last_stats["truncated"] is True

    under = ollama_chat("m", client=_client(handler_with(50)), num_ctx=100)
    under([{"role": "user", "content": "x"}])
    assert under.last_stats["truncated"] is False


def test_an_answer_stopped_by_the_token_cap_is_flagged():
    """An answer cut off by num_predict can drop its citation at the end —
    `done_reason == "length"` is the only signal that happened."""
    def handler(request):
        return httpx.Response(200, json=_chat_body(done_reason="length"))

    chat = ollama_chat("m", client=_client(handler))
    chat([{"role": "user", "content": "x"}])

    assert chat.last_stats["cut"] is True
    assert chat.last_stats["done_reason"] == "length"

    def handler_stop(request):
        return httpx.Response(200, json=_chat_body(done_reason="stop"))

    finished = ollama_chat("m", client=_client(handler_stop))
    finished([{"role": "user", "content": "x"}])
    assert finished.last_stats["cut"] is False


def test_missing_stats_fields_are_none_or_false_never_a_keyerror():
    """A future or older Ollama build may omit a duration field; reading the
    reply must never blow up on a field it happened not to send."""
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "x"}})

    chat = ollama_chat("m", client=_client(handler))
    chat([{"role": "user", "content": "x"}])
    stats = chat.last_stats

    assert stats["prompt_tokens"] is None
    assert stats["output_tokens"] is None
    assert stats["total_s"] is None
    assert stats["load_s"] is None
    assert stats["prompt_s"] is None
    assert stats["output_s"] is None
    assert stats["done_reason"] is None
    assert stats["truncated"] is False
    assert stats["cut"] is False


def test_format_and_think_are_sent_only_when_asked_for():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_body())

    client = _client(handler)

    plain = ollama_chat("m", client=client)
    plain([{"role": "user", "content": "x"}])
    assert "format" not in captured[-1]
    assert "think" not in captured[-1]

    both = ollama_chat("m", client=client, fmt="json", think=False)
    both([{"role": "user", "content": "x"}])
    assert captured[-1]["format"] == "json"
    assert captured[-1]["think"] is False

    think_only = ollama_chat("m", client=client, think=True)
    think_only([{"role": "user", "content": "x"}])
    assert "format" not in captured[-1]
    assert captured[-1]["think"] is True


def test_an_unreachable_server_raises_generator_unavailable_not_system_exit():
    """`load_model` in generate.py raises SystemExit for a missing dependency
    — library code that does that cannot be caught by a caller that expects
    an exception, only worked around. This client must not repeat that."""
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    chat = ollama_chat("m", host="http://127.0.0.1:11434", client=_client(handler))

    try:
        chat([{"role": "user", "content": "x"}])
    except SystemExit:
        pytest.fail("must raise GeneratorUnavailable, not SystemExit")
    except GeneratorUnavailable as e:
        assert "127.0.0.1:11434" in str(e)
    else:
        pytest.fail("expected GeneratorUnavailable")


def test_a_missing_model_is_reported_with_the_servers_own_error():
    def handler(request):
        return httpx.Response(
            404, json={"error": "model 'ghost:1b' not found, try pulling it first"}
        )

    chat = ollama_chat("ghost:1b", client=_client(handler))
    with pytest.raises(GeneratorUnavailable) as exc_info:
        chat([{"role": "user", "content": "x"}])

    msg = str(exc_info.value)
    assert "not found" in msg
    assert "ghost:1b" in msg


def test_health_reports_how_much_of_a_loaded_model_sits_on_the_gpu():
    """The 1050 Ti has 4 GB — whether a model fits on it, or spills onto the
    much slower CPU path, is exactly what `size_vram / size` answers."""
    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.20.3"})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [
                {"name": "qwen3:4b", "size": 4_000_000_000, "size_vram": 3_000_000_000},
            ]})
        return httpx.Response(404, json={"error": "not found"})  # pragma: no cover

    info = health(client=_client(handler))
    assert info["reachable"] is True
    assert info["version"] == "0.20.3"
    assert info["loaded"] == [{
        "name": "qwen3:4b", "size": 4_000_000_000, "size_vram": 3_000_000_000,
        "gpu_share": 0.75,
    }]

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    unreachable = health(client=_client(down))
    assert unreachable["reachable"] is False
    assert isinstance(unreachable["error"], str)


def test_ollama_chat_returns_a_chat_instance_without_touching_the_network():
    """Building a client must be safe even when no server is running — the
    network call happens on the first `__call__`, not at construction."""
    chat = ollama_chat("qwen3:4b")
    assert isinstance(chat, OllamaChat)
    assert chat.model == "qwen3:4b"
    assert chat.last_stats is None
