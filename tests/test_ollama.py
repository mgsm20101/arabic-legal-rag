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
    run_metadata,
)

NUM_PREDICT = 128  # a fixed stand-in; Run 4/5's real cap is generate.MAX_NEW_TOKENS


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
    # `host` is explicit, not the module default: an OLLAMA_HOST already set
    # in this shell must not change what this test asserts about the URL.
    chat = ollama_chat(
        "qwen3:4b", host="http://127.0.0.1:11434",
        client=_client(handler), num_predict=64, num_ctx=2048,
    )
    chat(messages)

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    body = captured["body"]
    assert body["model"] == "qwen3:4b"
    assert body["messages"] == messages
    assert body["stream"] is False
    assert body["keep_alive"] == "30m"
    assert body["options"]["temperature"] == 0
    assert body["options"]["seed"] == 0
    assert body["options"]["num_predict"] == 64
    assert body["options"]["num_ctx"] == 2048


def test_the_reply_text_and_its_timings_are_returned():
    def handler(request):
        return httpx.Response(200, json=_chat_body(message={"content": "  جواب نظيف  "}))

    chat = ollama_chat("qwen2.5:7b-instruct", client=_client(handler), num_predict=NUM_PREDICT)
    text = chat([{"role": "user", "content": "س"}])

    assert text == "جواب نظيف"
    stats = chat.last_stats
    assert stats["total_s"] == 2.0
    assert stats["load_s"] == 0.5
    assert stats["prompt_s"] == 0.3
    assert stats["output_s"] == 1.2
    assert stats["prompt_tokens"] == 10
    assert stats["output_tokens"] == 20


def test_every_call_is_recorded_not_just_the_latest():
    """Run 5 retries once on invalid JSON, so a caller needs every call's
    stats for one question, not only whatever `last_stats` holds after the
    last one — reading only the latest would silently drop a cut first
    attempt the moment a retry follows it."""
    def handler(request):
        return httpx.Response(200, json=_chat_body(eval_count=len(chat.calls) + 1))

    chat = ollama_chat("m", client=_client(handler), num_predict=NUM_PREDICT)
    assert chat.calls == []

    chat([{"role": "user", "content": "1"}])
    chat([{"role": "user", "content": "2"}])

    assert len(chat.calls) == 2
    assert chat.calls[0]["output_tokens"] == 1
    assert chat.calls[1]["output_tokens"] == 2
    assert chat.last_stats == chat.calls[-1]


def test_a_prompt_that_fills_the_context_window_is_flagged_not_silently_truncated():
    """Ollama truncates a prompt longer than num_ctx and still answers — a
    full context window is the only visible trace of that happening."""
    def handler_with(count):
        def handler(request):
            return httpx.Response(200, json=_chat_body(prompt_eval_count=count))
        return handler

    full = ollama_chat("m", client=_client(handler_with(100)), num_predict=NUM_PREDICT, num_ctx=100)
    full([{"role": "user", "content": "x"}])
    assert full.last_stats["truncated"] is True

    under = ollama_chat("m", client=_client(handler_with(50)), num_predict=NUM_PREDICT, num_ctx=100)
    under([{"role": "user", "content": "x"}])
    assert under.last_stats["truncated"] is False


def test_an_answer_stopped_by_the_token_cap_is_flagged():
    """An answer cut off by num_predict can drop its citation at the end —
    `done_reason == "length"` is the only signal that happened."""
    def handler(request):
        return httpx.Response(200, json=_chat_body(done_reason="length"))

    chat = ollama_chat("m", client=_client(handler), num_predict=NUM_PREDICT)
    chat([{"role": "user", "content": "x"}])

    assert chat.last_stats["cut"] is True
    assert chat.last_stats["done_reason"] == "length"

    def handler_stop(request):
        return httpx.Response(200, json=_chat_body(done_reason="stop"))

    finished = ollama_chat("m", client=_client(handler_stop), num_predict=NUM_PREDICT)
    finished([{"role": "user", "content": "x"}])
    assert finished.last_stats["cut"] is False


def test_missing_stats_fields_are_none_or_false_never_a_keyerror():
    """A future or older Ollama build may omit a duration field; reading the
    reply must never blow up on a field it happened not to send."""
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "x"}})

    chat = ollama_chat("m", client=_client(handler), num_predict=NUM_PREDICT)
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

    plain = ollama_chat("m", client=client, num_predict=NUM_PREDICT)
    plain([{"role": "user", "content": "x"}])
    assert "format" not in captured[-1]
    assert "think" not in captured[-1]

    both = ollama_chat("m", client=client, num_predict=NUM_PREDICT, fmt="json", think=False)
    both([{"role": "user", "content": "x"}])
    assert captured[-1]["format"] == "json"
    assert captured[-1]["think"] is False

    think_only = ollama_chat("m", client=client, num_predict=NUM_PREDICT, think=True)
    think_only([{"role": "user", "content": "x"}])
    assert "format" not in captured[-1]
    assert captured[-1]["think"] is True


def test_an_unreachable_server_raises_generator_unavailable_not_system_exit():
    """`generate.load_model` treats a missing dependency and a bad checkpoint
    the same way, as `GeneratorUnavailable` — library code that instead
    exits the process cannot be caught by a caller that expects an
    exception, only worked around. This client must not repeat that."""
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    chat = ollama_chat("m", host="http://127.0.0.1:11434", client=_client(handler), num_predict=NUM_PREDICT)

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

    chat = ollama_chat("ghost:1b", client=_client(handler), num_predict=NUM_PREDICT)
    with pytest.raises(GeneratorUnavailable) as exc_info:
        chat([{"role": "user", "content": "x"}])

    msg = str(exc_info.value)
    assert "not found" in msg
    assert "ghost:1b" in msg


def test_a_non_dict_error_body_still_raises_generator_unavailable():
    """The server's error body is not guaranteed to be a `{"error": ...}`
    object — a JSON list has no `.get`, and plain text has no `.json()` at
    all. Neither should crash the error path itself with an AttributeError
    or ValueError; both must still surface as GeneratorUnavailable."""
    def as_list(request):
        return httpx.Response(500, json=["boom", "internal error"])

    with pytest.raises(GeneratorUnavailable):
        ollama_chat("m", client=_client(as_list), num_predict=NUM_PREDICT)(
            [{"role": "user", "content": "x"}]
        )

    def as_text(request):
        return httpx.Response(500, content=b"internal server error")

    with pytest.raises(GeneratorUnavailable) as exc_info:
        ollama_chat("m", client=_client(as_text), num_predict=NUM_PREDICT)(
            [{"role": "user", "content": "x"}]
        )
    assert "internal server error" in str(exc_info.value)


def _ps_handler(models):
    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.20.3"})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": models})
        return httpx.Response(404, json={"error": "not found"})  # pragma: no cover
    return handler


def test_health_reports_how_much_of_a_loaded_model_sits_on_the_gpu():
    """The 1050 Ti has 4 GB — whether a model fits on it, or spills onto the
    much slower CPU path, is exactly what `size_vram / size` answers."""
    handler = _ps_handler([
        {"name": "qwen3:4b", "size": 4_000_000_000, "size_vram": 3_000_000_000},
    ])
    info = health(client=_client(handler))
    assert info["reachable"] is True
    assert info["version"] == "0.20.3"
    assert info["loaded"] == [{
        "name": "qwen3:4b", "size": 4_000_000_000, "size_vram": 3_000_000_000,
        "gpu_share": 0.75, "digest": None, "quantization": None,
    }]

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    unreachable = health(client=_client(down))
    assert unreachable["reachable"] is False
    assert isinstance(unreachable["error"], str)


def test_health_captures_digest_and_quantization_for_each_loaded_model():
    handler = _ps_handler([
        {"name": "gemma3:4b", "size": 4_000_000_000, "size_vram": 4_000_000_000,
         "digest": "fe20cdd6d1f4abc123", "details": {"quantization_level": "Q4_K_M"}},
    ])
    info = health(client=_client(handler))
    assert info["loaded"][0]["digest"] == "fe20cdd6d1f4abc123"
    assert info["loaded"][0]["quantization"] == "Q4_K_M"


def test_health_reports_no_gpu_share_when_size_is_missing_or_zero():
    """0.0 would claim a precise measurement ("nothing is on the GPU"); the
    honest answer when `size` is absent or zero is that there is nothing to
    divide by, not a share of zero."""
    handler = _ps_handler([{"name": "m", "size_vram": 0}])
    info = health(client=_client(handler))
    assert info["loaded"][0]["gpu_share"] is None


def test_health_does_not_raise_on_a_malformed_json_response():
    """A byte-truncated or non-JSON body from the server must not turn a
    health check into an unhandled exception."""
    def handler(request):
        return httpx.Response(200, content=b"{not valid json")

    info = health(client=_client(handler))
    assert info["reachable"] is False
    assert isinstance(info["error"], str)


def test_health_does_not_raise_on_well_formed_json_of_the_wrong_shape():
    """`.json()` succeeding does not mean the shape this function assumes is
    actually there. A JSON list where `/api/ps` should send an object,
    `{"models": null}`, and a string sitting where a model object belongs
    are all valid JSON — none of them raise the `ValueError` the existing
    malformed-body handling already caught, so each would previously reach
    `.get`/the `for` loop and blow up with AttributeError or TypeError
    instead of a clean `reachable: False`."""
    def _handler(ps_body):
        def handler(request):
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.20.3"})
            return httpx.Response(200, json=ps_body)
        return handler

    ps_is_a_list = health(client=_client(_handler(["not", "an", "object"])))
    assert ps_is_a_list["reachable"] is False
    assert isinstance(ps_is_a_list["error"], str)

    models_is_null = health(client=_client(_handler({"models": None})))
    assert models_is_null["reachable"] is False
    assert isinstance(models_is_null["error"], str)

    entry_is_a_string = health(client=_client(_handler({"models": ["oops"]})))
    assert entry_is_a_string["reachable"] is False
    assert isinstance(entry_is_a_string["error"], str)


def test_run_metadata_matches_a_tagged_model_by_exact_name():
    handler = _ps_handler([
        {"name": "gemma3:4b", "size": 4_000_000_000, "size_vram": 2_000_000_000,
         "digest": "abc123def456", "details": {"quantization_level": "Q4_K_M"}},
    ])
    chat = ollama_chat("gemma3:4b", client=_client(handler), num_predict=NUM_PREDICT)

    meta = run_metadata(chat)
    assert meta["reachable"] is True
    assert meta["model"] == "gemma3:4b"
    assert meta["ollama_version"] == "0.20.3"
    assert meta["gpu_share"] == 0.5
    assert meta["digest"] == "abc123def456"
    assert meta["quantization"] == "Q4_K_M"
    assert meta["error"] is None


def test_run_metadata_matches_an_untagged_spec_via_its_latest_alias():
    """`ollama:llama3` (no explicit tag) is what Ollama itself calls
    `llama3:latest` once loaded — the untagged spec must still find it."""
    handler = _ps_handler([
        {"name": "llama3:latest", "size": 4_000_000_000, "size_vram": 4_000_000_000},
    ])
    chat = ollama_chat("llama3", client=_client(handler), num_predict=NUM_PREDICT)

    meta = run_metadata(chat)
    assert meta["gpu_share"] == 1.0


def test_run_metadata_reports_no_gpu_share_when_size_is_missing():
    handler = _ps_handler([{"name": "m"}])
    chat = ollama_chat("m", client=_client(handler), num_predict=NUM_PREDICT)

    meta = run_metadata(chat)
    assert meta["reachable"] is True
    assert meta["gpu_share"] is None


def test_run_metadata_is_unreachable_without_raising():
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    chat = ollama_chat("m", client=_client(down), num_predict=NUM_PREDICT)

    meta = run_metadata(chat)
    assert meta["reachable"] is False
    assert meta["gpu_share"] is None
    assert isinstance(meta["error"], str)


def test_ollama_chat_returns_a_chat_instance_without_touching_the_network(monkeypatch):
    """Building a client must be safe even when no server is running — the
    network call happens on the first `__call__`, not at construction.

    `httpx.Client` itself is monkeypatched to raise if constructed, so this
    proves construction never creates one, rather than merely trusting the
    absence of a crash as evidence of it.
    """
    def _must_not_construct(*args, **kwargs):
        raise AssertionError("ollama_chat must not construct an httpx.Client")
    monkeypatch.setattr(httpx, "Client", _must_not_construct)

    chat = ollama_chat("qwen3:4b", num_predict=NUM_PREDICT)

    assert isinstance(chat, OllamaChat)
    assert chat.model == "qwen3:4b"
    assert chat.last_stats is None
    assert chat.calls == []
