"""How the local app is built and started — Phase 5, task 2.

`RateLimiter` on its own (a sliding window per client, bounded memory, safe
across threads), `build_from_env` (an ollama: model or nothing, one pipeline,
no network at startup, a warning for a non-loopback Ollama host) and `main`
(a warning for a non-loopback bind address). Nothing here serves a request,
loads a model, or opens a socket.
"""

import logging
import sys
import threading
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("fastapi")

import legalrag.ollama as ollama_mod  # noqa: E402
import legalrag.webapp as webapp  # noqa: E402
from legalrag.claims import ClaimsGenerator  # noqa: E402
from legalrag.webapp import RateLimiter  # noqa: E402
from stubs import ScriptedChat  # noqa: E402


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_the_rate_limiter_slides_its_window_per_client():
    clock = FakeClock()
    limiter = RateLimiter(2, 10.0, clock=clock)

    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert limiter.check("a") == pytest.approx(10.0)
    assert limiter.check("b") is None, "every client has its own budget"
    clock.now += 4.0
    assert limiter.check("a") == pytest.approx(6.0), "a refused request does not extend the wait"
    clock.now += 6.0
    assert limiter.check("a") is None


def test_the_rate_limiter_forgets_the_least_recently_seen_client_when_full():
    limiter = RateLimiter(1, 60.0, clock=FakeClock(), max_keys=2)

    assert limiter.check("a") is None and limiter.check("b") is None
    assert limiter.check("a") is not None  # refused, and now the most recently seen
    assert limiter.check("c") is None      # a third client: "b" is forgotten
    assert limiter.check("a") is not None, "the client still being refused was kept"
    assert limiter.check("b") is None, "the forgotten client starts over"


def test_the_rate_limiter_admits_exactly_its_budget_across_threads():
    limiter = RateLimiter(50, 60.0, clock=FakeClock())
    admitted = []
    start = threading.Barrier(8, timeout=10)

    def hammer():
        start.wait()
        for _ in range(20):
            if limiter.check("same-client") is None:
                admitted.append(1)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(admitted) == 50


def test_the_rate_limiter_refuses_a_meaningless_configuration():
    for args in ((0, 60.0), (5, 0.0), (5, -1.0)):
        with pytest.raises(ValueError):
            RateLimiter(*args)
    with pytest.raises(ValueError):
        RateLimiter(5, 60.0, max_keys=0)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A data directory under tmp_path, the default model, a loopback Ollama
    host, and a generator factory that records every call instead of resolving a model."""
    built = []

    def build_generators(spec, contract):
        built.append((spec, contract))
        return ClaimsGenerator(model=ScriptedChat([]), relevance_model=ScriptedChat([]))

    monkeypatch.setattr(webapp, "build_generators", build_generators)
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("LEGALRAG_MODEL", raising=False)
    monkeypatch.setattr(ollama_mod, "OLLAMA_HOST", "http://127.0.0.1:11434")
    return built


def test_build_from_env_refuses_a_model_that_is_not_ollama(env, monkeypatch, tmp_path):
    for spec in ("hf:Qwen/Qwen2.5-1.5B-Instruct", "gemma3:4b", "ollama:", "ollama:  "):
        monkeypatch.setenv("LEGALRAG_MODEL", spec)
        with pytest.raises(webapp.AppConfigError, match="ollama:"):
            webapp.build_from_env()

    assert env == []
    assert not (tmp_path / "data").exists(), "refused before anything was created"


def test_build_from_env_builds_one_pipeline_and_opens_no_connection(env, monkeypatch, tmp_path, caplog):
    def no_connections(*args, **kwargs):
        raise AssertionError("an HTTP client was built at startup")

    monkeypatch.setattr(httpx, "Client", no_connections)

    with caplog.at_level(logging.WARNING, logger="legalrag.webapp"):
        app = webapp.build_from_env()

    assert env == [("ollama:gemma3:4b", "gated")]
    assert (tmp_path / "data" / "docs").is_dir()
    assert {route.path for route in app.routes} >= {"/api/chat", "/api/documents", "/api/health", "/"}
    assert "loopback" not in caplog.text


def test_build_from_env_warns_when_the_ollama_host_is_not_loopback(env, monkeypatch, caplog):
    monkeypatch.setattr(ollama_mod, "OLLAMA_HOST", "http://192.168.1.20:11434")

    with caplog.at_level(logging.WARNING, logger="legalrag.webapp"):
        webapp.build_from_env()

    assert "not a loopback address" in caplog.text
    assert env == [("ollama:gemma3:4b", "gated")], "a warning, not a refusal"


@pytest.fixture
def served(monkeypatch):
    calls = []
    monkeypatch.setattr(webapp, "build_from_env", lambda: "the app")
    monkeypatch.setattr(webapp.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    return calls


def test_main_serves_on_loopback_by_default(served, capsys):
    assert webapp.main([]) == 0

    assert [(app, kwargs["host"], kwargs["port"]) for app, kwargs in served] == [("the app", "127.0.0.1", 8000)]
    assert "WARNING" not in capsys.readouterr().out


def test_main_warns_when_the_bind_address_is_not_loopback(served, capsys):
    assert webapp.main(["--host", "0.0.0.0", "--port", "8765"]) == 0

    assert [(kwargs["host"], kwargs["port"]) for _, kwargs in served] == [("0.0.0.0", 8765)]
    assert "WARNING" in capsys.readouterr().out


def test_main_stops_with_the_reason_when_the_configuration_is_refused(served, monkeypatch, capsys):
    def refuse():
        raise webapp.AppConfigError("LEGALRAG_MODEL must be an ollama: model spec")

    monkeypatch.setattr(webapp, "build_from_env", refuse)

    assert webapp.main([]) == 2
    assert served == []
    assert "LEGALRAG_MODEL must be an ollama: model spec" in capsys.readouterr().out
