"""Local Ollama chat client — the second generation runtime (ADR-023).

`generate.load_model` runs a checkpoint in-process through `transformers`;
this talks HTTP to a local Ollama server instead, which may already have the
model loaded or will load it on the first request. Same shape as
`load_model`'s return value — callable messages -> str — so `Generator`
does not know or care which runtime it holds; `generate.resolve_model`
is the switch between them.

Determinism (temperature 0, seed 0) is fixed here as a literal, not imported
from `generate.py`: `generate.py` imports `ollama_chat` from this module, so
this module importing `TEMPERATURE` back from `generate.py` would be a cycle.
`generate.TEMPERATURE` carries the same reasoning and the same value.

Two response fields matter beyond the answer text, both because they are
silent failure modes of the real server rather than hypothetical ones:
Ollama truncates a prompt longer than `num_ctx` and still answers — a full
context window (`truncated`) is the only visible trace — and a generation
stopped by `num_predict` looks identical to one that finished on its own
unless `done_reason` is checked (`cut`), which matters here because an answer
cut off mid-sentence can drop its citation at the end.
"""

from __future__ import annotations

import os

import httpx

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Greedy and reproducible, same reasoning as generate.TEMPERATURE: a sampled
# answer would make the citation numbers vary between runs, and B1 is a claim
# about the system, not about one sample of it. Duplicated as a literal
# rather than imported — see the module docstring for why.
TEMPERATURE = 0.0
SEED = 0

# Ollama's own health-check calls are local and small; they should fail fast
# rather than hang for the same 600s a generation call is allowed.
HEALTH_TIMEOUT = 10.0


class GeneratorUnavailable(RuntimeError):
    """The Ollama server could not be reached, or rejected the request (for
    example: the model named has not been pulled)."""


class OllamaChat:
    """Callable messages -> str backed by a local Ollama server.

    Construction touches no network — it only records configuration — so
    building one to decide which runtime a `--model` spec means
    (`generate.resolve_model`) is safe even when no server is running. The
    first `__call__` creates the underlying `httpx.Client` if none was given,
    and every call after that reuses it.
    """

    def __init__(
        self,
        model: str,
        host: str,
        *,
        num_predict: int,
        num_ctx: int,
        fmt: dict | str | None,
        think: bool | None,
        timeout: float,
        client: httpx.Client | None,
    ) -> None:
        self.model = model
        self.host = host
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.fmt = fmt
        self.think = think
        self.timeout = timeout
        self._client = client
        self.last_stats: dict | None = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def __call__(self, messages: list[dict]) -> str:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": TEMPERATURE,
                "seed": SEED,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        if self.fmt is not None:
            body["format"] = self.fmt
        if self.think is not None:
            body["think"] = self.think

        client = self._ensure_client()
        try:
            resp = client.post(f"{self.host}/api/chat", json=body)
        except httpx.TransportError as e:
            raise GeneratorUnavailable(
                f"cannot reach Ollama at {self.host} ({e}). "
                "Start the Ollama server and try again."
            ) from e

        if not (200 <= resp.status_code < 300):
            raise GeneratorUnavailable(_error_message(resp, self.model, self.host))

        data = resp.json()
        self.last_stats = _stats(data, self.num_ctx)
        return data["message"]["content"].strip()


def _error_message(resp: httpx.Response, model: str, host: str) -> str:
    try:
        detail = resp.json().get("error", resp.text)
    except ValueError:
        detail = resp.text
    return f"Ollama at {host} rejected model '{model}' (HTTP {resp.status_code}): {detail}"


def _ns_to_s(data: dict, key: str) -> float | None:
    value = data.get(key)
    return value / 1e9 if value is not None else None


def _stats(data: dict, num_ctx: int) -> dict:
    prompt_tokens = data.get("prompt_eval_count")
    done_reason = data.get("done_reason")
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": data.get("eval_count"),
        "total_s": _ns_to_s(data, "total_duration"),
        "load_s": _ns_to_s(data, "load_duration"),
        "prompt_s": _ns_to_s(data, "prompt_eval_duration"),
        "output_s": _ns_to_s(data, "eval_duration"),
        "done_reason": done_reason,
        "truncated": prompt_tokens is not None and prompt_tokens >= num_ctx,
        "cut": done_reason == "length",
    }


def ollama_chat(
    model: str,
    host: str | None = None,
    *,
    num_predict: int = 512,
    num_ctx: int = 4096,
    fmt: dict | str | None = None,
    think: bool | None = None,
    timeout: float = 600.0,
    client: httpx.Client | None = None,
) -> OllamaChat:
    return OllamaChat(
        model=model,
        host=host or OLLAMA_HOST,
        num_predict=num_predict,
        num_ctx=num_ctx,
        fmt=fmt,
        think=think,
        timeout=timeout,
        client=client,
    )


def health(host: str | None = None, client: httpx.Client | None = None) -> dict:
    """Is the server up, and how much of each loaded model sits on the GPU.

    `size_vram / size` on a 4 GB card is the number that says whether a
    candidate model in Run 4 runs fast (on the GPU) or falls back to the
    much slower CPU path — not something `/api/chat`'s own timings reveal
    on their own.
    """
    h = host or OLLAMA_HOST
    owns_client = client is None
    c = client if client is not None else httpx.Client(timeout=HEALTH_TIMEOUT)
    try:
        try:
            version_resp = c.get(f"{h}/api/version")
            ps_resp = c.get(f"{h}/api/ps")
        except httpx.TransportError as e:
            return {"reachable": False, "error": str(e)}

        version = None
        if 200 <= version_resp.status_code < 300:
            version = version_resp.json().get("version")

        loaded = []
        if 200 <= ps_resp.status_code < 300:
            for m in ps_resp.json().get("models", []):
                size = m.get("size") or 0
                size_vram = m.get("size_vram") or 0
                loaded.append({
                    "name": m.get("name"),
                    "size": size,
                    "size_vram": size_vram,
                    "gpu_share": (size_vram / size) if size else 0.0,
                })

        return {"reachable": True, "version": version, "loaded": loaded}
    finally:
        if owns_client:
            c.close()
