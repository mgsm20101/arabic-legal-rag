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

DEFAULT_PORT = 11434
DEFAULT_HOST = f"http://127.0.0.1:{DEFAULT_PORT}"

# Bind-all addresses: right for a server to listen on, wrong for a client to
# connect to (Windows refuses a connection to 0.0.0.0 outright).
_BIND_ALL = {"0.0.0.0": "127.0.0.1", "::": "::1"}


def normalize_host(value: str) -> str:
    """The base URL a client should use for an Ollama host string.

    `OLLAMA_HOST` is shared with the Ollama server itself, where it is a bind
    address, so the same variable often holds `0.0.0.0`, or `host:port` with
    no scheme. A missing scheme becomes `http://`, a bind-all address becomes
    loopback, a missing or unusable port becomes 11434, and trailing slashes
    go. An empty value is the local default. Never raises: this runs at import.
    """
    text = value.strip()
    if not text:
        return DEFAULT_HOST
    scheme, sep, rest = text.partition("://")
    if sep:
        scheme = scheme.lower()
    else:
        scheme, rest = "http", text
    authority, _, path = rest.partition("/")
    userinfo, at, hostport = authority.rpartition("@")
    host, port = _split_host_port(hostport)
    host = _BIND_ALL.get(host, host) or "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    path = path.rstrip("/")
    return f"{scheme}://{userinfo}{at}{host}:{port}" + (f"/{path}" if path else "")


def _split_host_port(hostport: str) -> tuple[str, int]:
    """`host[:port]`, `[ipv6][:port]` or a bare IPv6 address -> (host, port)."""
    if hostport.startswith("["):
        host, _, after = hostport[1:].partition("]")
        port = after[1:] if after.startswith(":") else ""
    elif hostport.count(":") == 1:
        host, _, port = hostport.partition(":")
    else:  # no port at all, or a bare IPv6 address, which leaves no room for one
        host, port = hostport, ""
    usable = port.isascii() and port.isdigit() and 0 < int(port) <= 65535
    return host, int(port) if usable else DEFAULT_PORT


OLLAMA_HOST = normalize_host(os.environ.get("OLLAMA_HOST", ""))

# A generation may take minutes to answer; connecting to a local server may
# not. Without its own connect timeout, an unreachable host held a request
# open for the whole generation timeout instead of failing in seconds.
CONNECT_TIMEOUT = 5.0

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
        num_gpu: int | None = None,
    ) -> None:
        self.model = model
        self.host = normalize_host(host)
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.fmt = fmt
        self.think = think
        self.timeout = timeout
        self._client = client
        # Layers offloaded to the GPU. None leaves the choice to Ollama, which
        # decides per server session from the VRAM free at load time — and a
        # different split changes floating-point results enough to change a
        # greedy answer (EVIDENCE E3c). Pinning it is what the variance
        # experiment tests; the env var lets a harness pin it for a subprocess.
        self.num_gpu = num_gpu if num_gpu is not None else num_gpu_from_env()
        self.last_stats: dict | None = None
        # Every call's stats, in order — not just the latest. Run 5 retries
        # once on invalid JSON, so a single question can make more than one
        # call, and `last_stats` alone would hide a cut first attempt behind
        # a successful retry.
        self.calls: list[dict] = []

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=CONNECT_TIMEOUT))
        return self._client

    def close(self) -> None:
        """Close the underlying `httpx.Client`, freeing its connection pool —
        the same cleanup `ollama.health` already does for its own client
        (`finally: if owns_client: c.close()`), which this one never had. A
        safe no-op when `__call__` was never made (no client to close), and
        `httpx.Client.close()` is itself safe to call more than once."""
        if self._client is not None:
            self._client.close()

    def __call__(self, messages: list[dict]) -> str:
        resp = self._post(self._request_body(messages))
        if not (200 <= resp.status_code < 300):
            raise GeneratorUnavailable(_error_message(resp, self.model, self.host))

        data, content = self._parse_reply(resp)
        stats = _stats(data, self.num_ctx)
        self.last_stats = stats
        self.calls.append(stats)
        return content.strip()

    def _request_body(self, messages: list[dict]) -> dict:
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
        if self.num_gpu is not None:
            body["options"]["num_gpu"] = self.num_gpu
        if self.fmt is not None:
            body["format"] = self.fmt
        if self.think is not None:
            body["think"] = self.think
        return body

    def _post(self, body: dict) -> httpx.Response:
        client = self._ensure_client()
        try:
            return client.post(f"{self.host}/api/chat", json=body)
        except (httpx.ReadTimeout, httpx.WriteTimeout) as e:
            # Connected, then no reply in time: the server is up but slow or
            # stuck (a model still loading, a prompt too big for the machine),
            # which "start the Ollama server" would not fix.
            raise GeneratorUnavailable(
                f"Ollama at {self.host} accepted the request but did not answer "
                f"in time ({e!r})."
            ) from e
        except httpx.TransportError as e:
            raise GeneratorUnavailable(
                f"cannot reach Ollama at {self.host} ({e}). "
                "Start the Ollama server and try again."
            ) from e

    def _parse_reply(self, resp: httpx.Response) -> tuple[dict, str]:
        try:
            data = resp.json()
        except ValueError as e:
            # A 200 status is not proof of a usable body — a truncated
            # stream or a proxy error page can still arrive with status 200.
            # Every other failure mode this client sees already becomes
            # `GeneratorUnavailable`; a non-JSON 200 body must not be the
            # one gap that instead raises a raw `json.JSONDecodeError`.
            raise GeneratorUnavailable(
                f"Ollama at {self.host} sent a 200 response that is not "
                f"JSON: {e}"
            ) from e

        # `.get`/`isinstance`, not `data["message"]["content"]` directly:
        # a well-formed-but-wrong-shaped body (no "message" key, "message"
        # not an object, "content" missing or not a string) must raise
        # `GeneratorUnavailable` the same way a malformed body does, not
        # `KeyError`/`AttributeError` out of a caller that expects only
        # `GeneratorUnavailable` from this method.
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise GeneratorUnavailable(
                f"Ollama at {self.host} sent a 200 response with no usable "
                "'message.content' string"
            )
        return data, content


def _error_message(resp: httpx.Response, model: str, host: str) -> str:
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    # The server's error body is not guaranteed to be a `{"error": ...}`
    # object — Ollama itself always sends that shape, but a proxy or a
    # different failure in front of it (a JSON list, plain text) would not.
    # Building the error message must not itself raise.
    detail = payload.get("error", resp.text) if isinstance(payload, dict) else resp.text
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
    num_predict: int,
    num_ctx: int = 4096,
    fmt: dict | str | None = None,
    think: bool | None = None,
    timeout: float = 600.0,
    client: httpx.Client | None = None,
    num_gpu: int | None = None,
) -> OllamaChat:
    """Build an `OllamaChat` for `model` on a local Ollama server.

    `num_predict` has no default: a silent one here would silently pick this
    project's own token cap for an unrelated caller. `host` defaults to
    `OLLAMA_HOST` (the module constant, itself `$OLLAMA_HOST` or localhost)
    when not given explicitly.
    """
    return OllamaChat(
        model=model,
        host=host or OLLAMA_HOST,
        num_predict=num_predict,
        num_ctx=num_ctx,
        fmt=fmt,
        think=think,
        timeout=timeout,
        client=client,
        num_gpu=num_gpu,
    )


NUM_GPU_ENV = "LEGALRAG_OLLAMA_NUM_GPU"


def num_gpu_from_env() -> int | None:
    """`$LEGALRAG_OLLAMA_NUM_GPU` as a layer count, or None when unset.

    A malformed value raises rather than being ignored: a run that believed
    it had pinned GPU placement and silently had not would record the very
    variance it was meant to rule out.
    """
    import os

    raw = os.environ.get(NUM_GPU_ENV, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"{NUM_GPU_ENV} must be >= 0, got {value}")
    return value


def health(host: str | None = None, client: httpx.Client | None = None) -> dict:
    """Is the server up, and what does it currently have loaded.

    `size_vram / size` on a 4 GB card is the number that says whether a
    candidate model in Run 4 runs fast (on the GPU) or falls back to the
    much slower CPU path — not something `/api/chat`'s own timings reveal on
    their own. `digest` and `quantization` (`details.quantization_level`)
    are recorded too, so a saved run's metadata can say *which* build of
    "gemma3:4b" actually answered — Ollama updates a tag's bytes in place on
    a re-pull, so the name alone does not pin that down.
    """
    h = normalize_host(host) if host else OLLAMA_HOST
    owns_client = client is None
    c = client if client is not None else httpx.Client(timeout=HEALTH_TIMEOUT)
    try:
        try:
            # `timeout=` is passed explicitly on both calls, not left to
            # `c`'s own default: `run_metadata` below reuses a running
            # `OllamaChat`'s client rather than building a fresh one, and
            # that client's default timeout is the *generation* timeout (up
            # to 600s) — a health check hanging that long instead of failing
            # fast defeats the point of a separate, short HEALTH_TIMEOUT.
            version_resp = c.get(f"{h}/api/version", timeout=HEALTH_TIMEOUT)
            ps_resp = c.get(f"{h}/api/ps", timeout=HEALTH_TIMEOUT)
        except httpx.TransportError as e:
            return {"reachable": False, "error": str(e)}

        try:
            # `.json()` succeeding is not the same as the *shape* being what
            # the rest of this function assumes. A well-formed but wrong
            # shape (a bare JSON list, `{"models": null}`, a string sitting
            # where a model object belongs) would otherwise raise
            # AttributeError/TypeError deep in the loop below, uncaught by
            # the `except ValueError` this function already had for a
            # non-JSON body. Raising ValueError from the same isinstance
            # checks routes both failure modes through one handler instead
            # of two.
            version_payload = _json_object(version_resp, "/api/version")
            ps_payload = _json_object(ps_resp, "/api/ps")
            version = version_payload.get("version") if version_payload is not None else None
            loaded = _loaded_models(ps_payload) if ps_payload is not None else []
        except ValueError as e:
            # `.json()` on a truncated or non-JSON body raises here too. A
            # health check reporting a badly-formed or wrong-shaped response
            # as an unhandled exception would be worse than just saying so.
            return {"reachable": False, "error": f"malformed response from Ollama: {e}"}

        return {"reachable": True, "version": version, "loaded": loaded}
    finally:
        if owns_client:
            c.close()


def _json_object(resp: httpx.Response, path: str) -> dict | None:
    """A 2xx response's body, which must be a JSON object; None for any other
    status. ValueError, as from `.json()` itself, when the body is not one."""
    if not 200 <= resp.status_code < 300:
        return None
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"'{path}' response is not a JSON object")
    return payload


def _loaded_models(ps_payload: dict) -> list[dict]:
    models = ps_payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("'models' is not a list")
    return [_loaded_model(m) for m in models]


def _loaded_model(m: object) -> dict:
    """One `/api/ps` entry, as `health` reports it."""
    if not isinstance(m, dict):
        raise ValueError("a 'models' entry is not a JSON object")
    # Not `or 0`: that would turn a MISSING or explicitly
    # null `size`/`size_vram` into the integer 0, and a
    # missing value dividing cleanly to `gpu_share: 0.0`
    # reads as "measured zero, ran on CPU" — a real
    # (if misleading) claim — rather than "unknown", which is
    # what a missing field actually means. An explicit 0 (a
    # model that really did report no VRAM use) is left
    # alone: it is still a number, so it still divides.
    size = m.get("size")
    size_vram = m.get("size_vram")
    details = m.get("details") or {}
    if not isinstance(details, dict):
        details = {}
    # Ollama itself always sends `size`/`size_vram` as
    # numbers, but nothing else guarantees it (a proxy, a
    # different server version) — dividing a string like
    # "4GB" would raise TypeError instead of the clean
    # `gpu_share: None` a value this function cannot use
    # should produce.
    size_is_number = isinstance(size, (int, float)) and not isinstance(size, bool)
    vram_is_number = (
        isinstance(size_vram, (int, float)) and not isinstance(size_vram, bool)
    )
    can_divide = size_is_number and vram_is_number and size
    return {
        "name": m.get("name"),
        "size": size,
        "size_vram": size_vram,
        # None, not 0.0: a share of zero is a measurement: a
        # missing, zero, or non-numeric `size`/`size_vram`
        # is the absence of one.
        "gpu_share": (size_vram / size) if can_divide else None,
        "digest": m.get("digest"),
        "quantization": details.get("quantization_level"),
    }


def _matches_spec(loaded_name: str, spec_name: str) -> bool:
    """Ollama reports an untagged pull's name with `:latest` appended, so
    `ollama:llama3` (spec_name `"llama3"`) must still find `"llama3:latest"`
    in `/api/ps` — but only when the spec itself carried no tag; a spec that
    already names a tag must match that tag exactly."""
    if loaded_name == spec_name:
        return True
    return ":" not in spec_name and loaded_name == f"{spec_name}:latest"


def run_metadata(chat: OllamaChat, client: httpx.Client | None = None) -> dict:
    """What actually served this run, worth pinning to disk next to its
    answers: the Ollama build, how much of the model sat on the GPU, and
    enough of the model's identity (digest, quantization) that "gemma3:4b"
    a month from now is provably comparable to today's — a tag can be
    re-pulled to different bytes, but a digest cannot.

    Reuses `chat`'s own client when `client` is not given, rather than
    `health`'s usual fresh one: `chat` may hold a test double (or simply an
    already-open connection to the right host), and defaulting to a brand
    new real `httpx.Client` here would silently bypass it.
    """
    info = health(host=chat.host, client=client if client is not None else chat._ensure_client())
    if not info.get("reachable"):
        return {
            "model": chat.model, "ollama_version": None, "gpu_share": None,
            "digest": None, "quantization": None,
            "reachable": False, "error": info.get("error"),
        }

    match = next(
        (m for m in info.get("loaded", []) if _matches_spec(m.get("name") or "", chat.model)),
        None,
    )
    return {
        "model": chat.model,
        "ollama_version": info.get("version"),
        "gpu_share": match.get("gpu_share") if match else None,
        "digest": match.get("digest") if match else None,
        "quantization": match.get("quantization") if match else None,
        "num_gpu_requested": chat.num_gpu,
        "reachable": True,
        "error": None,
    }
