"""What every request to the local app meets before any route (ADR-023).

`Guard` is pure ASGI middleware. It refuses, in this order:
1. a Host header that is not a loopback name (`ALLOWED_HOSTS`): 400. This is
   what defeats DNS rebinding;
2. a POST, PUT, PATCH or DELETE without `X-LegalRAG: 1`, or with an Origin
   that is not this app's own: 403. A cross-site page cannot add that header
   without a CORS preflight, and no CORS is configured;
3. an upload or a chat over its client's rate limit: 429, with Retry-After;
4. any request framed by Transfer-Encoding: 411. The server sizes such a body
   by its chunks, so a Content-Length beside it is a number nothing enforces;
5. a POST whose declared length is missing (411) or too large (413).
All of that happens before a byte of the body is read. Whatever it lets
through reads its body through a counter that stops it with 413 once it
passes its cap, whatever the headers claimed, and gets `SECURITY_HEADERS` on
its response.

The error shape lives here too, since the guard answers with it:
`{"error": code, "message_ar": sentence}`, a stable code and a Modern
Standard Arabic sentence, never exception text.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from . import library as lib
from .pipeline import MAX_QUESTION_CHARS, MIN_QUESTION_CHARS

ALLOWED_HOSTS = ("127.0.0.1", "localhost")
APP_HEADER = "X-LegalRAG"          # required on every POST/DELETE (value "1")
CHAT_PATH = "/api/chat"
UPLOAD_PATH = "/api/documents"
MULTIPART_OVERHEAD = 1024 * 1024   # boundaries and part headers around the file itself
MAX_JSON_BYTES = 64 * 1024         # a 500-character question and 20 ids, with room to spare

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}

MESSAGES_AR = {
    "unsupported_file": "نوع الملف غير مدعوم. ارفع ملف PDF أو TXT.",
    "file_too_large": "حجم الملف أكبر من الحد المسموح (20 ميجابايت).",
    "pdf_too_large": "تعذّرت معالجة الملف: عدد صفحاته أو وقت استخراج نصه أكبر من الحد المسموح.",
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
# `invalid_request` when the question's length, and nothing else, is the problem.
QUESTION_MESSAGE_AR = f"السؤال يجب أن يكون بين {MIN_QUESTION_CHARS} و{MAX_QUESTION_CHARS} حرف."

_STATUS = {
    "unsupported_file": 400, "no_text": 400, "invalid_request": 400, "forbidden": 403,
    "not_found": 404, "file_too_large": 413, "pdf_too_large": 413, "rate_limited": 429, "internal": 500,
    "generator_unavailable": 503, "encoder_unavailable": 503,
}
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def error_response(code: str, status: int | None = None, *, message: str | None = None,
                   headers: dict[str, str] | None = None) -> JSONResponse:
    """The one error shape. The security headers are set here as well as by
    `Guard`: the catch-all 500 handler answers from outside every middleware."""
    return JSONResponse(
        {"error": code, "message_ar": message or MESSAGES_AR[code]},
        status_code=status or _STATUS[code],
        headers={**SECURITY_HEADERS, **(headers or {})},
    )


class RateLimiter:
    """Sliding-window limit per client key; thread-safe; bounded memory; injectable clock.

    At most `max_keys` clients are remembered: when a new one arrives, the one
    seen least recently is forgotten, and starts over if it returns."""

    def __init__(self, max_events: int, window_s: float, clock: Callable[[], float] = time.monotonic,
                 max_keys: int = 1000):
        if max_events < 1 or window_s <= 0 or max_keys < 1:
            raise ValueError("max_events and max_keys must be at least 1, and window_s positive")
        self.max_events = max_events
        self.window_s = window_s
        self._clock = clock
        self._max_keys = max_keys
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """None, and the event is counted; else the seconds until one is allowed."""
        now = self._clock()
        with self._lock:
            events = self._events.get(key)
            if events is None:
                if len(self._events) >= self._max_keys:
                    self._events.popitem(last=False)
                events = self._events[key] = deque()
            else:
                self._events.move_to_end(key)
            while events and events[0] <= now - self.window_s:
                events.popleft()
            if len(events) >= self.max_events:
                return events[0] + self.window_s - now
            events.append(now)
            return None


class Guard:
    """The middleware; see the module docstring for what it refuses, in order."""

    def __init__(self, app, limits: dict[str, RateLimiter]):
        self.app = app
        self.limits = limits  # POST path -> its limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                present = {bytes(name).lower() for name, _ in message.get("headers", [])}
                extra = [(name.lower().encode("latin-1"), value.encode("latin-1"))
                         for name, value in SECURITY_HEADERS.items()
                         if name.lower().encode("latin-1") not in present]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        refusal = self._refusal(scope)
        if refusal is not None:
            await refusal(scope, receive, send_with_headers)
        else:
            await self.app(scope, _capped(receive, *_body_cap(scope)), send_with_headers)

    def _refusal(self, scope) -> JSONResponse | None:
        headers = _first_headers(scope)
        hostname, port = _split_host_header(headers.get("host", ""))
        if hostname not in ALLOWED_HOSTS:
            return error_response("invalid_request")
        method = scope["method"]
        if method in _UNSAFE_METHODS:
            origin = headers.get("origin")
            if headers.get(APP_HEADER.lower()) != "1" or (origin is not None and origin not in _origins(port)):
                return error_response("forbidden")
        if method == "POST":
            wait = self._wait(scope)
            if wait is not None:
                return error_response("rate_limited", headers={"Retry-After": str(max(1, math.ceil(wait)))})
        if "transfer-encoding" in headers:
            return error_response("invalid_request", 411)  # framed by chunks, not by Content-Length
        return _length_refusal(scope, headers.get("content-length")) if method == "POST" else None

    def _wait(self, scope) -> float | None:
        """Seconds until this client may POST here again, or None. The key is the
        connection's own address, never a header a client can write."""
        limiter = self.limits.get(scope["path"])
        if limiter is None:
            return None
        client = scope.get("client")
        return limiter.check(client[0] if client else "unknown")


def _first_headers(scope) -> dict[str, str]:
    """Header names lowercased; the first occurrence of a repeated one wins, as in Starlette."""
    found: dict[str, str] = {}
    for name, value in scope.get("headers", []):
        found.setdefault(name.decode("latin-1").lower(), value.decode("latin-1"))
    return found


def _split_host_header(value: str) -> tuple[str, int | None]:
    """`host[:port]` -> (lowercased host, port or None). An IPv6 literal stays
    whole, and a port that is not a number empties the host: both are refused."""
    text = value.strip().lower()
    if text.startswith("["):
        return text, None
    host, colon, port = text.partition(":")
    if not colon or not port:
        return host, None
    if not (port.isascii() and port.isdigit()):
        return "", None
    return host, int(port)


def _origins(port: int | None) -> set[str]:
    """What a browser sends as Origin from this app's own page; port 80 is left out."""
    port = port or 80
    origins = {f"http://{host}:{port}" for host in ALLOWED_HOSTS}
    return origins | ({f"http://{host}" for host in ALLOWED_HOSTS} if port == 80 else set())


def _length_refusal(scope, declared: str | None) -> JSONResponse | None:
    if declared is None:
        return error_response("invalid_request", 411)
    if not (declared.isascii() and declared.isdigit()):
        return error_response("invalid_request")
    cap, code = _body_cap(scope)
    return error_response(code, 413) if int(declared) > cap else None


def _body_cap(scope) -> tuple[int, str]:
    """The most body bytes a request may carry, and the error code past that."""
    if scope["method"] == "POST" and scope["path"] == UPLOAD_PATH:
        return lib.MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD, "file_too_large"
    return MAX_JSON_BYTES, "invalid_request"


class BodyTooLarge(HTTPException):
    """A body past its cap, raised while it is read. An HTTPException, so
    FastAPI's body parsing re-raises it instead of turning it into a 400; the
    app answers it as `code`, with status 413."""

    def __init__(self, code: str):
        super().__init__(status_code=413)
        self.code = code


def _capped(receive, cap: int, code: str):
    """`receive`, counting the body bytes the app takes: past `cap` it raises
    BodyTooLarge, whatever Content-Length or Transfer-Encoding claimed."""
    taken = 0

    async def capped():
        nonlocal taken
        message = await receive()
        if message["type"] == "http.request":
            taken += len(message.get("body", b""))
            if taken > cap:
                raise BodyTooLarge(code)
        return message

    return capped
