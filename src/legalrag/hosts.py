"""The Host headers a local server accepts (ADR-023).

Standard library only, so the legacy `server.py` shares it without importing
FastAPI. Binding to 127.0.0.1 stops remote connections but not DNS rebinding,
so both servers also check the Host header itself.
"""

from __future__ import annotations

ALLOWED_HOSTS = ("127.0.0.1", "localhost")


def split_host_header(value: str) -> tuple[str, int | None]:
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
