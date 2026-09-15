"""The local app's HTTP API — the P1 demo layer (ADR-023, ADR-025).

    python tasks.py app [--host 127.0.0.1] [--port 8000]

One `Library` and one `Pipeline` behind FastAPI: upload a PDF or TXT, list
and soft-delete documents, ask about them, check the generator, and load the
chat page from an explicit allowlist of files under ui/.

Single user, on this machine. `web_guard.Guard` sees every request first
(a loopback Host; the app header and a same-origin Origin on anything that
changes state; rate limits; the declared body size) and puts the security
headers on every response. Every error is `{"error": code, "message_ar":
sentence}`, mapped from exceptions in one place (`_add_error_handlers`):
exception text is logged here and never sent.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, StrictStr, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import library as lib
from . import ollama
from .claims import build_generators
from .generate import parse_model_spec
from .library import DocMeta, DocumentNotFound, EncoderUnavailable, Library, LibraryError
from .ollama import GeneratorUnavailable
from .pipeline import MAX_QUESTION_CHARS, MIN_QUESTION_CHARS, Pipeline
from .web_guard import (  # noqa: F401  ALLOWED_HOSTS, APP_HEADER, SECURITY_HEADERS: this module's contract too
    ALLOWED_HOSTS,
    APP_HEADER,
    CHAT_PATH,
    MESSAGES_AR,
    QUESTION_MESSAGE_AR,
    SECURITY_HEADERS,
    UPLOAD_PATH,
    Guard,
    RateLimiter,
    error_response,
)

logger = logging.getLogger(__name__)

MAX_DOC_IDS = 20
UI_DIR = Path(__file__).resolve().parents[2] / "ui"

_JS = "text/javascript; charset=utf-8"
# Exact URL path -> (file under the UI directory, content type). One route
# each: nothing else under ui/ is reachable, and no content type is guessed.
UI_FILES: dict[str, tuple[str, str]] = {
    "/": ("app.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", _JS),
    "/js/copy.js": ("js/copy.js", _JS),
    "/js/api.js": ("js/api.js", _JS),
    "/js/dom.js": ("js/dom.js", _JS),
    "/js/documents.js": ("js/documents.js", _JS),
    "/js/chat.js": ("js/chat.js", _JS),
}

DEFAULT_DATA_DIR = "data/app"
DEFAULT_MODEL = "ollama:gemma3:4b"
CHAT_LIMIT = (20, 60.0)            # per client address: requests, seconds
UPLOAD_LIMIT = (10, 60.0)


class AppConfigError(RuntimeError):
    """The environment names a configuration the app refuses to start with."""


class InvalidQuestion(Exception):
    """The question's length is outside what the pipeline answers."""


class ChatRequest(BaseModel):
    question: StrictStr  # its length is the pipeline's to enforce
    doc_ids: list[StrictStr] | None = Field(default=None, max_length=MAX_DOC_IDS)

    @field_validator("doc_ids")
    @classmethod
    def _well_formed(cls, doc_ids: list[str] | None) -> list[str] | None:
        if doc_ids is not None and not all(lib.DOC_ID.fullmatch(doc_id) for doc_id in doc_ids):
            raise ValueError("a document id is malformed")
        return doc_ids


_DOCUMENT_FIELDS = ("doc_id", "title", "kind", "suffix", "pages", "chunks", "size_bytes", "created_at")


def _document(meta: DocMeta) -> dict:
    return {name: getattr(meta, name) for name in _DOCUMENT_FIELDS}


def create_app(library, pipeline, *, health_probe: Callable[[], dict] | None = None,
               chat_limiter: RateLimiter | None = None, upload_limiter: RateLimiter | None = None,
               ui_dir: Path = UI_DIR) -> FastAPI:
    """The app over `library` and `pipeline`. `health_probe` answers
    `{"reachable", "model", "gpu_share"}` (see `ollama_probe`); without one the
    generator reads as unreachable. Only those keys, each of its own type,
    ever reach the client."""
    app = FastAPI(title="arabic-legal-rag", docs_url=None, redoc_url=None, openapi_url=None)
    limits = {
        CHAT_PATH: chat_limiter if chat_limiter is not None else RateLimiter(*CHAT_LIMIT),
        UPLOAD_PATH: upload_limiter if upload_limiter is not None else RateLimiter(*UPLOAD_LIMIT),
    }
    app.add_middleware(Guard, limits=limits)
    _add_error_handlers(app)
    for url, (name, content_type) in UI_FILES.items():
        app.add_api_route(url, _ui_file(Path(ui_dir) / name, content_type), methods=["GET"],
                          include_in_schema=False)

    @app.get("/api/health")
    def health() -> dict:
        generator = _generator_status(health_probe)
        return {"status": "ok" if generator["reachable"] else "degraded", "generator": generator,
                "documents": len(library.documents())}

    @app.get(UPLOAD_PATH)
    def documents() -> dict:
        return {"documents": [_document(meta) for meta in library.documents()]}

    @app.post(UPLOAD_PATH, status_code=201)
    def upload(file: Annotated[UploadFile, File()]) -> dict:
        # One byte past the limit is enough to know. The client's name for the
        # file only ever becomes a title (Library.add), never part of a path.
        data = file.file.read(lib.MAX_UPLOAD_BYTES + 1)
        if len(data) > lib.MAX_UPLOAD_BYTES:
            raise lib.FileTooLarge("the uploaded file is over the limit")
        return _document(library.add(data, file.filename or ""))

    @app.delete(UPLOAD_PATH + "/{doc_id}")
    def delete(doc_id: str) -> dict:
        if not lib.DOC_ID.fullmatch(doc_id):
            raise DocumentNotFound("a malformed document id")  # the library is never asked
        library.soft_delete(doc_id)
        return {"deleted": doc_id}

    @app.post(CHAT_PATH)
    def chat(body: ChatRequest) -> dict:
        try:
            result = pipeline.ask(body.question, body.doc_ids)
        except ValueError as e:
            if MIN_QUESTION_CHARS <= len(body.question.strip()) <= MAX_QUESTION_CHARS:
                raise  # not the question: a fault on this side, answered as `internal`
            raise InvalidQuestion() from e
        return result.to_dict()

    return app


def _ui_file(path: Path, content_type: str) -> Callable[[], Response]:
    def serve() -> Response:
        try:
            content = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return error_response("not_found")  # listed, but not written yet
        return Response(content, media_type=content_type)
    return serve


def _generator_status(probe: Callable[[], dict] | None) -> dict:
    """The generator as /api/health reports it, whatever the probe returned or raised."""
    info: object = {}
    if probe is not None:
        try:
            info = probe()
        except Exception:  # an outside service: health must still answer
            logger.warning("the generator health probe failed", exc_info=True)
    if not isinstance(info, dict):
        info = {}
    model, share = info.get("model"), info.get("gpu_share")
    numeric = isinstance(share, (int, float)) and not isinstance(share, bool) and math.isfinite(share)
    return {"reachable": info.get("reachable") is True,
            "model": model if isinstance(model, str) else None,
            "gpu_share": float(share) if numeric else None}


def _add_error_handlers(app: FastAPI) -> None:
    """Every exception a request can raise, mapped to its error response in one place."""

    async def library_error(request, exc: LibraryError) -> JSONResponse:
        code = exc.code if exc.code in MESSAGES_AR else "invalid_request"
        logger.info("%s %s refused (%s): %s", request.method, request.url.path, code, exc)
        return error_response(code)

    async def unavailable(request, exc: Exception) -> JSONResponse:
        code = "encoder_unavailable" if isinstance(exc, EncoderUnavailable) else "generator_unavailable"
        logger.error("%s %s failed (%s)", request.method, request.url.path, code, exc_info=exc)
        return error_response(code)

    async def invalid_question(request, exc: InvalidQuestion) -> JSONResponse:
        return error_response("invalid_request", message=QUESTION_MESSAGE_AR)

    async def invalid_input(request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's own 422 body echoes the submitted input; this one names nothing.
        logger.info("%s %s refused: the request did not validate", request.method, request.url.path)
        return error_response("invalid_request")

    async def http_error(request, exc: StarletteHTTPException) -> JSONResponse:
        status = exc.status_code
        code = {404: "not_found", 413: "file_too_large"}.get(status, "internal" if status >= 500 else "invalid_request")
        allow = exc.headers.get("Allow") if exc.headers else None
        return error_response(code, status, headers={"Allow": allow} if allow else None)

    async def internal(request, exc: Exception) -> JSONResponse:
        logger.error("%s %s failed unexpectedly", request.method, request.url.path, exc_info=exc)
        return error_response("internal")

    for exc_type, handler in ((LibraryError, library_error), (EncoderUnavailable, unavailable),
                              (GeneratorUnavailable, unavailable), (InvalidQuestion, invalid_question),
                              (RequestValidationError, invalid_input), (StarletteHTTPException, http_error),
                              (Exception, internal)):
        app.add_exception_handler(exc_type, handler)


# --- starting it -------------------------------------------------------------------


def ollama_probe(spec: str) -> Callable[[], dict]:
    """/api/health's check of an ollama: generator: is the server up, and how
    much of this model sits on the GPU. Opens no connection until first called."""
    _, name = parse_model_spec(spec)
    chat = ollama.ollama_chat(name, num_predict=1)  # never generates: a host, a name, one reused client

    def probe() -> dict:
        meta = ollama.run_metadata(chat)
        return {"reachable": meta["reachable"], "model": spec, "gpu_share": meta["gpu_share"]}

    return probe


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def build_from_env() -> FastAPI:
    """LEGALRAG_DATA_DIR (default data/app) and LEGALRAG_MODEL (default
    ollama:gemma3:4b). Exactly one Pipeline: its lock is what keeps two
    questions from interleaving calls on the one model."""
    spec = os.environ.get("LEGALRAG_MODEL") or DEFAULT_MODEL
    try:
        runtime, _ = parse_model_spec(spec)
    except ValueError:
        runtime = None
    if runtime != "ollama":
        raise AppConfigError(
            f"LEGALRAG_MODEL must be an ollama: model spec such as {DEFAULT_MODEL}, not {spec!r}. "
            "The app answers through a local Ollama server (ADR-023)."
        )
    host = urlsplit(ollama.OLLAMA_HOST).hostname or ""
    if not _is_loopback(host):
        logger.warning("OLLAMA_HOST %s is not a loopback address: questions and document text will "
                       "leave this machine, and ADR-023 keeps the demo fully local", ollama.OLLAMA_HOST)
    library = Library(Path(os.environ.get("LEGALRAG_DATA_DIR") or DEFAULT_DATA_DIR))
    pipeline = Pipeline(library, build_generators(spec, "gated"))
    return create_app(library, pipeline, health_probe=ollama_probe(spec))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app", allow_abbrev=False,
                                     description="The local app: upload a document and ask it (ADR-023).")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    try:
        args = parser.parse_args(list(argv) if argv is not None else [])
    except SystemExit as e:  # argparse has already printed the usage and the error
        return e.code if isinstance(e.code, int) else 2

    if not _is_loopback(args.host):
        print(f"WARNING: --host {args.host} is not a loopback address, so other machines can reach "
              "this app. Use it only inside a container whose port is published on 127.0.0.1.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        app = build_from_env()
    except AppConfigError as e:
        print(str(e))
        return 2
    print(f"\n  http://127.0.0.1:{args.port}\n\nCtrl+C to stop.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
