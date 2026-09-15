"""The on-disk document library behind the P1 demo layer (ADR-023).

    root/docs/<doc_id>/{source.pdf | source.txt, meta.json, chunks.jsonl, embeddings.npz}
    root/tmp/<uuid4 hex>/    an upload being staged

`doc_id` is the first 12 hex digits of the upload's sha256; the same bytes,
checked against the full hash, are always the same document, and re-uploading
a soft-deleted one restores it. The client's filename only ever becomes a
display title: every path is built from `doc_id` and fixed names (the layout
itself is `docstore`'s). Made to sit under an HTTP layer — a client mistake
raises a `LibraryError` with a stable `code`, and nothing here ends the process.

One process per root: the lock serializes threads, not processes. A second
`Library` opened on the root by mistake clears only staging older than
STALE_STAGING_SECONDS, and an upload whose staging lost a file is refused.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path

from . import dense, pdf_text
from .chunking import Chunk, chunk_document
from .dense import DenseIndex, Encoder
from .docstore import CHUNKS, EMBEDDINGS, META, DocMeta, clear_stale, missing
from .docstore import read_chunks, read_meta, write_chunks, write_meta

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_TEXT_CHARS = 200
ALLOWED_SUFFIXES = (".pdf", ".txt")
DOC_ID = re.compile(r"^[0-9a-f]{12}$")
# A .txt is decoded strictly in each of these in turn: UTF-8 (a BOM dropped), then Windows-1256.
TEXT_ENCODINGS = ("utf-8-sig", "cp1256")
# Staging untouched this long when a Library opens belongs to an upload that died.
STALE_STAGING_SECONDS = 3600
# A display title, not a path: long enough to recognise a file by.
MAX_TITLE_CHARS = 120

# Unicode's Bidi_Control characters, ARABIC LETTER MARK (U+061C) included:
# invisible, and able to make a title read as something it is not.
_BIDI_CONTROLS = frozenset(
    map(chr, (0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)))
)
_PATH_SEPARATORS = re.compile(r"[\\/]")
# What loading a stored index raises when its files are not what the library wrote.
_LOAD_ERRORS = (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile)


class LibraryError(Exception):
    """Raised on purpose, with a stable `code` for an HTTP layer to map."""
    code = "library_error"


class UnsupportedFile(LibraryError):
    """A bad suffix, no %PDF- magic, a .txt with NUL bytes or undecodable text, an unreadable PDF."""
    code = "unsupported_file"


class FileTooLarge(LibraryError):
    code = "file_too_large"


class NoTextLayer(LibraryError):
    """Under MIN_TEXT_CHARS of extracted text: most likely a scanned PDF. OCR is out of scope."""
    code = "no_text"


class DocumentNotFound(LibraryError):
    """A malformed id, an unknown one, or a soft-deleted document."""
    code = "not_found"


class StorageError(LibraryError):
    """The library's own files are not what it wrote or expects: staging that lost a file,
    an index that will not load, a directory it did not skip at open, a stored document
    whose full sha256 differs. A server-side fault, never the client's."""
    code = "internal"


class EncoderUnavailable(RuntimeError):
    """The embedding model cannot be loaded: a server configuration problem, not a client error."""


@dataclass(frozen=True)
class LibraryHit:
    chunk: Chunk
    doc_title: str
    score: float


@dataclass(frozen=True)
class _DocumentIndex:
    dense: DenseIndex
    chunks: dict[str, Chunk]  # by chunk id, which is what a dense Hit carries


class Library:
    """Uploaded documents on disk, each with its own dense index.

    One lock guards the in-memory registry and every final move or metadata
    write. The slow part of an upload — extraction, chunking, embedding —
    runs outside it, in a staging directory of its own.
    """

    def __init__(self, root: Path, encoder: Encoder | None = None, model_name: str = dense.DEFAULT_MODEL):
        self.root = Path(root)
        self.model_name = model_name
        # ONE encoder behind every document's index: the injected one, or the
        # real model, loaded the first time something is embedded.
        self._encoder = encoder if encoder is not None else _LazyEncoder(model_name)
        self._docs = self.root / "docs"
        self._tmp = self.root / "tmp"
        self._lock = threading.Lock()
        self._metas: dict[str, DocMeta] = {}
        self._indexes: dict[str, _DocumentIndex] = {}
        self._skipped: set[str] = set()  # document directories found unservable at open

        self._docs.mkdir(parents=True, exist_ok=True)
        self._tmp.mkdir(parents=True, exist_ok=True)
        clear_stale(self._tmp, STALE_STAGING_SECONDS)
        for doc_dir in sorted(self._docs.iterdir()):
            if not DOC_ID.fullmatch(doc_dir.name):
                continue
            meta, problem = read_meta(doc_dir)
            if meta is None:  # kept on disk: re-uploading these bytes replaces it
                logger.warning("skipping stored document %s: %s", doc_dir.name, problem)
                self._skipped.add(doc_dir.name)
            else:
                self._metas[meta.doc_id] = meta

    def add(self, data: bytes, filename: str) -> DocMeta:
        """Store and index an upload, or return the document these exact bytes already
        are, restored first if it was soft-deleted. Checked before anything touches disk:
        size, suffix, PDF magic, then text decoding. Raises FileTooLarge, UnsupportedFile,
        NoTextLayer, EncoderUnavailable or StorageError."""
        suffix = _checked_suffix(data, filename)
        text = _decode_text(data) if suffix == ".txt" else None
        sha256 = hashlib.sha256(data).hexdigest()
        doc_id = sha256[:12]
        with self._lock:
            if doc_id in self._metas:
                return self._restore(doc_id, sha256)

        title = _display_title(filename, suffix)
        staging = self._tmp / uuid.uuid4().hex
        try:
            staging.mkdir(parents=True)
            source = staging / f"source{suffix}"
            source.write_bytes(data)
            pages = _pdf_pages(source) if text is None else text.split("\f")
            chars = sum(len(run) for page in pages for run in page.split())
            if chars < MIN_TEXT_CHARS:
                raise NoTextLayer(f"only {chars} characters of text were found; at least "
                                  f"{MIN_TEXT_CHARS} are needed (a scanned PDF needs OCR first)")
            kind, chunks = chunk_document(doc_id, pages, title)
            if not chunks:
                raise NoTextLayer("no readable text was found in the document")
            write_chunks(staging / CHUNKS, chunks)
            DenseIndex(_index_docs(chunks, title), encoder=self._encoder,
                       cache_path=staging / EMBEDDINGS, model_name=self.model_name).save()
            meta = DocMeta(doc_id=doc_id, title=title, kind=kind, suffix=suffix,
                           size_bytes=len(data), sha256=sha256, pages=len(pages),
                           chunks=len(chunks), chars=chars, created_at=_now())
            write_meta(staging / META, meta)
            with self._lock:
                if doc_id in self._metas:  # an identical upload finished while this one embedded
                    return self._restore(doc_id, sha256)
                self._install(staging, meta)
                return meta
        finally:
            shutil.rmtree(staging, ignore_errors=True)  # already moved away once installed

    def documents(self) -> list[DocMeta]:
        """Every document not soft-deleted, oldest first."""
        with self._lock:
            live = [m for m in self._metas.values() if not m.deleted]
        return sorted(live, key=lambda m: (m.created_at, m.doc_id))

    def get(self, doc_id: str) -> DocMeta:
        """`DocumentNotFound` if `doc_id` is malformed, unknown or deleted."""
        with self._lock:
            return self._live(doc_id)

    def soft_delete(self, doc_id: str) -> DocMeta:
        """Hide a document from `documents`, `get` and `search`. No file is
        deleted; uploading the same bytes again restores it."""
        with self._lock:
            meta = replace(self._live(doc_id), deleted=True, deleted_at=_now())
            write_meta(self._docs / meta.doc_id / META, meta)
            self._metas[meta.doc_id] = meta
            self._indexes.pop(meta.doc_id, None)
            return meta

    def search(self, query: str, k: int = 5, doc_ids: list[str] | None = None) -> list[LibraryHit]:
        """The top `k` chunks for `query` across every live document, or exactly `doc_ids`
        (each a live document, or DocumentNotFound). Ranked by score, descending; a tie goes
        to the older document, then the lower chunk number. A document whose index will not
        load is logged and skipped by an unscoped search, and DocumentNotFound to a scoped one."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        with self._lock:
            if doc_ids is None:
                scope = [m for m in self._metas.values() if not m.deleted]
            else:
                scope = [self._live(doc_id) for doc_id in dict.fromkeys(doc_ids)]

        ranked: list[tuple[tuple, LibraryHit]] = []
        for meta in scope:
            try:
                index = self._index(meta)
            except StorageError as e:
                if doc_ids is not None:
                    raise DocumentNotFound("no such document") from e
                logger.warning("search skipped document %s: %s", meta.doc_id, e)
                continue
            for hit in index.dense.search(query, k):
                chunk = index.chunks[hit.id]
                order = (-hit.score, meta.created_at, chunk.number, meta.doc_id)
                ranked.append((order, LibraryHit(chunk=chunk, doc_title=meta.title, score=hit.score)))
        ranked.sort(key=lambda pair: pair[0])
        return [hit for _, hit in ranked[:k]]

    def _install(self, staging: Path, meta: DocMeta) -> None:
        """Move a complete staging directory to docs/<doc_id> and register it; the caller
        holds the lock. A directory already there is replaced only if it was skipped at
        open: moved aside, put back if the move fails, deleted once the new one is in place."""
        lost = missing(staging, (f"source{meta.suffix}", CHUNKS, EMBEDDINGS, META))
        if lost:  # cleared mid-upload, and saving the embeddings recreated the directory
            raise StorageError(f"the staged upload of {meta.doc_id} lost {', '.join(lost)}")
        target = self._docs / meta.doc_id
        aside = None
        if os.path.lexists(target):
            if meta.doc_id not in self._skipped:
                raise StorageError(f"docs/{meta.doc_id} appeared after the library opened; not replacing it")
            aside = self._tmp / uuid.uuid4().hex
            os.replace(target, aside)
        try:
            os.replace(staging, target)
        except BaseException:
            if aside is not None:
                os.replace(aside, target)  # the skipped directory goes back where it was
            raise
        if aside is not None:
            shutil.rmtree(aside, ignore_errors=True)
        self._skipped.discard(meta.doc_id)
        self._metas[meta.doc_id] = meta

    def _live(self, doc_id: object) -> DocMeta:
        """The live document `doc_id`; the caller holds the lock."""
        valid = isinstance(doc_id, str) and DOC_ID.fullmatch(doc_id)
        meta = self._metas.get(doc_id) if valid else None
        if meta is None or meta.deleted:
            raise DocumentNotFound("no such document")
        return meta

    def _restore(self, doc_id: str, sha256: str) -> DocMeta:
        """The stored document these bytes are, undeleted if it was deleted; the caller holds
        the lock. doc_id keeps 48 bits of the hash, so the full sha256 must agree first."""
        meta = self._metas[doc_id]
        if meta.sha256 != sha256:
            raise StorageError(f"document {doc_id} is stored for other bytes: its sha256 differs")
        if meta.deleted:
            meta = replace(meta, deleted=False, deleted_at=None)
            write_meta(self._docs / doc_id / META, meta)
            self._metas[doc_id] = meta
        return meta

    def _index(self, meta: DocMeta) -> _DocumentIndex:
        """`meta`'s dense index: cached, or loaded from its chunks and saved embeddings —
        nothing is embedded when the fingerprint matches. StorageError if they will not load."""
        with self._lock:
            cached = self._indexes.get(meta.doc_id)
        if cached is not None:
            return cached

        doc_dir = self._docs / meta.doc_id
        try:
            chunks = read_chunks(doc_dir / CHUNKS)
            index = DenseIndex(_index_docs(chunks, meta.title), encoder=self._encoder,
                               cache_path=doc_dir / EMBEDDINGS, model_name=self.model_name)
        except _LOAD_ERRORS as e:
            raise StorageError(f"the index of document {meta.doc_id} will not load: {e}") from e
        loaded = _DocumentIndex(dense=index, chunks={c.id: c for c in chunks})
        with self._lock:
            current = self._metas.get(meta.doc_id)
            if current is None or current.deleted:
                return loaded  # deleted while it loaded: serve this search, cache nothing
            return self._indexes.setdefault(meta.doc_id, loaded)


class _LazyEncoder:
    """The real model, loaded on the first `encode`, so opening a library never pays for torch."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def encode(self, texts, **kwargs):
        with self._lock:
            if self._model is None:
                self._model = _load_model(self.model_name)
        return self._model.encode(texts, **kwargs)


def _load_model(model_name: str):
    """`dense.load_model`, with every way it fails turned into EncoderUnavailable.
    It ends the process when an import fails (torch missing under an installed
    sentence-transformers, say), which inside a server would take the server down;
    OSError is weights it cannot find or read, e.g. offline with no cached copy."""
    if find_spec("sentence_transformers") is None:
        raise EncoderUnavailable("sentence-transformers is not installed, so documents cannot be embedded: "
                                 "python tasks.py setup (or: pip install -r requirements.txt)")
    try:
        return dense.load_model(model_name)
    except (SystemExit, ImportError, OSError) as e:
        raise EncoderUnavailable(f"the embedding model {model_name!r} could not be loaded: {e}") from e


def _checked_suffix(data: bytes, filename: str | None) -> str:
    """The upload's suffix, once its size, suffix and PDF magic all pass."""
    if not data:
        raise UnsupportedFile("the upload is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise FileTooLarge(f"the upload is {len(data):,} bytes; the limit is {MAX_UPLOAD_BYTES:,}")
    suffix = _suffix(filename)
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedFile("only .pdf and .txt files can be uploaded")
    if suffix == ".pdf" and not data.startswith(b"%PDF-"):
        raise UnsupportedFile("the file is named .pdf but is not a PDF")
    return suffix


def _decode_text(data: bytes) -> str:
    """Decoded strictly in each of TEXT_ENCODINGS in turn."""
    if b"\x00" in data:
        raise UnsupportedFile("a .txt file must not contain NUL bytes")
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # cp1256 maps all 256 byte values, so under the default list the NUL check is
    # what refuses a binary file; this refuses what a stricter list cannot decode.
    raise UnsupportedFile("the text is not in an accepted encoding")


def _pdf_pages(path: Path) -> list[str]:
    try:
        # keep_latin=True: the default drops EVERY Latin glyph — right for the
        # bilingual statute corpus, whose English column is a translation, but
        # an uploaded Arabic document would lose its inline terms (VPN, Wi-Fi,
        # Microsoft Teams). Known limit of this mode: a side-by-side bilingual
        # page is not split, so its English column is chunked with the Arabic.
        return pdf_text.extract_pages(path, keep_latin=True)
    except Exception as e:  # a damaged PDF raises whatever pdfminer hits first
        raise UnsupportedFile("the PDF could not be read") from e


def _basename(filename: str | None) -> str:
    """The last component of a client-supplied name, whatever separator its OS used."""
    return _PATH_SEPARATORS.split(filename or "")[-1]


def _suffix(filename: str | None) -> str:
    _, dot, extension = _basename(filename).lower().rpartition(".")
    return f".{extension}" if dot else ""


def _display_title(filename: str | None, suffix: str) -> str:
    """The base name without control or bidi-control characters, whitespace
    collapsed, at most MAX_TITLE_CHARS. Display only — never part of a path."""
    name = "".join(ch for ch in _basename(filename)
                   if unicodedata.category(ch) != "Cc" and ch not in _BIDI_CONTROLS)
    title = " ".join(name.split())[:MAX_TITLE_CHARS].rstrip()
    return title or f"document{suffix}"


def _index_docs(chunks: list[Chunk], title: str) -> list[dict]:
    return [{"id": c.id, "number": c.number, "text": c.text, "law_name": title} for c in chunks]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
