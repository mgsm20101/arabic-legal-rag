"""The on-disk document library behind the P1 demo layer (ADR-023).

    root/docs/<doc_id>/{source.pdf | source.txt, meta.json, chunks.jsonl, embeddings.npz}
    root/tmp/<uuid4 hex>/    an upload being staged; cleared whenever a Library opens

`doc_id` is the first 12 hex digits of the upload's sha256, so the same
bytes are always the same document, and re-uploading a soft-deleted one
restores it. The client's filename only ever becomes a display title: every
path is built from `doc_id` and fixed names. Made to sit under an HTTP
layer — a client mistake raises a `LibraryError` with a stable `code`, and
nothing here ends the process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import unicodedata
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path

from . import dense, pdf_text
from .chunking import Chunk, chunk_document
from .dense import DenseIndex, Encoder

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_TEXT_CHARS = 200
ALLOWED_SUFFIXES = (".pdf", ".txt")
DOC_ID = re.compile(r"^[0-9a-f]{12}$")

# A display title, not a path: long enough to recognise a file by.
MAX_TITLE_CHARS = 120

# Unicode's Bidi_Control characters, ARABIC LETTER MARK (U+061C) included:
# invisible, and able to make a title read as something it is not.
_BIDI_CONTROLS = frozenset(
    map(chr, (0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)))
)
_PATH_SEPARATORS = re.compile(r"[\\/]")


class LibraryError(Exception):
    """A request the client can correct; `code` is stable, for an HTTP layer to map."""
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


class EncoderUnavailable(RuntimeError):
    """sentence-transformers is missing: a server configuration problem, not a client error."""


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    kind: str            # "statute" | "generic"
    suffix: str          # ".pdf" | ".txt"
    size_bytes: int
    sha256: str
    pages: int
    chunks: int
    chars: int           # non-whitespace characters extracted: the count held to MIN_TEXT_CHARS
    created_at: str      # UTC ISO-8601, seconds
    deleted: bool = False
    deleted_at: str | None = None


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

        self._docs.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self._tmp, ignore_errors=True)  # an upload that died is never resumed
        self._tmp.mkdir(parents=True, exist_ok=True)
        for doc_dir in sorted(self._docs.iterdir()):
            meta = _read_meta(doc_dir)
            if meta is not None:
                self._metas[meta.doc_id] = meta

    def add(self, data: bytes, filename: str) -> DocMeta:
        """Store and index an upload, or return the document these exact bytes
        already are, restored first if it was soft-deleted. Checked before
        anything touches disk: size, suffix, PDF magic, then text decoding.
        Raises FileTooLarge, UnsupportedFile, NoTextLayer or EncoderUnavailable."""
        suffix = _checked_suffix(data, filename)
        text = _decode_text(data) if suffix == ".txt" else None
        sha256 = hashlib.sha256(data).hexdigest()
        doc_id = sha256[:12]
        with self._lock:
            if doc_id in self._metas:
                return self._restore(doc_id)

        title = _display_title(filename, suffix)
        staging = self._tmp / uuid.uuid4().hex
        aside: Path | None = None
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
            _write_chunks(staging / "chunks.jsonl", chunks)
            DenseIndex(_index_docs(chunks, title), encoder=self._encoder,
                       cache_path=staging / "embeddings.npz", model_name=self.model_name).save()
            meta = DocMeta(doc_id=doc_id, title=title, kind=kind, suffix=suffix,
                           size_bytes=len(data), sha256=sha256, pages=len(pages),
                           chunks=len(chunks), chars=chars, created_at=_now())
            _write_json_atomic(staging / "meta.json", asdict(meta))

            with self._lock:
                if doc_id in self._metas:  # an identical upload finished while this one embedded
                    return self._restore(doc_id)
                target = self._docs / doc_id
                if target.exists():  # skipped at open (no readable meta.json): these bytes replace it
                    aside = self._tmp / uuid.uuid4().hex
                    os.replace(target, aside)
                os.replace(staging, target)
                self._metas[doc_id] = meta
                return meta
        finally:
            for leftover in (staging, aside):
                if leftover is not None and leftover.exists():
                    shutil.rmtree(leftover, ignore_errors=True)

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
            _write_json_atomic(self._docs / meta.doc_id / "meta.json", asdict(meta))
            self._metas[meta.doc_id] = meta
            self._indexes.pop(meta.doc_id, None)
            return meta

    def search(self, query: str, k: int = 5, doc_ids: list[str] | None = None) -> list[LibraryHit]:
        """The top `k` chunks for `query` across every live document, or exactly
        `doc_ids` (each a live document, or DocumentNotFound). Ranked by score,
        descending; a tie goes to the older document, then the lower chunk number."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        with self._lock:
            if doc_ids is None:
                scope = [m for m in self._metas.values() if not m.deleted]
            else:
                scope = [self._live(doc_id) for doc_id in dict.fromkeys(doc_ids)]

        ranked: list[tuple[tuple, LibraryHit]] = []
        for meta in scope:
            index = self._index(meta)
            for hit in index.dense.search(query, k):
                chunk = index.chunks[hit.id]
                order = (-hit.score, meta.created_at, chunk.number, meta.doc_id)
                ranked.append((order, LibraryHit(chunk=chunk, doc_title=meta.title, score=hit.score)))
        ranked.sort(key=lambda pair: pair[0])
        return [hit for _, hit in ranked[:k]]

    def _live(self, doc_id: object) -> DocMeta:
        """The live document `doc_id`; the caller holds the lock."""
        valid = isinstance(doc_id, str) and DOC_ID.fullmatch(doc_id)
        meta = self._metas.get(doc_id) if valid else None
        if meta is None or meta.deleted:
            raise DocumentNotFound("no such document")
        return meta

    def _restore(self, doc_id: str) -> DocMeta:
        """Stored document `doc_id`, undeleted if it was deleted; the caller holds the lock."""
        meta = self._metas[doc_id]
        if meta.deleted:
            meta = replace(meta, deleted=False, deleted_at=None)
            _write_json_atomic(self._docs / doc_id / "meta.json", asdict(meta))
            self._metas[doc_id] = meta
        return meta

    def _index(self, meta: DocMeta) -> _DocumentIndex:
        """`meta`'s dense index: cached, or loaded from its chunks and saved
        embeddings — nothing is embedded when the saved fingerprint matches."""
        with self._lock:
            cached = self._indexes.get(meta.doc_id)
        if cached is not None:
            return cached

        doc_dir = self._docs / meta.doc_id
        chunks = _read_chunks(doc_dir / "chunks.jsonl")
        index = DenseIndex(_index_docs(chunks, meta.title), encoder=self._encoder,
                           cache_path=doc_dir / "embeddings.npz", model_name=self.model_name)
        loaded = _DocumentIndex(dense=index, chunks={c.id: c for c in chunks})
        with self._lock:
            current = self._metas.get(meta.doc_id)
            if current is None or current.deleted:
                return loaded  # deleted while it loaded: serve this search, cache nothing
            return self._indexes.setdefault(meta.doc_id, loaded)


class _LazyEncoder:
    """The real model, loaded on the first `encode`, so opening a library never
    pays for torch. `dense.load_model` ends the process when sentence-transformers
    is missing, which would take a server down with it: the package is looked for first."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def encode(self, texts, **kwargs):
        with self._lock:
            if self._model is None:
                if find_spec("sentence_transformers") is None:
                    raise EncoderUnavailable(
                        "sentence-transformers is not installed, so documents cannot be embedded: "
                        "python tasks.py setup (or: pip install -r requirements.txt)")
                self._model = dense.load_model(self.model_name)
        return self._model.encode(texts, **kwargs)


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
    """Strict UTF-8 (a BOM is dropped), or failing that strict Windows-1256,
    the other encoding Arabic text files arrive in."""
    if b"\x00" in data:
        raise UnsupportedFile("a .txt file must not contain NUL bytes")
    for encoding in ("utf-8-sig", "cp1256"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # cp1256 maps all 256 byte values, so in practice the NUL check is what
    # refuses a binary file; this stays for a codec that ever does not.
    raise UnsupportedFile("the text is neither UTF-8 nor Windows-1256")


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


def _read_meta(doc_dir: Path) -> DocMeta | None:
    """A stored document's metadata, or None for anything else — a stray file, a
    foreign name, a missing or unreadable meta.json. Skipped, never fatal."""
    if not doc_dir.is_dir() or not DOC_ID.fullmatch(doc_dir.name):
        return None
    try:
        meta = DocMeta(**json.loads((doc_dir / "meta.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None
    return meta if meta.doc_id == doc_dir.name else None


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def _read_chunks(path: Path) -> list[Chunk]:
    # Split on "\n" only: JSON escapes it inside strings, but not U+2028, which splitlines() splits on.
    lines = path.read_text(encoding="utf-8").split("\n")
    return [Chunk(**json.loads(line)) for line in lines if line.strip()]


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write beside `path`, then os.replace over it: readers see old or new, never half."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
