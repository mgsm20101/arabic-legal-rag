"""A stored document's dense index, and the one encoder every index shares (ADR-023).

`library` decides which documents are live and what a failure means to its
caller; this module builds the index a search reads, from a document's
chunks.jsonl and embeddings.npz, and loads the embedding model the first time
anything is embedded.
"""

from __future__ import annotations

import threading
import zipfile
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from . import dense
from .chunking import Chunk
from .dense import DenseIndex, Encoder
from .docstore import CHUNKS, EMBEDDINGS, read_chunks


class EncoderUnavailable(RuntimeError):
    """The embedding model cannot be loaded: a server configuration problem, not a client error."""


class IndexDamaged(Exception):
    """A stored document's chunks.jsonl or embeddings.npz is not what the library wrote."""


@dataclass(frozen=True)
class DocumentIndex:
    dense: DenseIndex
    chunks: dict[str, Chunk]  # by chunk id, which is what a dense Hit carries


def index_docs(chunks: list[Chunk], title: str) -> list[dict]:
    """What a DenseIndex is built over: one entry per chunk."""
    return [{"id": c.id, "number": c.number, "text": c.text, "law_name": title} for c in chunks]


def load_index(doc_dir: Path, title: str, chunk_count: int, encoder: Encoder, model_name: str) -> DocumentIndex:
    """A stored document's index, from its chunks.jsonl and saved embeddings, never by embedding a
    passage. IndexDamaged when either file is not what the library wrote, a chunks file holding
    other than `chunk_count` chunks included; any other error is a bug, and raised as itself."""
    try:
        chunks = read_chunks(doc_dir / CHUNKS)
    except (OSError, ValueError, TypeError) as e:  # TypeError: a line of JSON that is not a chunk
        raise IndexDamaged(f"chunks.jsonl will not load: {e}") from e
    if len(chunks) != chunk_count:
        raise IndexDamaged(f"chunks.jsonl holds {len(chunks)} chunks, not {chunk_count}")
    try:
        index = DenseIndex(index_docs(chunks, title), encoder=_QueriesOnly(encoder),
                           cache_path=doc_dir / EMBEDDINGS, model_name=model_name)
    except (EOFError, zipfile.BadZipFile) as e:  # what np.load raises past DenseIndex's own cache check
        raise IndexDamaged(f"embeddings.npz will not load: {e}") from e
    return DocumentIndex(dense=index, chunks={c.id: c for c in chunks})


class _QueriesOnly:
    """The shared encoder as a stored index may use it: for queries only. A stored document's
    passages were embedded when it was added; an index asking to embed them again found its
    saved embeddings unusable, and re-embedding on every search, never saved, would pass that
    damage off as a slow search."""

    def __init__(self, encoder: Encoder):
        self._encoder = encoder

    def encode(self, texts, **kwargs):
        texts = list(texts)
        if any(text.startswith(dense.PASSAGE_PREFIX) for text in texts):
            raise IndexDamaged("embeddings.npz did not load, so its passages would be embedded again")
        return self._encoder.encode(texts, **kwargs)


class LazyEncoder:
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
    """`dense.load_model`, with every way it fails as EncoderUnavailable: it ends the process when
    an import fails (torch missing under sentence-transformers, say), which would take a server
    down, and raises OSError for weights it cannot read, e.g. offline with no cached copy."""
    if find_spec("sentence_transformers") is None:
        raise EncoderUnavailable("sentence-transformers is not installed, so documents cannot be embedded: "
                                 "python tasks.py setup (or: pip install -r requirements.txt)")
    try:
        return dense.load_model(model_name)
    except (SystemExit, ImportError, OSError) as e:
        raise EncoderUnavailable(f"the embedding model {model_name!r} could not be loaded: {e}") from e
