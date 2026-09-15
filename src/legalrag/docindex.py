"""A stored document's dense index, and the one encoder every index shares (ADR-023).

`library` decides which documents are live and what a failure means to its
caller; this module builds the index a search reads, from a document's
chunks.jsonl and embeddings.npz, and loads the embedding model the first time
anything is embedded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from . import dense
from .chunking import Chunk
from .dense import DenseIndex, Encoder
from .docstore import CHUNKS, EMBEDDINGS, read_chunks


class EncoderUnavailable(RuntimeError):
    """The embedding model cannot be loaded: a server configuration problem, not a client error."""


@dataclass(frozen=True)
class DocumentIndex:
    dense: DenseIndex
    chunks: dict[str, Chunk]  # by chunk id, which is what a dense Hit carries


def index_docs(chunks: list[Chunk], title: str) -> list[dict]:
    """What a DenseIndex is built over: one entry per chunk."""
    return [{"id": c.id, "number": c.number, "text": c.text, "law_name": title} for c in chunks]


def load_index(doc_dir: Path, title: str, encoder: Encoder, model_name: str) -> DocumentIndex:
    """A stored document's index, from its chunks.jsonl and its saved embeddings."""
    chunks = read_chunks(doc_dir / CHUNKS)
    index = DenseIndex(index_docs(chunks, title), encoder=encoder,
                       cache_path=doc_dir / EMBEDDINGS, model_name=model_name)
    return DocumentIndex(dense=index, chunks={c.id: c for c in chunks})


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
