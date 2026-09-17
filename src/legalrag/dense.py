"""Dense retrieval baseline (ADR-015).

The number this produces only means something next to the BM25 number, which is
why BM25 came first. What this file mostly does is refuse to make the three
wiring mistakes that would make a dense run *look* finished while measuring
something else:

* it embeds ``text``, never ``search_text``. The search index is deliberately
  lossy (ADR-004) — clitics stripped, hamza folded, diacritics gone — because
  that is what makes lexical matching work across Arabic inflection. A
  transformer trained on natural Arabic reads that field as damage.
* it applies the E5 ``query:`` / ``passage:`` prefixes. E5 is asymmetric; drop
  them and quality falls with no error anywhere.
* it fingerprints the corpus into the embedding cache, so a cache built before
  a re-ingest is rebuilt instead of silently answering for the old corpus.

The model is downloaded once and then used offline. Nothing in the test suite
loads it: ``tests/test_dense.py`` injects a stub encoder, so a fresh clone with
no model still runs green in under a second (M1/A1).
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Protocol

from .retrieve import Hit

CORPUS_PATH = Path("data/processed/articles.jsonl")
CACHE_PATH = Path("data/processed/embeddings.npz")

# intfloat/multilingual-e5-base — retrieval-trained, multilingual, 768-dim.
# The rationale and the alternatives considered are in DECISIONS.md ADR-015.
DEFAULT_MODEL = "intfloat/multilingual-e5-base"

# E5 was trained with these exact strings. They are not decoration.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Encoder(Protocol):
    """Anything with SentenceTransformer's ``encode``. Kept narrow so the tests
    can inject a stub and never touch the network."""

    def encode(self, texts: list[str], **kwargs): ...


def corpus_fingerprint(docs: list[dict]) -> str:
    """Content hash of the corpus the embeddings belong to.

    Keyed on id + verbatim text, in order. A re-ingest that changes one article
    changes this, which is what makes a stale cache detectable rather than
    invisible.
    """
    h = hashlib.sha256()
    for d in docs:
        h.update(d["id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(d["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_model(name: str = DEFAULT_MODEL):
    """Load the sentence-transformers model, with an actionable error.

    Import is local: ``sentence_transformers`` pulls in torch, which costs
    seconds. Nothing else in the project should pay that just to import a
    module.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "sentence-transformers is not installed.\n"
            "  python tasks.py setup      (or: pip install -r requirements.txt)"
        ) from e
    return SentenceTransformer(name)


class DenseIndex:
    """Cosine similarity over article-level embeddings."""

    def __init__(
        self,
        docs: list[dict],
        encoder: Encoder | None = None,
        cache_path: Path | None = None,
        query_prefix: str = QUERY_PREFIX,
        passage_prefix: str = PASSAGE_PREFIX,
        model_name: str = DEFAULT_MODEL,
    ):
        import numpy as np

        self.docs = docs
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model_name = model_name
        self.cache_path = cache_path
        self.fingerprint = corpus_fingerprint(docs)
        self._encoder = encoder
        self.from_cache = False

        cached = self._load_cache() if cache_path else None
        if cached is not None:
            self.embeddings = cached
            self.from_cache = True
            return

        if not docs:
            self.embeddings = np.zeros((0, 0), dtype="float32")
            return

        passages = [self.passage_prefix + d["text"] for d in docs]
        self.embeddings = _l2_normalize(np.asarray(self.encoder.encode(passages)))

    # -- model ------------------------------------------------------------

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = load_model(self.model_name)
        return self._encoder

    # -- cache ------------------------------------------------------------

    def _load_cache(self):
        import numpy as np

        if not self.cache_path or not self.cache_path.exists():
            return None
        try:
            with np.load(self.cache_path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if meta.get("fingerprint") != self.fingerprint:
                    return None  # corpus changed — rebuild, do not answer for the old one
                if meta.get("model") != self.model_name:
                    return None  # different model — vectors are not comparable
                if meta.get("passage_prefix") != self.passage_prefix:
                    return None
                embeddings = z["embeddings"]
                # The metadata can match while the array itself is not a usable (N, D) matrix —
                # np.load does not raise just because what it hands back is empty, misshapen, the
                # wrong dtype, or full of NaN. Each check below is a rebuild, not a crash, same as
                # every check above: a bad cache is "don't trust this," never an exception.
                if embeddings.ndim != 2:
                    return None  # not a (N, D) matrix at all
                if not np.issubdtype(embeddings.dtype, np.floating):
                    return None  # not embedding vectors
                if embeddings.shape[0] != len(self.docs):
                    return None  # row count does not match the corpus on disk (0 == 0 for an
                    # empty corpus is fine — that is the same shape __init__'s own no-cache
                    # empty-corpus branch produces)
                if not np.isfinite(embeddings).all():
                    return None  # NaN/Inf smuggled into an otherwise loadable cache
                return embeddings
        except (KeyError, ValueError, OSError):
            return None  # unreadable cache is a rebuild, not a crash

    def save(self) -> None:
        import numpy as np

        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(
            {
                "fingerprint": self.fingerprint,
                "model": self.model_name,
                "passage_prefix": self.passage_prefix,
                "articles": len(self.docs),
            },
            ensure_ascii=False,
        )
        # Written to a temp file, then os.replace into place: a reader sees the old cache or the
        # new one, never one truncated by a process that died mid-write (docstore.write_json_atomic
        # and ingest._write_articles_atomic use the same pattern). The temp name keeps the .npz
        # suffix on purpose — np.savez appends one to any path that lacks it, which would otherwise
        # write the data somewhere other than the path os.replace is about to move.
        tmp = self.cache_path.with_name(f".{self.cache_path.stem}.{uuid.uuid4().hex}.tmp.npz")
        try:
            np.savez(tmp, embeddings=self.embeddings, meta=np.array(meta))
            os.replace(tmp, self.cache_path)
        finally:
            tmp.unlink(missing_ok=True)

    # -- search -----------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[Hit]:
        # The query goes to the model verbatim. `search_normalize` belongs to
        # BM25 and would strip exactly the morphology the model reads (ADR-015).
        # Checked before encoding, not after: an empty index must never touch
        # `self.encoder`, which lazily loads the real model on first access.
        if not self.docs or self.embeddings.size == 0:
            return []
        return self.search_with_vector(encode_query(self.encoder, query, self.query_prefix), k)

    def search_with_vector(self, vec, k: int = 5) -> list[Hit]:
        """The ranking half of `search`, taking an already-encoded, already-normalized query
        vector — see `encode_query`. Pulled apart so a caller ranking the same query across many
        indexes (`Library.search`) can encode it once and reuse the vector here."""
        import numpy as np

        if not self.docs or self.embeddings.size == 0:
            return []

        scores = self.embeddings @ vec

        order = np.argsort(-scores)[:k]
        hits: list[Hit] = []
        for i in order:
            d = self.docs[int(i)]
            hits.append(
                Hit(
                    id=d["id"],
                    law_name=d.get("law_name", ""),
                    number=d["number"],
                    score=round(float(scores[int(i)]), 4),
                    snippet=d["text"][:280],
                )
            )
        return hits


def encode_query(encoder: Encoder, query: str, query_prefix: str = QUERY_PREFIX):
    """The L2-normalized query vector `DenseIndex.search` uses internally — pulled out so a
    caller ranking the same query across many indexes (`Library.search`) can compute it once
    and pass it to each index's `search_with_vector` instead of re-encoding the same string."""
    import numpy as np

    return _l2_normalize(np.asarray(encoder.encode([query_prefix + query])))[0]


def _l2_normalize(matrix):
    """Unit-length rows, so a dot product is cosine similarity."""
    import numpy as np

    arr = np.asarray(matrix, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector stays zero instead of becoming NaN
    return arr / norms


def load_docs(path: Path = CORPUS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_dense_index(
    path: Path = CORPUS_PATH,
    cache_path: Path | None = CACHE_PATH,
    model_name: str = DEFAULT_MODEL,
) -> DenseIndex | None:
    docs = load_docs(path)
    return DenseIndex(docs, cache_path=cache_path, model_name=model_name) if docs else None
