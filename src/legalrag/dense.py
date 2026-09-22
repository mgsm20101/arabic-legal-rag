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
* it splits an article that does not fit the encoder's token window instead of
  handing it over whole. ``SentenceTransformer.encode`` truncates at
  ``max_seq_length`` in silence: no exception, no warning, just a vector that
  never saw the end of the article — indistinguishable downstream from a vector
  that did. On Law 151/2020 that hid 8.3% of the corpus, including more than
  half of the definitions article. Windows carry the article's own id, so
  everything above this file still ranks and cites whole articles.

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

# Tokens a window shares with the one before it. A sentence that straddles a
# boundary is whole in the next window, so no fact is lost at a seam — which is
# why the split does not also need to hunt for sentence ends.
WINDOW_OVERLAP_TOKENS = 64


def verbatim(text: str) -> str:
    """The spelling both sides of the dense path use: the corpus's own (ADR-015).

    Folding letter forms here was tried and measured, and it lost: it bought
    perfect invariance to a typist's spelling and paid 0.100 recall@5 on the
    canonical questions (EVAL.md Run 9, ADR-027). This stays the default until
    a run beats 0.767/0.588, not because folding is a bad idea.

    It remains a parameter for one reason: the cache records which spelling its
    vectors are in. Without that name, vectors built under a fold would be
    silently reused for verbatim queries, and every comparison would be a little
    wrong with nothing to show for it.
    """
    return text


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


def encoder_limits(encoder) -> tuple[object, int] | None:
    """The tokenizer and token ceiling ``encode`` will actually apply, or None.

    Read off the model rather than hardcoded, because a hardcoded 512 keeps
    claiming a ceiling that a different encoder does not have. None means this
    encoder cannot say — a stub in the tests, or ``docindex.LazyEncoder``, which
    refuses to load torch until something actually embeds. Callers treat None as
    "embed whole passages", which is what this file did before windows existed.
    """
    tokenizer = getattr(encoder, "tokenizer", None)
    max_seq = getattr(encoder, "max_seq_length", None)
    if tokenizer is None or not isinstance(max_seq, int) or max_seq <= 0:
        return None
    return tokenizer, max_seq


def _token_spans(tokenizer, text: str) -> list[tuple[int, int]] | None:
    """Character span of every token in `text`, or None if this tokenizer cannot say.

    Special tokens map to empty spans and are dropped: what is wanted here is
    where the real text sits, not how the model frames it.
    """
    try:
        encoded = tokenizer(
            text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]
    except (TypeError, ValueError, KeyError, NotImplementedError):
        return None
    return [(int(s), int(e)) for s, e in offsets if e > s]


def window_budget(tokenizer, max_seq: int, prefix: str = PASSAGE_PREFIX) -> int:
    """How many tokens of article text fit beside the prefix and the special tokens."""
    try:
        specials = len(tokenizer.encode("", add_special_tokens=True))
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    except (TypeError, ValueError, NotImplementedError):
        specials, prefix_tokens = 2, 4
    return max(1, max_seq - specials - prefix_tokens)


def token_windows(
    tokenizer,
    budget: int,
    text: str,
    overlap: int = WINDOW_OVERLAP_TOKENS,
) -> list[tuple[int, int]]:
    """Character spans covering `text`, each within `budget` tokens.

    One span covering everything when the text already fits — the common case,
    and the one that must stay byte-identical to what was embedded before.
    """
    spans = _token_spans(tokenizer, text)
    if spans is None or len(spans) <= budget:
        return [(0, len(text))]

    stride = max(1, budget - max(0, overlap))
    windows: list[tuple[int, int]] = []
    start_token = 0
    while True:
        end_token = min(start_token + budget, len(spans))
        start_char = spans[start_token][0]
        # The final window runs to the end of the string so trailing text outside
        # any token span (whitespace, a stray mark) still belongs to something.
        last = end_token >= len(spans)
        end_char = len(text) if last else spans[end_token - 1][1]
        windows.append((start_char, end_char))
        if last:
            return windows
        start_token += stride


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
    """Cosine similarity over article embeddings — several per article when one
    does not fit the encoder's token window (ADR-026). An article still scores
    and ranks as a single result either way; `rows` is what maps a vector back
    to the article it came from."""

    def __init__(
        self,
        docs: list[dict],
        encoder: Encoder | None = None,
        cache_path: Path | None = None,
        query_prefix: str = QUERY_PREFIX,
        passage_prefix: str = PASSAGE_PREFIX,
        model_name: str = DEFAULT_MODEL,
        window_overlap: int = WINDOW_OVERLAP_TOKENS,
        fold=verbatim,
    ):
        import numpy as np

        self.docs = docs
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model_name = model_name
        self.cache_path = cache_path
        self.window_overlap = window_overlap
        # Applied to the query and to every passage, or it does nothing at all
        # (ADR-027). `verbatim` by default — the measured fold lost; see there.
        self.fold = fold
        self.fingerprint = corpus_fingerprint(docs)
        self._encoder = encoder
        self.from_cache = False
        # One entry per embedding row, holding the index of the article it came
        # from. Identity while every article fits in one window.
        self.rows: list[int] = []
        self.windowed = False

        cached = self._load_cache() if cache_path else None
        if cached is not None:
            self.embeddings, self.rows = cached
            # Read back off the row map, not left at its default: a cached index
            # is windowed or not for the same reason a freshly built one is, and
            # a status line that says otherwise is worse than no status line.
            self.windowed = len(self.rows) > len(self.docs)
            self.from_cache = True
            return

        if not docs:
            self.embeddings = np.zeros((0, 0), dtype="float32")
            self.rows = []
            return

        passages, self.rows = self._passages()
        self.embeddings = _l2_normalize(np.asarray(self.encoder.encode(passages)))

    def _passages(self) -> tuple[list[str], list[int]]:
        """The strings to embed, and the article each one belongs to.

        One string per article while the article fits the encoder's window, so
        a corpus of ordinary articles embeds exactly as it always has. An
        article that does not fit becomes several overlapping windows, all
        carrying its index — the alternative is `encode` dropping its tail
        without telling anyone.
        """
        limits = encoder_limits(self.encoder)
        if limits is None:
            self.windowed = False
            return (
                [self.passage_prefix + self.fold(d["text"]) for d in self.docs],
                list(range(len(self.docs))),
            )

        tokenizer, max_seq = limits
        budget = window_budget(tokenizer, max_seq, self.passage_prefix)
        passages: list[str] = []
        rows: list[int] = []
        for i, doc in enumerate(self.docs):
            # Folded before splitting, not after: the fold can change the string's
            # length, and a window cut on the raw text would then land somewhere
            # else in the folded one.
            text = self.fold(doc["text"])
            for start, end in token_windows(tokenizer, budget, text, self.window_overlap):
                passages.append(self.passage_prefix + text[start:end])
                rows.append(i)
        self.windowed = len(passages) > len(self.docs)
        return passages, rows

    # -- model ------------------------------------------------------------

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = load_model(self.model_name)
        return self._encoder

    # -- cache ------------------------------------------------------------

    def _fold_name(self) -> str:
        """What the cache records about the spelling its vectors were built in.

        A cache written under a different fold is rejected and rebuilt — which
        is the point: its vectors are in one spelling while queries would arrive
        in another, and every comparison between them would be quietly slightly
        wrong. Both rejected experiments (ADR-027, ADR-028) tripped this guard on
        the way back out, which is the only reason reverting them was a one-line
        change rather than a hunt for a stale index.
        """
        return getattr(self.fold, "__name__", repr(self.fold))

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
                if meta.get("window_overlap") != self.window_overlap:
                    return None  # different windows — these rows are not the ones this index wants
                if meta.get("fold") != self._fold_name():
                    return None  # embedded from a different spelling than this index will query in
                embeddings = z["embeddings"]
                # A cache written before windows existed has no `rows` at all, and
                # the KeyError below is caught as "rebuild": exactly right, since its
                # vectors are the truncated ones this index exists to stop using.
                rows = z["rows"]
                # The metadata can match while the array itself is not a usable (N, D) matrix —
                # np.load does not raise just because what it hands back is empty, misshapen, the
                # wrong dtype, or full of NaN. Each check below is a rebuild, not a crash, same as
                # every check above: a bad cache is "don't trust this," never an exception.
                if embeddings.ndim != 2:
                    return None  # not a (N, D) matrix at all
                if not np.issubdtype(embeddings.dtype, np.floating):
                    return None  # not embedding vectors
                if not np.isfinite(embeddings).all():
                    return None  # NaN/Inf smuggled into an otherwise loadable cache
                if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
                    return None  # not a row -> article map
                if rows.shape[0] != embeddings.shape[0]:
                    return None  # a vector with no article, or an article with no vector
                if len(self.docs) == 0:
                    return embeddings, []  # same shape as __init__'s own empty-corpus branch
                if rows.size == 0 or rows.min() < 0 or rows.max() >= len(self.docs):
                    return None  # points outside the corpus on disk
                if len(set(rows.tolist())) != len(self.docs):
                    return None  # an article with no window at all would be unreachable
                return embeddings, rows.tolist()
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
                "window_overlap": self.window_overlap,
                "windows": len(self.rows),
                "fold": self._fold_name(),
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
            np.savez(
                tmp,
                embeddings=self.embeddings,
                meta=np.array(meta),
                rows=np.asarray(self.rows, dtype="int32"),
            )
            os.replace(tmp, self.cache_path)
        finally:
            tmp.unlink(missing_ok=True)

    # -- search -----------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[Hit]:
        # The query goes through the same fold the passages did, whatever it is:
        # a fold on one side only moves the query away from its own corpus.
        # `search_normalize` is never that fold — it belongs to BM25 and strips
        # exactly the morphology the model reads (ADR-015).
        # Checked before encoding, not after: an empty index must never touch
        # `self.encoder`, which lazily loads the real model on first access.
        if not self.docs or self.embeddings.size == 0:
            return []
        vec = encode_query(self.encoder, query, self.query_prefix, fold=self.fold)
        return self.search_with_vector(vec, k)

    def search_with_vector(self, vec, k: int = 5) -> list[Hit]:
        """The ranking half of `search`, taking an already-encoded, already-normalized query
        vector — see `encode_query`. Pulled apart so a caller ranking the same query across many
        indexes (`Library.search`) can encode it once and reuse the vector here."""
        import numpy as np

        if not self.docs or self.embeddings.size == 0:
            return []

        scores = self.embeddings @ vec

        # An article scores as its best window. Anything other than the maximum
        # would let a long article's unrelated paragraphs dilute the one that
        # answers the question — a short article would then outrank it for being
        # short, which is the opposite of what the split is for.
        #
        # Worth knowing before adding another channel to this max: ADR-028 tried
        # a second vector per article (a folded spelling) and the max re-ranked
        # the corpus, because each article gained a different amount from its
        # extra channel and ranking reads only the differences.
        best = np.full(len(self.docs), -np.inf, dtype="float64")
        np.maximum.at(best, np.asarray(self.rows, dtype="int64"), scores)

        order = np.argsort(-best)[:k]
        hits: list[Hit] = []
        for i in order:
            index = int(i)
            if not np.isfinite(best[index]):
                continue  # an article with no window is not a result
            d = self.docs[index]
            hits.append(
                Hit(
                    id=d["id"],
                    law_name=d.get("law_name", ""),
                    number=d["number"],
                    score=round(float(best[index]), 4),
                    snippet=d["text"][:280],
                )
            )
        return hits


def encode_query(
    encoder: Encoder,
    query: str,
    query_prefix: str = QUERY_PREFIX,
    fold=verbatim,
):
    """The L2-normalized query vector `DenseIndex.search` uses internally — pulled out so a
    caller ranking the same query across many indexes (`Library.search`) can compute it once
    and pass it to each index's `search_with_vector` instead of re-encoding the same string.

    `fold` must be the one the passages were embedded with, or the two sides are
    answering in different spellings (ADR-027)."""
    import numpy as np

    return _l2_normalize(np.asarray(encoder.encode([query_prefix + fold(query)])))[0]


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
