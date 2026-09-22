"""The encoder's token window — T05.

``SentenceTransformer.encode`` truncates at ``max_seq_length`` without raising,
so an article longer than the window loses its tail to a vector that looks
exactly like a healthy one. Nothing downstream can tell the two apart: the hit
still carries the article, the citation still resolves, the score is still a
plausible number. On Law 151/2020 that hid 8.3% of the corpus, including more
than half of the definitions article.

These tests pin the split that stops it, and — just as important — pin that a
corpus of ordinary articles still embeds exactly one vector per article. A fix
that quietly re-cut every short article too would invalidate every measured run
in EVAL.md for a reason nobody would look for.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.dense import (  # noqa: E402
    PASSAGE_PREFIX,
    DenseIndex,
    encoder_limits,
    token_windows,
    window_budget,
)

np = pytest.importorskip("numpy")


class WordTokenizer:
    """One token per whitespace-separated word, with a fast tokenizer's offsets.

    Word-per-token keeps every boundary in these tests arithmetic instead of
    model-dependent: a budget of N means N words.
    """

    def __init__(self, offsets: bool = True):
        self.offsets = offsets

    def spans(self, text):
        spans, i = [], 0
        for word in text.split(" "):
            if word:
                spans.append((i, i + len(word)))
            i += len(word) + 1
        return spans

    def __call__(self, text, add_special_tokens=False, truncation=False,
                 return_offsets_mapping=False):
        if not self.offsets or not return_offsets_mapping:
            raise NotImplementedError("offset mapping unavailable")
        return {"offset_mapping": self.spans(text)}

    def encode(self, text, add_special_tokens=False, truncation=False):
        return list(range(len(self.spans(text)) + (2 if add_special_tokens else 0)))


class StubEncoder:
    """Records what it was asked to embed; returns deterministic vectors."""

    def __init__(self, vectors=None):
        self.seen: list[str] = []
        self.vectors = vectors or {}

    def encode(self, texts, **kwargs):
        texts = list(texts)
        self.seen.extend(texts)
        out = []
        for t in texts:
            for key, vec in self.vectors.items():
                if key in t:
                    out.append(vec)
                    break
            else:
                out.append([0.0, 0.0])
        return np.array(out, dtype="float32")


class WindowedEncoder(StubEncoder):
    """A stub that can say where it truncates, as a real model can."""

    def __init__(self, max_seq_length=10, vectors=None, offsets=True):
        super().__init__(vectors)
        self.tokenizer = WordTokenizer(offsets=offsets)
        self.max_seq_length = max_seq_length


def words(n, word="w"):
    return " ".join(f"{word}{i}" for i in range(n))


SHORT_DOCS = [
    {"id": "law-1", "law_name": "L", "number": 1, "text": words(3, "a")},
    {"id": "law-2", "law_name": "L", "number": 2, "text": words(3, "b")},
]

LONG_DOCS = [
    {"id": "law-1", "law_name": "L", "number": 1, "text": words(30, "a")},
    {"id": "law-2", "law_name": "L", "number": 2, "text": words(3, "b")},
]


# -- reading the ceiling off the model ---------------------------------------


def test_encoder_limits_is_none_when_the_encoder_cannot_say():
    """A stub, or LazyEncoder before it loads torch. None means "embed whole"."""
    assert encoder_limits(StubEncoder()) is None


def test_encoder_limits_reads_the_model_rather_than_a_constant():
    limits = encoder_limits(WindowedEncoder(max_seq_length=99))
    assert limits is not None and limits[1] == 99


def test_window_budget_leaves_room_for_the_prefix_and_special_tokens():
    tok = WordTokenizer()
    assert window_budget(tok, 100, PASSAGE_PREFIX) == 100 - 2 - 1


def test_window_budget_never_returns_a_useless_zero():
    assert window_budget(WordTokenizer(), 1, PASSAGE_PREFIX) >= 1


# -- cutting the windows ------------------------------------------------------


def test_text_within_the_budget_is_one_window_covering_everything():
    text = words(5)
    assert token_windows(WordTokenizer(), 10, text) == [(0, len(text))]


def test_a_tokenizer_without_offsets_falls_back_to_one_window():
    """Better a whole passage than windows cut at guessed positions."""
    text = words(50)
    assert token_windows(WordTokenizer(offsets=False), 10, text) == [(0, len(text))]


def test_long_text_is_split_into_windows_that_cover_the_whole_string():
    text = words(50)
    windows = token_windows(WordTokenizer(), 10, text, overlap=2)
    assert len(windows) > 1
    assert windows[0][0] == 0
    assert windows[-1][1] == len(text), "the end of the article must belong to some window"


def test_windows_overlap_so_nothing_is_lost_at_a_seam():
    text = words(50)
    windows = token_windows(WordTokenizer(), 10, text, overlap=3)
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:], strict=False):
        assert next_start < prev_end, "a gap between windows would drop the text between them"


def test_every_window_stays_within_the_budget():
    tok = WordTokenizer()
    text = words(50)
    for start, end in token_windows(tok, 10, text, overlap=2):
        assert len(tok.spans(text[start:end])) <= 10


def test_zero_overlap_still_makes_progress_instead_of_looping():
    text = words(30)
    windows = token_windows(WordTokenizer(), 10, text, overlap=0)
    assert windows[-1][1] == len(text)
    assert len(windows) == 3


def test_an_overlap_at_or_above_the_budget_still_terminates():
    """A misconfigured overlap must not hang the build."""
    text = words(30)
    windows = token_windows(WordTokenizer(), 5, text, overlap=99)
    assert windows[-1][1] == len(text)


# -- what the index builds from them -----------------------------------------


def test_a_corpus_that_fits_embeds_one_vector_per_article_as_before():
    enc = WindowedEncoder(max_seq_length=10)
    index = DenseIndex(SHORT_DOCS, encoder=enc)
    assert len(enc.seen) == len(SHORT_DOCS)
    assert index.rows == [0, 1]
    assert not index.windowed


def test_an_oversized_article_is_embedded_as_several_windows():
    enc = WindowedEncoder(max_seq_length=10)
    index = DenseIndex(LONG_DOCS, encoder=enc)
    assert index.windowed
    assert index.rows.count(0) > 1, "the long article should own more than one window"
    assert index.rows.count(1) == 1


def test_every_window_still_carries_the_passage_prefix():
    enc = WindowedEncoder(max_seq_length=10)
    DenseIndex(LONG_DOCS, encoder=enc)
    assert all(text.startswith(PASSAGE_PREFIX) for text in enc.seen)


def test_the_tail_of_a_long_article_reaches_the_encoder():
    """The whole point of T05: the last words must be inside some window."""
    enc = WindowedEncoder(max_seq_length=10)
    DenseIndex(LONG_DOCS, encoder=enc)
    assert any("a29" in text for text in enc.seen)


def test_an_encoder_that_cannot_say_keeps_embedding_whole_articles():
    enc = StubEncoder()
    index = DenseIndex(LONG_DOCS, encoder=enc)
    assert len(enc.seen) == len(LONG_DOCS)
    assert index.rows == [0, 1]
    assert not index.windowed


# -- what the index does with them -------------------------------------------


def test_an_article_is_ranked_by_its_best_window_not_its_average():
    """A long article whose one relevant window matches must not be diluted by
    its others — otherwise a short article wins for being short."""
    enc = WindowedEncoder(max_seq_length=10, vectors={"a29": [1.0, 0.0], "b0": [0.6, 0.0]})
    index = DenseIndex(LONG_DOCS, encoder=enc)
    hits = index.search_with_vector(np.array([1.0, 0.0], dtype="float32"), k=2)
    assert hits[0].id == "law-1"


def test_windows_do_not_produce_duplicate_hits_for_one_article():
    enc = WindowedEncoder(max_seq_length=10)
    index = DenseIndex(LONG_DOCS, encoder=enc)
    hits = index.search_with_vector(np.array([1.0, 0.0], dtype="float32"), k=5)
    assert [h.id for h in hits] == list(dict.fromkeys(h.id for h in hits))
    assert len(hits) <= len(LONG_DOCS)


def test_a_hit_still_carries_the_whole_article_not_the_window():
    """Everything above this file cites articles; windows are an encoder detail."""
    enc = WindowedEncoder(max_seq_length=10)
    index = DenseIndex(LONG_DOCS, encoder=enc)
    hits = index.search_with_vector(np.array([1.0, 0.0], dtype="float32"), k=1)
    assert hits[0].id == "law-1"
    assert hits[0].snippet == LONG_DOCS[0]["text"][:280]


def test_k_still_counts_articles_not_windows():
    enc = WindowedEncoder(max_seq_length=10)
    index = DenseIndex(LONG_DOCS, encoder=enc)
    assert len(index.search_with_vector(np.array([1.0, 0.0], dtype="float32"), k=1)) == 1


# -- the cache has to carry the row map --------------------------------------


def _rewrite(cache, embeddings=None, rows=None, drop_rows=False):
    """Rewrite a saved cache, keeping its metadata exactly as a real save wrote it."""
    with np.load(cache, allow_pickle=False) as z:
        meta = str(z["meta"])
        kept_embeddings = z["embeddings"]
        kept_rows = z["rows"]
    payload = {
        "embeddings": kept_embeddings if embeddings is None else embeddings,
        "meta": np.array(meta),
    }
    if not drop_rows:
        # dtype is passed through, not coerced: one test writes a float map on purpose.
        payload["rows"] = kept_rows if rows is None else np.asarray(rows)
    np.savez(cache, **payload)


def test_a_windowed_cache_round_trips(tmp_path):
    cache = tmp_path / "emb.npz"
    built = DenseIndex(LONG_DOCS, encoder=WindowedEncoder(), cache_path=cache)
    built.save()

    enc = WindowedEncoder()
    loaded = DenseIndex(LONG_DOCS, encoder=enc, cache_path=cache)
    assert loaded.from_cache
    assert enc.seen == [], "a valid cache must not re-embed anything"
    assert loaded.rows == built.rows


def test_a_cached_index_still_knows_it_is_windowed(tmp_path):
    """`windowed` is what the probe prints to say whether the blind spot is
    live. Left at its default on the cache path, it would report a fixed index
    as broken — on every run after the first, which is every real run."""
    cache = tmp_path / "emb.npz"
    DenseIndex(LONG_DOCS, encoder=WindowedEncoder(), cache_path=cache).save()

    loaded = DenseIndex(LONG_DOCS, encoder=WindowedEncoder(), cache_path=cache)
    assert loaded.from_cache
    assert loaded.windowed


def test_a_cached_index_of_short_articles_is_not_reported_as_windowed(tmp_path):
    cache = tmp_path / "emb.npz"
    DenseIndex(SHORT_DOCS, encoder=WindowedEncoder(), cache_path=cache).save()

    loaded = DenseIndex(SHORT_DOCS, encoder=WindowedEncoder(), cache_path=cache)
    assert loaded.from_cache
    assert not loaded.windowed


def test_a_cache_written_before_windows_existed_is_rebuilt(tmp_path):
    """Its vectors are the truncated ones; reusing them would undo the fix."""
    cache = tmp_path / "emb.npz"
    DenseIndex(LONG_DOCS, encoder=WindowedEncoder(), cache_path=cache).save()
    _rewrite(cache, drop_rows=True)

    enc = WindowedEncoder()
    index = DenseIndex(LONG_DOCS, encoder=enc, cache_path=cache)
    assert enc.seen, "a pre-window cache was reused instead of rebuilt"
    assert not index.from_cache


def test_a_cache_built_with_a_different_overlap_is_rebuilt(tmp_path):
    cache = tmp_path / "emb.npz"
    DenseIndex(LONG_DOCS, encoder=WindowedEncoder(), cache_path=cache, window_overlap=2).save()

    enc = WindowedEncoder()
    index = DenseIndex(LONG_DOCS, encoder=enc, cache_path=cache, window_overlap=5)
    assert enc.seen, "windows cut differently are not interchangeable"
    assert not index.from_cache


def test_a_row_map_shorter_than_the_embeddings_is_rejected(tmp_path):
    cache = tmp_path / "emb.npz"
    DenseIndex(LONG_DOCS, encoder=WindowedEncoder(), cache_path=cache).save()
    _rewrite(cache, rows=np.array([0], dtype="int32"))

    enc = WindowedEncoder()
    index = DenseIndex(LONG_DOCS, encoder=enc, cache_path=cache)
    assert enc.seen and not index.from_cache


def test_a_row_map_pointing_outside_the_corpus_is_rejected(tmp_path):
    cache = tmp_path / "emb.npz"
    DenseIndex(SHORT_DOCS, encoder=WindowedEncoder(), cache_path=cache).save()
    _rewrite(cache, rows=np.array([0, 7], dtype="int32"))

    enc = WindowedEncoder()
    index = DenseIndex(SHORT_DOCS, encoder=enc, cache_path=cache)
    assert enc.seen and not index.from_cache


def test_a_row_map_that_leaves_an_article_unreachable_is_rejected(tmp_path):
    """Two vectors for one article and none for the other: loadable, and wrong."""
    cache = tmp_path / "emb.npz"
    DenseIndex(SHORT_DOCS, encoder=WindowedEncoder(), cache_path=cache).save()
    _rewrite(cache, rows=np.array([0, 0], dtype="int32"))

    enc = WindowedEncoder()
    index = DenseIndex(SHORT_DOCS, encoder=enc, cache_path=cache)
    assert enc.seen and not index.from_cache


def test_a_non_integer_row_map_is_rejected(tmp_path):
    cache = tmp_path / "emb.npz"
    DenseIndex(SHORT_DOCS, encoder=WindowedEncoder(), cache_path=cache).save()
    _rewrite(cache, rows=np.array([0.0, 1.0], dtype="float32"))

    enc = WindowedEncoder()
    index = DenseIndex(SHORT_DOCS, encoder=enc, cache_path=cache)
    assert enc.seen and not index.from_cache
