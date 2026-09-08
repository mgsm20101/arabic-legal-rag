"""Dense retrieval tests (ADR-015).

None of these download a model. They inject a stub encoder, because the three
things most likely to go wrong here are wiring mistakes, not model quality —
and every one of them fails *silently*, producing a plausible-looking dense
number that is simply wrong:

1. Embedding ``search_text`` instead of ``text``. ADR-004's aggressive
   normalizer strips clitics, hamza forms and diacritics on purpose so BM25 can
   match inflections. A transformer is trained on natural Arabic; feeding it
   the stripped field degrades the vectors and the run ends with the wrong
   conclusion — "dense does not help on Arabic" — attributed to the model.
2. Dropping the E5 ``query:`` / ``passage:`` prefixes. E5 is asymmetric and was
   trained with them; without them retrieval quality drops with no error.
3. Reading a stale embedding cache after the corpus was re-ingested. That
   reports a number about the *previous* corpus, which is the exact failure
   ADR-010 exists to prevent on the question side.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.dense import DenseIndex, corpus_fingerprint  # noqa: E402
from legalrag.fusion import reciprocal_rank_fusion  # noqa: E402
from legalrag.retrieve import Hit, recall_at_k, reciprocal_rank  # noqa: E402

np = pytest.importorskip("numpy")


DOCS = [
    {
        "id": "law-7",
        "law_name": "قانون حماية البيانات الشخصية",
        "number": 7,
        "text": "يلتزم المتحكم بإبلاغ المركز خلال اثنتين وسبعين ساعة بخرق البيانات.",
        "search_text": "لتزم متحكم ابلاغ مركز خلال اثنتين وسبعين ساعه خرق بيانات",
    },
    {
        "id": "law-8",
        "law_name": "قانون حماية البيانات الشخصية",
        "number": 8,
        "text": "على المتحكم تعيين مسؤول لحماية البيانات الشخصية داخل المنشأة.",
        "search_text": "متحكم تعيين مسءول حمايه بيانات شخصيه داخل منشاه",
    },
]


class StubEncoder:
    """Records what it was asked to embed; returns deterministic vectors.

    Vector = one dimension per doc index, so ``search`` has a predictable
    winner without any real semantics involved.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.seen: list[str] = []
        self.vectors = vectors or {}

    def encode(self, texts, **kwargs):
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


# ---------------------------------------------------------------------------
# 1. the field that gets embedded
# ---------------------------------------------------------------------------


def test_dense_embeds_verbatim_text_not_the_search_index():
    """ADR-015. The search index is lossy by design (ADR-004); the model is not.

    `search_text` here has hamza folded ("مسءول") and clitics stripped
    ("لتزم"). Handing that to a transformer is not normalization, it is damage.
    """
    enc = StubEncoder()
    DenseIndex(DOCS, encoder=enc)

    embedded = " ".join(enc.seen)
    assert "يلتزم المتحكم بإبلاغ المركز" in embedded, "verbatim text was not embedded"
    assert "مسءول" not in embedded, "the aggressive search_text field was embedded"
    assert "لتزم متحكم" not in embedded


def test_query_is_normalized_for_bm25_but_not_for_dense():
    """The same query text takes two different paths on purpose."""
    enc = StubEncoder()
    idx = DenseIndex(DOCS, encoder=enc)
    enc.seen.clear()

    idx.search("هل الشركة مطالبة بتعيين موظف مسؤول عن الخصوصية؟", k=2)

    asked = " ".join(enc.seen)
    assert "مسؤول" in asked, "the query reached the model stripped of its hamza"


# ---------------------------------------------------------------------------
# 2. E5 asymmetry
# ---------------------------------------------------------------------------


def test_passages_carry_the_passage_prefix():
    enc = StubEncoder()
    DenseIndex(DOCS, encoder=enc)
    assert all(t.startswith("passage: ") for t in enc.seen), enc.seen[:2]


def test_query_carries_the_query_prefix():
    enc = StubEncoder()
    idx = DenseIndex(DOCS, encoder=enc)
    enc.seen.clear()
    idx.search("متى يجب الإبلاغ عن الخرق؟", k=2)
    assert enc.seen and enc.seen[0].startswith("query: "), enc.seen


def test_the_two_prefixes_are_not_the_same():
    """E5 is asymmetric. A symmetric wiring is a silent quality loss."""
    enc = StubEncoder()
    idx = DenseIndex(DOCS, encoder=enc)
    passage_prefixed = list(enc.seen)
    enc.seen.clear()
    idx.search("سؤال", k=1)
    assert enc.seen[0].split(":")[0] != passage_prefixed[0].split(":")[0]


def test_prefixes_are_configurable_for_a_non_e5_model():
    """A model that was not trained with prefixes must be able to opt out."""
    enc = StubEncoder()
    DenseIndex(DOCS, encoder=enc, query_prefix="", passage_prefix="")
    assert all(not t.startswith("passage: ") for t in enc.seen)
    assert DOCS[0]["text"] in enc.seen


# ---------------------------------------------------------------------------
# 3. search behaviour and Hit compatibility
# ---------------------------------------------------------------------------


def test_search_returns_the_nearest_document_first():
    enc = StubEncoder(
        {
            "تعيين مسؤول": [0.0, 1.0],  # law-8 passage
            "إبلاغ المركز": [1.0, 0.0],  # law-7 passage
            "موظف مسؤول": [0.0, 1.0],  # the query, pointing at law-8
        }
    )
    idx = DenseIndex(DOCS, encoder=enc)
    hits = idx.search("هل يلزم تعيين موظف مسؤول عن الخصوصية؟", k=2)
    assert hits[0].id == "law-8"


def test_dense_hits_are_shape_compatible_with_bm25_hits():
    """`recall_at_k` and `reciprocal_rank` must work unchanged on dense hits."""
    enc = StubEncoder({"إبلاغ المركز": [1.0, 0.0], "الخرق": [1.0, 0.0]})
    idx = DenseIndex(DOCS, encoder=enc)
    hits = idx.search("متى يجب الإبلاغ عن الخرق؟", k=2)

    assert all(isinstance(h, Hit) for h in hits)
    assert recall_at_k(["law-7"], hits) == 1.0
    assert reciprocal_rank(["law-7"], hits) == 1.0


def test_k_caps_the_number_of_hits():
    idx = DenseIndex(DOCS, encoder=StubEncoder())
    assert len(idx.search("أي سؤال", k=1)) == 1


def test_empty_corpus_returns_no_hits_instead_of_raising():
    idx = DenseIndex([], encoder=StubEncoder())
    assert idx.search("أي سؤال", k=5) == []


# ---------------------------------------------------------------------------
# 4. the stale-cache guard
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_an_article_changes():
    other = [dict(DOCS[0]), dict(DOCS[1])]
    other[1]["text"] = other[1]["text"] + " ويصدر بذلك قرار من الوزير."
    assert corpus_fingerprint(DOCS) != corpus_fingerprint(other)


def test_fingerprint_changes_when_an_article_is_dropped():
    assert corpus_fingerprint(DOCS) != corpus_fingerprint(DOCS[:1])


def test_fingerprint_is_stable_across_runs():
    assert corpus_fingerprint(DOCS) == corpus_fingerprint(list(DOCS))


def test_cache_round_trips_without_calling_the_model_again(tmp_path):
    cache = tmp_path / "emb.npz"
    enc = StubEncoder({"إبلاغ": [1.0, 0.0], "تعيين": [0.0, 1.0]})

    first = DenseIndex(DOCS, encoder=enc, cache_path=cache)
    first.save()
    calls = len(enc.seen)

    reused = DenseIndex(DOCS, encoder=enc, cache_path=cache)
    assert len(enc.seen) == calls, "the cache was ignored and the model re-ran"
    assert np.allclose(reused.embeddings, first.embeddings)


def test_a_stale_cache_is_rejected_not_silently_reused(tmp_path):
    """The number must belong to the corpus that is on disk right now."""
    cache = tmp_path / "emb.npz"
    DenseIndex(DOCS, encoder=StubEncoder(), cache_path=cache).save()

    changed = [dict(DOCS[0]), dict(DOCS[1])]
    changed[0]["text"] = "نص مختلف تماماً بعد إعادة الاستيعاب."

    enc = StubEncoder()
    rebuilt = DenseIndex(changed, encoder=enc, cache_path=cache)
    assert enc.seen, "a cache from a different corpus was reused"
    assert rebuilt.fingerprint == corpus_fingerprint(changed)


# ---------------------------------------------------------------------------
# 5. reciprocal rank fusion (ADR-016)
# ---------------------------------------------------------------------------


def _hit(doc_id: str, score: float = 1.0) -> Hit:
    return Hit(id=doc_id, law_name="x", number=1, score=score, snippet="")


def test_rrf_puts_a_doc_found_by_both_retrievers_first():
    lexical = [_hit("a"), _hit("b")]
    dense = [_hit("c"), _hit("b")]
    fused = reciprocal_rank_fusion([lexical, dense], k=5)
    assert fused[0].id == "b", [h.id for h in fused]


def test_rrf_keeps_documents_found_by_only_one_retriever():
    fused = reciprocal_rank_fusion([[_hit("a")], [_hit("c")]], k=5)
    assert {h.id for h in fused} == {"a", "c"}


def test_rrf_ignores_score_scale():
    """Why RRF and not a weighted sum: BM25 scores are unbounded, cosine is not.

    A weighted sum needs a calibration constant fitted on the dev set — a
    hyperparameter tuned on 15 questions is a way to overfit, not to measure.
    RRF reads ranks only, so multiplying one retriever's scores by 1000 must
    change nothing.
    """
    lexical = [_hit("a", 0.01), _hit("b", 0.005)]
    dense = [_hit("b", 0.9), _hit("a", 0.8)]
    inflated = [_hit("a", 10_000.0), _hit("b", 5_000.0)]

    assert [h.id for h in reciprocal_rank_fusion([lexical, dense], k=5)] == [
        h.id for h in reciprocal_rank_fusion([inflated, dense], k=5)
    ]


def test_rrf_respects_k():
    fused = reciprocal_rank_fusion([[_hit("a"), _hit("b"), _hit("c")]], k=2)
    assert len(fused) == 2


def test_rrf_on_no_results_is_empty_not_an_error():
    assert reciprocal_rank_fusion([[], []], k=5) == []


# ---------------------------------------------------------------------------
# 6. cross-encoder reranking (ADR-017)
# ---------------------------------------------------------------------------


class StubCrossEncoder:
    """Scores a pair by how many query words appear in the passage."""

    def __init__(self):
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.seen.extend(pairs)
        return [
            sum(1 for w in q.split() if w in passage) / max(len(q.split()), 1)
            for q, passage in pairs
        ]


LONG_TAIL = "الفقرة الثانية التي تحمل الحكم الفعلي ولا تظهر في المقتطف القصير."
TEXTS = {
    "law-7": DOCS[0]["text"],
    "law-8": DOCS[1]["text"] + " " + "حشو " * 200 + LONG_TAIL,
}


def test_reranker_reads_the_full_article_not_the_snippet():
    """ADR-017. `Hit.snippet` is a 280-char display field.

    Scoring on it truncates every article past the first paragraph — and in a
    law the operative clause is very often in the second one.
    """
    from legalrag.rerank import rerank

    ce = StubCrossEncoder()
    hits = [_hit("law-8")]
    hits[0].snippet = TEXTS["law-8"][:280]

    rerank("الفقرة الثانية", hits, ce, texts=TEXTS, k=1)

    scored_passage = ce.seen[0][1]
    assert LONG_TAIL in scored_passage, "the reranker only saw the truncated snippet"


def test_reranker_reorders_candidates():
    from legalrag.rerank import rerank

    candidates = [_hit("law-7"), _hit("law-8")]
    out = rerank("تعيين مسؤول لحماية البيانات", candidates, StubCrossEncoder(),
                 texts=TEXTS, k=2)
    assert out[0].id == "law-8", [h.id for h in out]


def test_reranker_cannot_recover_an_article_absent_from_the_candidates():
    """The candidate stage's recall is a hard ceiling on the reranked result."""
    from legalrag.rerank import rerank

    out = rerank("تعيين مسؤول", [_hit("law-7")], StubCrossEncoder(), texts=TEXTS, k=5)
    assert [h.id for h in out] == ["law-7"]


def test_reranker_caps_at_k():
    from legalrag.rerank import rerank

    out = rerank("أي سؤال", [_hit("law-7"), _hit("law-8")], StubCrossEncoder(),
                 texts=TEXTS, k=1)
    assert len(out) == 1


def test_reranking_nothing_returns_nothing():
    from legalrag.rerank import rerank

    assert rerank("سؤال", [], StubCrossEncoder(), texts=TEXTS, k=5) == []
