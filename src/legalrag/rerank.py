"""Cross-encoder reranking (ADR-017).

A bi-encoder embeds the query and the article separately and compares two
vectors; a cross-encoder reads the pair together and scores it directly. That
is strictly more informative and strictly more expensive — which is exactly why
it reranks a short candidate list instead of scoring the corpus.

The candidate list comes from hybrid fusion. The reranker can only reorder what
it is given: if the right article is not in the candidates, no reranker
recovers it. So recall@k of the *candidate* stage caps everything this can do,
and that ceiling is reported in the ablation table next to the result.

One detail worth stating because it is easy to get wrong and impossible to see
afterwards: the reranker is given the article's **full text**, not ``Hit
.snippet``. The snippet is a 280-character display field for the UI. Scoring on
it would silently truncate every article past the first paragraph — and the
answer to a legal question is very often in the second one.
"""

from __future__ import annotations

from dataclasses import replace

from .retrieve import Hit

# Multilingual MS MARCO cross-encoder — Arabic is in its training mix, it is
# small enough to run on CPU, and it is the standard reranking baseline.
# Alternatives considered in DECISIONS.md ADR-017.
DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# How many hybrid candidates get reranked. 20 of 56 articles: wide enough that
# the ceiling is meaningfully above k=5, narrow enough to stay quick on CPU.
CANDIDATE_DEPTH = 20


def load_reranker(name: str = DEFAULT_RERANKER):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "sentence-transformers is not installed.\n"
            "  python tasks.py setup      (or: pip install -r requirements.txt)"
        ) from e
    return CrossEncoder(name)


def rerank(
    query: str,
    candidates: list[Hit],
    model,
    texts: dict[str, str],
    k: int = 5,
) -> list[Hit]:
    """Reorder ``candidates`` by cross-encoder score and keep the top ``k``.

    ``texts`` maps article id -> full verbatim article text. The query and the
    article go to the model verbatim, for the same reason as ADR-015: the model
    was trained on natural text, and ``search_normalize`` strips the morphology
    it reads.
    """
    if not candidates:
        return []

    pairs = [(query, texts.get(c.id, c.snippet)) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores), key=lambda cs: (-float(cs[1]), cs[0].id)
    )
    return [replace(hit, score=round(float(score), 4)) for hit, score in ranked[:k]]
