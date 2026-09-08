"""Rank fusion for hybrid retrieval (ADR-016).

Reciprocal Rank Fusion, not a weighted score sum.

BM25 scores are unbounded and corpus-dependent; cosine similarity lives in
[-1, 1] and, for E5, clusters tightly near the top. Combining them by value
needs a weight fitted to this corpus — and fitting a weight on a 15-question
dev set is a way to overfit, not a way to measure. RRF reads ranks only, so it
has one constant that does not depend on either score scale.

    score(d) = Σ  1 / (RRF_K + rank_of_d_in_list_i)

RRF_K = 60 is the value from the original paper (Cormack et al., 2009); it
damps the top ranks so a single retriever cannot dominate on its own.
"""

from __future__ import annotations

from .retrieve import Hit

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[Hit]], k: int = 5, rrf_k: int = RRF_K
) -> list[Hit]:
    """Fuse several ranked lists into one, keeping the top ``k``.

    The returned ``Hit.score`` is the RRF score, not either retriever's score:
    the two are not on the same scale and pretending otherwise is what this
    function exists to avoid.
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)
            # Keep the first Hit seen for a document; the fields we care about
            # (id, number, snippet) are identical across retrievers.
            best.setdefault(hit.id, hit)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[Hit] = []
    for doc_id, score in ordered[:k]:
        h = best[doc_id]
        out.append(
            Hit(
                id=h.id,
                law_name=h.law_name,
                number=h.number,
                score=round(score, 6),
                snippet=h.snippet,
            )
        )
    return out
