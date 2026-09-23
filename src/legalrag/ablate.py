"""Four-configuration retrieval ablation — PRD M1/A4.

    BM25  ·  dense  ·  hybrid (RRF)  ·  hybrid + cross-encoder rerank

Deliberately a separate entry point from ``evaluate``. ``tasks.py eval`` must
keep running on a fresh clone with no model and no torch, in under a second
(M1/A1); importing sentence-transformers there would cost seconds and a 1 GB
download. This module is the one that is allowed to be slow.

Every configuration is scored on the SAME questions with the SAME k, and the
reranked row carries its candidate-stage ceiling next to it — a reranker can
only reorder what fusion handed it, so its result is bounded by recall@depth of
that stage, and a reranked number without that ceiling is unreadable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .dense import CACHE_PATH, CORPUS_PATH, DEFAULT_MODEL, DenseIndex, load_docs
from .evaluate import binding_problem, corpus_laws, load_meta, load_questions
from .fusion import reciprocal_rank_fusion
from .rerank import CANDIDATE_DEPTH, DEFAULT_RERANKER, rerank
from .retrieve import BM25Index, Hit, recall_at_k, reciprocal_rank

EVAL_K = 5


def _avg(values) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def _fmt(value: float | None, width: int = 8) -> str:
    return f"{value:.3f}".rjust(width) if value is not None else "—".rjust(width)


class Config:
    """One retrieval configuration, scored end to end."""

    def __init__(self, name: str, search, note: str = ""):
        self.name = name
        self.search = search
        self.note = note
        self.recall: float | None = None
        self.mrr: float | None = None
        self.seconds = 0.0
        self.per_category: dict[str, tuple[float | None, float | None]] = {}
        # Per-question (recall, RR), in the order the questions were scored.
        # Kept because callers that want intervals need the individual values,
        # and re-deriving them meant searching the whole set a second time —
        # about six seconds per question on the reranked configuration.
        self.per_question: list[tuple[float, float]] = []

    def run(self, questions, k: int = EVAL_K) -> None:
        rows: list[tuple[str, float, float]] = []
        start = time.perf_counter()
        for q in questions:
            hits = self.search(q.question, k)
            rows.append(
                (
                    q.category,
                    recall_at_k(q.expected_articles, hits),
                    reciprocal_rank(q.expected_articles, hits),
                )
            )
        self.seconds = time.perf_counter() - start
        self.per_question = [(r, m) for _, r, m in rows]
        self.recall = _avg(r for _, r, _ in rows)
        self.mrr = _avg(m for _, _, m in rows)
        for cat in ("direct", "multi_article", "colloquial"):
            sub = [r for r in rows if r[0] == cat]
            self.per_category[cat] = (
                _avg(r for _, r, _ in sub),
                _avg(m for _, _, m in sub),
            )


def candidate_ceiling(questions, search, depth: int) -> float | None:
    """Recall of the candidate stage — the hard cap on any reranking of it."""
    return _avg(
        recall_at_k(q.expected_articles, search(q.question, depth)) for q in questions
    )


def build_configs(docs: list[dict], questions, k: int = EVAL_K) -> list[Config]:
    bm25 = BM25Index(docs)

    print(f"loading {DEFAULT_MODEL} …", flush=True)
    t = time.perf_counter()
    dense = DenseIndex(docs, cache_path=CACHE_PATH)
    dense.save()
    src = "cache" if dense.from_cache else "model"
    print(f"  embeddings from {src} ({time.perf_counter() - t:.1f}s)", flush=True)

    def hybrid(query: str, kk: int) -> list[Hit]:
        return reciprocal_rank_fusion(
            [bm25.search(query, CANDIDATE_DEPTH), dense.search(query, CANDIDATE_DEPTH)],
            k=kk,
        )

    print(f"loading {DEFAULT_RERANKER} …", flush=True)
    t = time.perf_counter()
    from .rerank import load_reranker

    cross = load_reranker()
    print(f"  ready ({time.perf_counter() - t:.1f}s)", flush=True)

    texts = {d["id"]: d["text"] for d in docs}

    def hybrid_reranked(query: str, kk: int) -> list[Hit]:
        return rerank(query, hybrid(query, CANDIDATE_DEPTH), cross, texts=texts, k=kk)

    return [
        Config("BM25", bm25.search, "lexical only, no model"),
        Config("dense", dense.search, DEFAULT_MODEL.split("/")[-1]),
        Config("hybrid (RRF)", hybrid, f"BM25 + dense, depth {CANDIDATE_DEPTH}"),
        Config(
            "hybrid + rerank",
            hybrid_reranked,
            DEFAULT_RERANKER.split("/")[-1],
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    questions, errors = load_questions()
    if errors:
        for e in errors:
            print(f"  - {e}")
        return 1

    meta = load_meta()
    problem = binding_problem(meta, corpus_laws())
    if problem:
        print(f"BLOCKED: {problem}")
        return 3

    docs = load_docs(CORPUS_PATH)
    if not docs:
        print("corpus not ingested. Run `python tasks.py ingest --law \"…\"` first.")
        return 2

    answerable = [q for q in questions if q.answerable]
    unverified = [q for q in answerable if q.ref_status != "verified"]
    if unverified:
        print(
            f"BLOCKED: {len(unverified)} answerable questions are not `verified`.\n"
            "Run `python tasks.py verify-refs --write` first — an ablation over "
            "unverified ground truth compares four models against a ruler that "
            "has not been checked (ADR-005)."
        )
        return 4

    print("=" * 68)
    print("  arabic-legal-rag · retrieval ablation · PRD M1/A4")
    print("=" * 68)
    print(f"\ncorpus     : {len(docs)} articles · {meta.get('corpus_law', '?')} "
          f"[{meta.get('corpus_law_status', '?')}]")
    print(f"questions  : {len(answerable)} answerable of {len(questions)} "
          f"({len(questions) - len(answerable)} abstention, not scored on retrieval)")
    print(f"k          : {EVAL_K}\n")

    configs = build_configs(docs, answerable)
    for c in configs:
        c.run(answerable, EVAL_K)

    ceiling = candidate_ceiling(answerable, configs[2].search, CANDIDATE_DEPTH)

    print("\n  configuration      recall@5      MRR     sec/q    notes")
    print("  " + "-" * 72)
    base = configs[0].recall
    for c in configs:
        delta = ""
        if base and c.recall is not None and c is not configs[0]:
            delta = f"  ({c.recall - base:+.3f} vs BM25)"
        print(f"  {c.name:16s} {_fmt(c.recall)} {_fmt(c.mrr, 8)}  "
              f"{c.seconds / max(len(answerable), 1):7.3f}    {c.note}{delta}")
    print("  " + "-" * 72)
    print(f"\n  candidate ceiling (recall@{CANDIDATE_DEPTH} of hybrid) : {_fmt(ceiling).strip()}")
    print("  The reranker can only reorder those candidates; it cannot exceed this.")

    print("\n  per category (recall@5)")
    print(f"  {'configuration':16s} {'direct':>9} {'multi_art':>10} {'colloquial':>11}")
    print("  " + "-" * 50)
    for c in configs:
        cells = "".join(
            _fmt(c.per_category.get(cat, (None, None))[0], w)
            for cat, w in (("direct", 10), ("multi_article", 11), ("colloquial", 12))
        )
        print(f"  {c.name:16s}{cells}")

    print("\ncost: local CPU inference, no API calls. One-time model download "
          "(~530 MB e5-base + ~470 MB reranker); offline afterwards.")
    print("Small sample — read the caveat in EVAL.md before quoting any of this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
