"""Retrieval baselines.

BM25 first, on purpose. Most Arabic RAG write-ups jump straight to embeddings;
lexical search is the honest baseline a dense model has to beat, and on
inflected Arabic with a shared legal vocabulary it is often stubbornly good.
A dense number with no lexical number next to it says nothing.

The implementation is dependency-free by default (rank_bm25 is used when it is
installed, purely to cross-check the built-in one). Everything is indexed on
``search_text`` — the aggressively normalized field — never on the verbatim
text. See DECISIONS.md ADR-004.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .normalize import search_normalize

CORPUS_PATH = Path("data/processed/articles.jsonl")

K1 = 1.5
B = 0.75


@dataclass
class Hit:
    id: str
    law_name: str
    number: int
    score: float
    snippet: str


class BM25Index:
    """Okapi BM25 over article-level chunks."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.tokens = [d["search_text"].split() for d in docs]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        self.tf: list[Counter] = [Counter(t) for t in self.tokens]
        df: Counter = Counter()
        for counter in self.tf:
            df.update(counter.keys())
        n = len(docs)
        # BM25 idf with the +0.5 smoothing; floored at a small positive value so
        # a term present in every article cannot drag a score negative.
        self.idf = {
            term: max(math.log((n - freq + 0.5) / (freq + 0.5) + 1.0), 1e-6)
            for term, freq in df.items()
        }

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q_terms = search_normalize(query).split()
        if not q_terms:
            return []

        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self.tf):
            if not any(t in tf for t in q_terms):
                continue
            length = self.lengths[i] or 1
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * length / (self.avg_len or 1))
                score += self.idf.get(term, 0.0) * (f * (K1 + 1)) / denom
            if score > 0:
                scores.append((score, i))

        scores.sort(reverse=True)
        hits: list[Hit] = []
        for score, i in scores[:k]:
            d = self.docs[i]
            hits.append(
                Hit(
                    id=d["id"],
                    law_name=d.get("law_name", ""),
                    number=d["number"],
                    score=round(score, 4),
                    snippet=d["text"][:280],
                )
            )
        return hits


def load_index(path: Path = CORPUS_PATH) -> BM25Index | None:
    if not path.exists():
        return None
    docs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return BM25Index(docs) if docs else None


def recall_at_k(expected: list[str], hits: list[Hit]) -> float:
    """Fraction of expected articles that appear in the returned hits.

    A multi-article question scores 0.5 when one of its two articles is found —
    partial credit, because "found one of two" is genuinely half-right and
    collapsing it to 0 hides real progress.
    """
    if not expected:
        return 0.0
    found = sum(1 for e in expected if any(h.id == e for h in hits))
    return found / len(expected)


def reciprocal_rank(expected: list[str], hits: list[Hit]) -> float:
    """1/rank of the FIRST expected article; 0 if none retrieved."""
    for rank, h in enumerate(hits, 1):
        if h.id in expected:
            return 1.0 / rank
    return 0.0
