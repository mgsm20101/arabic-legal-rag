"""Lexical-overlap guard for the evaluation set.

The failure this prevents
-------------------------
A question written by copying the article's own wording is trivially retrievable
by any lexical method. The score that comes out measures the question set, not
the system. This is the exact defect that disqualified the public generated QA
set (DECISIONS.md ADR-008) — so the same standard is enforced on our own
hand-written questions rather than assumed.

The metric
----------
``containment`` = |content words of the question that also appear in the
answering article| / |content words of the question|.

Containment, not Jaccard: the article is far longer than the question, so
Jaccard would be dominated by article length and stay near zero for every
question, measuring nothing. Containment answers the question that matters —
"how much of this question is just copied from the passage?"

A legal question cannot reach 0.0 (it has to name the concept the article is
about) and should not approach 1.0. The threshold lives in
``evals/retrieval/meta.json``.
"""

from __future__ import annotations

from .normalize import search_normalize

# High-frequency Arabic function words + question words. Excluded because they
# say nothing about copying: every question and every article contains them.
STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "او", "أو", "و", "ما", "لا", "ان", "أن", "إن", "كان", "يكون",
    "هل", "كم", "كيف", "متى", "اين", "أين", "ليه", "ايه", "إيه", "ولا",
    "يجب", "هو", "هي", "به", "بها", "له", "لها", "كل", "اي", "أي", "عند",
    "بعد", "قبل", "بين", "غير", "قد", "لو", "عشان", "علي", "ده", "دي",
}


def content_tokens(text: str) -> list[str]:
    """Normalized tokens with stopwords and 1-2 letter fragments removed."""
    return [t for t in search_normalize(text).split() if len(t) > 2 and t not in STOPWORDS]


def containment(question: str, article_text: str) -> float:
    """Share of the question's content words that also appear in the article."""
    q = content_tokens(question)
    if not q:
        return 0.0
    a = set(content_tokens(article_text))
    return sum(1 for t in q if t in a) / len(q)


def leaked_terms(question: str, article_text: str) -> list[str]:
    """The specific words copied from the article — what to rewrite."""
    a = set(content_tokens(article_text))
    seen, out = set(), []
    for t in content_tokens(question):
        if t in a and t not in seen:
            seen.add(t)
            out.append(t)
    return out
