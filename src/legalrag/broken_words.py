"""Broken-word detector for the ingested corpus (ADR-018).

A word split by a space that does not belong — ``عشر ين`` for ``عشرين``,
``تو افر`` for ``توافر`` — is invisible in a spot check and expensive in
retrieval: BM25 indexes the two halves as two tokens, so a query containing the
whole word matches neither. The effect is a lexical baseline reported lower
than it deserves, with no error anywhere to explain why.

ADR-013 recorded this defect class as "0 remaining". Running this over the
corpus on 2026-09-08 found 20 instances across 13 articles, which is why the
check now lives in code instead of in a claim.

The evidence rule is deliberately conservative and needs no dictionary: flag
``a b`` only when the joined form ``ab`` **already occurs elsewhere in the same
corpus as a standalone token**. The corpus is its own lexicon. A legitimate
two-word sequence like ``أو كل`` can still slip through when ``أوكل`` happens
to exist — those are reported, not silently dropped, because a human deciding
"this one is fine" is cheap and a missed split is not.

Run: ``python tasks.py broken-words``
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

CORPUS_PATH = Path("data/processed/articles.jsonl")
STRIP = "،.:؛()\"'“”"

# Longest fragment that can plausibly be the tail of a split word. Above this
# the pair is far more likely to be two real words that happen to concatenate.
MAX_TAIL = 4


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def token_counts(docs: list[dict]) -> Counter:
    vocab: Counter = Counter()
    for d in docs:
        for tok in d["text"].split():
            tok = tok.strip(STRIP)
            if tok:
                vocab[tok] += 1
    return vocab


def find_broken_words(docs: list[dict], max_tail: int = MAX_TAIL) -> list[dict]:
    """Adjacent token pairs whose joined form exists elsewhere in the corpus."""
    vocab = token_counts(docs)
    out: list[dict] = []
    for d in docs:
        toks = d["text"].split()
        for i in range(len(toks) - 1):
            a = toks[i].strip(STRIP)
            b = toks[i + 1].strip(STRIP)
            if not a or not b or len(a) < 2 or len(b) > max_tail:
                continue
            joined = a + b
            if vocab.get(joined, 0) >= 1:
                out.append(
                    {
                        "article": d["id"],
                        "broken": f"{a} {b}",
                        "joined": joined,
                        "joined_seen": vocab[joined],
                    }
                )
    return out


def main(argv: list[str] | None = None) -> int:
    docs = load_corpus()
    if not docs:
        print("corpus not ingested. Run `python tasks.py ingest --law \"…\"` first.")
        return 2

    hits = find_broken_words(docs)
    by_article = Counter(h["article"] for h in hits)

    print(f"corpus     : {len(docs)} articles")
    print(f"broken     : {len(hits)} instances across {len(by_article)} articles\n")

    seen: set[str] = set()
    for h in hits:
        if h["broken"] in seen:
            continue
        seen.add(h["broken"])
        print(f"  {h['article']:10} '{h['broken']}'  ->  '{h['joined']}' "
              f"(joined form seen {h['joined_seen']}x)")

    if hits:
        print(
            "\nEach of these is two BM25 tokens instead of one, so the lexical "
            "baseline is reported lower than it deserves.\n"
            "Origin is not settled — see ADR-018. Some are in the source PDF, "
            "some come from our reconstruction."
        )
    return 0 if not hits else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
