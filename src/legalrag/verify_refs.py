"""Ground-truth reference checker.

Every ``expected_articles`` entry in the question set is a claim about the
corpus: "the answer to this question lives in article N". A wrong claim
silently corrupts every number the project will ever report, and it is exactly
the kind of error that is invisible once retrieval scores start moving.

This script checks two things per question:

1. the referenced article EXISTS in ``data/processed/articles.jsonl``;
2. every ``expected_keywords`` entry actually APPEARS in that article's text.

Questions that pass both are flipped to ``ref_status: "verified"`` when run
with ``--write``. Nothing else in the project is allowed to consume a question
whose ``ref_status`` is still ``unverified``.

Run: ``python tasks.py verify-refs [--write]``  (alias: ``make verify-refs``)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .evaluate import (
    CORPUS_PATH,
    QUESTIONS_PATH,
    binding_problem,
    corpus_laws,
    load_meta,
    load_questions,
)
from .normalize import search_normalize
from .overlap import containment, leaked_terms


def load_corpus() -> dict[str, dict]:
    if not CORPUS_PATH.exists():
        sys.exit(
            f"corpus not found at {CORPUS_PATH}.\n"
            "Run `python tasks.py ingest` first (and read data/raw/README.md if it is empty)."
        )
    corpus: dict[str, dict] = {}
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            corpus[rec["id"]] = rec
    return corpus


def main(argv: list[str]) -> int:
    write = "--write" in argv
    corpus = load_corpus()
    questions, errors = load_questions(split=None)  # validates held-out rows too
    if errors:
        for e in errors:
            print(f"  - {e}")
        return 1

    meta = load_meta()
    problem = binding_problem(meta, corpus_laws())
    if problem:
        print(f"BLOCKED: {problem}")
        return 3

    bands = meta.get("overlap_band", {})
    default_hi = float(meta.get("max_lexical_overlap", 0.55))
    verified: list[str] = []
    failed: list[tuple[str, str]] = []
    leaky: list[tuple[str, float, list[str]]] = []

    for q in questions:
        if not q.answerable:
            verified.append(q.id)  # nothing to point at, nothing to verify
            continue

        problems: list[str] = []
        present = [art for art in q.expected_articles if art in corpus]
        for art in q.expected_articles:
            if art not in corpus:
                problems.append(f"article {art} not in corpus")

        # Keywords are checked against the UNION of the expected articles: a
        # multi-article question is answered by the set, not by each member.
        combined_search = " ".join(corpus[a]["search_text"] for a in present)
        combined_text = " ".join(corpus[a]["text"] for a in present)
        for kw in q.expected_keywords:
            if search_normalize(kw) not in combined_search:
                problems.append(f"keyword {kw!r} not found in {q.expected_articles}")

        # Leakage guard (ADR-009): a question that copies the article's wording
        # is retrieved for the wrong reason and inflates every lexical score.
        if present:
            score = containment(q.question, combined_text)
            lo, hi = bands.get(q.category, [0.0, default_hi])
            if score > hi:
                leaky.append((q.id, round(score, 2), "HIGH",
                              leaked_terms(q.question, combined_text)))
            elif score < lo:
                # No lexical anchor at all: BM25 cannot match it, so the
                # baseline is crushed for a reason that is not about retrieval.
                # Paraphrase robustness is what `colloquial` is for.
                leaky.append((q.id, round(score, 2), "LOW", []))

        if problems:
            failed.append((q.id, "; ".join(problems)))
        else:
            verified.append(q.id)

    print(f"verified : {len(verified)}/{len(questions)}")
    if failed:
        print("\nFAILED — the ground truth is wrong, not the model:")
        for qid, why in failed:
            print(f"  {qid}: {why}")
    if leaky:
        print("\nOUT OF OVERLAP BAND (ADR-009) — not stamped verified:")
        for qid, score, kind, terms in leaky:
            if kind == "HIGH":
                print(f"  {qid}: {score} too high · copied: {' '.join(terms[:8])}")
            else:
                print(f"  {qid}: {score} too low · no lexical anchor — move to `colloquial` "
                      f"or add one concrete term from the article's subject")

    if write:
        lines = []
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.strip().startswith("//"):
                lines.append(line)
                continue
            rec = json.loads(line)
            leaky_ids = {q for q, _, _, _ in leaky}
            ok = rec["id"] in verified and rec["id"] not in leaky_ids
            rec["ref_status"] = "verified" if ok else "unverified"
            lines.append(json.dumps(rec, ensure_ascii=False))
        QUESTIONS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote ref_status back to {QUESTIONS_PATH}")

    return 0 if not (failed or leaky) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
