"""Evaluation harness.

Deliberately written BEFORE any retrieval code (see PRD.md §5 and the
eval-first rule in DECISIONS.md ADR-001). Today it loads the question set,
enforces its schema and prints the scoreboard with an empty result column.
When a retriever lands it fills the same table — the table shape never changes,
so runs stay comparable across the whole project.

Run: ``python tasks.py eval``  (alias: ``make eval``)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

QUESTIONS_PATH = Path("evals/retrieval/questions.jsonl")
META_PATH = Path("evals/retrieval/meta.json")
CORPUS_PATH = Path("data/processed/articles.jsonl")


def load_meta(path: Path = META_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_laws(path: Path = CORPUS_PATH) -> list[str]:
    if not path.exists():
        return []
    names = {
        json.loads(line).get("law_name", "")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return sorted(n for n in names if n)


def binding_problem(meta: dict, laws: list[str]) -> str | None:
    """Refuse to score when the question set was written for a different law.

    Without this check, running the 151/2020 questions against a corpus of
    قانون المرافعات produces a low score that looks like a retrieval problem.
    It is not — it is a mismatched eval set, and nothing downstream would ever
    reveal that.
    """
    target = (meta.get("corpus_law") or "").strip()
    if not target:
        return None
    if not laws:
        return None  # nothing ingested yet; the empty-corpus message covers it
    if not any(target in law for law in laws):
        return (
            f"corpus/question mismatch: questions target {target!r}, "
            f"ingested corpus is {laws!r}. Re-target the questions or ingest the "
            f"matching law — scoring is refused."
        )
    return None

EVAL_K = 5


def _avg(values) -> str:
    vals = list(values)
    return f"{sum(vals) / len(vals):.3f}" if vals else "—"


def score_questions(questions, k: int = EVAL_K) -> list[tuple[str, float, float]]:
    """(category, recall@k, RR) for every ANSWERABLE question.

    Unanswerable questions are left out rather than scored 0: retrieval always
    returns something, so scoring an abstention question on recall measures the
    generator's job with the retriever's ruler. They are M2's test (PRD §5).
    """
    from .retrieve import load_index, recall_at_k, reciprocal_rank

    index = load_index()
    if index is None:
        return []
    scored = []
    for q in questions:
        if not q.answerable:
            continue
        hits = index.search(q.question, k)
        scored.append(
            (
                q.category,
                recall_at_k(q.expected_articles, hits),
                reciprocal_rank(q.expected_articles, hits),
            )
        )
    return scored


CATEGORIES = {"direct", "multi_article", "out_of_corpus", "colloquial"}
SPLITS = {"dev", "test"}
DEFAULT_SPLIT = "dev"
REQUIRED_FIELDS = {
    "id", "category", "question", "expected_articles",
    "expected_keywords", "answerable", "ref_status", "split",
}


@dataclass
class Question:
    id: str
    category: str
    question: str
    expected_articles: list[str]
    expected_keywords: list[str]
    answerable: bool
    ref_status: str
    # Required in the FILE (see REQUIRED_FIELDS) so no row defaults into `dev`
    # silently; defaulted on the object so constructing one ad hoc stays cheap.
    split: str = DEFAULT_SPLIT
    notes: str = ""


def load_questions(
    path: Path = QUESTIONS_PATH, split: str | None = DEFAULT_SPLIT
) -> tuple[list[Question], list[str]]:
    """Load the question set, returning only `split` (None returns every row).

    The default is `dev` on purpose, and every caller in this repo takes it.
    The `test` rows exist to answer one question later — does a configuration
    chosen on `dev` hold on questions that never informed it — and that answer
    is only worth having if the rows stayed unseen. A held-out split kept by
    convention is not held out; making it invisible unless a caller names it
    means forgetting to exclude it is not a way to leak it.

    `verify-refs` is the one caller that passes None, because a row it never
    validates is a row with no `ref_status` — and that check reads the article
    text, not any retrieval or generation output.
    """
    if split is not None and split not in SPLITS:
        return [], [f"unknown split {split!r}; expected one of {sorted(SPLITS)}"]
    if not path.exists():
        return [], [f"missing question file: {path}"]

    questions: list[Question] = []
    errors: list[str] = []
    seen: set[str] = set()

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
            continue

        missing = REQUIRED_FIELDS - raw.keys()
        if missing:
            errors.append(f"line {lineno}: missing fields {sorted(missing)}")
            continue
        if raw["category"] not in CATEGORIES:
            errors.append(f"{raw['id']}: unknown category {raw['category']!r}")
            continue
        if raw["split"] not in SPLITS:
            errors.append(f"{raw['id']}: unknown split {raw['split']!r}")
            continue
        if raw["id"] in seen:
            errors.append(f"{raw['id']}: duplicate id")
            continue
        if raw["answerable"] and not raw["expected_articles"]:
            errors.append(f"{raw['id']}: answerable=true but no expected_articles")
            continue
        if not raw["answerable"] and raw["expected_articles"]:
            errors.append(f"{raw['id']}: answerable=false must have empty expected_articles")
            continue

        seen.add(raw["id"])
        if split is not None and raw["split"] != split:
            continue  # after validation: a malformed held-out row is still an error
        questions.append(Question(**{k: raw[k] for k in Question.__annotations__ if k in raw}))

    return questions, errors


def question_set_fingerprint(questions) -> str:
    """Identify the exact eval set a run scored, so a result cannot outlive it.

    Covers what a saved answer is only interpretable against: which questions
    were asked, what each one expects, and whether it was answerable at all.
    Wording is deliberately included — a reworded question under the same id is
    a different question, and re-using an old answer for it would be silent.

    Not a security hash; it exists so that promoting a stale run into the
    registry fails loudly instead of stamping today's commit onto answers
    generated against a question set that no longer exists.
    """
    parts = [
        "|".join(
            [
                q.id,
                q.category,
                q.split,
                "1" if q.answerable else "0",
                ",".join(sorted(q.expected_articles)),
                q.question.strip(),
            ]
        )
        for q in sorted(questions, key=lambda q: q.id)
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def corpus_size(path: Path = CORPUS_PATH) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    questions, errors = load_questions()

    print("=" * 68)
    print("  arabic-legal-rag — retrieval evaluation")
    print("=" * 68)

    if errors:
        print("\nSCHEMA ERRORS:")
        for e in errors:
            print(f"  - {e}")
        print("\nfix the question set before running anything else.")
        return 1

    meta = load_meta()
    laws = corpus_laws()
    n = corpus_size()
    print(f"\ncorpus     : {n} articles" if n else "\ncorpus     : NOT INGESTED (run `python tasks.py ingest`)")
    if laws:
        print(f"law        : {' · '.join(laws)}")
    unverified = sum(1 for q in questions if q.ref_status != "verified")
    print(f"questions  : {len(questions)} loaded, {unverified} with unverified article refs")
    print(f"split      : {meta.get('split', '?')}  ·  target law: {meta.get('corpus_law', '?')} "
          f"[{meta.get('corpus_law_status', '?')}]")

    problem = binding_problem(meta, laws)
    if problem:
        print(f"\nBLOCKED: {problem}")
        return 3

    by_cat = Counter(q.category for q in questions)
    scores = score_questions(questions)

    print("\n  category         count   scored   recall@5      MRR")
    print("  " + "-" * 56)
    for cat in ("direct", "multi_article", "colloquial", "out_of_corpus"):
        rows = [s for s in scores if s[0] == cat]
        n = len(rows)
        print(f"  {cat:16s} {by_cat.get(cat, 0):5d}   {n:6d}   "
              f"{_avg(r for _, r, _ in rows):>8}   {_avg(m for _, _, m in rows):>6}")
    print("  " + "-" * 56)
    print(f"  {'TOTAL':16s} {len(questions):5d}   {len(scores):6d}   "
          f"{_avg(r for _, r, _ in scores):>8}   {_avg(m for _, _, m in scores):>6}")

    if not scores:
        print(f"\n{len(questions)} questions loaded, 0 scored — no corpus ingested yet.")
    else:
        skipped = len(questions) - len(scores)
        print(f"\nBM25 baseline · k=5 · dev split · {len(scores)} scored"
              f"{f', {skipped} abstention questions not scored on retrieval' if skipped else ''}.")
    if unverified:
        print(f"WARNING: {unverified} questions have unverified article refs. Run `python tasks.py verify-refs`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
