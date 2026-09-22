"""Adversarial probes — one named failure mode per case.

Separate from ``evaluate`` and from ``ablate`` on purpose, and separate from the
dev split on purpose. The numbers here are not a score: each case asserts that
one specific failure mode exists, and its result reads as yes/no on that
assertion. Mixing them into ``eval`` would turn "this failure is reachable" into
"the system is N% good", which is the one reading this file cannot support —
the questions were written after reading the article, so they are circular by
construction (``evals/adversarial/meta.json`` says so in full).

What keeps a probe honest is that it checks itself before it scores anything:

* ``evidence`` must appear verbatim in the expected article. If it does not, the
  case is stale — the corpus moved under it — and it prints INVALID instead of a
  failure. A broken ruler must never read as a broken system.
* a ``tail_evidence`` case must have its evidence land *past* the encoder's token
  ceiling, measured with the encoder's own tokenizer rather than a hardcoded 512.
  If a fix later brings that text inside the window, the case prints NOT-A-PROBE
  and asks to be re-read — it does not quietly start passing for a new reason.

That second check is the whole point. ``SentenceTransformer.encode`` truncates
silently: no exception, no warning, just a vector that never saw the end of the
article. Nothing downstream can tell that apart from a vector that did.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .dense import (
    CACHE_PATH,
    CORPUS_PATH,
    PASSAGE_PREFIX,
    DenseIndex,
    _token_spans,
    encoder_limits,
    load_docs,
    token_windows,
    window_budget,
)
from .evaluate import binding_problem, corpus_laws, load_meta, load_questions
from .retrieve import recall_at_k, reciprocal_rank

CASES_PATH = Path("evals/adversarial/cases.jsonl")
META_PATH = Path("evals/adversarial/meta.json")

PROBE_K = 5
# How deep to look when reporting a rank. Past this the rank is reported as "out".
RANK_DEPTH = 20

# Verdicts that are about the case, not about the system.
INVALID = "INVALID"
NOT_A_PROBE = "NOT-A-PROBE"


# -- how an Egyptian typist actually writes ----------------------------------
#
# These are substitutions, not normalizations: they take the corpus's spelling
# and produce the one a person types on a phone or a keyboard without the hamza
# keys. Every one of them is a form that appears constantly in real Arabic
# input, so a retriever that only works on the first column works only in tests.
#
# The dense path takes the query verbatim on purpose (ADR-015), which means the
# user's spelling reaches the model exactly as typed. That is the design under
# test here, not an oversight.

TYPIST_FORMS: dict[str, dict[str, str]] = {
    # «الدعائية» -> «الدعائيه». The single most common one: ة and ه sit on the
    # same key and Egyptian typing drops the dots constantly.
    "ta_marbuta": {"ة": "ه"},
    # «على» -> «علي». Also constant, and it is the same key without the dots.
    "alef_maqsura": {"ى": "ي"},
    # «الأفراد» -> «الافراد». Hamza needs a modifier key that most people skip.
    "bare_alef": {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"},
    # «مسؤول» -> «مسوول», «هيئة» -> «هيية». Rarer than the others, and included
    # because the seated hamzas are the ones a phone keyboard hides deepest.
    "hamza_seat": {"ؤ": "و", "ئ": "ي"},
}

# Everything at once: not a separate failure mode, but the realistic case. A
# person who skips the hamza key skips it everywhere in the same sentence.
ALL_FORMS = "all"


def typist_variant(text: str, form: str) -> str:
    """`text` as someone would type it under `form`. Unknown form returns it unchanged."""
    if form == ALL_FORMS:
        for mapping in TYPIST_FORMS.values():
            for source, target in mapping.items():
                text = text.replace(source, target)
        return text
    for source, target in TYPIST_FORMS.get(form, {}).items():
        text = text.replace(source, target)
    return text


def sweep_forms() -> list[str]:
    return [*TYPIST_FORMS, ALL_FORMS]


@dataclass
class Case:
    id: str
    probe: str
    question: str
    expected_articles: list[str]
    evidence: str = ""
    distractor_articles: list[str] = field(default_factory=list)
    variants: list[dict] = field(default_factory=list)
    base: str = ""
    why: str = ""


def load_cases(path: Path = CASES_PATH) -> tuple[list[Case], list[str]]:
    """Cases plus the problems found while reading them. `//` lines are comments."""
    if not path.exists():
        return [], [f"{path} not found"]

    cases: list[Case] = []
    errors: list[str] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as e:
            errors.append(f"{path}:{n}: {e}")
            continue
        case_id = raw.get("id", f"line-{n}")
        if case_id in seen:
            errors.append(f"{path}:{n}: duplicate case id {case_id}")
            continue
        seen.add(case_id)
        cases.append(
            Case(
                id=case_id,
                probe=raw.get("probe", ""),
                question=raw.get("question", ""),
                expected_articles=raw.get("expected_articles") or [],
                evidence=raw.get("evidence", ""),
                distractor_articles=raw.get("distractor_articles") or [],
                variants=raw.get("variants") or [],
                base=raw.get("base", ""),
                why=raw.get("why", ""),
            )
        )
    return cases, errors


# -- the encoder's blind spot -------------------------------------------------


def visible_chars(tokenizer, max_seq: int, text: str, prefix: str = PASSAGE_PREFIX) -> int | None:
    """Where the first window ends, or None if `text` fits in one.

    Deliberately `dense`'s own splitter rather than a second measurement of the
    same boundary: "past the cut" then means exactly "past the first window",
    which is the thing the index acts on. Two implementations of one boundary
    would eventually disagree, and the probe would be arguing with the code it
    is supposed to be checking.
    """
    budget = window_budget(tokenizer, max_seq, prefix)
    windows = token_windows(tokenizer, budget, text, overlap=0)
    return None if len(windows) == 1 else windows[0][1]


def offsets_available(tokenizer, sample: str = "aa bb") -> bool:
    """Whether this tokenizer can say where its tokens sit in the string.

    Without offsets `token_windows` returns one window covering everything —
    correct as a conservative fallback for the index, but it makes every article
    look like it fits, which would quietly turn every tail probe into "nothing
    to see here".
    """
    return _token_spans(tokenizer, sample) is not None


def check_case(case: Case, texts: dict[str, str], cuts: dict[str, int | None]) -> str | None:
    """A problem with the case itself, or None if it is fit to score."""
    if not case.expected_articles:
        return f"{INVALID}: no expected_articles"
    missing = [a for a in case.expected_articles if a not in texts]
    if missing:
        return f"{INVALID}: expected article not in corpus: {', '.join(missing)}"

    if case.probe == "orthographic_variant":
        if not case.variants:
            return f"{INVALID}: orthographic_variant with no variants"
        return None

    if not case.evidence:
        return f"{INVALID}: no evidence"

    holder = case.expected_articles[0]
    position = texts[holder].find(case.evidence)
    if position < 0:
        return f"{INVALID}: evidence not found verbatim in {holder}"

    if case.probe == "tail_evidence":
        cut = cuts.get(holder)
        if cut is None:
            return f"{NOT_A_PROBE}: {holder} is no longer truncated by the encoder"
        if position < cut:
            return (
                f"{NOT_A_PROBE}: evidence sits at char {position}, inside the "
                f"first {cut} the encoder reads"
            )
    return None


# -- scoring ------------------------------------------------------------------


def rank_of(article: str, hits) -> int | None:
    for i, hit in enumerate(hits, 1):
        if hit.id == article:
            return i
    return None


def _rank_cell(rank: int | None, k: int = PROBE_K) -> str:
    if rank is None:
        return "out"
    return f"#{rank}" if rank <= k else f"#{rank}!"


def run_tail_and_distractor(case: Case, search) -> dict:
    hits = search(case.question, RANK_DEPTH)
    expected = case.expected_articles[0]
    expected_rank = rank_of(expected, hits)
    distractors = {d: rank_of(d, hits) for d in case.distractor_articles}
    found = expected_rank is not None and expected_rank <= PROBE_K
    outranked = [
        d
        for d, r in distractors.items()
        if r is not None and (expected_rank is None or r < expected_rank)
    ]
    return {
        "id": case.id,
        "probe": case.probe,
        "expected": expected,
        "expected_rank": expected_rank,
        "distractors": distractors,
        "outranked_by": outranked,
        "passed": found and not outranked,
    }


def run_orthographic(case: Case, search) -> dict:
    expected = case.expected_articles[0]
    forms = [("as written", case.question)] + [
        (v.get("form", "variant"), v.get("question", "")) for v in case.variants
    ]
    ranks = [(form, rank_of(expected, search(q, RANK_DEPTH))) for form, q in forms if q]
    values = [r for _, r in ranks]
    stable = len(set(values)) == 1
    all_in_k = all(r is not None and r <= PROBE_K for r in values)
    return {
        "id": case.id,
        "probe": case.probe,
        "expected": expected,
        "base": case.base,
        "ranks": ranks,
        "stable": stable,
        "passed": stable and all_in_k,
    }


RUNNERS = {
    "tail_evidence": run_tail_and_distractor,
    "near_miss_distractor": run_tail_and_distractor,
    "orthographic_variant": run_orthographic,
}


# -- the orthographic sweep ---------------------------------------------------


def sweep_orthography(questions, search, k: int = PROBE_K) -> list[dict]:
    """Every answerable dev question re-asked in each typist spelling.

    The hand-written `orthographic_variant` cases show the failure exists. This
    shows how big it is, on the same questions and the same ruler the adopted
    number (`ablate`) is measured with — three cases cannot tell the difference
    between one unlucky question and a retriever that drops whenever someone
    types without the hamza key.

    The dev questions are read, never written: this derives variants in memory
    and leaves `evals/retrieval/` untouched, so Run 1-8 stay comparable.
    """
    rows: list[dict] = []
    baseline_ranks: dict[str, list] = {}

    for form in ["as written", *sweep_forms()]:
        recalls, rrs = [], []
        moved, left_top_k, entered_top_k = 0, 0, 0
        for q in questions:
            text = q.question if form == "as written" else typist_variant(q.question, form)
            hits = search(text, RANK_DEPTH)
            recalls.append(recall_at_k(q.expected_articles, hits[:k]))
            rrs.append(reciprocal_rank(q.expected_articles, hits[:k]))
            ranks = [rank_of(a, hits) for a in q.expected_articles]
            if form == "as written":
                baseline_ranks[q.id] = ranks
                continue
            before = baseline_ranks[q.id]
            if ranks != before:
                moved += 1
            for was, now in zip(before, ranks, strict=True):
                was_in = was is not None and was <= k
                now_in = now is not None and now <= k
                if was_in and not now_in:
                    left_top_k += 1
                elif now_in and not was_in:
                    entered_top_k += 1
        rows.append({
            "form": form,
            "recall": sum(recalls) / len(recalls) if recalls else None,
            "mrr": sum(rrs) / len(rrs) if rrs else None,
            "moved": moved,
            "left_top_k": left_top_k,
            "entered_top_k": entered_top_k,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = argv or []

    cases, errors = load_cases()
    if errors:
        for e in errors:
            print(f"  - {e}")
        return 1
    if not cases:
        print(f"no cases in {CASES_PATH}")
        return 1

    meta = load_meta()
    problem = binding_problem(meta, corpus_laws())
    if problem:
        print(f"BLOCKED: {problem}")
        return 3

    docs = load_docs(CORPUS_PATH)
    if not docs:
        print('corpus not ingested. Run `python tasks.py ingest --law "…"` first.')
        return 2

    print("=" * 68)
    print("  arabic-legal-rag · adversarial probes")
    print("=" * 68)
    print(f"\ncorpus     : {len(docs)} articles · {meta.get('corpus_law', '?')}")
    print(f"cases      : {len(cases)}")
    print(f"k          : {PROBE_K}")
    print("\nNot a score. Each case asserts one failure mode — read "
          "evals/adversarial/meta.json first.\n")

    index = DenseIndex(docs, cache_path=CACHE_PATH)
    index.save()

    texts = {d["id"]: d["text"] for d in docs}
    limits = encoder_limits(index.encoder)
    if limits is None:
        print("BLOCKED: this encoder exposes no tokenizer or max_seq_length, so a "
              "tail probe cannot prove where the cut falls. Refusing to print a "
              "number that would be a guess.")
        return 5
    tokenizer, max_seq = limits
    if not offsets_available(tokenizer):
        print("BLOCKED: this tokenizer reports no character offsets, so the window "
              "boundary cannot be located. Every tail case would read as 'fits "
              "fine', which is the one wrong answer worse than no answer.")
        return 5
    cuts = {
        doc_id: visible_chars(tokenizer, max_seq, text) for doc_id, text in texts.items()
    }
    oversized = {k: v for k, v in cuts.items() if v is not None}
    beyond_first = sum(len(texts[k]) - v for k, v in oversized.items())
    corpus_chars = sum(len(t) for t in texts.values())
    print(f"encoder    : max_seq_length={max_seq} tokens")
    print(f"over one window : {len(oversized)} of {len(texts)} articles · "
          f"{beyond_first} chars past the first window "
          f"({100 * beyond_first / max(corpus_chars, 1):.1f}% of the corpus)")
    if oversized:
        listing = ", ".join(
            f"{k} (+{len(texts[k]) - v})" for k, v in sorted(oversized.items())
        )
        print(f"                  {listing}")
    # Whether that text is reachable at all is the difference between a probe
    # that is measuring a live bug and one that is confirming a fix.
    if index.windowed:
        print(f"index      : {len(index.rows)} windows over {len(texts)} articles — "
              "text past the first window IS embedded")
    else:
        print(f"index      : one vector per article — the {beyond_first} chars above are "
              "NOT embedded, and nothing downstream can tell")

    results: list[dict] = []
    skipped: list[tuple[Case, str]] = []
    for case in cases:
        trouble = check_case(case, texts, cuts)
        if trouble:
            skipped.append((case, trouble))
            continue
        runner = RUNNERS.get(case.probe)
        if runner is None:
            skipped.append((case, f"{INVALID}: unknown probe {case.probe!r}"))
            continue
        results.append(runner(case, index.search))

    by_probe: dict[str, list[dict]] = {}
    for r in results:
        by_probe.setdefault(r["probe"], []).append(r)

    for probe, rows in by_probe.items():
        print(f"\n  {probe}")
        print("  " + "-" * 66)
        if probe == "orthographic_variant":
            for r in rows:
                cells = "  ".join(f"{form}={_rank_cell(rank)}" for form, rank in r["ranks"])
                mark = "ok  " if r["passed"] else "FAIL"
                # Plain "->" on purpose: this prints to a Windows console whose
                # code page is cp1256 here, which has no U+2192 and raises on it.
                print(f"  {mark} {r['id']} (base {r['base']} -> {r['expected']})")
                print(f"       {cells}")
        else:
            for r in rows:
                mark = "ok  " if r["passed"] else "FAIL"
                line = f"  {mark} {r['id']}  {r['expected']}={_rank_cell(r['expected_rank'])}"
                if r["distractors"]:
                    line += "   distractor " + " ".join(
                        f"{d}={_rank_cell(rank)}" for d, rank in r["distractors"].items()
                    )
                print(line)
                if r["outranked_by"]:
                    print(f"       outranked by {', '.join(r['outranked_by'])}")
        failed = sum(1 for r in rows if not r["passed"])
        print(f"  {len(rows) - failed} of {len(rows)} passed")

    if skipped:
        print("\n  cases not scored — the case, not the system")
        print("  " + "-" * 66)
        for case, trouble in skipped:
            print(f"  {case.id}: {trouble}")

    if "--sweep" in argv:
        questions, question_errors = load_questions()
        if question_errors:
            for e in question_errors:
                print(f"  - {e}")
            return 1
        answerable = [q for q in questions if q.answerable]
        print("\n  orthographic sweep — every dev question, re-asked as typed")
        print("  " + "-" * 66)
        print(f"  {'spelling':16} {'recall@5':>9} {'MRR':>7} {'moved':>7} "
              f"{'left top5':>10} {'entered':>8}")
        sweep = sweep_orthography(answerable, index.search)
        for row in sweep:
            recall = f"{row['recall']:.3f}" if row["recall"] is not None else "—"
            mrr = f"{row['mrr']:.3f}" if row["mrr"] is not None else "—"
            counts = ("—", "—", "—") if row["form"] == "as written" else (
                str(row["moved"]), str(row["left_top_k"]), str(row["entered_top_k"]))
            print(f"  {row['form']:16} {recall:>9} {mrr:>7} {counts[0]:>7} "
                  f"{counts[1]:>10} {counts[2]:>8}")
        print(f"  {len(answerable)} answerable dev questions. `moved` counts questions whose "
              "ranks changed at all,")
        print("  which is the readable signal here — recall@5 on this sample moves in steps "
              f"of {1 / max(len(answerable), 1):.3f}.")

    total_failed = sum(1 for r in results if not r["passed"])
    print(f"\n  scored {len(results)} · passed {len(results) - total_failed} · "
          f"failed {total_failed} · not scored {len(skipped)}")
    print("\n  `#7!` means the article was found but outside k. A failure here is a "
          "reachable weakness, not a percentage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
