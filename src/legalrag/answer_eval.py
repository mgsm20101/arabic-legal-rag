"""End-to-end answer evaluation — PRD M2/B1 and M2/B2.

Retrieval is the dense configuration, because Run 2 measured it as the winner
(0.767 against BM25's 0.333) and because a generation number computed over a
retriever nobody chose is unreadable.

**B2 has a trap, and this is designed against it.** "Abstention accuracy on
``out_of_corpus`` >= 80%" is satisfied trivially by a model that abstains on
everything: it scores 100% and answers nothing. So the false-abstention rate on
*answerable* questions is printed next to it, always, and neither number is
quotable alone. The pair is the metric.

**B1 counts three failures apart** — fabricated, ungrounded, uncited — because
they say different things about where the system broke. See ``cite``.

One more number that costs nothing and says a lot: whether the answer cites the
article the eval set says it should. That is end-to-end correctness as far as
this project measures it, and it is bounded above by retrieval — the generator
cannot cite what the retriever never handed it.

Run: ``python tasks.py answer-eval``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .cite import audit
from .dense import CACHE_PATH, CORPUS_PATH, DenseIndex, load_docs
from .evaluate import binding_problem, corpus_laws, load_meta, load_questions
from .generate import DEFAULT_MODEL, Generator

TOP_K = 5

RUNS = Path("runs")
ROWS_PATH = RUNS / "answer_eval.json"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.1%}" if d else f"{n}/0 = -"


def article_number(doc_id: str) -> int | None:
    """``law-7`` -> 7. Issuance articles are numbered on their own sequence and
    are not part of the answerable set in batch one."""
    book, _, num = doc_id.partition("-")
    return int(num) if book == "law" and num.isdigit() else None


def run(questions, docs, generator: Generator, k: int = TOP_K) -> list[dict]:
    index = DenseIndex(docs, cache_path=CACHE_PATH)
    index.save()
    texts = {d["id"]: d["text"] for d in docs}
    corpus_numbers = {n for n in (article_number(d["id"]) for d in docs) if n}

    rows: list[dict] = []
    for q in questions:
        hits = index.search(q.question, k)
        articles = [
            {"number": article_number(h.id), "text": texts[h.id]}
            for h in hits
            if article_number(h.id) is not None
        ]
        t = time.perf_counter()
        ans = generator.answer(q.question, articles)
        elapsed = time.perf_counter() - t

        retrieved = {a["number"] for a in articles}
        result = audit(ans.text, corpus_numbers, retrieved,
                       context=[a["text"] for a in articles])
        expected = {n for n in (article_number(a) for a in q.expected_articles) if n}

        rows.append(
            {
                "id": q.id,
                "category": q.category,
                "answerable": q.answerable,
                "text": ans.text,
                "seconds": elapsed,
                "retrieved": sorted(retrieved),
                "expected": sorted(expected),
                "cited_expected": bool(expected & set(result["cited"])),
                **result,
            }
        )
        state = "ABSTAIN" if result["abstained"] else "answer "
        cited = result["cited"] or "-"
        print(f"  {q.id} [{q.category:13}] {elapsed:5.1f}s  {state}  cited {cited}",
              flush=True)
    return rows


def report(rows: list[dict]) -> int:
    answerable = [r for r in rows if r["answerable"]]
    out_of_corpus = [r for r in rows if not r["answerable"]]

    print("\n" + "=" * 68)
    print("  B1 - grounding: no answer may cite an article it was not given")
    print("=" * 68)
    scored = [r for r in answerable if not r["abstained"]]
    fabricated = [r for r in scored if r["fabricated"]]
    ungrounded = [r for r in scored if r["ungrounded"]]
    uncited = [r for r in scored if r["uncited"]]
    clean = [r for r in scored if r["grounded"]]

    print(f"  answered (of {len(answerable)} answerable)  : {len(scored)}")
    print(f"  fully grounded                     : {_pct(len(clean), len(scored))}")
    line = f"  cited an article not in the law    : {len(fabricated)}"
    if fabricated:
        line += "   " + str([(r["id"], r["fabricated"]) for r in fabricated])
    print(line)
    line = f"  cited a real article NOT retrieved : {len(ungrounded)}"
    if ungrounded:
        line += "   " + str([(r["id"], r["ungrounded"]) for r in ungrounded])
    print(line)
    line = f"  made a claim with no citation      : {len(uncited)}"
    if uncited:
        line += "   " + str([r["id"] for r in uncited])
    print(line)
    copied = [r for r in scored if r.get("copied")]
    line = f"  citation was inside copied statute : {len(copied)}"
    if copied:
        line += "   " + str([(r["id"], r["copied"]) for r in copied])
    print(line)
    if copied:
        print("      (the law citing itself in text the model reproduced -")
        print("       not the model grounding an answer; excluded from `cited`)")

    # PRD B1 reads «صفر إجابة **بلا استشهاد** بمادة موجودة فعلا في المتن» — zero
    # answers *without* a citation to an article that really exists. An answer
    # carrying no citation at all is a violation, not a neutral outcome.
    #
    # Getting this wrong is the same species of error as the B2 trap one
    # section below: a verdict of `not fabricated and not ungrounded` is passed
    # trivially by a model that never cites anything, and the first run of this
    # eval printed exactly that — PASS, on 12 of 14 answers with no citation.
    # Silence is not grounding.
    b1 = not fabricated and not ungrounded and not uncited
    verdict = "PASS" if b1 else "FAIL"
    print(f"\n  B1: {verdict} - every answer must carry a citation, it must resolve to a")
    print("      real article, and that article must have been retrieved. An answer")
    print("      with no citation fails: not citing is not the same as not lying.")

    print("\n" + "=" * 68)
    print("  B2 - abstention, with the false-abstention rate beside it")
    print("=" * 68)
    correct_abstain = [r for r in out_of_corpus if r["abstained"]]
    false_abstain = [r for r in answerable if r["abstained"]]
    print(f"  abstained on out_of_corpus   : {_pct(len(correct_abstain), len(out_of_corpus))}"
          "    (criterion: >= 80%)")
    print(f"  abstained on answerable      : {_pct(len(false_abstain), len(answerable))}"
          "    (abstaining on everything scores 100% above)")
    b2 = bool(out_of_corpus) and len(correct_abstain) / len(out_of_corpus) >= 0.80
    verdict = "PASS" if b2 else "FAIL"
    print(f"\n  B2: {verdict} - unreadable without the line above it.")

    print("\n" + "=" * 68)
    print("  beyond the criteria")
    print("=" * 68)
    hit = [r for r in scored if r["cited_expected"]]
    strict = [r for r in scored if r["strict"]]
    print(f"  cited the expected article         : {_pct(len(hit), len(scored))}"
          "    (bounded above by retrieval)")
    print(f"  used the requested bracket form    : {_pct(len(strict), len(scored))}")
    mean_s = sum(r["seconds"] for r in rows) / max(len(rows), 1)
    print(f"  mean seconds per answer            : {mean_s:.1f}")

    print("\nSmall sample. Read the caveats in EVAL.md before quoting any of this,")
    print("starting with the one that matters most: the corpus is not the gazette text.")
    return 0 if (b1 and b2) else 1


def main(argv: list[str] | None = None) -> int:
    argv = argv or []

    if "--report-only" in argv:
        if not ROWS_PATH.exists():
            print(f"no saved run at {ROWS_PATH}. Run `python tasks.py answer-eval` first.")
            return 2
        rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
        # Re-audit rather than replay the stored verdicts. The audit has been
        # wrong twice already — once on B1's wording, once on citations lifted
        # out of copied statute text — and each time the answers themselves
        # were still good. Recomputing from the text means a fixed checker
        # costs a second instead of another 45 minutes of CPU.
        docs = load_docs(CORPUS_PATH)
        by_number = {
            n: d["text"]
            for d in docs
            if (n := article_number(d["id"])) is not None
        }
        corpus_numbers = set(by_number)
        for r in rows:
            retrieved = set(r["retrieved"])
            r.update(audit(
                r["text"], corpus_numbers, retrieved,
                context=[by_number[n] for n in retrieved if n in by_number],
            ))
            r["cited_expected"] = bool(set(r["expected"]) & set(r["cited"]))
        print(f"re-audited {len(rows)} saved answers from {ROWS_PATH} "
              "(no model loaded)\n")
        return report(rows)

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
        print("corpus not ingested. Run `python tasks.py ingest --law ...` first.")
        return 2

    unverified = [q for q in questions if q.answerable and q.ref_status != "verified"]
    if unverified:
        print(
            f"BLOCKED: {len(unverified)} answerable questions are not `verified`. "
            "Run `python tasks.py verify-refs --write` first (ADR-005)."
        )
        return 4

    print("=" * 68)
    print("  arabic-legal-rag - answer evaluation - PRD M2/B1 + M2/B2")
    print("=" * 68)
    print(f"\ncorpus     : {len(docs)} articles - {meta.get('corpus_law', '?')}")
    n_ooc = sum(1 for q in questions if not q.answerable)
    print(f"questions  : {len(questions)} ({n_ooc} out_of_corpus)")
    print(f"retrieval  : dense, top-{TOP_K} (Run 2 winner)")
    print(f"generator  : {DEFAULT_MODEL}, local CPU, greedy\n")

    print(f"loading {DEFAULT_MODEL} ...", flush=True)
    t = time.perf_counter()
    generator = Generator()
    _ = generator.model
    print(f"  ready ({time.perf_counter() - t:.1f}s)\n", flush=True)

    rows = run(questions, docs, generator)

    # Persist before reporting. The first run of this eval cost 45 minutes of
    # CPU and kept none of the answers, so the one question worth asking after
    # it — *what did the model actually write?* — could not be answered without
    # paying for the whole run again. The report is cheap to recompute; the
    # generation is not.
    RUNS.mkdir(exist_ok=True)
    ROWS_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nanswers saved to {ROWS_PATH} ({len(rows)} rows)")

    return report(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
