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
import re
import sys
import time
from datetime import date
from pathlib import Path

from .cite import audit
from .dense import CACHE_PATH, CORPUS_PATH, DenseIndex, load_docs
from .evaluate import binding_problem, corpus_laws, load_meta, load_questions
from .generate import DEFAULT_MODEL, Generator, model_source, resolve_model
from .ollama import GeneratorUnavailable, run_metadata

TOP_K = 5

RUNS = Path("runs")
ROWS_PATH = RUNS / "answer_eval.json"


def rows_path(spec: str) -> Path:
    """Where a run's raw answers are saved.

    The default spec keeps Run 3's exact file name, so `--report-only` with
    no `--model` keeps reproducing that saved run. Any other spec gets its
    own file, named from the spec itself — otherwise a second model's run
    would silently overwrite the first's saved answers.
    """
    if spec == "hf:" + DEFAULT_MODEL:
        return ROWS_PATH
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", spec)
    return RUNS / f"answer_eval-{slug}.json"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.1%}" if d else f"{n}/0 = -"


def article_number(doc_id: str) -> int | None:
    """``law-7`` -> 7. Issuance articles are numbered on their own sequence and
    are not part of the answerable set in batch one."""
    book, _, num = doc_id.partition("-")
    return int(num) if book == "law" and num.isdigit() else None


def _stats_for(model, before: int) -> list[dict] | None:
    """Every call `model` made since `before` — a count of its `calls` list
    taken just before the question was asked — or None if this runtime
    keeps no such list at all (the hf: runtime is a plain function).

    Counting from a call-count snapshot, not from whether the question's
    articles were empty, is what stays correct once one question can make
    more than one call: Run 5 retries once on invalid JSON, so `calls`
    holds both the cut first attempt and the retry, where reading only
    `last_stats` would silently drop the first one. Must be called *after*
    the question's `generator.answer` returns, so `before` refers to a
    count taken strictly earlier.
    """
    calls = getattr(model, "calls", None)
    if calls is None:
        return None
    return calls[before:]


def _calls_of(row: dict) -> list[dict]:
    """A row's `stats`, normalized to a list of per-call dicts regardless of
    whether it was saved as Run 4's single dict or the list this module now
    writes — `report()` must keep reading Run 4's saved files unchanged."""
    s = row.get("stats")
    if s is None:
        return []
    return [s] if isinstance(s, dict) else s


def run(questions, docs, generator: Generator, model_spec: str, k: int = TOP_K,
        index: DenseIndex | None = None) -> list[dict]:
    if index is None:
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
        model = generator.model
        before = len(getattr(model, "calls", []))
        t = time.perf_counter()
        ans = generator.answer(q.question, articles)
        elapsed = time.perf_counter() - t
        stats = _stats_for(model, before)

        retrieved = {a["number"] for a in articles}
        result = audit(ans.text, corpus_numbers, retrieved,
                       context=[a["text"] for a in articles])
        expected = {n for n in (article_number(a) for a in q.expected_articles) if n}

        row = {
            "id": q.id,
            "category": q.category,
            "answerable": q.answerable,
            "model": model_spec,
            "text": ans.text,
            "seconds": elapsed,
            "retrieved": sorted(retrieved),
            "expected": sorted(expected),
            "cited_expected": bool(expected & set(result["cited"])),
            **result,
        }
        if stats is not None:
            row["stats"] = stats
        rows.append(row)
        state = "ABSTAIN" if result["abstained"] else "answer "
        cited = result["cited"] or "-"
        print(f"  {q.id} [{q.category:13}] {elapsed:5.1f}s  {state}  cited {cited}",
              flush=True)
    return rows


def _read_run_meta(model_spec: str) -> dict | None:
    """The sibling `.meta.json` an `ollama:` run wrote next to its rows, or
    None when there isn't one (an `hf:` run, or an `ollama:` run saved
    before this file existed) — absence is not an error here."""
    meta_path = RUNS / f"{rows_path(model_spec).stem}.meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        return None


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
    model_spec = rows[0]["model"] if rows and "model" in rows[0] else "hf:" + DEFAULT_MODEL
    print(f"  model                              : {model_spec}")
    hit = [r for r in scored if r["cited_expected"]]
    strict = [r for r in scored if r["strict"]]
    print(f"  cited the expected article         : {_pct(len(hit), len(scored))}"
          "    (bounded above by retrieval)")
    print(f"  used the requested bracket form    : {_pct(len(strict), len(scored))}")
    mean_s = sum(r["seconds"] for r in rows) / max(len(rows), 1)
    print(f"  mean seconds per answer            : {mean_s:.1f}")

    # Only an `ollama:` run ever carries `stats` at all (the hf: runtime
    # exposes no such attribute — see `_stats_for`, above). Within one such
    # run, a question answered without calling the model (no articles
    # retrieved) has `stats == []`, which is falsy; the rest have at least
    # one call. `_calls_of` normalizes Run 4's single dict and the list this
    # module now writes into the same shape, so aggregation below does not
    # need to know which format a given row was saved in.
    stat_rows = [r for r in rows if r.get("stats")]
    all_calls = [c for r in stat_rows for c in _calls_of(r)]
    if all_calls:
        if any(c.get("load_s") is not None for c in all_calls):
            adjusted = [
                r["seconds"] - sum(c.get("load_s") or 0 for c in _calls_of(r))
                for r in rows
            ]
            mean_adj = sum(adjusted) / max(len(adjusted), 1)
            print(f"  mean seconds excl. load            : {mean_adj:.1f}")

        with_rate = [c for c in all_calls
                     if c.get("output_tokens") is not None and c.get("output_s")]
        if with_rate:
            tokens = sum(c["output_tokens"] for c in with_rate)
            secs = sum(c["output_s"] for c in with_rate)
            print(f"  mean output tokens/s               : {tokens / secs:.1f}")
        n_truncated = sum(1 for c in all_calls if c.get("truncated"))
        n_cut = sum(1 for c in all_calls if c.get("cut"))
        print(f"  prompts truncated by num_ctx       : {n_truncated}")
        print(f"  answers cut by the token cap       : {n_cut}")

        # More than one call for a question is a retry (Run 5 retries once
        # on invalid JSON) — worth surfacing on its own, since a model that
        # retries often is slower than its mean seconds alone would suggest.
        retries = sum(max(0, len(_calls_of(r)) - 1) for r in stat_rows)
        print(f"  model calls                        : {len(all_calls)}")
        print(f"  retries (more than 1 call)         : {retries}")

    meta = _read_run_meta(model_spec)
    if meta is not None:
        if meta.get("gpu_share") is not None:
            print(f"  GPU share                          : {meta['gpu_share']:.0%}")
        if meta.get("quantization"):
            print(f"  quantization                       : {meta['quantization']}")
        if meta.get("digest"):
            print(f"  digest                             : {meta['digest'][:12]}")

    print("\nSmall sample. Read the caveats in EVAL.md before quoting any of this,")
    print("starting with the one that matters most: the corpus is not the gazette text.")
    return 0 if (b1 and b2) else 1


def _weights_line(model_spec: str) -> str | None:
    """The `weights` header line for an `hf:` spec, or None when there is
    nothing sensible to show.

    A spec with no repo name after `hf:` (`"hf:"`, or all whitespace) is
    invalid — `resolve_model` rejects it — but that happens later, in the
    timed "loading ..." block below. Calling `model_source` on it here first
    would print a blank `weights    :` line before the real error, and
    `Path("").is_dir()` resolving to the current directory makes that not
    even reliably blank. Validate before resolving, not after.
    """
    if not model_spec.startswith("hf:"):
        return None
    repo = model_spec.partition(":")[2]
    if not repo.strip():
        return None
    return f"weights    : {model_source(repo)}"


def _write_ollama_meta(model_spec: str, chat, rows_file: Path) -> Path:
    """Record what actually served an `ollama:` run alongside its answers.

    `run_metadata` looks the loaded model up by name from `/api/ps` rather
    than trusting anything the request itself claimed: whether a candidate
    model fit on a 4 GB card, and which build's digest answered, are facts
    about the server at run time, not something the client can assert.
    """
    meta = run_metadata(chat)
    meta["model"] = model_spec  # the full spec, consistent with rows' "model"
    meta["date"] = date.today().isoformat()
    meta_path = RUNS / f"{rows_file.stem}.meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"model metadata saved to {meta_path}")
    return meta_path


def main(argv: list[str] | None = None) -> int:
    argv = argv or []

    model_spec = "hf:" + DEFAULT_MODEL
    if "--model" in argv:
        i = argv.index("--model")
        if i + 1 >= len(argv):
            print("--model requires a value, e.g. --model ollama:qwen3:4b")
            return 2
        model_spec = argv[i + 1]
    rows_file = rows_path(model_spec)

    if "--report-only" in argv:
        if not rows_file.exists():
            print(f"no saved run at {rows_file}. "
                  f"Run `python tasks.py answer-eval --model {model_spec}` first.")
            return 2
        rows = json.loads(rows_file.read_text(encoding="utf-8"))
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
        print(f"re-audited {len(rows)} saved answers from {rows_file} "
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
    print(f"generator  : {model_spec}")
    weights_line = _weights_line(model_spec)
    if weights_line is not None:
        print(weights_line)
    print()

    print(f"loading {model_spec} ...", flush=True)
    t = time.perf_counter()
    try:
        model = resolve_model(model_spec)
    except ValueError as e:
        # A bad --model spec, caught here specifically rather than with the
        # broader try below: a ValueError from deep inside `run()` would be a
        # real bug, and reporting it as "bad model spec" would hide that.
        print(str(e))
        return 2

    try:
        generator = Generator(model=model)
        print(f"  ready ({time.perf_counter() - t:.1f}s)\n", flush=True)
        rows = run(questions, docs, generator, model_spec)
    except GeneratorUnavailable as e:
        # No traceback: this is an environment problem (server not running,
        # model not pulled), not a bug, and the message already says what to
        # do about it.
        print(str(e))
        return 5

    # Persist before reporting. The first run of this eval cost 45 minutes of
    # CPU and kept none of the answers, so the one question worth asking after
    # it — *what did the model actually write?* — could not be answered without
    # paying for the whole run again. The report is cheap to recompute; the
    # generation is not.
    RUNS.mkdir(exist_ok=True)
    rows_file.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nanswers saved to {rows_file} ({len(rows)} rows)")

    if model_spec.startswith("ollama:"):
        _write_ollama_meta(model_spec, model, rows_file)

    return report(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
