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

from .answer_report import report
from .cite import audit
from .dense import CACHE_PATH, CORPUS_PATH, DenseIndex, load_docs
from .evaluate import binding_problem, corpus_laws, load_meta, load_questions
from .generate import DEFAULT_MODEL, Generator, model_source, parse_model_spec, resolve_model
from .ollama import GeneratorUnavailable, run_metadata

TOP_K = 5

RUNS = Path("runs")
ROWS_PATH = RUNS / "answer_eval.json"
DEFAULT_SPEC = "hf:" + DEFAULT_MODEL  # Run 3's model, as a --model spec


def rows_path(spec: str) -> Path:
    """Where a run's raw answers are saved.

    The default spec keeps Run 3's exact file name, so `--report-only` with
    no `--model` keeps reproducing that saved run. Any other spec gets its
    own file, named from the spec itself — otherwise a second model's run
    would silently overwrite the first's saved answers.
    """
    if spec == DEFAULT_SPEC:
        return ROWS_PATH
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", spec)
    return RUNS / f"answer_eval-{slug}.json"


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


def run(questions, docs, generator: Generator, model_spec: str, k: int = TOP_K,
        index: DenseIndex | None = None) -> list[dict]:
    if index is None:
        index = DenseIndex(docs, cache_path=CACHE_PATH)
        index.save()
    texts = {d["id"]: d["text"] for d in docs}
    corpus_numbers = {n for n in (article_number(d["id"]) for d in docs) if n}

    # Provenance belongs on the row, not only in the console header
    # (`_weights_line`): a saved run should be able to answer "what did
    # this actually load from?" on its own. Same for every row in one run,
    # so it is resolved once rather than per question.
    prefix, name = parse_model_spec(model_spec)
    weights = model_source(name) if prefix == "hf" else None

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
        if weights is not None:
            row["weights"] = weights
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


def _meta_for_rows(rows: list[dict]) -> dict | None:
    """Which `.meta.json` belongs beside a report of `rows` — resolved from
    the model *actually recorded in the rows themselves*, not necessarily
    whatever `--model` the CLI was given: under a reported mismatch
    (`_rows_model_mismatch`) `--report-only` still reports on what is
    actually saved, and the meta shown must describe the same run.

    `report()`/`report_claims()` used to resolve this themselves; pulled out
    here so `answer_report.py` never needs to know how a model spec maps to
    a file on disk (`rows_path` lives in this module, not that one).
    """
    model_spec = rows[0]["model"] if rows and "model" in rows[0] else DEFAULT_SPEC
    return _read_run_meta(model_spec)


def _weights_line(model_spec: str) -> str | None:
    """The `weights` header line for an `hf:` spec, or None when there is
    nothing sensible to show.

    A spec with no repo name after `hf:` (`"hf:"`, or all whitespace) is
    invalid — `parse_model_spec` rejects it, `main` now checks that before
    this is ever called — but this stays defensive on its own: calling
    `model_source` on such a spec would print a blank `weights    :` line,
    and `Path("").is_dir()` resolving to the current directory makes that
    not even reliably blank.
    """
    if not model_spec.startswith("hf:"):
        return None
    try:
        _, repo = parse_model_spec(model_spec)
    except ValueError:
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


def _rows_model_mismatch(rows: list[dict], model_spec: str) -> str | None:
    """The model recorded in `rows`'s first row, when it differs from
    `model_spec` — None when they agree, `rows` is empty, or `rows` predates
    the "model" field entirely (Run 3's saved shape, before this existed).
    """
    if not rows:
        return None
    existing = rows[0].get("model")
    if existing is None or existing == model_spec:
        return None
    return existing


def _check_rows_collision(rows_file: Path, model_spec: str) -> str | None:
    """None if it is safe to (over)write `rows_file` for `model_spec` — a
    message naming the model actually saved there otherwise.

    Two different specs can slugify to the same file name: `rows_path` maps
    every character outside [A-Za-z0-9._-] to '-', so a spec built from one
    kind of separator can collide with one built from another. Silently
    overwriting a previous run's saved answers under those circumstances
    would corrupt a different run's history.
    """
    if not rows_file.exists():
        return None
    try:
        existing = json.loads(rows_file.read_text(encoding="utf-8"))
    except ValueError:
        return None
    mismatch = _rows_model_mismatch(existing, model_spec)
    if mismatch is None:
        return None
    return (
        f"{rows_file} already holds answers for {mismatch!r}, not "
        f"{model_spec!r}. Refusing to overwrite — two different --model "
        f"specs must not share a rows file."
    )


def _reaudit(rows: list[dict], docs: list[dict]) -> list[dict]:
    """Recompute every row's citation verdict from its saved text against
    the current corpus, in place — not a replay of the stored verdicts.

    The audit itself has been wrong twice already: once on B1's wording,
    once on citations lifted out of copied statute text — and each time the
    saved answers were still good, only the check was not. Recomputing from
    the text means a fixed checker costs a second; the generation that
    produced the text can cost up to 45 minutes of CPU.
    """
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
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = argv or []

    model_spec = DEFAULT_SPEC
    if "--model" in argv:
        i = argv.index("--model")
        if i + 1 >= len(argv):
            print("--model requires a value, e.g. --model ollama:qwen3:4b")
            return 2
        model_spec = argv[i + 1]

    try:
        parse_model_spec(model_spec)
    except ValueError as e:
        # Validated here, before anything else loads: a typo in --model
        # should not cost a corpus load, a question load, or a wasted run.
        # `resolve_model` (below) re-parses the same spec to dispatch, but
        # by then it is known good, so it cannot raise ValueError again.
        print(str(e))
        return 2

    rows_file = rows_path(model_spec)

    if "--report-only" in argv:
        if not rows_file.exists():
            print(f"no saved run at {rows_file}. "
                  f"Run `python tasks.py answer-eval --model {model_spec}` first.")
            return 2
        rows = json.loads(rows_file.read_text(encoding="utf-8"))
        mismatch = _rows_model_mismatch(rows, model_spec)
        if mismatch is not None:
            # A warning, not a refusal: --report-only changes nothing on
            # disk, so there is nothing to protect by blocking it — but
            # silently reporting another model's answers as this one's
            # would be misleading.
            print(f"WARNING: {rows_file} holds answers for {mismatch!r}, not "
                  f"{model_spec!r} — reporting on what is actually saved there.")
        rows = _reaudit(rows, load_docs(CORPUS_PATH))
        print(f"re-audited {len(rows)} saved answers from {rows_file} "
              "(no model loaded)\n")
        return report(rows, meta=_meta_for_rows(rows))

    collision = _check_rows_collision(rows_file, model_spec)
    if collision is not None:
        print(collision)
        return 2

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
        generator = Generator(model=model)
        print(f"  ready ({time.perf_counter() - t:.1f}s)\n", flush=True)
        rows = run(questions, docs, generator, model_spec)
    except GeneratorUnavailable as e:
        # No traceback: this is an environment problem (missing dependency,
        # bad checkpoint, unreachable Ollama server, model not pulled), not
        # a bug, and the message already says what to do about it.
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

    return report(rows, meta=_meta_for_rows(rows))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
