"""End-to-end answer evaluation over dense top-5 retrieval.

Three answer contracts, chosen with ``--contract``:

``gated``  relevance step, then claims JSON, then `cite.gate`: what the app runs.
``json``   claims JSON and `cite.gate`, no relevance step.
``text``   free text with inline citations, checked by `cite.audit`. Legacy:
           `generate.Generator` and this contract are kept to reproduce E3.

Abstention on out-of-corpus questions is always printed beside the
false-abstention rate on answerable ones: a model that abstains on everything
scores 100% on the first. Fabricated, ungrounded and uncited are counted apart
(see `cite`). Raw rows go to ``runs/``; `evals/answer_eval_result.py` promotes
them to the registry.

Run: ``python tasks.py answer-eval``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from .answer_report import report, report_claims
from .cite import audit, gate
from .claims import ClaimsGenerator, build_generators
from .dense import CACHE_PATH, CORPUS_PATH, DenseIndex, corpus_fingerprint, load_docs
from .evaluate import (
    binding_problem,
    corpus_laws,
    load_meta,
    load_questions,
    question_set_fingerprint,
)
from .envcheck import observed as observed_environment
from .generate import DEFAULT_MODEL, Generator, model_source, parse_model_spec, resolve_model
from .ollama import NUM_GPU_ENV, GeneratorUnavailable, run_metadata
from .provenance import commit_state

TOP_K = 5

RUNS = Path("runs")
ROWS_PATH = RUNS / "answer_eval.json"
DEFAULT_SPEC = "hf:" + DEFAULT_MODEL  # Run 3's model, as a --model spec

# text: free text (legacy, EVAL.md Runs 3-4); json: claims + gate (Run 5);
# gated: relevance step + claims + gate (Run 6).
CONTRACTS = ("text", "json", "gated")


def rows_path(spec: str, contract: str = "text") -> Path:
    """Where a run's raw answers are saved: one file per (spec, contract).

    The default spec under "text" keeps Run 3's file name so `--report-only`
    still reproduces it; other contracts get a `-{contract}` suffix.
    """
    if contract != "text":
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", spec)
        return RUNS / f"answer_eval-{slug}-{contract}.json"
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
    """Every call `model` made since the `before` count, or None if it keeps no `calls` list.

    Counted from a snapshot, so a retried question keeps both calls.
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
    fingerprint = corpus_fingerprint(docs)

    # Where the weights came from, stamped on every row.
    prefix, name = parse_model_spec(model_spec)
    weights = model_source(name) if prefix == "hf" else None

    rows: list[dict] = []
    for q in questions:
        hits = index.search(q.question, k)
        articles = [
            {"id": h.id, "number": article_number(h.id), "text": texts[h.id]}
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
            "retrieved_ids": [a["id"] for a in articles],
            "expected": sorted(expected),
            "cited_expected": bool(expected & set(result["cited"])),
            "corpus_fingerprint": fingerprint,
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


def run_claims(questions, docs, generator: ClaimsGenerator, model_spec: str,
               k: int = TOP_K, index: DenseIndex | None = None,
               contract: str = "json") -> list[dict]:
    """`run()` for the claims contracts: same retrieval, then `cite.gate` on each answer.

    Call stats come from `ans.calls`, which include the relevance model's calls.
    `contract` ("json" or "gated") is stamped on each row; the logic is the same.
    """
    if index is None:
        index = DenseIndex(docs, cache_path=CACHE_PATH)
        index.save()
    texts = {d["id"]: d["text"] for d in docs}
    fingerprint = corpus_fingerprint(docs)

    rows: list[dict] = []
    for q in questions:
        hits = index.search(q.question, k)
        articles = [
            {"id": h.id, "number": article_number(h.id), "text": texts[h.id]}
            for h in hits
            if article_number(h.id) is not None
        ]
        source_numbers = [a["number"] for a in articles]
        source_texts = [a["text"] for a in articles]
        source_ids = [a["id"] for a in articles]

        t = time.perf_counter()
        ans = generator.answer(q.question, source_texts)
        elapsed = time.perf_counter() - t

        expected = {n for n in (article_number(a) for a in q.expected_articles) if n}
        gate_result = gate(ans.parsed, source_numbers, source_texts)

        row = {
            "id": q.id,
            "category": q.category,
            "answerable": q.answerable,
            "model": model_spec,
            "contract": contract,
            "question": q.question,
            "source_numbers": source_numbers,
            "source_ids": source_ids,
            "expected": sorted(expected),
            "raw": ans.raw,
            "parsed": ans.parsed,
            "attempts": ans.attempts,
            "schema_failure": ans.schema_failure,
            "seconds": elapsed,
            "gate": gate_result,
            "corpus_fingerprint": fingerprint,
            "relevance": ans.relevance,
            "abstain_reason": ans.abstain_reason,
            "calls": ans.calls,
        }
        rows.append(row)

        n_claims = len(gate_result["kept"]) + len(gate_result["dropped"])
        print(f"  {q.id} [{q.category:13}] {elapsed:5.1f}s  "
              f"{gate_result['status']:9}  kept {len(gate_result['kept'])}/{n_claims}",
              flush=True)
    return rows


def _read_run_meta(model_spec: str, contract: str = "text") -> dict | None:
    """The sibling `.meta.json` an `ollama:` run wrote next to its rows, or
    None when there isn't one (an `hf:` run, or an `ollama:` run saved
    before this file existed) — absence is not an error here."""
    meta_path = RUNS / f"{rows_path(model_spec, contract).stem}.meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _meta_for_rows(rows: list[dict], contract: str = "text") -> dict | None:
    """The `.meta.json` for the model recorded in `rows`, which may differ from `--model`."""
    model_spec = rows[0]["model"] if rows and "model" in rows[0] else DEFAULT_SPEC
    return _read_run_meta(model_spec, contract)


def _weights_line(model_spec: str) -> str | None:
    """The `weights` header line for a valid `hf:` spec, else None."""
    if not model_spec.startswith("hf:"):
        return None
    try:
        _, repo = parse_model_spec(model_spec)
    except ValueError:
        return None
    return f"weights    : {model_source(repo)}"


def _write_run_meta(
    model_spec: str, rows_file: Path, questions, ollama_meta: dict | None = None
) -> Path:
    """Record what served the run and which eval set it scored.

    `ollama_meta` is what the server reported at run time (`run_metadata`).
    The question-set fingerprint lets the promoter refuse rows generated
    against an older question set.
    """
    meta = dict(ollama_meta or {})
    meta["model"] = model_spec  # the full spec, consistent with rows' "model"
    meta["date"] = date.today().isoformat()
    # This process's interpreter: the promoter runs later, from another one.
    meta["runtime"] = observed_environment()
    meta.update(commit_state())  # recorded now: the promoter runs later, maybe at another commit
    meta["question_set"] = {
        "fingerprint": question_set_fingerprint(questions),
        "n": len(questions),
        "ids": [q.id for q in sorted(questions, key=lambda q: q.id)],
    }
    meta_path = RUNS / f"{rows_file.stem}.meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"run metadata saved to {meta_path}")
    return meta_path


def _rows_model_mismatch(rows: list[dict], model_spec: str) -> str | None:
    """The model recorded in `rows`, when it differs from `model_spec`; else None."""
    if not rows:
        return None
    existing = rows[0].get("model")
    if existing is None or existing == model_spec:
        return None
    return existing


def _rows_contract_mismatch(rows: list[dict], contract: str) -> str | None:
    """The contract recorded in `rows` (default "text"), when it differs; else None."""
    if not rows:
        return None
    existing = rows[0].get("contract", "text")
    if existing == contract:
        return None
    return existing


def _check_rows_collision(rows_file: Path, model_spec: str, contract: str = "text") -> str | None:
    """None if it is safe to (over)write `rows_file` for `(model_spec,
    contract)` — a message naming what is actually saved there otherwise.

    Different specs can slugify to the same file name; overwriting another
    run's answers would corrupt its history, so model and contract are both checked.
    """
    if not rows_file.exists():
        return None
    try:
        existing = json.loads(rows_file.read_text(encoding="utf-8"))
    except ValueError:
        return None
    model_mismatch = _rows_model_mismatch(existing, model_spec)
    contract_mismatch = _rows_contract_mismatch(existing, contract)
    if model_mismatch is None and contract_mismatch is None:
        return None
    holds = []
    if model_mismatch is not None:
        holds.append(f"model {model_mismatch!r}")
    if contract_mismatch is not None:
        holds.append(f"contract {contract_mismatch!r}")
    return (
        f"{rows_file} already holds answers for " + " and ".join(holds) +
        f", not model {model_spec!r} / contract {contract!r}. Refusing to "
        f"overwrite — two different runs must not share a rows file."
    )


def _reaudit(rows: list[dict], docs: list[dict]) -> list[dict]:
    """New rows with the citation verdict recomputed from saved text, no model needed.

    Raises ValueError if a row retrieved an article this corpus lacks: the
    corpus is not the one the run saw.
    """
    by_number = {
        n: d["text"]
        for d in docs
        if (n := article_number(d["id"])) is not None
    }
    corpus_numbers = set(by_number)
    new_rows = []
    for r in rows:
        retrieved = set(r["retrieved"])
        missing = sorted(retrieved - corpus_numbers)
        if missing:
            raise ValueError(
                f"row {r.get('id')!r} retrieved article(s) {missing} that "
                "the loaded corpus does not contain — this corpus does not "
                "match the one this run saw."
            )
        result = audit(
            r["text"], corpus_numbers, retrieved,
            context=[by_number[n] for n in retrieved],
        )
        new_row = {**r, **result}
        new_row["cited_expected"] = bool(set(r["expected"]) & set(result["cited"]))
        new_rows.append(new_row)
    return new_rows


def _regate(rows: list[dict], docs: list[dict]) -> list[dict]:
    """New claims rows with `gate` re-applied to the saved answers, no model needed.

    Source text comes from the current corpus by article number. A number the
    corpus lacks raises ValueError rather than gating against empty text.
    """
    by_number = {
        n: d["text"]
        for d in docs
        if (n := article_number(d["id"])) is not None
    }
    new_rows = []
    for r in rows:
        source_numbers = r["source_numbers"]
        missing = sorted({n for n in source_numbers if n is not None and n not in by_number})
        if missing:
            raise ValueError(
                f"row {r.get('id')!r} names article(s) {missing} that the "
                "loaded corpus does not contain — this corpus does not "
                "match the one this run saw."
            )
        source_texts = [("" if n is None else by_number[n]) for n in source_numbers]
        new_rows.append({**r, "gate": gate(r["parsed"], source_numbers, source_texts)})
    return new_rows


@dataclass(frozen=True)
class _Contract:
    """Per-contract hooks: build the generator, run it, re-score saved rows, report."""
    build: Callable[[str], object]
    run: Callable[[list, list[dict], object, str], list[dict]]
    rescore: Callable[[list[dict], list[dict]], list[dict]]
    report: Callable[[list[dict], dict | None], int]


def _run_text(questions, docs, generator: Generator, model_spec: str) -> list[dict]:
    return run(questions, docs, generator, model_spec)


def _run_json(questions, docs, generator: ClaimsGenerator, model_spec: str) -> list[dict]:
    return run_claims(questions, docs, generator, model_spec, contract="json")


def _run_gated(questions, docs, generator: ClaimsGenerator, model_spec: str) -> list[dict]:
    return run_claims(questions, docs, generator, model_spec, contract="gated")


def _report_gated(rows: list[dict], meta: dict | None) -> int:
    # Refuses a verdict when no row shows the relevance step ran.
    return report_claims(
        rows, meta=meta, contract_name="Run 6", pre_registration_commit="d3f39c3",
        expect_relevance=True)


CONTRACT_TABLE: dict[str, _Contract] = {
    "text": _Contract(
        build=lambda spec: Generator(model=resolve_model(spec)),
        run=_run_text, rescore=_reaudit, report=report,
    ),
    "json": _Contract(
        build=lambda spec: build_generators(spec, "json"),
        run=_run_json, rescore=_regate, report=report_claims,
    ),
    "gated": _Contract(
        build=lambda spec: build_generators(spec, "gated"),
        run=_run_gated, rescore=_regate, report=_report_gated,
    ),
}


class _ArgumentParser(argparse.ArgumentParser):
    """An ArgumentParser whose errors go to stdout, like every other message here."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stdout)
        print(f"{self.prog}: error: {message}")
        raise SystemExit(2)


def _build_arg_parser() -> _ArgumentParser:
    parser = _ArgumentParser(
        prog="answer-eval",
        description="End-to-end answer evaluation — PRD M2/B1 + M2/B2.",
        # A typo must not expand to --overwrite.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model", default=DEFAULT_SPEC,
        help=f"a model spec, e.g. ollama:gemma3:4b (default: {DEFAULT_SPEC})",
    )
    parser.add_argument(
        "--contract", choices=list(CONTRACTS), default="text",
        help="the answer contract to use",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="re-score a previously saved run; no model is loaded",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="let a full run replace an existing rows file",
    )
    parser.add_argument(
        "--num-gpu", type=int, default=None, metavar="LAYERS",
        help="pin the number of layers Ollama offloads to the GPU "
             "(default: Ollama decides per server session)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else []

    parser = _build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # Usage and error already printed; an unknown flag must not start a run.
        return e.code if isinstance(e.code, int) else 2

    model_spec = args.model
    contract = args.contract  # already restricted to CONTRACTS by `choices=`

    try:
        parse_model_spec(model_spec)
    except ValueError as e:
        # Before anything loads: a typo should cost nothing.
        print(str(e))
        return 2

    if contract != "text" and not model_spec.startswith("ollama:"):
        # Schema-constrained decoding needs Ollama; checked before anything loads.
        print(f"--contract {contract} needs an ollama: model (got {model_spec!r}) — "
              "schema-constrained decoding is not available on the "
              "transformers path here")
        return 2

    if args.num_gpu is not None:
        if not model_spec.startswith("ollama:"):
            print(f"--num-gpu needs an ollama: model (got {model_spec!r})")
            return 2
        if args.num_gpu < 0:
            print(f"--num-gpu must be >= 0 (got {args.num_gpu})")
            return 2
        # Via the environment, so every chat this run builds carries the pin.
        os.environ[NUM_GPU_ENV] = str(args.num_gpu)

    rows_file = rows_path(model_spec, contract)

    if args.report_only:
        if not rows_file.exists():
            print(f"no saved run at {rows_file}. "
                  f"Run `python tasks.py answer-eval --model {model_spec} "
                  f"--contract {contract}` first.")
            return 2
        rows = json.loads(rows_file.read_text(encoding="utf-8"))

        contract_mismatch = _rows_contract_mismatch(rows, contract)
        if contract_mismatch is not None:
            # Refused: rows of another contract lack the fields re-scoring needs.
            print(f"{rows_file} holds {contract_mismatch!r}-contract rows, "
                  f"not {contract!r} — re-run with "
                  f"--contract {contract_mismatch} instead.")
            return 2

        mismatch = _rows_model_mismatch(rows, model_spec)
        if mismatch is not None:
            # Warned, not refused: --report-only writes nothing.
            print(f"WARNING: {rows_file} holds answers for {mismatch!r}, not "
                  f"{model_spec!r} — reporting on what is actually saved there.")

        docs = load_docs(CORPUS_PATH)
        if not docs:
            # Re-scoring against no corpus once changed a published count silently.
            print("corpus not ingested — cannot re-score against nothing. "
                  "Run `python tasks.py ingest --law ...` first.")
            return 2

        saved_fp = rows[0].get("corpus_fingerprint") if rows else None
        if saved_fp is None:
            # Runs 3-5 predate fingerprints; they must stay reportable.
            print(f"WARNING: {rows_file} predates corpus fingerprints — the "
                  "saved answers cannot be verified against this corpus. "
                  "Proceeding anyway.")
        else:
            current_fp = corpus_fingerprint(docs)
            if saved_fp != current_fp:
                print(f"{rows_file} was generated against a different corpus "
                      f"(fingerprint {saved_fp[:12]} != {current_fp[:12]}). "
                      "Refusing to re-score against a corpus it never saw — "
                      "re-run the full evaluation instead.")
                return 2

        handlers = CONTRACT_TABLE[contract]
        try:
            # Relevance decisions are saved on each row and never re-run.
            rows = handlers.rescore(rows, docs)
            verb = "re-audited" if contract == "text" else "re-gated"
            print(f"{verb} {len(rows)} saved answers from {rows_file} "
                  "(no model loaded)\n")
            return handlers.report(rows, _meta_for_rows(rows, contract))
        except ValueError as e:
            # A corpus mismatch in rows older than the fingerprint check.
            print(f"cannot re-score {rows_file} against the loaded corpus: {e}")
            return 2

    collision = _check_rows_collision(rows_file, model_spec, contract)
    if collision is not None:
        print(collision)
        return 2

    if rows_file.exists() and not args.overwrite:
        # The rows in runs/ are the only record of a measurement.
        print(f"{rows_file} already holds a saved run. The rows in runs/ "
              "are the only record of that measurement — pass --overwrite "
              "to replace it.")
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
    print(f"contract   : {contract}")
    weights_line = _weights_line(model_spec)
    if weights_line is not None:
        print(weights_line)
    print()

    handlers = CONTRACT_TABLE[contract]

    print(f"loading {model_spec} ...", flush=True)
    t = time.perf_counter()
    try:
        generator = handlers.build(model_spec)
        print(f"  ready ({time.perf_counter() - t:.1f}s)\n", flush=True)
        rows = handlers.run(questions, docs, generator, model_spec)
    except GeneratorUnavailable as e:
        # An environment problem, not a bug: the message says what to do.
        print(str(e))
        return 5

    # Saved before reporting: generation is expensive, the report is not.
    RUNS.mkdir(exist_ok=True)
    rows_file.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nanswers saved to {rows_file} ({len(rows)} rows)")

    # Written for every backend: its question-set fingerprint guards promotion.
    ollama_meta = None
    if model_spec.startswith("ollama:"):
        # The relevance chat uses the same server and model, so one record covers both.
        ollama_meta = run_metadata(generator.model)
    _write_run_meta(model_spec, rows_file, questions, ollama_meta)

    return handlers.report(rows, _meta_for_rows(rows, contract))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
