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

import argparse
import json
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
from .evaluate import binding_problem, corpus_laws, load_meta, load_questions
from .generate import DEFAULT_MODEL, Generator, model_source, parse_model_spec, resolve_model
from .ollama import GeneratorUnavailable, run_metadata

TOP_K = 5

RUNS = Path("runs")
ROWS_PATH = RUNS / "answer_eval.json"
DEFAULT_SPEC = "hf:" + DEFAULT_MODEL  # Run 3's model, as a --model spec

# "text": Run 3/4's free text. "json": Run 5's claims + gate. "gated": Run 6
# — the same claims contract behind a yes/no relevance step (EVAL.md,
# "Run 6"). Order matters only for --help/argparse's choices listing.
CONTRACTS = ("text", "json", "gated")


def rows_path(spec: str, contract: str = "text") -> Path:
    """Where a run's raw answers are saved.

    The default spec (under the "text" contract, the only one it can ever
    appear under — see `main`) keeps Run 3's exact file name, so
    `--report-only` with no `--model`/`--contract` keeps reproducing that
    saved run. Any other (spec, contract) pair gets its own file, named
    from the spec itself — otherwise a second run would silently overwrite
    a previous one's saved answers. Any non-"text" contract gets a
    `-{contract}` suffix, so `contract="json"` (Run 5) keeps its established
    `-json` suffix exactly, and `contract="gated"` (Run 6) gets its own
    `-gated` suffix that cannot collide with either.
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
    fingerprint = corpus_fingerprint(docs)

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
    """The claims-JSON sibling of `run()` — Run 5's contract (EVAL.md,
    commit ecd37f8) and, when `generator` carries a `relevance_model`, Run
    6's gated version of it (EVAL.md, "Run 6") — the same retrieval as
    `run()` (dense top-k, law articles only, kept in rank order); the answer
    shape and the model-free check applied to it are what differ — `cite.gate`
    runs immediately, so a saved row already carries its own verdict the way
    a text-contract row's `cited`/`fabricated`/... do (`audit`, called
    inside `run()`).

    Per-call stats come from the answer object (`ans.calls`), already
    stage-tagged ("relevance" / "claims") by `ClaimsGenerator.answer` —
    not from `generator.model.calls`, which would only ever see the claims
    model's own calls and silently miss every relevance call Run 6 makes
    through a SEPARATE model object.

    `contract` is stamped onto every row as-is ("json" or "gated") — this
    function's own logic never branches on it; only which fields a saved
    row carries downstream (`relevance`, `abstain_reason`) actually differs,
    and those come from `ans` regardless of which contract produced them
    (`None` under Run 5's contract, since `ClaimsGenerator.answer` returns
    exactly that when it was built with no `relevance_model`).
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
    return _read_run_meta(model_spec, contract)


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


def _rows_contract_mismatch(rows: list[dict], contract: str) -> str | None:
    """Analogous to `_rows_model_mismatch`, for `contract` — rows saved
    before `--contract` existed carry no such field at all, and count as
    `"text"`, the only contract there was then."""
    if not rows:
        return None
    existing = rows[0].get("contract", "text")
    if existing == contract:
        return None
    return existing


def _check_rows_collision(rows_file: Path, model_spec: str, contract: str = "text") -> str | None:
    """None if it is safe to (over)write `rows_file` for `(model_spec,
    contract)` — a message naming what is actually saved there otherwise.

    Two different specs can slugify to the same file name: `rows_path` maps
    every character outside [A-Za-z0-9._-] to '-', so a spec built from one
    kind of separator can collide with one built from another — and, since
    `contract="json"` only changes the file name by a fixed `-json` suffix,
    a `text` run of some model literally named `...-json` could in
    principle land on the same path as a `json` run of a different model.
    Silently overwriting a previous run's saved answers under either
    circumstance would corrupt a different run's history, so both fields
    are checked, not just the model.
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
    """Recompute every row's citation verdict from its saved text against
    the current corpus — a new list of new row dicts, never a mutation of
    `rows` or the dicts in it, so a caller keeps its own copy of whatever it
    passed in.

    The audit itself has been wrong twice already: once on B1's wording,
    once on citations lifted out of copied statute text — and each time the
    saved answers were still good, only the check was not. Recomputing from
    the text means a fixed checker costs a second; the generation that
    produced the text can cost up to 45 minutes of CPU.

    Raises `ValueError` if a row's `retrieved` names an article number this
    `docs` does not contain — the corpus does not match the one the run saw,
    and silently dropping that article from `context` (or, worse, treating
    it as `""`) would misclassify a real citation as fabricated instead of
    surfacing the mismatch.
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
    """Recompute every claims row's gate verdict from its saved `parsed`
    answer and `source_numbers` — the claims-JSON sibling of `_reaudit`, for
    the same reason (the gate is model-free specifically so a fixed gate
    never needs Ollama again) and in the same style: a new list of new row
    dicts, never a mutation of `rows` or the dicts in it.

    Source *text* is looked up fresh from the current corpus by article
    number, exactly as `_reaudit` does — a saved row keeps `source_numbers`,
    not the article text itself, so the gate's copied-source check
    (`cite.copied_from_context`) always runs against the corpus as it is
    now, not a second copy frozen at generation time. `None` (a source with
    no article number of its own — an issuance article) has no number to
    look up and stays `""`, same as before; a real number this corpus does
    not contain is a different situation entirely — a corpus mismatch — and
    raises `ValueError` rather than silently gating against empty text (this
    is the exact bug that changed Run 5's re-gated coverage 15 -> 14 with no
    error at all).
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
    """Everything that varies by `--contract`, keyed by name — how to build
    a generator from a model spec, how to run it over the question set, how
    to re-score saved rows against a freshly-loaded corpus, and how to
    report the result. Replaces the `contract == "json"` branches that used
    to be scattered through `main` one at a time — Run 6 ("gated") only
    needed a third branch of each, not a rewrite of the branching itself.
    """
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
    # Run 6 reuses Run 5's exact gate and report — only the header (and,
    # inside it, the pre-registered relevance lines it now also prints) says
    # which pre-registration these numbers were measured against.
    return report_claims(
        rows, meta=meta, contract_name="Run 6", pre_registration_commit="d3f39c3")


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
    """`argparse.ArgumentParser`, except a parse error prints to stdout and
    raises `SystemExit(2)` instead of argparse's default of stderr plus a
    direct `sys.exit`.

    stdout because every other diagnostic this module prints — a bad
    `--model` spec, a rows collision, an incompatible contract — already
    goes to stdout via a plain `print`, and the tests that pin those
    messages read `capsys.readouterr().out`; splitting CLI errors across two
    streams for no reason would just make them harder to find. `SystemExit`
    is still raised, not swallowed here, because `main` below is the one
    place that catches it and turns it into a return code — this class
    exists to be *what* argparse raises, not to hide the raise.
    """

    def error(self, message: str) -> None:
        self.print_usage(sys.stdout)
        print(f"{self.prog}: error: {message}")
        raise SystemExit(2)


def _build_arg_parser() -> _ArgumentParser:
    parser = _ArgumentParser(
        prog="answer-eval",
        description="End-to-end answer evaluation — PRD M2/B1 + M2/B2.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else []

    parser = _build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse itself already printed usage + the error (via
        # `_ArgumentParser.error` above) before raising this — an unknown
        # flag (a `--report-only` typo, say) must exit 2 here rather than
        # silently being ignored the way the hand-rolled `"--x" in argv`
        # parsing this replaces would have, which is exactly how a
        # `--report_only` typo used to start a full run by accident.
        return e.code if isinstance(e.code, int) else 2

    model_spec = args.model
    contract = args.contract  # already restricted to CONTRACTS by `choices=`

    try:
        parse_model_spec(model_spec)
    except ValueError as e:
        # Validated here, before anything else loads: a typo in --model
        # should not cost a corpus load, a question load, or a wasted run.
        # `resolve_model` (below) re-parses the same spec to dispatch, but
        # by then it is known good, so it cannot raise ValueError again.
        print(str(e))
        return 2

    if contract != "text" and not model_spec.startswith("ollama:"):
        # Same reasoning as `generate.resolve_model`'s own refusal — checked
        # again here, before anything loads, so a mismatched pair costs
        # nothing rather than failing deep inside the "loading ..." block.
        print(f"--contract {contract} needs an ollama: model (got {model_spec!r}) — "
              "schema-constrained decoding is not available on the "
              "transformers path here")
        return 2

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
            # Unlike a model mismatch (below), this is a refusal, not a
            # warning: a text-contract row has no `parsed`/`source_numbers`
            # at all, so proceeding into `_regate` would not produce a
            # merely-misleading report, it would raise `KeyError` deep
            # inside it.
            print(f"{rows_file} holds {contract_mismatch!r}-contract rows, "
                  f"not {contract!r} — re-run with "
                  f"--contract {contract_mismatch} instead.")
            return 2

        mismatch = _rows_model_mismatch(rows, model_spec)
        if mismatch is not None:
            # A warning, not a refusal: --report-only changes nothing on
            # disk, so there is nothing to protect by blocking it — but
            # silently reporting another model's answers as this one's
            # would be misleading.
            print(f"WARNING: {rows_file} holds answers for {mismatch!r}, not "
                  f"{model_spec!r} — reporting on what is actually saved there.")

        docs = load_docs(CORPUS_PATH)
        if not docs:
            # Today's bug this guards against: `_regate`'s old
            # `by_number.get(n, "")` silently treated every source as empty
            # text against an empty corpus, which is how re-gating Run 5
            # with no corpus loaded changed its coverage number 15 -> 14
            # with no error printed at all.
            print("corpus not ingested — cannot re-score against nothing. "
                  "Run `python tasks.py ingest --law ...` first.")
            return 2

        saved_fp = rows[0].get("corpus_fingerprint") if rows else None
        if saved_fp is None:
            # True of every row saved before this fingerprint field existed
            # (Run 3/4/5) — must warn and proceed, not refuse, or none of
            # those saved runs could ever be reported on again.
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
            # The gate is model-free by design so a fixed gate never needs
            # Ollama again (`_regate`) — "gated" re-applies it exactly like
            # "json": the relevance decision is saved data on each row
            # (`relevance`, `abstain_reason`), never re-run here.
            rows = handlers.rescore(rows, docs)
            verb = "re-audited" if contract == "text" else "re-gated"
            print(f"{verb} {len(rows)} saved answers from {rows_file} "
                  "(no model loaded)\n")
            return handlers.report(rows, _meta_for_rows(rows, contract))
        except ValueError as e:
            # A missing article number from `_regate`/`_reaudit` — a corpus
            # that does not match the one this run saw, slipping past the
            # fingerprint check above only because these particular saved
            # rows predate it.
            print(f"cannot re-score {rows_file} against the loaded corpus: {e}")
            return 2

    collision = _check_rows_collision(rows_file, model_spec, contract)
    if collision is not None:
        print(collision)
        return 2

    if rows_file.exists() and not args.overwrite:
        # Unconditional on top of the mismatch check above: before this fix,
        # a full run of the SAME model/contract silently overwrote its own
        # previously saved rows, and the rows in `runs/` are the only record
        # of a measurement — so any existing file now blocks a full run
        # unless `--overwrite` says the replacement is intentional.
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
        # `generator.model` is the claims chat for every contract — "gated"'s
        # relevance chat talks to the same Ollama server and the same model
        # name (EVAL.md, "Run 6": relevance and claims are the same model,
        # different token caps only), so its GPU share/digest/quantization
        # are identical and do not need their own separate metadata file.
        _write_ollama_meta(model_spec, generator.model, rows_file)

    return handlers.report(rows, _meta_for_rows(rows, contract))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
