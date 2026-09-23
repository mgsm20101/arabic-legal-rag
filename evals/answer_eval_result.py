"""The end-to-end answer evaluation as a registry artifact.

    python evals/answer_eval_result.py [--model SPEC] [--contract text|json|gated]

`tasks.py answer-eval` prints a report for a human and saves its raw answers to
`runs/`, which is git-ignored. A number nobody can open the raw answers for is
not a quotable number, so this does two things:

1. copies the raw rows — every answer, every citation, every verdict — into
   `evals/registry/`, where they are tracked;
2. writes a summary beside them carrying the generation protocol in full:
   runtime, model digest, quantization, decoding parameters, the contract, the
   retriever underneath, and how the answers were scored.

It computes nothing the printed report does not already compute. The counts
here come from the same row fields `answer_report.report` reads, so the file
and the report cannot disagree about a run.

**The scoring is a program, not a judge.** `legalrag.cite.audit` resolves every
citation against the corpus and against the exact article set the retriever
handed the model. There is no LLM in the loop, so there is no judge model,
judge prompt or judge temperature to report — and the verdicts replay
identically from the saved rows with `answer-eval --report-only`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag.answer_eval import TOP_K, _read_run_meta, rows_path  # noqa: E402
from legalrag.answer_report import (  # noqa: E402
    B2_MIN,
    CITED_EXPECTED_MIN,
    FABRICATED_MAX,
)
from legalrag.dense import CORPUS_PATH, load_docs  # noqa: E402
from legalrag.evaluate import (  # noqa: E402
    load_meta,
    load_questions,
    question_set_fingerprint,
)
from legalrag.envcheck import load_recorded, mismatches  # noqa: E402
from legalrag.envcheck import require as require_environment  # noqa: E402
from legalrag.generate import MAX_NEW_TOKENS, TEMPERATURE, parse_model_spec  # noqa: E402

REGISTRY = ROOT / "evals" / "registry"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _source_is_clean() -> bool:
    """Uncommitted changes outside this run's own output directory."""
    return all(
        line[3:].strip().strip('"').startswith("evals/registry/")
        for line in _git("status", "--porcelain").splitlines()
    )


def _ollama_identity(name: str) -> dict:
    """The digest and quantization of the Ollama model that actually answered.

    Read from the running server rather than assumed from the tag: `gemma3:4b`
    is a moving pointer, and two machines can hold different bytes under it.
    Returns what it can and says nothing it could not confirm.
    """
    try:
        import httpx

        tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=10).json()
    except Exception:
        return {}
    for model in tags.get("models", []):
        if model.get("name") == name:
            return {
                "digest": model.get("digest", "")[:12] or None,
                "quantization": model.get("details", {}).get("quantization_level"),
                "parameter_size": model.get("details", {}).get("parameter_size"),
            }
    return {}


def _pct(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def summarise(rows: list[dict], spec: str, contract: str) -> dict:
    """The same three populations `answer_report.report` splits rows into, and
    the same verdicts — counted here instead of printed."""
    answerable = [r for r in rows if r["answerable"]]
    out_of_corpus = [r for r in rows if not r["answerable"]]
    scored = [r for r in answerable if not r["abstained"]]

    fabricated = [r for r in scored if r["fabricated"]]
    ungrounded = [r for r in scored if r["ungrounded"]]
    uncited = [r for r in scored if r["uncited"]]
    grounded = [r for r in scored if r["grounded"]]
    cited_expected = [r for r in scored if r["cited_expected"]]
    strict_form = [r for r in scored if r["strict"]]

    correct_abstain = [r for r in out_of_corpus if r["abstained"]]
    false_abstain = [r for r in answerable if r["abstained"]]

    b1 = not fabricated and not ungrounded and not uncited
    b2_rate = _pct(len(correct_abstain), len(out_of_corpus))
    b2 = bool(out_of_corpus) and (b2_rate or 0) >= B2_MIN

    runtime, name = parse_model_spec(spec)
    seconds = [r["seconds"] for r in rows if r.get("seconds")]

    return {
        "metric": "end-to-end answer quality — grounding, abstention, citation",
        "source_commit_sha": _git("rev-parse", "HEAD"),
        "worktree_clean": _source_is_clean(),
        **require_environment(),
        "generation_protocol": {
            "runtime": runtime,
            "model": name,
            **_ollama_identity(name),
            "decoding": "greedy",
            "temperature": TEMPERATURE,
            "seed": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "answer_contract": contract,
            "retrieval_under_test": f"dense (multilingual-e5-base), top-{TOP_K}",
            "prompt": "legalrag.generate.SYSTEM + USER — Arabic, fixed, in the repo",
        },
        "scoring": {
            "type": "deterministic program — no LLM judge",
            "module": "legalrag.cite.audit",
            "replay": f"python tasks.py answer-eval --model {spec} --report-only",
            "rubric": {
                "fabricated": "cited an article number that is not in the corpus",
                "ungrounded": "cited a real article the retriever did not return",
                "uncited": "made a claim carrying no citation at all",
                "grounded": "none of the three above",
                "cited_expected": "cited the article the eval set names — "
                                  "bounded above by retrieval",
                "strict_form": "used the requested [مادة N] bracket form",
            },
            "pre_registered_thresholds": {
                "fabricated_max": FABRICATED_MAX,
                "abstention_on_out_of_corpus_min": B2_MIN,
                "cited_expected_min": CITED_EXPECTED_MIN,
                "note": "fixed in EVAL.md before this run — see answer_report.py",
            },
        },
        "dataset": {
            "corpus_articles": len(load_docs(CORPUS_PATH)),
            "corpus_fingerprint": rows[0].get("corpus_fingerprint"),
            "questions_total": len(rows),
            "answerable": len(answerable),
            "out_of_corpus": len(out_of_corpus),
            "split": load_meta().get("split"),
        },
        "results": {
            "answered_of_answerable": [len(scored), len(answerable)],
            "fully_grounded": _pct(len(grounded), len(scored)),
            "cited_expected_article": _pct(len(cited_expected), len(scored)),
            "used_requested_citation_form": _pct(len(strict_form), len(scored)),
            "abstained_on_out_of_corpus": b2_rate,
            "falsely_abstained_on_answerable": _pct(len(false_abstain), len(answerable)),
            "seconds_per_answer": {
                "median": round(statistics.median(seconds), 1) if seconds else None,
                "total": round(sum(seconds), 1) if seconds else None,
                "n": len(seconds),
            },
        },
        "failure_categories": {
            "fabricated": [[r["id"], r["fabricated"]] for r in fabricated],
            "ungrounded": [[r["id"], r["ungrounded"]] for r in ungrounded],
            "uncited": [r["id"] for r in uncited],
            "false_abstention": [r["id"] for r in false_abstain],
            "missed_abstention": [r["id"] for r in out_of_corpus if not r["abstained"]],
        },
        "verdicts": {
            "B1_grounding": "PASS" if b1 else "FAIL",
            "B2_abstention": "PASS" if b2 else "FAIL",
            "reading": (
                "B2 alone is meaningless: a model that abstains on everything "
                "scores 1.0. Read it only beside falsely_abstained_on_answerable."
            ),
        },
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _stale_against_current_questions(model_spec: str, contract: str) -> str | None:
    """Refuse to stamp the current commit onto answers from a different eval set.

    This promoter does not run anything: it reads whatever rows are sitting in
    the git-ignored `runs/` directory and writes them into the registry under
    HEAD. That separation is deliberate — a measurement and its promotion are
    different commits — but it means the rows' provenance was assumed rather
    than checked, and the assumption broke the first time the question set
    grew. The promoter published a row reading `questions_total: 20` beside
    `split: {dev: 40}`, under a commit that had never produced those answers.

    A missing fingerprint is refused too, not waved through. Rows saved before
    the fingerprint existed are indistinguishable from rows saved against a
    question set that has since changed, and "indistinguishable" is precisely
    the case this guard exists for.
    """
    questions, errors = load_questions()
    if errors:
        return "question set does not load; fix that before promoting:\n  - " + (
            "\n  - ".join(errors)
        )

    meta = _read_run_meta(model_spec, contract) or {}
    recorded = meta.get("question_set")
    rerun = (
        f"re-run it:  python tasks.py answer-eval --model {model_spec} "
        f"--contract {contract} --overwrite"
    )
    if not recorded or not recorded.get("fingerprint"):
        return (
            "the saved run carries no question-set fingerprint, so which questions\n"
            "produced it cannot be established. It predates the fingerprint or was\n"
            "written by hand; either way it cannot back a registry row.\n"
            f"  {rerun}"
        )

    if recorded["fingerprint"] == question_set_fingerprint(questions):
        return None

    saved_ids = set(recorded.get("ids") or [])
    now_ids = {q.id for q in questions}
    added = sorted(now_ids - saved_ids)
    dropped = sorted(saved_ids - now_ids)

    def _names(ids: list[str]) -> str:
        return ", ".join(ids[:6]) + (" …" if len(ids) > 6 else "")

    detail = []
    if added:
        detail.append(f"  {len(added)} added since that run: {_names(added)}")
    if dropped:
        detail.append(f"  {len(dropped)} no longer present: {_names(dropped)}")
    if not detail:
        detail.append("  same ids, but wording or expected articles changed")

    return (
        f"the saved run scored a different question set "
        f"({recorded.get('n', '?')} questions) from the one on disk "
        f"({len(questions)} questions).\n"
        + "\n".join(detail)
        + "\nPromoting it would stamp HEAD onto answers HEAD never produced.\n"
        + f"  {rerun}"
    )


def _produced_elsewhere(model_spec: str, contract: str) -> str | None:
    """Refuse rows generated by an interpreter other than the recorded one.

    The promoter's own process is checked when it stamps the file; this checks
    the run it is promoting, from the runtime `answer-eval` wrote into the
    run's metadata. A run with no recorded runtime is refused for the same
    reason a run with no question fingerprint is: it cannot be told apart
    from one that ran somewhere else.
    """
    meta = _read_run_meta(model_spec, contract) or {}
    seen = meta.get("runtime")
    rerun = (f"re-run it with the recorded interpreter:  python tasks.py answer-eval "
             f"--model {model_spec} --contract {contract} --overwrite")
    if not seen:
        return ("the saved run records no runtime, so the environment that produced it "
                "cannot be established.\n  " + rerun)
    problems = mismatches(load_recorded(), seen)
    if problems:
        return ("the saved run was produced outside the recorded environment:\n  - "
                + "\n  - ".join(problems) + "\n  " + rerun)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ollama:gemma3:4b")
    ap.add_argument("--contract", default="text", choices=("text", "json", "gated"))
    a = ap.parse_args()

    saved = rows_path(a.model, a.contract)
    if not saved.exists():
        print(f"no saved run at {saved} — run `python tasks.py answer-eval "
              f"--model {a.model}` first.")
        return 2

    stale = _stale_against_current_questions(a.model, a.contract)
    if stale is not None:
        print(stale)
        return 2

    foreign = _produced_elsewhere(a.model, a.contract)
    if foreign is not None:
        print(foreign)
        return 2

    rows = json.loads(saved.read_text(encoding="utf-8"))
    payload = summarise(rows, a.model, a.contract)
    sha = payload["source_commit_sha"][:8]

    REGISTRY.mkdir(parents=True, exist_ok=True)
    raw = REGISTRY / f"answer_rows_{sha}.json"
    raw.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    payload["raw_judgments"] = f"evals/registry/{raw.name}"

    out = REGISTRY / f"answer_eval_{sha}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    r = payload["results"]
    print(f"  grounded            {r['fully_grounded']}")
    print(f"  cited expected      {r['cited_expected_article']}")
    print(f"  abstained (OOC)     {r['abstained_on_out_of_corpus']}")
    print(f"  false abstention    {r['falsely_abstained_on_answerable']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"wrote {raw.relative_to(ROOT)}")
    if not payload["worktree_clean"]:
        print("WARNING: worktree was dirty — this sha does not describe what ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
