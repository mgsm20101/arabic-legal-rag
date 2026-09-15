"""App eval — the upload pipeline measured on the app-dev split (ADR-023).

    python tasks.py app-eval [--retrieval-only] [--doc pdf|txt] [--model SPEC] [--overwrite]

Retrieval is scored in every mode. The document goes into a fresh `Library`
in a temporary directory, exactly as an upload would, and every answerable
question in `evals/app/questions.jsonl` is searched top-k:

* hit@5 — `meta.json`'s definition: a top-5 chunk whose text contains every
  expected keyword, both sides passed through `evaluation_normalize`;
* page-hit@5 — a top-5 chunk on one of the question's expected pages; n/a
  for a one-page document, where every chunk is on page 1.

Without --retrieval-only, the whole `Pipeline` (Run 6's gated contract)
runs over all 15 questions and every row is saved the moment it exists.
That run calls a model: its adoption rule belongs in EVAL.md before it runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from .claims import build_generators
from .generate import parse_model_spec
from .library import EncoderUnavailable, Library
from .normalize import evaluation_normalize
from .ollama import GeneratorUnavailable
from .pipeline import TOP_K, Pipeline

APP_DIR = Path("evals/app")
RUNS = Path("runs")
DEFAULT_SPEC = "ollama:gemma3:4b"
DOCS = ("pdf", "txt")


def load_questions(path: Path | None = None) -> list[dict]:
    path = path if path is not None else APP_DIR / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rows_path(spec: str, doc: str) -> Path:
    """Where a full run's rows are saved: one file per model spec and document."""
    return RUNS / f"app_eval-{re.sub(r'[^A-Za-z0-9._-]', '-', spec)}-{doc}.json"


def retrieval_hit(texts: list[str], keywords: list[str]) -> bool:
    """`meta.json`'s hit: ONE chunk whose text holds every keyword, both sides
    through `evaluation_normalize`. No keyword is no evidence, never a hit."""
    wanted = [evaluation_normalize(k) for k in keywords]
    if not wanted:
        return False
    return any(all(k in evaluation_normalize(text) for k in wanted) for text in texts)


def score_retrieval(library, questions: list[dict], k: int = TOP_K) -> list[dict]:
    """One row per answerable question: hit@k, page-hit@k, and the pages of
    the chunks retrieved, in rank order."""
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        hits = library.search(q["question"], k)
        pages = [h.chunk.page for h in hits]
        rows.append({
            "id": q["id"],
            "category": q["category"],
            "hit": retrieval_hit([h.chunk.text for h in hits], q["expected_keywords"]),
            "page_hit": bool(set(pages) & set(q["expected_pages"])),
            "pages": pages,
        })
    return rows


def print_retrieval(meta, rows: list[dict]) -> None:
    one_page = meta.pages == 1
    print(f"document   : {meta.title} ({meta.kind}, {meta.pages} page{'' if one_page else 's'}, "
          f"{meta.chunks} chunks)")
    print(f"retrieval  : dense top-{TOP_K} over the document's chunks\n")
    for r in rows:
        page = "n/a" if one_page else ("yes" if r["page_hit"] else "no")
        print(f"  {r['id']}  {r['category']:<10}  hit@{TOP_K} {'yes' if r['hit'] else 'no':<3}  "
              f"page-hit@{TOP_K} {page:<3}  pages {','.join(map(str, r['pages']))}")
    total = len(rows)
    print()
    print(f"hit@{TOP_K}      : {sum(r['hit'] for r in rows)}/{total}")
    print(f"page-hit@{TOP_K} : " + ("n/a" if one_page else f"{sum(r['page_hit'] for r in rows)}/{total}"))
    print("missed     : " + (", ".join(r["id"] for r in rows if not r["hit"]) or "none"))


def run_answers(pipeline: Pipeline, questions: list[dict], path: Path) -> list[dict]:
    """Ask every question. After EACH one, `path` is atomically replaced with
    every row so far, so a run that dies on question 9 keeps questions 1-8."""
    rows: list[dict] = []
    for q in questions:
        result = pipeline.ask(q["question"])
        rows.append({
            "id": q["id"],
            "category": q["category"],
            "answerable": q["answerable"],
            "question": q["question"],
            "status": result.status,
            "abstain_reason": result.abstain_reason,
            "claims": result.claims,
            "source_chunk_ids": [s.chunk_id for s in result.sources],
            "source_pages": [s.page for s in result.sources],
            "dropped": result.dropped,
            "timings_ms": result.timings_ms,
        })
        _write_rows(path, rows)
        print(f"  {q['id']}  {result.status:<9}  kept {len(result.claims)}  "
              f"{result.timings_ms['total'] / 1000:5.1f}s", flush=True)
    return rows


def summarize(rows: list[dict], questions: list[dict]) -> dict:
    """The full run's numbers, each over the questions it is about; which
    questions are answerable comes from the question set, not the rows."""
    by_id = {q["id"]: q for q in questions}
    answerable = [r for r in rows if by_id[r["id"]]["answerable"]]
    out_of_doc = [r for r in rows if not by_id[r["id"]]["answerable"]]
    n_answerable = sum(1 for q in questions if q["answerable"])
    n_out_of_doc = len(questions) - n_answerable

    def page_grounded(row: dict) -> bool:
        expected = set(by_id[row["id"]]["expected_pages"])
        return any(row["source_pages"][n - 1] in expected
                   for claim in row["claims"] for n in claim["sources"])

    return {
        "coverage": (sum(1 for r in answerable if r["claims"]), n_answerable),
        "page_grounded": (sum(1 for r in answerable if page_grounded(r)), n_answerable),
        "out_of_doc_abstentions": (sum(1 for r in out_of_doc if r["status"] == "abstained"), n_out_of_doc),
        "false_abstentions": (sum(1 for r in answerable if r["status"] == "abstained"), n_answerable),
        "fabricated": sum(r["dropped"]["fabricated"] for r in rows),
        "mean_seconds": sum(r["timings_ms"]["total"] for r in rows) / len(rows) / 1000 if rows else 0.0,
    }


def print_summary(summary: dict, one_page: bool) -> None:
    def share(pair: tuple[int, int]) -> str:
        return f"{pair[0]}/{pair[1]}"

    grounded = "n/a" if one_page else share(summary["page_grounded"])
    print()
    print(f"coverage (answerable, >=1 kept claim)         : {share(summary['coverage'])}")
    print(f"page-grounded (kept claim cites expected page): {grounded}")
    print(f"out-of-doc abstentions                        : {share(summary['out_of_doc_abstentions'])}")
    print(f"false abstentions (answerable)                : {share(summary['false_abstentions'])}")
    print(f"fabricated (total)                            : {summary['fabricated']}")
    print(f"mean seconds per question                     : {summary['mean_seconds']:.1f}")


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Replace `path` with `rows` atomically: written beside it, then moved
    over it, so a write that fails midway leaves the previous rows whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # already gone once the replace succeeded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app-eval", description="The upload pipeline on the app-dev split (ADR-023).",
        allow_abbrev=False,  # "--o" must never pass for --overwrite
    )
    parser.add_argument("--retrieval-only", action="store_true",
                        help="score retrieval and stop; no generator is ever built")
    parser.add_argument("--doc", choices=DOCS, default="pdf",
                        help="which rendering of policy_ar to upload (default: pdf)")
    parser.add_argument("--model", default=None,
                        help=f"model spec for the full run (default: $LEGALRAG_MODEL, else {DEFAULT_SPEC})")
    parser.add_argument("--overwrite", action="store_true",
                        help="let a full run replace its saved rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(list(argv) if argv is not None else [])
    except SystemExit as e:  # argparse has already printed the usage and the error
        return e.code if isinstance(e.code, int) else 2

    spec = args.model or os.environ.get("LEGALRAG_MODEL") or DEFAULT_SPEC
    path = rows_path(spec, args.doc)
    if not args.retrieval_only:
        # Everything that can refuse a full run does so before anything loads.
        try:
            parse_model_spec(spec)
        except ValueError as e:
            print(str(e))
            return 2
        if not spec.startswith("ollama:"):
            print(f"the gated claims contract needs an ollama: model, not {spec!r}")
            return 2
        if path.exists() and not args.overwrite:
            print(f"{path} already holds a saved run. Pass --overwrite to replace it.")
            return 2

    questions = load_questions()
    source = APP_DIR / f"policy_ar.{args.doc}"
    print("=" * 68)
    print(f"  arabic-legal-rag - app eval - app-dev split - {source.name}")
    print("=" * 68)
    with tempfile.TemporaryDirectory(prefix="legalrag-app-eval-", ignore_cleanup_errors=True) as tmp:
        try:
            library = Library(Path(tmp))
            meta = library.add(source.read_bytes(), source.name)
            print_retrieval(meta, score_retrieval(library, questions))
        except EncoderUnavailable as e:
            print(str(e))
            return 5
        if args.retrieval_only:
            return 0

        print(f"\nanswers    : {spec}, gated contract, {len(questions)} questions -> {path}")
        pipeline = Pipeline(library, build_generators(spec, "gated"))
        try:
            rows = run_answers(pipeline, questions, path)
        except GeneratorUnavailable as e:
            print(str(e))
            return 5

    print_summary(summarize(rows, questions), one_page=meta.pages == 1)
    return 0
