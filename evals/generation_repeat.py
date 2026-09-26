"""Does the generation run repeat? — computed from saved answer rows.

    python evals/generation_repeat.py \
        --rows evals/registry/answer_rows_83384b2b.json \
        --rows evals/registry/answer_rows_4fe53a93.json \
        --rows runs/answer_eval-ollama-gemma3-4b.json

Compares two or more runs of the same questions, same model digest, greedy
decoding. For each run it reports the headline counts; for each pair it reports
how many questions got the same retrieval and the same generated text, and which
questions flipped a verdict (grounded, abstained).

Written because EVIDENCE once said a re-run "reproduced it exactly". That was
two runs inside one Ollama server session. Across a server restart the retrieval
was identical on every question and the generated text was not, and the
pre-registered abstention criterion passed in one run and failed in the other.
A verdict that flips between runs of the same code is a property of the
measurement, and it belongs in the registry next to the verdict.

Rows files outside `evals/registry/` (the git-ignored `runs/`) are embedded in
the output, so nothing this file reports depends on an untracked file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag.provenance import git as _git  # noqa: E402
from legalrag.provenance import worktree_clean as _source_is_clean  # noqa: E402

REGISTRY = ROOT / "evals" / "registry"
ABSTENTION_CRITERION = 0.80  # EVAL.md, pre-registered: abstain on >= 80% of out_of_corpus


def summarise(rows: list[dict]) -> dict:
    """Headline counts, with answered questions as the grounding denominator."""
    answerable = [r for r in rows if r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    ooc = [r for r in rows if not r["answerable"]]
    abstained_ooc = sum(r["abstained"] for r in ooc)
    return {
        "answered": len(answered),
        "answerable": len(answerable),
        "grounded": sum(r["grounded"] for r in answered),
        "fabricated": sum(bool(r["fabricated"]) for r in answered),
        "cited_not_retrieved": sum(bool(r["ungrounded"]) for r in answered),
        "uncited": sum(bool(r["uncited"]) for r in answered),
        "false_abstention": len(answerable) - len(answered),
        "abstained_out_of_corpus": abstained_ooc,
        "out_of_corpus": len(ooc),
        "abstention_criterion_met": bool(ooc) and abstained_ooc / len(ooc) >= ABSTENTION_CRITERION,
    }


def compare(a: list[dict], b: list[dict]) -> dict:
    """Question-by-question agreement between two runs of the same question set."""
    ra, rb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if ra.keys() != rb.keys():
        raise SystemExit("the two runs did not answer the same questions")
    ids = sorted(ra)
    return {
        "questions": len(ids),
        "same_retrieval": sum(ra[q]["retrieved_ids"] == rb[q]["retrieved_ids"] for q in ids),
        "same_text": sum(ra[q]["text"] == rb[q]["text"] for q in ids),
        "grounded_flipped": [q for q in ids if ra[q]["answerable"] and not ra[q]["abstained"]
                             and not rb[q]["abstained"] and ra[q]["grounded"] != rb[q]["grounded"]],
        "abstained_flipped": [q for q in ids if ra[q]["abstained"] != rb[q]["abstained"]],
    }


def _label(path: Path) -> str:
    return path.stem if path.parent == REGISTRY else f"{path.parent.name}/{path.name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", action="append", required=True, type=Path,
                    help="an answer-rows file; give two or more")
    ap.add_argument("--note", action="append", default=[],
                    help="LABEL=TEXT provenance for one of the rows files")
    a = ap.parse_args()
    if len(a.rows) < 2:
        raise SystemExit("give at least two --rows files")

    runs = {}
    for path in a.rows:
        path = (ROOT / path) if not path.is_absolute() else path
        rows = json.loads(path.read_text(encoding="utf-8"))
        runs[_label(path)] = {"path": path, "rows": rows}

    notes = dict(n.split("=", 1) for n in a.note)
    out = {
        "metric": "generation repeatability — same questions, same model digest, greedy",
        "source_commit_sha": _git("rev-parse", "HEAD"),
        "worktree_clean": _source_is_clean(),
        "command": "python evals/generation_repeat.py "
                   + " ".join(f"--rows {p.as_posix()}" for p in a.rows),
        "runs": {
            label: {
                "rows_file": run["path"].relative_to(ROOT).as_posix(),
                "provenance": notes.get(label),
                **summarise(run["rows"]),
                **({"embedded_rows": run["rows"]} if run["path"].parent != REGISTRY else {}),
            }
            for label, run in runs.items()
        },
        "pairs": {
            f"{x} vs {y}": compare(runs[x]["rows"], runs[y]["rows"])
            for x, y in combinations(runs, 2)
        },
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    for label, s in out["runs"].items():
        print(f"{label:40s} grounded {s['grounded']}/{s['answered']}  fabricated {s['fabricated']}  "
              f"abstained {s['abstained_out_of_corpus']}/{s['out_of_corpus']}  "
              f"false-abst {s['false_abstention']}/{s['answerable']}  "
              f"B2 {'PASS' if s['abstention_criterion_met'] else 'FAIL'}")
    for pair, c in out["pairs"].items():
        print(f"{pair}\n    same retrieval {c['same_retrieval']}/{c['questions']}  "
              f"same text {c['same_text']}/{c['questions']}  "
              f"grounded flipped {len(c['grounded_flipped'])}  abstained flipped {c['abstained_flipped']}")

    dest = REGISTRY / f"generation_repeat_{out['source_commit_sha'][:8]}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}  worktree_clean={out['worktree_clean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
