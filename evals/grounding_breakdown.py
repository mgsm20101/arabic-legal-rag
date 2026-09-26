"""Where a grounding change came from — computed from saved answer rows.

    python evals/grounding_breakdown.py
    python evals/grounding_breakdown.py --rows evals/registry/answer_rows_83384b2b.json \
        --baseline evals/registry/answer_rows_46797bef.json

Splits fully-grounded answers by category and by question batch, and counts how
many answers the token cap cut. It reads saved rows only — no model, no
retrieval — so it replays exactly.

**The denominator is answered questions, never answerable ones.** An answerable
question the model declined carries `grounded: true` in its row: it made no
claim, so it cited nothing wrong. Counting it as grounded is how the first
version of this breakdown (hand-computed, no script) reported 12 grounded when
the headline said 11/29 — the one false abstention, Q014, had been added to the
colloquial and batch-1 numerators. A false abstention is a failure, and it is
reported in its own column rather than hidden in either direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag.provenance import git as _git  # noqa: E402
from legalrag.provenance import worktree_clean as _source_is_clean  # noqa: E402

REGISTRY = ROOT / "evals" / "registry"
DEFAULT_ROWS = REGISTRY / "answer_rows_83384b2b.json"
DEFAULT_BASELINE = REGISTRY / "answer_rows_46797bef.json"
BATCH_1_LAST_ID = 20  # Q001–Q020 were the original set; Q021+ were added with the 60-question expansion


def _batch(row: dict) -> str:
    return "batch_1_Q001_Q020" if int(row["id"][1:]) <= BATCH_1_LAST_ID else "batch_2_Q021_plus"


def _was_cut(row: dict) -> bool:
    return any(s.get("cut") or s.get("truncated") for s in row.get("stats") or [])


def _tally(rows: list[dict], key) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {"answerable": 0, "answered": 0, "grounded": 0, "uncited": 0,
                 "cut_by_token_cap": 0, "false_abstention": 0}
    )
    for row in rows:
        if not row["answerable"]:
            continue
        cell = out[key(row)]
        cell["answerable"] += 1
        if row["abstained"]:
            cell["false_abstention"] += 1
            continue
        cell["answered"] += 1
        cell["grounded"] += bool(row["grounded"])
        cell["uncited"] += bool(row["uncited"])
        cell["cut_by_token_cap"] += _was_cut(row)
    return dict(sorted(out.items()))


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tag(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    # relative paths are relative to the repository, whatever the shell's directory
    args.rows, args.baseline = ((p if p.is_absolute() else ROOT / p).resolve()
                                for p in (args.rows, args.baseline))

    rows, baseline = _load(args.rows), _load(args.baseline)
    answered = [r for r in rows if r["answerable"] and not r["abstained"]]
    cut = [r for r in answered if _was_cut(r)]
    baseline_b1 = _tally(baseline, _batch).get("batch_1_Q001_Q020", {})
    now_b1 = _tally(rows, _batch)["batch_1_Q001_Q020"]

    sha = _git("rev-parse", "HEAD")
    result = {
        "check": "where-the-grounding-drop-came-from",
        "command": "python evals/grounding_breakdown.py",
        "source_commit_sha": sha,
        "worktree_clean": _source_is_clean(),
        "rows": str(args.rows.relative_to(ROOT)).replace("\\", "/"),
        "rows_produced_at": _tag(args.rows),
        "baseline_rows": str(args.baseline.relative_to(ROOT)).replace("\\", "/"),
        "denominator": "answered questions; an answerable question the model declined is a "
                       "false_abstention, never a grounded answer",
        "headline": {
            "answerable": sum(r["answerable"] for r in rows),
            "answered": len(answered),
            "grounded": sum(bool(r["grounded"]) for r in answered),
        },
        "by_category": _tally(rows, lambda r: r["category"]),
        "by_batch": _tally(rows, _batch),
        "batch_1_against_baseline": {
            "baseline": {k: baseline_b1.get(k, 0) for k in ("answered", "grounded", "false_abstention")},
            "now": {k: now_b1[k] for k in ("answered", "grounded", "false_abstention")},
        },
        "token_cap": {
            "answers_cut": len(cut),
            "cut_and_uncited": sum(bool(r["uncited"]) for r in cut),
            "uncited_total": sum(bool(r["uncited"]) for r in answered),
        },
        "supersedes": {
            "file": "evals/registry/grounding_breakdown_83384b2b.json",
            "why": "hand-computed with no script, and counted the one false abstention (Q014) "
                   "as grounded: colloquial read 5/9 and batch 1 read 8/15 where the answered "
                   "counts are 4/8 and 7/14. multi_article (0/9) and the token-cap figures "
                   "are unaffected.",
        },
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out = args.out or REGISTRY / f"grounding_breakdown_{sha[:8]}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    h = result["headline"]
    print(f"grounded {h['grounded']}/{h['answered']} answered ({h['answerable']} answerable)")
    for name, c in result["by_category"].items():
        print(f"  {name:<14} {c['grounded']}/{c['answered']}  uncited {c['uncited']}"
              f"  cut {c['cut_by_token_cap']}  false_abstention {c['false_abstention']}")
    b = result["batch_1_against_baseline"]
    print(f"  batch 1: {b['now']['grounded']}/{b['now']['answered']} now vs "
          f"{b['baseline']['grounded']}/{b['baseline']['answered']} baseline")
    print(f"wrote {out}  worktree_clean={result['worktree_clean']}")
    return 0 if result["worktree_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
