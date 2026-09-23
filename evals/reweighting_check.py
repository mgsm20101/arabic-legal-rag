"""Did the retrievers change between two ablation runs, or did the question mix?

    python evals/reweighting_check.py
    python evals/reweighting_check.py --new evals/registry/ablation_<sha>.json \
        --old evals/registry/ablation_bd04e8e8.json

Takes the per-category Recall@5 from the newer run and re-weights it by the
category mix of the older run's scored questions. If that brings the old
aggregate picture back, the change in the aggregate was a composition effect,
not a change in retrieval behaviour.

Both mixes are computed from `evals/retrieval/questions.jsonl`, not typed in:
the older run scored the answerable dev questions up to Q020, the newer one
all answerable dev questions. The file also reports, position by position,
whether the re-weighted ordering matches the old one — because "reproduces the
old ordering" is exactly the kind of sentence that is true at the ends of a
ranking and false in the middle, and the first hand-computed version of this
check said it without looking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "evals" / "registry"
sys.path.insert(0, str(ROOT / "src"))

from legalrag.evaluate import load_questions  # noqa: E402

DEFAULT_OLD = REGISTRY / "ablation_bd04e8e8.json"
OLD_SET_LAST_ID = 20  # the 15-question run scored the answerable dev questions Q001–Q020


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _source_is_clean() -> bool:
    return all(
        line[3:].strip().strip('"').startswith("evals/registry/")
        for line in _git("status", "--porcelain").splitlines()
    )


def _mix(categories: list[str]) -> dict[str, float]:
    counts = Counter(categories)
    total = sum(counts.values())
    return {c: round(n / total, 3) for c, n in sorted(counts.items())}


def reweight(per_category: dict[str, float], mix: dict[str, float]) -> float:
    return round(sum(per_category[c] * w for c, w in mix.items()), 3)


def ranking(values: dict[str, float]) -> list[str]:
    return [name for name, _ in sorted(values.items(), key=lambda kv: -kv[1])]


def _latest_ablation() -> Path:
    """The most recently *measured* ablation, by the timestamp inside the file.
    File mtimes are useless here: a fresh clone gives every file the same one."""
    def measured(path: Path) -> str:
        return json.loads(path.read_text(encoding="utf-8")).get("measured_at", "")
    return max(REGISTRY.glob("ablation_*.json"), key=measured)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new", type=Path, default=None)
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    new_path = args.new or _latest_ablation()

    questions, errors = load_questions()
    if errors:
        print("\n".join(errors))
        return 1
    scored = [q for q in questions if q.answerable]
    old_mix = _mix([q.category for q in scored if int(q.id[1:]) <= OLD_SET_LAST_ID])
    new_mix = _mix([q.category for q in scored])

    old = {r["configuration"]: r for r in json.loads(args.old.read_text(encoding="utf-8"))["results"]}
    new = {r["configuration"]: r for r in json.loads(new_path.read_text(encoding="utf-8"))["results"]}

    rows = [
        {
            "configuration": name,
            "old_actual": old[name]["recall_at_5"],
            "new_actual": new[name]["recall_at_5"],
            "new_rates_under_old_mix": reweight(new[name]["per_category_recall_at_5"], old_mix),
            "per_category_new": new[name]["per_category_recall_at_5"],
        }
        for name in new
    ]
    old_rank = ranking({r["configuration"]: r["old_actual"] for r in rows})
    reweighted_rank = ranking({r["configuration"]: r["new_rates_under_old_mix"] for r in rows})
    positions = [
        {"position": i + 1, "old": o, "reweighted": w, "match": o == w}
        for i, (o, w) in enumerate(zip(old_rank, reweighted_rank))
    ]

    sha = _git("rev-parse", "HEAD")
    result = {
        "check": "category-reweighting",
        "command": "python evals/reweighting_check.py",
        "source_commit_sha": sha,
        "worktree_clean": _source_is_clean(),
        "old_run": str(args.old.relative_to(ROOT)).replace("\\", "/"),
        "new_run": str(new_path.relative_to(ROOT)).replace("\\", "/"),
        "old_mix": old_mix,
        "new_mix": new_mix,
        "rows": rows,
        "ordering": {
            "old": old_rank,
            "new_rates_under_old_mix": reweighted_rank,
            "positions": positions,
            "all_positions_match": all(p["match"] for p in positions),
        },
        "supersedes": {
            "file": "evals/registry/reweighting_check_618693ef.json",
            "why": "hand-computed with no script, bound to a commit that is not on the public "
                   "main branch, and read as reproducing the old ordering when only the first "
                   "and last positions match",
        },
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out = args.out or REGISTRY / f"reweighting_check_{sha[:8]}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"old mix {old_mix}\nnew mix {new_mix}")
    for r in rows:
        print(f"  {r['configuration']:<16} old {r['old_actual']:.3f}  new {r['new_actual']:.3f}"
              f"  new@old-mix {r['new_rates_under_old_mix']:.3f}")
    for p in positions:
        print(f"  #{p['position']}  old {p['old']:<16} re-weighted {p['reweighted']:<16}"
              f" {'match' if p['match'] else 'DIFFERS'}")
    print(f"wrote {out.relative_to(ROOT)}  worktree_clean={result['worktree_clean']}")
    return 0 if result["worktree_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
