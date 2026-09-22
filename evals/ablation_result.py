"""The retrieval ablation as a JSON artifact, with bootstrap intervals.

    python evals/ablation_result.py [--out PATH] [--resamples N]

`tasks.py ablate` prints this table for a human. This writes the same numbers —
same `build_configs`, same `Config.run`, not a reimplementation — as a file a
registry row can point at, stamped with the commit that produced it.

It adds one thing the printed table does not have: a 95% interval on Recall@5.
With 15 scored questions the intervals are wide enough to overlap, and that is
the point. `0.767` written alone reads like three significant figures of
precision that 15 questions cannot support.

The interval is a percentile bootstrap over the per-question recall values, not
a Wilson interval: recall is fractional on `multi_article` questions (two
expected articles, one retrieved scores 0.5), so the Bernoulli assumption a
Wilson interval rests on does not hold here.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag.ablate import (  # noqa: E402
    CORPUS_PATH,
    EVAL_K,
    build_configs,
    candidate_ceiling,
)
from legalrag.dense import load_docs  # noqa: E402
from legalrag.evaluate import binding_problem, corpus_laws, load_meta, load_questions  # noqa: E402
from legalrag.rerank import CANDIDATE_DEPTH  # noqa: E402
from legalrag.retrieve import recall_at_k  # noqa: E402


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


def _bootstrap_ci(values: list[float], resamples: int, seed: int) -> dict:
    """Percentile bootstrap for the mean. Reports the seed so it replays exactly."""
    if not values:
        return {}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(rng.choices(values, k=n)) for _ in range(resamples)
    )
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return {
        "point": round(statistics.fmean(values), 3),
        "ci95_low": round(lo, 3),
        "ci95_high": round(hi, 3),
        "method": "percentile bootstrap over per-question values",
        "resamples": resamples,
        "seed": seed,
    }


def main(out: Path | None, resamples: int, seed: int) -> int:
    questions, errors = load_questions()
    if errors:
        for e in errors:
            print(f"  - {e}")
        return 1

    problem = binding_problem(load_meta(), corpus_laws())
    if problem:
        print(f"BLOCKED: {problem}")
        return 3

    docs = load_docs(CORPUS_PATH)
    if not docs:
        print('corpus not ingested — run `python tasks.py ingest --law "…"` first.')
        return 2

    answerable = [q for q in questions if q.answerable]
    configs = build_configs(docs, answerable, k=EVAL_K)

    results = []
    for cfg in configs:
        cfg.run(answerable, EVAL_K)
        per_q = [
            recall_at_k(q.expected_articles, cfg.search(q.question, EVAL_K))
            for q in answerable
        ]
        results.append(
            {
                "configuration": cfg.name,
                "note": cfg.note,
                "recall_at_5": round(cfg.recall, 3) if cfg.recall is not None else None,
                "mrr": round(cfg.mrr, 3) if cfg.mrr is not None else None,
                "recall_at_5_interval": _bootstrap_ci(per_q, resamples, seed),
                "per_category_recall_at_5": {
                    cat: (round(v[0], 3) if v[0] is not None else None)
                    for cat, v in cfg.per_category.items()
                },
            }
        )
        r = results[-1]
        ci = r["recall_at_5_interval"]
        print(
            f"  {cfg.name:<18} recall@5 {r['recall_at_5']:.3f} "
            f"[{ci['ci95_low']:.3f}–{ci['ci95_high']:.3f}]   MRR {r['mrr']:.3f}",
            flush=True,
        )

    hybrid = next(c for c in configs if c.name == "hybrid (RRF)")
    ceiling = candidate_ceiling(answerable, hybrid.search, CANDIDATE_DEPTH)

    payload = {
        "metric": "retrieval quality — four-configuration ablation",
        "source_commit_sha": _git("rev-parse", "HEAD"),
        "worktree_clean": _source_is_clean(),
        "environment_ref": "env-001",
        "dataset": {
            "corpus_articles": len(docs),
            "questions_written": len(questions),
            "questions_scored": len(answerable),
            "questions_abstention": len(questions) - len(answerable),
            "split": load_meta().get("split"),
            "k": EVAL_K,
            "candidate_depth": CANDIDATE_DEPTH,
        },
        "results": results,
        "candidate_ceiling_recall_at_20": round(ceiling, 3) if ceiling else None,
        "reading": (
            "Differences here are a direction, not a magnitude. With 15 scored "
            "questions the intervals overlap; 'dense leads' is supported, "
            "'dense leads by 0.433' is not."
        ),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    sha8 = payload["source_commit_sha"][:8]
    out = out or ROOT / "evals" / "registry" / f"ablation_{sha8}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    if not payload["worktree_clean"]:
        print("WARNING: source tree was dirty — this sha does not describe what ran.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()
    raise SystemExit(main(Path(a.out) if a.out else None, a.resamples, a.seed))
