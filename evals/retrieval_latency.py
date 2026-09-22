"""Per-query retrieval latency for each ablation configuration, under a stated protocol.

    python evals/retrieval_latency.py [--reps N] [--warmup N] [--out PATH]

`ablate` prints one wall-clock average per configuration. That is enough to say
"the reranker is far slower" and not enough to quote as a latency number: no
warm-up, one pass, no spread. This runs the same four configurations from
`ablate.build_configs` — the same code path, not a reimplementation — with the
model load and a fixed warm-up excluded, several repetitions per question, and
median/p95 reported beside n so the sample size is visible next to the number.

Writes a JSON result that carries its own source_commit_sha, so the row in
EVIDENCE.md points at the code that produced it rather than at whatever HEAD
happens to be when the registry is committed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag.ablate import CORPUS_PATH, EVAL_K, build_configs  # noqa: E402
from legalrag.dense import load_docs  # noqa: E402
from legalrag.evaluate import load_questions  # noqa: E402


WARMUP_QUERY = "ما هي مدة الإخطار؟"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo_relative(path: Path) -> str:
    """`path` as a repo-relative POSIX string, whether it arrives relative or absolute.

    CORPUS_PATH is relative to the working directory, not to ROOT, so calling
    Path.relative_to(ROOT) on it raises. Normalise instead of assuming.
    """
    resolved = path if path.is_absolute() else (Path.cwd() / path)
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.resolve().as_posix()


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile. Stated plainly because with n=45 the choice shows."""
    if not sorted_values:
        raise ValueError("no samples")
    rank = max(1, min(len(sorted_values), round(q * len(sorted_values) + 0.5)))
    return sorted_values[rank - 1]


def measure(reps: int, warmup: int) -> dict:
    questions, errors = load_questions()
    if errors:
        raise SystemExit("question set did not load: " + "; ".join(errors))
    answerable = [q for q in questions if q.answerable]

    docs = load_docs(CORPUS_PATH)
    if not docs:
        raise SystemExit('corpus not ingested — run `python tasks.py ingest --law "…"`')

    configs = build_configs(docs, answerable, k=EVAL_K)

    # Warm every configuration before timing any of them. Each lazily loads its
    # model on first search, and the configurations share objects — `hybrid`
    # calls the same DenseIndex `dense` does — so warming them one at a time and
    # timing in the same pass charges one configuration for work the next gets
    # free.
    for cfg in configs:
        for _ in range(warmup):
            cfg.search(WARMUP_QUERY, EVAL_K)

    # Interleave: one question through all four configurations, then the next.
    # Timing each configuration to completion in turn makes every result a
    # measurement of a different minute — any ambient load on the machine lands
    # entirely on whichever configuration was running at the time. Measured the
    # other way, an earlier pass of this benchmark put `hybrid` *below* `dense`,
    # which cannot happen: hybrid calls dense. Interleaving spreads drift across
    # all four instead of concentrating it in one.
    samples: dict[str, list[float]] = {c.name: [] for c in configs}
    for _ in range(reps):
        for q in answerable:
            for cfg in configs:
                t0 = time.perf_counter()
                cfg.search(q.question, EVAL_K)
                samples[cfg.name].append((time.perf_counter() - t0) * 1000.0)

    results = []
    for cfg in configs:
        s = sorted(samples[cfg.name])
        results.append(
            {
                "configuration": cfg.name,
                "note": cfg.note,
                "n": len(s),
                "median_ms": round(statistics.median(s), 1),
                "p95_ms": round(_percentile(s, 0.95), 1),
                "min_ms": round(s[0], 1),
                "max_ms": round(s[-1], 1),
                "mean_ms": round(statistics.fmean(s), 1),
            }
        )
        r = results[-1]
        print(
            f"  {cfg.name:<18} median {r['median_ms']:>9.1f} ms   "
            f"p95 {r['p95_ms']:>9.1f} ms   n={r['n']}",
            flush=True,
        )

    return {
        "metric": "retrieval latency per query",
        "source_commit_sha": _git("rev-parse", "HEAD"),
        "worktree_clean": _git("status", "--porcelain") == "",
        "environment_ref": "env-001",
        "dataset": {
            "corpus": _repo_relative(CORPUS_PATH),
            "articles": len(docs),
            "questions_scored": len(answerable),
            "k": EVAL_K,
        },
        "protocol": {
            "warm_up_runs_per_config": warmup,
            "warm_up_excluded": True,
            "model_load_excluded": True,
            "repetitions_per_question": reps,
            "samples_per_config": reps * len(answerable),
            "state": "warm",
            "order": "interleaved — one question through all configurations, then the next",
            "percentile_method": "nearest-rank",
            "concurrency": 1,
            "note": (
                "Single process. p95 over this many samples is coarse — read it "
                "with n, not alone. Ambient machine load is spread across "
                "configurations by interleaving, not eliminated."
            ),
        },
        "results": results,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    payload = measure(a.reps, a.warmup)
    sha8 = payload["source_commit_sha"][:8]
    out = Path(a.out) if a.out else ROOT / "evals" / "registry" / f"latency_{sha8}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    if not payload["worktree_clean"]:
        print("WARNING: worktree was dirty — this sha does not describe what ran.")
