"""Which commit produced a result file, and whether the worktree matched it.

`answer_eval` and every harness under `evals/` stamp `source_commit_sha` and
`worktree_clean` from here. Changes under `evals/registry/` do not make the
tree dirty: that is where the harnesses write their own output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWN_OUTPUT = "evals/registry/"


def git(*args: str, cwd: Path = ROOT) -> str:
    """`git <args>` run in `cwd`, its stdout stripped. Raises CalledProcessError on failure."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def worktree_clean(cwd: Path = ROOT) -> bool:
    """True when nothing outside `evals/registry/` is uncommitted, tracked or not."""
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return all(line[3:].strip().strip('"').startswith(OWN_OUTPUT) for line in status)


def commit_state(cwd: Path = ROOT) -> dict:
    """`{"source_commit_sha", "worktree_clean"}`, both None outside a git checkout."""
    try:
        sha = git("rev-parse", "HEAD", cwd=cwd)
        clean = worktree_clean(cwd)
    except (OSError, subprocess.CalledProcessError):
        return {"source_commit_sha": None, "worktree_clean": None}
    return {"source_commit_sha": sha, "worktree_clean": clean}
