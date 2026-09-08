#!/usr/bin/env python3
"""Task runner — the canonical entry point on every platform.

    python tasks.py ingest --law "…"      data/raw/*.pdf -> data/processed/articles.jsonl
    python tasks.py verify-refs [--write] check ground-truth article references
    python tasks.py eval                  load the eval set, print the scoreboard
    python tasks.py ablate                4 retrieval configs compared (M1/A4)
    python tasks.py broken-words          find words split by a stray space (ADR-018)
    python tasks.py serve [--port 8000]   local test page (retrieval + eval run)
    python tasks.py test                  run the test suite
    python tasks.py setup                 install dependencies
    python tasks.py all --law "…"         ingest -> verify-refs -> eval

`make <target>` does the same thing on Linux/CI; this file is what runs on
Windows, where `make` is usually absent. Both must stay in sync.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def _run_module(module: str, *args: str) -> int:
    sys.path.insert(0, str(SRC))
    if module == "legalrag.evaluate":
        from legalrag.evaluate import main
        return main()
    if module == "legalrag.ingest":
        from legalrag.ingest import main
        return main(list(args))
    if module == "legalrag.verify_refs":
        from legalrag.verify_refs import main
        return main(list(args))
    if module == "legalrag.broken_words":
        from legalrag.broken_words import main
        return main(list(args))
    if module == "legalrag.ablate":
        from legalrag.ablate import main
        return main(list(args))
    if module == "legalrag.server":
        from legalrag.server import main
        return main(list(args))
    raise ValueError(module)


def _subprocess(*cmd: str) -> int:
    env_note = f"  $ {' '.join(cmd)}"
    print(env_note)
    return subprocess.call(cmd, cwd=ROOT)


TASKS = {
    "eval": lambda a: _run_module("legalrag.evaluate"),
    "ablate": lambda a: _run_module("legalrag.ablate", *a),
    "broken-words": lambda a: _run_module("legalrag.broken_words", *a),
    "ingest": lambda a: _run_module("legalrag.ingest", *a),
    "serve": lambda a: _run_module("legalrag.server", *a),
    "verify-refs": lambda a: _run_module("legalrag.verify_refs", *a),
    "test": lambda a: _subprocess(sys.executable, "-m", "pytest", "-q"),
    "setup": lambda a: _subprocess(
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ),
}


def task_all(args: list[str]) -> int:
    """ingest -> verify-refs -> eval. `--law` is forwarded to ingest."""
    for name in ("ingest", "verify-refs", "eval"):
        print(f"\n>>> {name}")
        code = TASKS[name](args if name == "ingest" else [])
        if code != 0:
            print(f"\n{name} failed (exit {code}) — stopping.")
            return code
    return 0


TASKS["all"] = task_all


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    name = sys.argv[1]
    if name not in TASKS:
        print(f"unknown task: {name}\n")
        print(__doc__)
        return 1
    import os

    os.chdir(ROOT)  # every path in the project is relative to the repo root
    return TASKS[name](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
