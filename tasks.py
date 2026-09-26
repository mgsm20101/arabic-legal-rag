#!/usr/bin/env python3
"""Task runner — the canonical entry point on every platform.

    python tasks.py ingest --law "…"      data/raw/*.pdf -> data/processed/articles.jsonl
    python tasks.py verify-refs [--write] check ground-truth article references
    python tasks.py eval                  load the eval set, print the scoreboard
    python tasks.py ablate                4 retrieval configs compared (M1/A4)
    python tasks.py adversarial           named failure-mode probes (not a score)
    python tasks.py broken-words          find words split by a stray space (ADR-018)
    python tasks.py ocr-gate <json>       score an OCR engine on digit accuracy (ADR-019)
    python tasks.py ocr-to-raw <in> <out> OCR pages -> raw text for ingest (ADR-020)
    python tasks.py answer-eval [--model SPEC] [--contract text|json|gated] [--report-only] [--overwrite]
                                          end-to-end answers: citations + abstention (M2)
    python tasks.py app-eval [--retrieval-only] [--doc pdf|txt] [--model SPEC] [--overwrite]
                                          the upload pipeline on the app-dev split (ADR-023)
    python tasks.py serve [--port 8000]   legacy benchmark test page (BM25 + eval run)
    python tasks.py app [--host 127.0.0.1] [--port 8000]
                                          the local app: upload a document and ask it (ADR-023)
    python tasks.py reindex [--dry-run]   recompute stored documents' embeddings after an upgrade
    python tasks.py test                  run the test suite (after setup)
    python tasks.py setup                 install the app and test dependencies at the pinned versions
    python tasks.py all --law "…"         ingest -> verify-refs -> eval

The only runner: it needs nothing but Python, on every platform.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def _run_module(module: str, args: list[str] | None) -> int:
    """Import `legalrag.<module>` and call its `main`; `args=None` for a main that takes none."""
    sys.path.insert(0, str(SRC))
    main = importlib.import_module(f"legalrag.{module}").main
    return main() if args is None else main(list(args))


def _subprocess(*cmd: str) -> int:
    env_note = f"  $ {' '.join(cmd)}"
    print(env_note)
    return subprocess.call(cmd, cwd=ROOT)


# command -> module under src/legalrag/ whose main() it runs
MODULE_TASKS = {
    "ingest": "ingest",
    "verify-refs": "verify_refs",
    "ablate": "ablate",
    "adversarial": "adversarial",
    "broken-words": "broken_words",
    "ocr-gate": "ocr_gate",
    "ocr-to-raw": "ocr_text",
    "answer-eval": "answer_eval",
    "app-eval": "app_eval",
    "serve": "server",
    "app": "webapp",
    "reindex": "reindex",
}

TASKS = {
    "eval": lambda a: _run_module("evaluate", None),  # takes no arguments
    **{name: (lambda a, m=module: _run_module(m, a)) for name, module in MODULE_TASKS.items()},
    "test": lambda a: _subprocess(sys.executable, "-m", "pytest", "-q"),
    "setup": lambda a: _subprocess(
        sys.executable, "-m", "pip", "install", "-c", "constraints.txt", "-r", "requirements-dev.txt"
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
