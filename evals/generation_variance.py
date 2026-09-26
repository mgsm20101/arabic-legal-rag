"""How much does a generation run vary across Ollama server restarts?

    python evals/generation_variance.py --runs 5 --label auto
    python evals/generation_variance.py --runs 5 --label pinned --num-gpu 20

Each run stops every Ollama server, starts a fresh one with its log captured,
records the VRAM already in use by other processes, runs the full answer
evaluation, and keeps the rows, the metadata and the server's own report of
how many layers it offloaded to the GPU. The result file holds every distinct
set of answers, how many runs produced each, which questions changed, and the
range of every headline metric.

Why this exists. E3c showed that greedy decoding on one model digest repeated
exactly within a server session and differed on 26 of 40 answers across one.
Two sessions show the variance exists; they cannot size it, and they cannot
say where it comes from. The hypothesis is GPU placement: Ollama picks how
many layers to offload per session from the VRAM free at load time, and a
different CPU/GPU split changes floating-point results enough to change an
argmax. `--num-gpu` pins the split, so the same harness run with and without
it is the test.

Machine-specific by necessity: it restarts a local server. The binary comes
from `$LEGALRAG_OLLAMA_EXE`, and the server inherits this process's
environment (`OLLAMA_MODELS`, `OLLAMA_HOST`), so run it from the same shell
that normally starts Ollama. Runs are sequential and slow — each is a full
40-question generation run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "evals" / "registry"
RUNS = ROOT / "runs"
MODEL = "ollama:gemma3:4b"
ROWS_FILE = RUNS / "answer_eval-ollama-gemma3-4b.json"
META_FILE = RUNS / "answer_eval-ollama-gemma3-4b.meta.json"
HOST = "http://127.0.0.1:11434"
START_TIMEOUT_S = 180

sys.path.insert(0, str(ROOT / "src"))

from legalrag.provenance import git as _git  # noqa: E402
from legalrag.provenance import worktree_clean as _source_is_clean  # noqa: E402


def _repeat_module():
    spec = importlib.util.spec_from_file_location(
        "generation_repeat", ROOT / "evals" / "generation_repeat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- pure parts

_OFFLOAD_PATTERNS = (
    re.compile(r"offloaded (\d+)/(\d+) layers to GPU"),
    re.compile(r"layers\.offload=(\d+)"),
)


def parse_offload(log_text: str) -> dict:
    """The server's own statement of the CPU/GPU split, from its log.

    Every match is kept, not just the last: a model loaded twice in one session
    with different splits is itself a finding, and hiding it would hide that.
    """
    found = [f"{m.group(1)}/{m.group(2)}" for m in _OFFLOAD_PATTERNS[0].finditer(log_text)]
    if not found:
        found = [m.group(1) for m in _OFFLOAD_PATTERNS[1].finditer(log_text)]
    return {"offloaded_layers": found or None}


def signature(rows: list[dict]) -> str:
    """One hash for one run's generated text, question by question."""
    pairs = sorted((r["id"], r["text"]) for r in rows)
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def aggregate(runs: list[dict]) -> dict:
    """What varied across `runs` — each {"rows": [...], ...}.

    Distinct outputs are counted at two levels: whole runs (did this run
    reproduce another exactly?) and single questions (which answers move?).
    """
    rm = _repeat_module()
    sigs = [signature(r["rows"]) for r in runs]
    summaries = [rm.summarise(r["rows"]) for r in runs]

    ids = sorted(r["id"] for r in runs[0]["rows"])
    texts = {q: Counter() for q in ids}
    for run in runs:
        for row in run["rows"]:
            texts[row["id"]][row["text"]] += 1
    unstable = sorted(q for q, c in texts.items() if len(c) > 1)

    def span(key):
        values = [s[key] for s in summaries]
        return [min(values), max(values)]

    return {
        "runs": len(runs),
        "distinct_outputs": len(set(sigs)),
        "run_signatures": sigs,
        "questions": len(ids),
        "questions_with_identical_text_in_every_run": len(ids) - len(unstable),
        "questions_that_varied": unstable,
        "ranges": {k: span(k) for k in (
            "grounded", "answered", "fabricated", "cited_not_retrieved", "uncited",
            "false_abstention", "abstained_out_of_corpus")},
        "abstention_criterion_met_in": sum(s["abstention_criterion_met"] for s in summaries),
        "per_run": summaries,
    }


# ---------------------------------------------------------------- the server


def _server_up() -> bool:
    try:
        return httpx.get(f"{HOST}/api/tags", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def stop_ollama() -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    deadline = time.monotonic() + 30
    while _server_up() and time.monotonic() < deadline:
        time.sleep(1)
    if _server_up():
        raise SystemExit("an Ollama server is still answering after being stopped")


def start_ollama(exe: str, log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([exe, "serve"], stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(Path(exe).parent), env=dict(os.environ))
    deadline = time.monotonic() + START_TIMEOUT_S
    while not _server_up():
        if proc.poll() is not None or time.monotonic() > deadline:
            raise SystemExit(f"Ollama did not come up; see {log_path}")
        time.sleep(2)
    return proc


def vram_used_mib() -> int | None:
    """VRAM in use before the model loads — other processes' share of the card."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------- one run


def run_once(k: int, exe: str, workdir: Path, num_gpu: int | None) -> dict:
    stop_ollama()
    vram_before = vram_used_mib()
    server_log = workdir / f"run{k}.ollama.log"
    proc = start_ollama(exe, server_log)
    try:
        cmd = [sys.executable, str(ROOT / "tasks.py"), "answer-eval",
               "--model", MODEL, "--overwrite"]
        if num_gpu is not None:
            cmd += ["--num-gpu", str(num_gpu)]
        t0 = time.monotonic()
        with (workdir / f"run{k}.eval.log").open("w", encoding="utf-8", errors="replace") as out:
            code = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
        seconds = round(time.monotonic() - t0, 1)
    finally:
        proc.terminate()
        stop_ollama()

    # answer-eval exits 1 when a pre-registered criterion fails; that is a
    # result, not an error. Anything else, or no rows written, is an error.
    if code not in (0, 1) or not ROWS_FILE.exists():
        raise SystemExit(f"run {k}: answer-eval exited {code}; see {workdir}")
    rows_copy = workdir / f"run{k}.rows.json"
    meta_copy = workdir / f"run{k}.meta.json"
    shutil.copyfile(ROWS_FILE, rows_copy)
    shutil.copyfile(META_FILE, meta_copy)
    meta = json.loads(meta_copy.read_text(encoding="utf-8"))
    return {
        "run": k,
        "rows": json.loads(rows_copy.read_text(encoding="utf-8")),
        "vram_used_by_others_mib": vram_before,
        "gpu_share": meta.get("gpu_share"),
        "num_gpu_requested": meta.get("num_gpu_requested"),
        "ran_at": meta.get("source_commit_sha"),
        "worktree_clean": meta.get("worktree_clean"),
        "runtime": meta.get("runtime"),
        "wall_seconds": seconds,
        **parse_offload(server_log.read_text(encoding="utf-8", errors="replace")),
    }


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--label", required=True, help="names this arm in the output file")
    ap.add_argument("--num-gpu", type=int, default=None,
                    help="pin the GPU layer count; omit to let Ollama choose")
    a = ap.parse_args()

    from legalrag.envcheck import require as require_environment

    environment = require_environment()
    exe = os.environ.get("LEGALRAG_OLLAMA_EXE")
    if not exe or not Path(exe).exists():
        raise SystemExit("set LEGALRAG_OLLAMA_EXE to the ollama binary this machine serves from")
    if not _source_is_clean():
        raise SystemExit("worktree is dirty — commit first; the runs record the commit they ran at")
    if a.runs < 2:
        raise SystemExit("variance needs at least two runs")

    sha = _git("rev-parse", "HEAD")
    workdir = RUNS / "variance" / a.label
    workdir.mkdir(parents=True, exist_ok=True)

    runs = []
    for k in range(1, a.runs + 1):
        print(f"[{a.label}] run {k}/{a.runs} — restarting Ollama", flush=True)
        run = run_once(k, exe, workdir, a.num_gpu)
        if run["ran_at"] != sha:
            raise SystemExit(f"run {k} recorded commit {run['ran_at']}, expected {sha}")
        s = _repeat_module().summarise(run["rows"])
        print(f"    grounded {s['grounded']}/{s['answered']}  abstained "
              f"{s['abstained_out_of_corpus']}/{s['out_of_corpus']}  offload "
              f"{run['offloaded_layers']}  gpu_share {run['gpu_share']}  "
              f"vram_before {run['vram_used_by_others_mib']} MiB  sig {signature(run['rows'])}",
              flush=True)
        runs.append(run)

    agg = aggregate(runs)
    unique_rows = {}
    for run in runs:
        unique_rows.setdefault(signature(run["rows"]), run["rows"])

    out = {
        "metric": "generation variance across Ollama server restarts",
        "source_commit_sha": sha,
        "worktree_clean": _source_is_clean(),
        **environment,
        "command": " ".join(["python evals/generation_variance.py", f"--runs {a.runs}",
                             f"--label {a.label}"]
                            + ([f"--num-gpu {a.num_gpu}"] if a.num_gpu is not None else [])),
        "protocol": {
            "model": MODEL,
            "num_gpu": a.num_gpu if a.num_gpu is not None else "chosen by Ollama per session",
            "restart_before_every_run": True,
            "decoding": "greedy, temperature 0, seed 0, 128-token cap",
        },
        "summary": agg,
        "placement": [{k: run[k] for k in (
            "run", "offloaded_layers", "gpu_share", "num_gpu_requested",
            "vram_used_by_others_mib", "wall_seconds")} for run in runs],
        "rows_by_signature": unique_rows,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    dest = REGISTRY / f"generation_variance_{sha[:8]}_{a.label}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{agg['distinct_outputs']} distinct outputs in {agg['runs']} runs; "
          f"{agg['questions_with_identical_text_in_every_run']}/{agg['questions']} questions "
          f"identical in every run; B2 met in {agg['abstention_criterion_met_in']}/{agg['runs']}")
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
