"""Which `num_gpu` gives which GPU share? — one load per value, no evaluation.

    python evals/placement_probe.py --layers 0 2 4 6 8 10 12

For each layer count: restart Ollama, load the model with that pin through one
one-token request, and read back what the server did — its own "offloaded N/M
layers" line and `/api/ps`'s size_vram/size. Nothing is generated beyond one
token and nothing is scored.

It exists to translate a *recorded* GPU share into a layer count. The run at
83384b2b recorded "GPU share 9%" and no layer count; reproducing its placement
means finding the pin that yields 9%, measured rather than guessed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "evals" / "registry"
MODEL_NAME = "gemma3:4b"


def _variance():
    spec = importlib.util.spec_from_file_location(
        "generation_variance", ROOT / "evals" / "generation_variance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def closest(probes: list[dict], target_share: float) -> int:
    """The layer count whose measured share is nearest `target_share`;
    the smaller count on a tie (fewer layers is the conservative reading of
    a share recorded rounded to a whole percent)."""
    usable = [p for p in probes if p["gpu_share"] is not None]
    if not usable:
        raise SystemExit("no probe reported a GPU share")
    return min(usable, key=lambda p: (abs(p["gpu_share"] - target_share), p["num_gpu"]))["num_gpu"]


def probe(gv, exe: str, layers: int, workdir: Path) -> dict:
    gv.stop_ollama()
    log = workdir / f"probe{layers}.ollama.log"
    proc = gv.start_ollama(exe, log)
    try:
        httpx.post(f"{gv.HOST}/api/chat", timeout=600, json={
            "model": MODEL_NAME, "stream": False, "keep_alive": "5m",
            "messages": [{"role": "user", "content": "مرحبا"}],
            "options": {"num_predict": 1, "num_gpu": layers, "temperature": 0, "seed": 0},
        }).raise_for_status()
        loaded = httpx.get(f"{gv.HOST}/api/ps", timeout=10).json().get("models", [])
        m = next((x for x in loaded if x.get("name", "").startswith(MODEL_NAME)), None)
        share = round(m["size_vram"] / m["size"], 4) if m and m.get("size") else None
    finally:
        proc.terminate()
        gv.stop_ollama()
    return {"num_gpu": layers, "gpu_share": share,
            **gv.parse_offload(log.read_text(encoding="utf-8", errors="replace"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--target-share", type=float, default=None,
                    help="report the layer count whose share is nearest this")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from legalrag.envcheck import require as require_environment

    environment = require_environment()
    gv = _variance()
    exe = os.environ.get("LEGALRAG_OLLAMA_EXE")
    if not exe or not Path(exe).exists():
        raise SystemExit("set LEGALRAG_OLLAMA_EXE to the ollama binary this machine serves from")
    if not gv._source_is_clean():
        raise SystemExit("worktree is dirty — commit first")

    workdir = gv.RUNS / "variance" / "probe"
    workdir.mkdir(parents=True, exist_ok=True)
    probes = []
    for n in a.layers:
        p = probe(gv, exe, n, workdir)
        print(f"num_gpu {n:>2}  offloaded {p['offloaded_layers']}  gpu_share {p['gpu_share']}",
              flush=True)
        probes.append(p)

    sha = gv._git("rev-parse", "HEAD")
    out = {
        "metric": "GPU share as a function of the pinned layer count",
        "source_commit_sha": sha,
        "worktree_clean": gv._source_is_clean(),
        **environment,
        "command": "python evals/placement_probe.py --layers " + " ".join(map(str, a.layers))
                   + (f" --target-share {a.target_share}" if a.target_share is not None else ""),
        "model": MODEL_NAME,
        "probes": probes,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if a.target_share is not None:
        out["target_share"] = a.target_share
        out["closest_num_gpu"] = closest(probes, a.target_share)
        print(f"closest to {a.target_share}: num_gpu {out['closest_num_gpu']}")
    dest = REGISTRY / f"placement_probe_{sha[:8]}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
