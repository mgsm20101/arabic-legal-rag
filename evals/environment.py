"""Write evals/environment.json — the environment every measured number was produced in.

Run from the repository root:  python evals/environment.py

EVIDENCE.md names an `environment_ref` instead of repeating these fields on every
row. Regenerate this file whenever the machine, the interpreter or a pinned
dependency changes, and record the new ref on rows measured afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "environment.json" if False else ROOT / "evals" / "environment.json"

# Models resolved from the local Hugging Face cache; the revision is the snapshot
# actually on disk, not whatever `main` points at today.
HF_MODELS = {
    "embedding": "intfloat/multilingual-e5-base",
    "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _hf_revision(repo_id: str) -> str | None:
    """The snapshot on disk for `repo_id`, or None if it is not cached here."""
    cache = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache/huggingface/hub")
    ref = cache / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def _pkg_versions(names: list[str]) -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def _ollama_models() -> list[dict] | None:
    """Digest and quantization of what the running Ollama actually serves.

    The digest matters: a tag can be repointed, so `gemma3:4b` alone does not
    identify the weights a number was produced with.
    """
    try:
        import httpx

        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
        r.raise_for_status()
        return [
            {
                "name": m["name"],
                "digest": m["digest"],
                "parameter_size": m["details"].get("parameter_size"),
                "quantization": m["details"].get("quantization_level"),
            }
            for m in r.json().get("models", [])
        ]
    except Exception:
        return None


def _total_ram_gb() -> float | None:
    """Installed RAM in GiB, or None if the platform will not say.

    wmic is gone from current Windows builds, so ask CIM; fall back to the POSIX
    sysconf pair elsewhere.
    """
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=30,
            ).stdout
            digits = "".join(c for c in out if c.isdigit())
            return round(int(digits) / 1024**3, 1) if digits else None
        except Exception:
            return None
    try:
        return round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1
        )
    except (ValueError, OSError, AttributeError):
        return None


def build() -> dict:
    return {
        "environment_ref": "env-001",
        "generated": date.today().isoformat(),
        "generator": "python evals/environment.py",
        "hardware": {
            "cpu": platform.processor(),
            "logical_cores": os.cpu_count(),
            "total_ram_gb": _total_ram_gb(),
            "gpu": None,
            "note": "CPU-only. Retrieval and reranking run on CPU; no CUDA device present.",
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": platform.platform(),
        },
        "dependency_pins": {
            "constraints.txt": _sha256(ROOT / "constraints.txt"),
            "requirements.txt": _sha256(ROOT / "requirements.txt"),
            "requirements-dev.txt": _sha256(ROOT / "requirements-dev.txt"),
        },
        "packages": _pkg_versions(
            ["torch", "sentence-transformers", "transformers", "numpy",
             "rank-bm25", "pdfplumber", "fastapi", "httpx", "pytest"]
        ),
        "models": {
            role: {"repo_id": repo, "revision": _hf_revision(repo), "device": "cpu"}
            for role, repo in HF_MODELS.items()
        },
        "generator_models": {
            "served_by": "ollama @ http://127.0.0.1:11434",
            "available": _ollama_models(),
        },
        "decoding": {
            "note": "Retrieval and reranking are deterministic and take no decoding "
                    "parameters. Generation decoding is set per run and recorded on "
                    "that run's row in EVIDENCE.md, not here.",
            "seed": None,
            "temperature": None,
            "top_p": None,
        },
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    sys.exit(0)
