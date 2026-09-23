"""Refuse to stamp an environment_ref the running interpreter does not match.

Every registry row names `environment_ref: env-001` instead of repeating the
interpreter, the package versions and whether torch can see a GPU. That only
works if the harness checks the claim. It did not: the three measuring
harnesses wrote the string `"env-001"` unconditionally, so a run launched from a
shell whose `python` resolved to a different interpreter — Python 3.12 with a
CUDA build of torch 2.7, sentence-transformers 5.5 — was stamped as the
recorded Python 3.14 / torch 2.14+cpu environment. Its retrieval numbers came
out identical; its latency numbers came out 5-11x faster, because the reranker
had quietly run on the GPU. Nothing in the output file could have told a reader.

`require()` compares what is actually running with `evals/environment.json` and
raises before a single measurement is taken. `observed()` is also written into
every result, so the file says what ran rather than only what was meant to.
"""

from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "evals" / "environment.json"

# The packages whose version changes a measured number: the model stack for
# retrieval quality and latency, numpy for the arithmetic under both.
CHECKED_PACKAGES = ("torch", "sentence-transformers", "transformers", "numpy")


class EnvironmentMismatch(SystemExit):
    """Raised by `require` — a SystemExit so a harness stops with the message."""


def _base(version: str | None) -> str | None:
    """`2.14.0+cpu` -> `2.14.0`: the local build tag is compared separately,
    through whether torch can see a GPU, which is the part that moves latency."""
    return version.split("+", 1)[0] if version else version


def _torch_sees_gpu() -> bool | None:
    try:
        import torch
    except ImportError:
        return None
    return bool(torch.cuda.is_available())


def observed() -> dict:
    """What is actually running, in the same shape `environment.json` records."""
    packages = {}
    for name in CHECKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_sees_gpu": _torch_sees_gpu(),
    }


def mismatches(recorded: dict, seen: dict) -> list[str]:
    """Human-readable differences between a recorded environment and `seen`."""
    problems = []
    want_py = recorded.get("runtime", {}).get("python")
    if want_py != seen["python"]:
        problems.append(f"python {seen['python']} (recorded {want_py})")
    for name in CHECKED_PACKAGES:
        want = _base(recorded.get("packages", {}).get(name))
        got = _base(seen["packages"].get(name))
        if want != got:
            problems.append(f"{name} {got} (recorded {want})")
    want_gpu = recorded.get("hardware", {}).get("gpu", {}).get("visible_to_torch")
    if want_gpu is not None and seen["torch_sees_gpu"] != want_gpu:
        problems.append(
            f"torch sees a GPU: {seen['torch_sees_gpu']} (recorded {want_gpu}) — "
            "models would run on a different device"
        )
    return problems


def load_recorded(path: Path = ENV_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ref: str = "env-001", path: Path = ENV_PATH) -> dict:
    """Return `{"environment_ref", "runtime_observed"}` for a result file, or
    stop the harness when the running interpreter is not the recorded one."""
    recorded = load_recorded(path)
    if recorded.get("environment_ref") != ref:
        raise EnvironmentMismatch(
            f"{path.name} records {recorded.get('environment_ref')!r}, not {ref!r}"
        )
    seen = observed()
    problems = mismatches(recorded, seen)
    if problems:
        import sys

        raise EnvironmentMismatch(
            f"refusing to stamp {ref}: this interpreter is not the recorded environment.\n"
            f"  running: {sys.executable}\n  - "
            + "\n  - ".join(problems)
            + "\nRun the harness with the interpreter evals/environment.json describes, "
            "or record a new environment with `python evals/environment.py` and a new ref."
        )
    return {"environment_ref": ref, "runtime_observed": seen}
