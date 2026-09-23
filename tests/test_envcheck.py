"""A harness must not stamp an environment it is not running in."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag import envcheck  # noqa: E402

RECORDED = {
    "environment_ref": "env-001",
    "runtime": {"python": "3.14.6"},
    "packages": {"torch": "2.14.0", "sentence-transformers": "6.0.1",
                 "transformers": "5.16.1", "numpy": "2.5.1"},
    "hardware": {"gpu": {"visible_to_torch": False}},
}


def _seen(**over):
    seen = {
        "python": "3.14.6",
        "packages": {"torch": "2.14.0+cpu", "sentence-transformers": "6.0.1",
                     "transformers": "5.16.1", "numpy": "2.5.1"},
        "torch_sees_gpu": False,
    }
    seen.update(over)
    return seen


def test_matching_environment_has_no_mismatches():
    assert envcheck.mismatches(RECORDED, _seen()) == []


def test_local_build_tag_is_not_a_version_difference():
    """torch reports 2.14.0+cpu where the record says 2.14.0."""
    assert envcheck.mismatches(RECORDED, _seen()) == []


def test_the_interpreter_that_actually_ran_is_caught():
    """The real case: Python 3.12, CUDA torch 2.7, older model stack."""
    seen = _seen(
        python="3.12.10",
        packages={"torch": "2.7.1+cu118", "sentence-transformers": "5.5.1",
                  "transformers": "4.57.6", "numpy": "1.26.4"},
        torch_sees_gpu=True,
    )
    problems = envcheck.mismatches(RECORDED, seen)
    joined = "\n".join(problems)
    assert len(problems) == 6
    assert "python 3.12.10" in joined
    assert "torch 2.7.1" in joined
    assert "different device" in joined


def test_gpu_visibility_alone_is_a_mismatch():
    assert envcheck.mismatches(RECORDED, _seen(torch_sees_gpu=True))


def test_require_stops_on_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(RECORDED), encoding="utf-8")
    monkeypatch.setattr(envcheck, "observed", lambda: _seen(python="3.12.10"))
    with pytest.raises(envcheck.EnvironmentMismatch, match="refusing to stamp env-001"):
        envcheck.require("env-001", path)


def test_require_returns_what_ran(tmp_path, monkeypatch):
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(RECORDED), encoding="utf-8")
    monkeypatch.setattr(envcheck, "observed", lambda: _seen())
    stamp = envcheck.require("env-001", path)
    assert stamp["environment_ref"] == "env-001"
    assert stamp["runtime_observed"]["python"] == "3.14.6"


def test_require_refuses_an_unknown_ref(tmp_path):
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(RECORDED), encoding="utf-8")
    with pytest.raises(envcheck.EnvironmentMismatch, match="not 'env-002'"):
        envcheck.require("env-002", path)
