"""The rule that turns a recorded GPU share into a layer count."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "evals" / "placement_probe.py"
    spec = importlib.util.spec_from_file_location("placement_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _p(n, share):
    return {"num_gpu": n, "gpu_share": share}


def test_the_nearest_share_wins():
    pp = _module()
    assert pp.closest([_p(0, 0.0), _p(4, 0.07), _p(6, 0.11), _p(8, 0.15)], 0.09) == 4


def test_a_tie_goes_to_fewer_layers():
    pp = _module()
    assert pp.closest([_p(4, 0.07), _p(6, 0.11)], 0.09) == 4


def test_a_probe_with_no_share_is_skipped_not_read_as_zero():
    pp = _module()
    assert pp.closest([_p(0, None), _p(6, 0.10)], 0.0) == 6


def test_no_usable_probe_refuses():
    pp = _module()
    with pytest.raises(SystemExit):
        pp.closest([_p(0, None)], 0.09)
