"""The variance harness's pure parts: placement parsing and aggregation.

Restarting a server is not unit-testable; what the harness concludes from the
runs it collected is, and that is where a wrong number would come from.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "evals" / "generation_variance.py"
    spec = importlib.util.spec_from_file_location("generation_variance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(qid, text, *, answerable=True, abstained=False, grounded=True):
    return {"id": qid, "answerable": answerable, "abstained": abstained, "grounded": grounded,
            "fabricated": [], "ungrounded": [], "uncited": [] if grounded else ["c"],
            "text": text, "retrieved_ids": [1]}


def _run(texts, *, ooc_abstained=True, q2_grounded=True):
    return {"rows": [_row("Q1", texts[0]), _row("Q2", texts[1], grounded=q2_grounded),
                     _row("O1", "-", answerable=False, abstained=ooc_abstained, grounded=False)]}


def test_offload_is_read_from_the_server_log_every_time_it_appears():
    gv = _module()
    log = ("... load_tensors: offloaded 20/35 layers to GPU\n"
           "... load_tensors: offloaded 18/35 layers to GPU\n")
    assert gv.parse_offload(log) == {"offloaded_layers": ["20/35", "18/35"]}


def test_offload_falls_back_to_the_structured_form():
    gv = _module()
    assert gv.parse_offload('msg=x layers.offload=21 layers.split=""')["offloaded_layers"] == ["21"]


def test_no_offload_line_is_recorded_as_unknown_not_zero():
    gv = _module()
    assert gv.parse_offload("nothing here")["offloaded_layers"] is None


def test_signature_ignores_row_order_and_sees_one_changed_answer():
    gv = _module()
    a = _run(["x", "y"])["rows"]
    assert gv.signature(a) == gv.signature(list(reversed(a)))
    assert gv.signature(a) != gv.signature(_run(["x", "z"])["rows"])


def test_identical_runs_are_one_distinct_output():
    gv = _module()
    agg = gv.aggregate([_run(["x", "y"]) for _ in range(3)])
    assert agg["distinct_outputs"] == 1
    assert agg["questions_that_varied"] == []
    assert agg["ranges"]["grounded"] == [2, 2]


def test_variance_is_located_by_question_and_ranged_by_metric():
    gv = _module()
    agg = gv.aggregate([
        _run(["x", "y"]),
        _run(["x", "z"], q2_grounded=False, ooc_abstained=False),
        _run(["x", "y"]),
    ])
    assert agg["distinct_outputs"] == 2
    assert agg["questions_that_varied"] == ["Q2"]  # O1 abstained differently with the same text
    assert agg["ranges"]["grounded"] == [1, 2]
    assert agg["abstention_criterion_met_in"] == 2
