"""The repeat check counts agreement question by question, from saved rows only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "evals" / "generation_repeat.py"
    spec = importlib.util.spec_from_file_location("generation_repeat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(qid, *, answerable=True, abstained=False, grounded=True, text="t", retrieved=(1,)):
    return {"id": qid, "answerable": answerable, "abstained": abstained, "grounded": grounded,
            "fabricated": [], "ungrounded": [], "uncited": [] if grounded else ["claim"],
            "text": text, "retrieved_ids": list(retrieved)}


def test_a_false_abstention_is_not_counted_as_grounded():
    gr = _module()
    s = gr.summarise([_row("Q1"), _row("Q2", abstained=True, grounded=True)])
    assert (s["grounded"], s["answered"], s["false_abstention"]) == (1, 1, 1)


def test_abstention_criterion_is_eighty_percent_of_out_of_corpus():
    gr = _module()
    ooc = [_row(f"O{i}", answerable=False, abstained=i < 7, grounded=False) for i in range(10)]
    assert gr.summarise(ooc)["abstention_criterion_met"] is False
    ooc[7]["abstained"] = True
    assert gr.summarise(ooc)["abstention_criterion_met"] is True


def test_compare_separates_retrieval_from_text_and_lists_flips():
    gr = _module()
    a = [_row("Q1", text="x"), _row("Q2", text="y"), _row("O1", answerable=False, abstained=True)]
    b = [_row("Q1", text="x"), _row("Q2", text="z", grounded=False),
         _row("O1", answerable=False, abstained=False)]
    c = gr.compare(a, b)
    assert (c["same_retrieval"], c["same_text"]) == (3, 2)  # Q1 and O1 unchanged
    assert c["grounded_flipped"] == ["Q2"]
    assert c["abstained_flipped"] == ["O1"]


def test_compare_refuses_different_question_sets():
    gr = _module()
    with pytest.raises(SystemExit):
        gr.compare([_row("Q1")], [_row("Q2")])


def test_saved_runs_agree_on_retrieval_and_not_on_text():
    """The finding this script exists to keep: identical retrieval, different generation."""
    gr = _module()
    reg = ROOT / "evals" / "registry"
    a = json.loads((reg / "answer_rows_83384b2b.json").read_text(encoding="utf-8"))
    b = json.loads((reg / "answer_rows_4fe53a93.json").read_text(encoding="utf-8"))
    c = gr.compare(a, b)
    assert c["same_retrieval"] == c["questions"] == 40
    assert c["same_text"] == 14
