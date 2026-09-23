"""The grounding breakdown divides by answered questions, never answerable ones.

A declined answerable question has `grounded: true` in its row — it made no
claim, so it cited nothing wrong. The first, hand-computed breakdown counted it
as grounded and reported 12 where the headline said 11/29. These tests pin the
denominator so that cannot come back.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "evals" / "grounding_breakdown.py"
    spec = importlib.util.spec_from_file_location("grounding_breakdown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(qid, category="colloquial", *, answerable=True, abstained=False,
         grounded=True, uncited=(), cut=False):
    return {
        "id": qid, "category": category, "answerable": answerable,
        "abstained": abstained, "grounded": grounded, "uncited": list(uncited),
        "stats": [{"cut": cut, "truncated": cut}],
    }


def test_false_abstention_is_not_counted_as_grounded():
    gb = _module()
    rows = [
        _row("Q001", grounded=True),
        _row("Q002", abstained=True, grounded=True),  # declined — row still says grounded
    ]
    cell = gb._tally(rows, lambda r: r["category"])["colloquial"]
    assert cell["answerable"] == 2
    assert cell["answered"] == 1
    assert cell["grounded"] == 1
    assert cell["false_abstention"] == 1


def test_unanswerable_questions_are_ignored():
    gb = _module()
    rows = [_row("Q001"), _row("Q050", answerable=False, abstained=True)]
    assert gb._tally(rows, lambda r: r["category"])["colloquial"]["answerable"] == 1


def test_batches_split_at_q020():
    gb = _module()
    assert gb._batch(_row("Q020")) == "batch_1_Q001_Q020"
    assert gb._batch(_row("Q021")) == "batch_2_Q021_plus"


def test_uncited_and_cut_are_counted_on_answered_rows_only():
    gb = _module()
    rows = [
        _row("Q001", grounded=False, uncited=["claim"], cut=True),
        _row("Q002", abstained=True, grounded=True, cut=True),
    ]
    cell = gb._tally(rows, lambda r: r["category"])["colloquial"]
    assert cell["uncited"] == 1
    assert cell["cut_by_token_cap"] == 1


def test_published_rows_reproduce_the_headline():
    """Against the real saved run: the breakdown must agree with E3's 11/29."""
    gb = _module()
    rows = gb._load(gb.DEFAULT_ROWS)
    by_cat = gb._tally(rows, lambda r: r["category"])
    assert sum(c["answered"] for c in by_cat.values()) == 29
    assert sum(c["grounded"] for c in by_cat.values()) == 11
    assert by_cat["multi_article"]["grounded"] == 0
    assert (by_cat["colloquial"]["grounded"], by_cat["colloquial"]["answered"]) == (4, 8)


def test_a_relative_rows_path_is_resolved_against_the_repository(monkeypatch, tmp_path):
    """`--rows evals/registry/...` crashed in relative_to(ROOT) before this."""
    import sys
    gb = _module()
    out = tmp_path / "breakdown.json"
    monkeypatch.setattr(sys, "argv", ["grounding_breakdown.py", "--out", str(out),
                                      "--rows", "evals/registry/answer_rows_83384b2b.json"])
    gb.main()  # exit code reflects worktree cleanliness, not this behaviour
    assert json.loads(out.read_text(encoding="utf-8"))["rows"] == "evals/registry/answer_rows_83384b2b.json"
