"""The re-weighting check computes its mixes and reports every rank position."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "evals" / "reweighting_check.py"
    spec = importlib.util.spec_from_file_location("reweighting_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mix_is_a_share_of_the_scored_questions():
    rc = _module()
    assert rc._mix(["a", "a", "b", "c"]) == {"a": 0.5, "b": 0.25, "c": 0.25}


def test_reweight_is_the_mix_weighted_sum():
    rc = _module()
    rates = {"direct": 1.0, "multi_article": 0.5, "colloquial": 0.0}
    mix = {"direct": 0.25, "multi_article": 0.5, "colloquial": 0.25}
    assert rc.reweight(rates, mix) == 0.5


def test_ranking_is_descending():
    rc = _module()
    assert rc.ranking({"x": 0.2, "y": 0.9, "z": 0.5}) == ["y", "z", "x"]


def test_published_runs_match_at_the_ends_only():
    """The claim this check exists to keep honest: dense first and BM25 last
    come back under the old mix, the middle two do not."""
    rc = _module()
    import json

    old = {r["configuration"]: r["recall_at_5"]
           for r in json.loads((rc.DEFAULT_OLD).read_text(encoding="utf-8"))["results"]}
    new = json.loads((rc.REGISTRY / "ablation_618693ef.json").read_text(encoding="utf-8"))["results"]
    old_mix = {"colloquial": 0.533, "direct": 0.267, "multi_article": 0.2}
    reweighted = {r["configuration"]: rc.reweight(r["per_category_recall_at_5"], old_mix) for r in new}

    old_rank, new_rank = rc.ranking(old), rc.ranking(reweighted)
    assert old_rank[0] == new_rank[0] == "dense"
    assert old_rank[-1] == new_rank[-1] == "BM25"
    assert old_rank[1:3] != new_rank[1:3]
