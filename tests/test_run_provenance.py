"""A saved run must not outlive the question set it scored.

These cover a bug that actually shipped a registry row: the promoter reads
answers out of the git-ignored `runs/` directory and writes them into
`evals/registry/` under the current commit, and it did that for answers
generated days earlier against a 20-question set. The row it produced said
`questions_total: 20` and `split: {dev: 40}` in the same object, under a commit
that had never produced those answers.

The registry's whole value is that a number's `source_commit_sha` names the
code that produced it. A promoter that cannot tell stale rows from fresh ones
breaks that for every row it writes, so this is guarded rather than remembered.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag.evaluate import Question, question_set_fingerprint  # noqa: E402


def _q(qid="Q1", question="سؤال", expected=("law-1",), **over):
    kw = {
        "id": qid,
        "category": "direct",
        "question": question,
        "expected_articles": list(expected),
        "expected_keywords": ["كلمة"],
        "answerable": True,
        "ref_status": "VERIFIED",
        "split": "dev",
    }
    kw.update(over)
    return Question(**kw)


def _promoter():
    """Load the promoter script as a module; it is not an installed package."""
    path = ROOT / "evals" / "answer_eval_result.py"
    spec = importlib.util.spec_from_file_location("answer_eval_result", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- fingerprint


def test_reordering_the_questions_does_not_change_the_fingerprint():
    a = [_q("Q1"), _q("Q2", question="ثان")]
    assert question_set_fingerprint(a) == question_set_fingerprint(list(reversed(a)))


def test_rewording_a_question_changes_the_fingerprint():
    """Same id, different question. Re-using an old answer for it would be
    silent, which is the failure mode worth catching."""
    before = question_set_fingerprint([_q("Q1", question="ما مدة الإجازة؟")])
    after = question_set_fingerprint([_q("Q1", question="كم يوماً الإجازة؟")])

    assert before != after


def test_changing_what_a_question_expects_changes_the_fingerprint():
    before = question_set_fingerprint([_q("Q1", expected=("law-1",))])
    after = question_set_fingerprint([_q("Q1", expected=("law-1", "law-2"))])

    assert before != after


def test_adding_a_question_changes_the_fingerprint():
    before = question_set_fingerprint([_q("Q1")])
    after = question_set_fingerprint([_q("Q1"), _q("Q2")])

    assert before != after


def test_moving_a_question_between_splits_changes_the_fingerprint():
    """dev and test are different eval sets, so they are different fingerprints
    even when every question is otherwise identical."""
    before = question_set_fingerprint([_q("Q1", split="dev")])
    after = question_set_fingerprint([_q("Q1", split="test")])

    assert before != after


# --------------------------------------------------------------------- guard


@pytest.fixture
def promoter(monkeypatch):
    mod = _promoter()
    monkeypatch.setattr(mod, "load_questions", lambda *a, **k: ([_q("Q1"), _q("Q2")], []))
    return mod


def _meta(mod, monkeypatch, value):
    monkeypatch.setattr(mod, "_read_run_meta", lambda *a, **k: value)


def test_matching_fingerprint_promotes(promoter, monkeypatch):
    fp = question_set_fingerprint([_q("Q1"), _q("Q2")])
    _meta(promoter, monkeypatch, {"question_set": {"fingerprint": fp, "n": 2,
                                                   "ids": ["Q1", "Q2"]}})

    assert promoter._stale_against_current_questions("ollama:m", "text") is None


def test_a_run_with_no_fingerprint_is_refused(promoter, monkeypatch):
    """Absence is refused, not waved through: rows written before the
    fingerprint existed cannot be told apart from rows written against a
    question set that has since changed."""
    _meta(promoter, monkeypatch, {"model": "ollama:m", "date": "2026-09-23"})

    problem = promoter._stale_against_current_questions("ollama:m", "text")

    assert problem is not None
    assert "fingerprint" in problem
    assert "--overwrite" in problem


def test_a_missing_meta_file_is_refused(promoter, monkeypatch):
    _meta(promoter, monkeypatch, None)

    assert promoter._stale_against_current_questions("ollama:m", "text") is not None


def test_a_grown_question_set_is_refused_and_names_what_changed(promoter, monkeypatch):
    """The real case: 20 saved rows, 40 questions on disk."""
    _meta(promoter, monkeypatch, {
        "question_set": {"fingerprint": "stale", "n": 1, "ids": ["Q1"]},
    })

    problem = promoter._stale_against_current_questions("ollama:m", "text")

    assert problem is not None
    assert "Q2" in problem
    assert "1 added since that run" in problem


def test_a_shrunk_question_set_names_the_missing_ones(promoter, monkeypatch):
    _meta(promoter, monkeypatch, {
        "question_set": {"fingerprint": "stale", "n": 3,
                         "ids": ["Q1", "Q2", "Q9"]},
    })

    problem = promoter._stale_against_current_questions("ollama:m", "text")

    assert "Q9" in problem
    assert "no longer present" in problem


def test_same_ids_but_different_wording_still_refuses(promoter, monkeypatch):
    """The id sets match exactly, so the diff has nothing to name — the message
    still has to say why it refused."""
    _meta(promoter, monkeypatch, {
        "question_set": {"fingerprint": "stale", "n": 2, "ids": ["Q1", "Q2"]},
    })

    problem = promoter._stale_against_current_questions("ollama:m", "text")

    assert "wording or expected articles changed" in problem


def test_an_unloadable_question_set_refuses_before_anything_else(promoter, monkeypatch):
    monkeypatch.setattr(promoter, "load_questions", lambda *a, **k: ([], ["Q7: bad"]))

    problem = promoter._stale_against_current_questions("ollama:m", "text")

    assert "Q7: bad" in problem


# --------------------------------------------------- the meta is always written


def test_every_backend_writes_a_fingerprint_not_just_ollama(tmp_path, monkeypatch):
    """`hf:` runs need the guard as much as `ollama:` ones, and the metadata
    used to be written only for Ollama."""
    from legalrag import answer_eval as ae

    monkeypatch.setattr(ae, "RUNS", tmp_path)
    questions = [_q("Q1"), _q("Q2")]

    path = ae._write_run_meta("hf:some/model", tmp_path / "rows.json", questions)
    meta = json.loads(path.read_text(encoding="utf-8"))

    assert meta["model"] == "hf:some/model"
    assert meta["question_set"]["fingerprint"] == question_set_fingerprint(questions)
    assert meta["question_set"]["ids"] == ["Q1", "Q2"]


def test_ollama_details_survive_alongside_the_fingerprint(tmp_path, monkeypatch):
    from legalrag import answer_eval as ae

    monkeypatch.setattr(ae, "RUNS", tmp_path)

    path = ae._write_run_meta(
        "ollama:gemma3:4b", tmp_path / "rows.json", [_q("Q1")],
        {"digest": "abc", "quantization": "Q4_K_M", "gpu_share": 0.55},
    )
    meta = json.loads(path.read_text(encoding="utf-8"))

    assert meta["digest"] == "abc"
    assert meta["quantization"] == "Q4_K_M"
    assert "fingerprint" in meta["question_set"]


# ---------------------------------------------------------------- commit


def test_a_run_records_the_commit_it_ran_at(tmp_path, monkeypatch):
    import legalrag.answer_eval as ae

    monkeypatch.setattr(ae, "RUNS", tmp_path)
    path = ae._write_run_meta("hf:some/model", tmp_path / "rows.json", [_q("Q1")])
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert len(meta["source_commit_sha"]) == 40
    assert meta["worktree_clean"] in (True, False)


def _at(mod, monkeypatch, head):
    monkeypatch.setattr(mod, "_git", lambda *a: head)


def test_rows_generated_at_another_commit_are_refused(promoter, monkeypatch):
    _at(promoter, monkeypatch, "b" * 40)
    _meta(promoter, monkeypatch, {"source_commit_sha": "a" * 40, "worktree_clean": True})
    reason = promoter._committed_elsewhere("ollama:m", "text")
    assert reason is not None and "aaaaaaaa" in reason and "bbbbbbbb" in reason


def test_rows_with_no_recorded_commit_are_refused(promoter, monkeypatch):
    _at(promoter, monkeypatch, "b" * 40)
    _meta(promoter, monkeypatch, {"model": "ollama:m"})
    assert "no commit" in promoter._committed_elsewhere("ollama:m", "text")


def test_rows_from_an_edited_tree_are_refused(promoter, monkeypatch):
    _at(promoter, monkeypatch, "a" * 40)
    _meta(promoter, monkeypatch, {"source_commit_sha": "a" * 40, "worktree_clean": False})
    assert "edited tree" in promoter._committed_elsewhere("ollama:m", "text")


def test_rows_generated_at_head_from_a_clean_tree_promote(promoter, monkeypatch):
    _at(promoter, monkeypatch, "a" * 40)
    _meta(promoter, monkeypatch, {"source_commit_sha": "a" * 40, "worktree_clean": True})
    assert promoter._committed_elsewhere("ollama:m", "text") is None
