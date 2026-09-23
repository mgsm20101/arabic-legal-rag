"""The held-out split is a claim about reachability, so it is tested as one.

`EVAL.md` says the `test` rows have never been scored. That sentence is only
worth printing if forgetting to exclude them is not a way to include them —
which is what these tests check: the default is `dev`, naming the split is the
only way past it, and a row that lies about its split is an error rather than a
silent member of whatever split it was read into.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.evaluate import (  # noqa: E402
    DEFAULT_SPLIT,
    QUESTIONS_PATH,
    SPLITS,
    load_questions,
)

ROOT = Path(__file__).resolve().parents[1]


def _row(qid, split, **over):
    row = {
        "id": qid,
        "category": "direct",
        "question": "سؤال",
        "expected_articles": ["law-1"],
        "expected_keywords": ["كلمة"],
        "answerable": True,
        "ref_status": "VERIFIED",
        "split": split,
    }
    row.update(over)
    return row


def _write(tmp_path, rows):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_the_default_caller_never_sees_a_held_out_question(tmp_path):
    path = _write(tmp_path, [_row("Q1", "dev"), _row("Q2", "test")])

    questions, errors = load_questions(path)

    assert errors == []
    assert [q.id for q in questions] == ["Q1"]


def test_asking_for_the_test_split_by_name_is_the_only_way_in(tmp_path):
    path = _write(tmp_path, [_row("Q1", "dev"), _row("Q2", "test")])

    assert [q.id for q in load_questions(path, split="test")[0]] == ["Q2"]
    assert [q.id for q in load_questions(path, split=None)[0]] == ["Q1", "Q2"]


def test_a_row_with_no_split_is_rejected_rather_than_assumed_dev(tmp_path):
    """The dataclass defaults `split`; the file format must not.

    A row that omits it would otherwise join `dev` by accident, which is the
    one direction the guard cannot detect afterwards.
    """
    row = _row("Q1", "dev")
    del row["split"]
    path = _write(tmp_path, [row])

    questions, errors = load_questions(path, split=None)

    assert questions == []
    assert any("split" in e for e in errors)


def test_an_unknown_split_on_a_row_is_an_error(tmp_path):
    path = _write(tmp_path, [_row("Q1", "holdout")])

    questions, errors = load_questions(path, split=None)

    assert questions == []
    assert any("holdout" in e for e in errors)


def test_a_held_out_row_is_validated_even_though_it_is_filtered_out(tmp_path):
    """Filtering happens after validation, so the excluded rows still have to
    be well formed. A test split that silently rots is not held out — it is
    lost."""
    bad = _row("Q2", "test", answerable=True, expected_articles=[])
    path = _write(tmp_path, [_row("Q1", "dev"), bad])

    questions, errors = load_questions(path)  # default split: dev

    assert [q.id for q in questions] == ["Q1"]
    assert any("Q2" in e for e in errors)


def test_a_typo_in_the_split_argument_fails_loudly(tmp_path):
    """`split="dev "` must not quietly return nothing and read as "no questions
    matched"; an empty scoreboard is indistinguishable from a broken corpus."""
    path = _write(tmp_path, [_row("Q1", "dev")])

    questions, errors = load_questions(path, split="dev ")

    assert questions == []
    assert any("dev" in e for e in errors)


def test_the_shipped_question_set_actually_holds_rows_back():
    """Guards the property the registry depends on, on the real file."""
    every, errors = load_questions(ROOT / QUESTIONS_PATH, split=None)
    default, _ = load_questions(ROOT / QUESTIONS_PATH)

    assert errors == []
    assert {q.split for q in every} == SPLITS
    assert {q.split for q in default} == {DEFAULT_SPLIT}
    assert len(default) < len(every)
