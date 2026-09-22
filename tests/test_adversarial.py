"""Adversarial probe tests.

The probes exist to catch silent retrieval failures, so the thing most worth
testing is the probe's own honesty: a stale or mis-written case must announce
itself as a broken case and never be scored as a broken system. Every test here
uses a fake tokenizer — nothing downloads a model.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.adversarial import (  # noqa: E402
    INVALID,
    NOT_A_PROBE,
    Case,
    check_case,
    encoder_limits,
    load_cases,
    offsets_available,
    rank_of,
    run_orthographic,
    run_tail_and_distractor,
    visible_chars,
)
from legalrag.dense import PASSAGE_PREFIX  # noqa: E402
from legalrag.retrieve import Hit  # noqa: E402


class CharTokenizer:
    """One token per character, with the offsets a fast tokenizer would give.

    Character-per-token keeps every boundary in these tests arithmetic rather
    than model-dependent: a ceiling of N tokens means N-2 characters survive.
    """

    def __init__(self, offsets: bool = True):
        self.offsets = offsets

    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=False):
        if not self.offsets or not return_offsets_mapping:
            raise NotImplementedError("offset mapping unavailable")
        spans = [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)]
        return {"offset_mapping": spans}

    def encode(self, text, add_special_tokens=True, truncation=False):
        return list(range(len(text) + (2 if add_special_tokens else 0)))


class FakeEncoder:
    def __init__(self, max_seq_length=None, tokenizer=None):
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        if tokenizer is not None:
            self.tokenizer = tokenizer


def hits(*ids: str) -> list[Hit]:
    return [Hit(id=i, law_name="", number=n, score=1.0, snippet="") for n, i in enumerate(ids, 1)]


# -- reading the case file ----------------------------------------------------


def test_load_cases_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "// a comment\n"
        "\n"
        '{"id": "A", "probe": "tail_evidence", "question": "q", "expected_articles": ["law-1"]}\n'
        "   // indented comment\n",
        encoding="utf-8",
    )
    cases, errors = load_cases(path)
    assert errors == []
    assert [c.id for c in cases] == ["A"]


def test_load_cases_reports_a_duplicate_id_instead_of_keeping_both(tmp_path):
    path = tmp_path / "cases.jsonl"
    row = '{"id": "A", "probe": "tail_evidence", "question": "q", "expected_articles": ["law-1"]}'
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")
    cases, errors = load_cases(path)
    assert len(cases) == 1
    assert any("duplicate case id A" in e for e in errors)


def test_load_cases_reports_malformed_json_with_its_line_number(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "A"\n', encoding="utf-8")
    cases, errors = load_cases(path)
    assert cases == []
    assert errors and ":1:" in errors[0]


def test_missing_case_file_is_an_error_not_an_empty_run(tmp_path):
    cases, errors = load_cases(tmp_path / "absent.jsonl")
    assert cases == []
    assert errors


# -- where the encoder stops reading ------------------------------------------


def test_visible_chars_returns_none_when_nothing_is_cut():
    tok = CharTokenizer()
    assert visible_chars(tok, 500, "short text", prefix="") is None


def test_visible_chars_is_where_the_first_window_ends():
    tok = CharTokenizer()
    text = "a" * 100
    # one token per character, less two special tokens, no prefix
    assert visible_chars(tok, 50, text, prefix="") == 48


def test_visible_chars_leaves_room_for_the_passage_prefix():
    """The prefix rides in the same window, so it costs the article characters."""
    tok = CharTokenizer()
    text = "b" * 100
    assert visible_chars(tok, 50, text, prefix=PASSAGE_PREFIX) == 48 - len(PASSAGE_PREFIX)


def test_visible_chars_agrees_with_the_splitter_the_index_uses():
    """One boundary, one implementation — two would drift apart and the probe
    would end up arguing with the code it exists to check."""
    from legalrag.dense import token_windows, window_budget

    tok = CharTokenizer()
    text = "c" * 200
    budget = window_budget(tok, 60, PASSAGE_PREFIX)
    first_window_end = token_windows(tok, budget, text, overlap=0)[0][1]
    assert visible_chars(tok, 60, text, prefix=PASSAGE_PREFIX) == first_window_end


def test_a_tokenizer_without_offsets_is_refused_rather_than_read_as_fitting():
    """`token_windows` falls back to one whole window, which would make every
    article look like it fits and every tail probe silently pass."""
    tok = CharTokenizer(offsets=False)
    assert not offsets_available(tok)
    assert visible_chars(tok, 50, "c" * 100, prefix="") is None


def test_offsets_available_is_true_for_a_tokenizer_that_has_them():
    assert offsets_available(CharTokenizer())


def test_encoder_limits_is_none_for_an_encoder_that_cannot_say():
    assert encoder_limits(FakeEncoder()) is None
    assert encoder_limits(FakeEncoder(max_seq_length=512)) is None
    assert encoder_limits(FakeEncoder(tokenizer=CharTokenizer())) is None


def test_encoder_limits_reads_the_model_not_a_constant():
    tok = CharTokenizer()
    assert encoder_limits(FakeEncoder(max_seq_length=256, tokenizer=tok)) == (tok, 256)


# -- the probe checking itself ------------------------------------------------


def test_a_tail_case_whose_evidence_is_visible_is_not_a_probe():
    """The case has stopped testing what it claims — it must say so, not pass."""
    case = Case(
        id="T", probe="tail_evidence", question="q",
        expected_articles=["law-1"], evidence="needle",
    )
    texts = {"law-1": "needle" + "x" * 500}
    trouble = check_case(case, texts, {"law-1": 400})
    assert trouble is not None and trouble.startswith(NOT_A_PROBE)


def test_a_tail_case_whose_article_is_no_longer_truncated_is_not_a_probe():
    case = Case(
        id="T", probe="tail_evidence", question="q",
        expected_articles=["law-1"], evidence="needle",
    )
    trouble = check_case(case, {"law-1": "x" * 100 + "needle"}, {"law-1": None})
    assert trouble is not None and trouble.startswith(NOT_A_PROBE)


def test_a_tail_case_with_evidence_past_the_cut_is_scorable():
    case = Case(
        id="T", probe="tail_evidence", question="q",
        expected_articles=["law-1"], evidence="needle",
    )
    assert check_case(case, {"law-1": "x" * 500 + "needle"}, {"law-1": 400}) is None


def test_evidence_absent_from_the_corpus_invalidates_the_case():
    """A corpus re-ingest must break the ruler loudly, not read as a failure."""
    case = Case(
        id="T", probe="tail_evidence", question="q",
        expected_articles=["law-1"], evidence="gone",
    )
    trouble = check_case(case, {"law-1": "x" * 500}, {"law-1": 400})
    assert trouble is not None and trouble.startswith(INVALID)


def test_an_expected_article_outside_the_corpus_invalidates_the_case():
    case = Case(id="T", probe="tail_evidence", question="q", expected_articles=["law-99"])
    trouble = check_case(case, {"law-1": "text"}, {"law-1": None})
    assert trouble is not None and trouble.startswith(INVALID)


def test_an_orthographic_case_needs_variants():
    case = Case(id="O", probe="orthographic_variant", question="q", expected_articles=["law-1"])
    trouble = check_case(case, {"law-1": "text"}, {"law-1": None})
    assert trouble is not None and trouble.startswith(INVALID)


def test_an_orthographic_case_does_not_need_tail_evidence():
    case = Case(
        id="O", probe="orthographic_variant", question="q",
        expected_articles=["law-1"], variants=[{"form": "f", "question": "q2"}],
    )
    assert check_case(case, {"law-1": "text"}, {"law-1": None}) is None


# -- scoring ------------------------------------------------------------------


def test_rank_of_is_one_based_and_none_when_absent():
    assert rank_of("law-2", hits("law-1", "law-2")) == 2
    assert rank_of("law-9", hits("law-1")) is None


def test_expected_article_inside_k_passes():
    case = Case(id="T", probe="tail_evidence", question="q", expected_articles=["law-1"])
    result = run_tail_and_distractor(case, lambda q, k: hits("law-2", "law-1"))
    assert result["expected_rank"] == 2
    assert result["passed"]


def test_expected_article_found_but_outside_k_fails():
    case = Case(id="T", probe="tail_evidence", question="q", expected_articles=["law-1"])
    ranked = ["law-%d" % i for i in range(2, 9)] + ["law-1"]
    result = run_tail_and_distractor(case, lambda q, k: hits(*ranked))
    assert result["expected_rank"] == 8
    assert not result["passed"]


def test_a_distractor_above_the_expected_article_fails_even_inside_k():
    """Right number, wrong subject: the answer looks correct and is not."""
    case = Case(
        id="T", probe="tail_evidence", question="q",
        expected_articles=["law-1"], distractor_articles=["law-20"],
    )
    result = run_tail_and_distractor(case, lambda q, k: hits("law-20", "law-1"))
    assert result["expected_rank"] == 2
    assert result["outranked_by"] == ["law-20"]
    assert not result["passed"]


def test_a_distractor_below_the_expected_article_does_not_fail_the_case():
    case = Case(
        id="T", probe="tail_evidence", question="q",
        expected_articles=["law-1"], distractor_articles=["law-20"],
    )
    result = run_tail_and_distractor(case, lambda q, k: hits("law-1", "law-20"))
    assert result["outranked_by"] == []
    assert result["passed"]


def test_a_missing_expected_article_is_outranked_by_any_present_distractor():
    case = Case(
        id="T", probe="tail_evidence", question="q",
        expected_articles=["law-1"], distractor_articles=["law-20"],
    )
    result = run_tail_and_distractor(case, lambda q, k: hits("law-20", "law-3"))
    assert result["expected_rank"] is None
    assert result["outranked_by"] == ["law-20"]
    assert not result["passed"]


def test_orthographic_variants_that_all_rank_the_same_are_stable():
    case = Case(
        id="O", probe="orthographic_variant", question="q", expected_articles=["law-7"],
        variants=[{"form": "ta_marbuta", "question": "q2"}],
    )
    result = run_orthographic(case, lambda q, k: hits("law-7", "law-1"))
    assert result["stable"] and result["passed"]
    assert [form for form, _ in result["ranks"]] == ["as written", "ta_marbuta"]


def test_a_variant_that_moves_the_rank_fails_even_while_inside_k():
    """Same question, different spelling, different answer order — that is the finding."""
    case = Case(
        id="O", probe="orthographic_variant", question="q", expected_articles=["law-7"],
        variants=[{"form": "ta_marbuta", "question": "q2"}],
    )

    def search(query, k):
        return hits("law-7", "law-1") if query == "q" else hits("law-1", "law-7")

    result = run_orthographic(case, search)
    assert not result["stable"]
    assert not result["passed"]


def test_variants_stable_but_all_outside_k_still_fail():
    case = Case(
        id="O", probe="orthographic_variant", question="q", expected_articles=["law-7"],
        variants=[{"form": "f", "question": "q2"}],
    )
    ranked = ["law-%d" % i for i in range(1, 7)] + ["law-7"]
    result = run_orthographic(case, lambda q, k: hits(*ranked))
    assert result["stable"]
    assert not result["passed"]


# -- the shipped case file ----------------------------------------------------


def test_the_shipped_cases_parse_and_declare_a_known_probe():
    root = Path(__file__).resolve().parents[1]
    cases, errors = load_cases(root / "evals" / "adversarial" / "cases.jsonl")
    assert errors == []
    assert cases
    known = {"tail_evidence", "near_miss_distractor", "orthographic_variant"}
    for case in cases:
        assert case.probe in known, f"{case.id} declares unknown probe {case.probe!r}"
        assert case.expected_articles, f"{case.id} has no expected article"
        assert case.why, f"{case.id} does not say what failure it claims"
        if case.probe == "orthographic_variant":
            assert case.variants, f"{case.id} has no variants"
        else:
            assert case.evidence, f"{case.id} has no evidence"


def test_every_string_this_runner_prints_survives_a_windows_console():
    """A `->` that was a `→` crashed `python tasks.py adversarial` on this machine.

    Windows picks the console code page from the system locale; here that is
    cp1256, which has Arabic letters and `·` and `—` but no arrows. The failure
    is a traceback at the end of a run that already did all its work, and it
    only appears on the real entry point — importing `main` and calling it from
    a UTF-8 wrapper hides it completely.

    Scoped to the strings this module actually prints, not every literal in it:
    `TYPIST_FORMS` holds letters like `ٱ` as *data* and must keep them, exactly
    as `normalize` and `ocr_gate` keep Arabic-Indic digits in their matching
    rules. Data never reaches a terminal; arguments to `print` do.
    """
    import ast

    source = (Path(__file__).resolve().parents[1] / "src" / "legalrag" / "adversarial.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            continue
        if call.func.id != "print":
            continue
        for node in ast.walk(call):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                try:
                    node.value.encode("cp1256")
                except UnicodeEncodeError as e:
                    pytest.fail(
                        f"line {node.lineno}: {e.object[e.start:e.end]!r} cannot reach "
                        "a cp1256 console"
                    )


def test_the_shipped_meta_documents_every_probe_in_the_case_file():
    root = Path(__file__).resolve().parents[1]
    cases, _ = load_cases(root / "evals" / "adversarial" / "cases.jsonl")
    meta = json.loads((root / "evals" / "adversarial" / "meta.json").read_text(encoding="utf-8"))
    for case in cases:
        assert case.probe in meta["probes"], f"probe {case.probe!r} is undocumented"
