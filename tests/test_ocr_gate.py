"""OCR digit-gate tests (ADR-019).

The gate exists because the failure it catches is invisible. `المادة (١٢)`
misread as `١٢١` is still well-formed Arabic referring to a well-formed article
number — it just refers to the wrong one, forever, in a corpus where article
numbers are the primary key. These tests pin the scoring decisions that make
that failure show up as a failure.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag import ocr_gate  # noqa: E402
from legalrag.ocr_gate import (  # noqa: E402
    PASS_THRESHOLD,
    digit_tokens,
    drop_furniture,
    fold,
    load_ground_truth,
    run,
    score_page,
)

REPO = Path(__file__).resolve().parents[1]


def test_digit_tokens_are_maximal_runs_not_single_characters():
    assert digit_tokens("مادة (٤٧) والمواد ٣٦، ٣٧") == ["٤٧", "٣٦", "٣٧"]


def test_a_split_number_scores_as_a_miss_not_partial_credit():
    """easyocr read ٢٠٢٠ as `٢ ٠`. Three of four characters survived and the
    number did not. Character accuracy would call that 75%; it is 0%."""
    r = score_page(["٢٠٢٠"], digit_tokens("سنة ٢ ٠"))
    assert r["correct"] == 0
    assert r["missing"] == ["2020"]


def test_an_invented_number_is_reported_apart_from_a_missing_one():
    """`٣٦،` read as `٣٦١` both loses article 36 and invents article 361.
    The two failures have different consequences and are counted separately."""
    r = score_page(["٣٦", "٣٧"], digit_tokens("المواد ٣٦١ ٣٧"))
    assert r["missing"] == ["36"]
    assert r["spurious"] == ["361"]
    assert r["correct"] == 1


def test_a_deleted_number_is_a_miss_with_nothing_invented():
    """PaddleOCR drops digits rather than corrupting them — a different
    failure with a different signature, and the gate must tell them apart."""
    r = score_page(["٤", "١٢"], digit_tokens("مادة ( ) من هذا القانون"))
    assert r["missing"] == ["12", "4"]
    assert r["spurious"] == []


def test_arabic_indic_and_ascii_are_the_same_number():
    """An engine that transliterates ٤٧ to 47 has read the number correctly."""
    assert fold("٤٧") == "47"
    assert score_page(["٤٧"], digit_tokens("مادة (47)"))["correct"] == 1


def test_repeats_are_counted_not_collapsed():
    """Page 7 has ٤ twice — as an article header and as a list marker. An
    engine that reads one of them has read one of them."""
    r = score_page(["٤", "٤"], digit_tokens("مادة (٤)"))
    assert r["correct"] == 1
    assert r["missing"] == ["4"]


def test_a_page_key_may_be_written_three_ways():
    gt = {"p07": {"tokens": ["٤"], "critical": {}}}
    for key in ("p07", "07", "7"):
        assert run(gt, {key: "مادة (٤)"})["correct"] == 1, key


def test_a_missing_critical_token_fails_the_gate_despite_high_recall():
    """The one number that matters is not the average number. An engine can
    read a page nearly perfectly and still corrupt the cross-reference."""
    gt = {"p07": {"tokens": ["١", "٢", "٣", "٤", "١٢"],
                  "critical": {"١٢": "in-prose cross-reference"}}}
    result = run(gt, {"p07": "١ ٢ ٣ ٤ ١٢١"})
    assert result["recall"] >= 0.8
    assert result["critical_failures"] == ["p07:١٢"]


def test_a_clean_read_passes_the_threshold():
    gt = {"p07": {"tokens": ["٤", "١٢"], "critical": {"١٢": "x"}}}
    result = run(gt, {"p07": "مادة (٤) … المادة (١٢)"})
    assert result["recall"] == 1.0
    assert result["critical_failures"] == []
    assert result["recall"] >= PASS_THRESHOLD


def test_the_shipped_ground_truth_is_hand_read_not_engine_derived():
    """An engine cannot be its own ruler. This asserts the file is present and
    shaped as the scorer expects; the `_about` key records how it was made."""
    raw = json.loads((REPO / "evals/ocr/digit_ground_truth.json").read_text(encoding="utf-8"))
    assert "_about" in raw and "EYE" in raw["_about"].upper()

    gt = load_ground_truth(REPO / "evals/ocr/digit_ground_truth.json")
    assert gt, "ground truth is empty"
    for page, spec in gt.items():
        assert spec["tokens"], f"{page} has no tokens"
        assert all(digit_tokens(t) == [t] for t in spec["tokens"]), page
        for tok in spec.get("critical", {}):
            assert tok in spec["tokens"], f"{page}: critical {tok} not in tokens"


def test_extended_arabic_indic_digits_count_as_digits():
    """U+06F0-06F9 (۰۱۲) and U+0660-0669 (٠١٢) render almost identically and
    are different code points. surya emits both — and mixes them inside a
    single number. A gate that knows only the first block scores 16% of this
    document's digits as missing."""
    assert digit_tokens("مادة (۲۰)") == ["۲۰"]
    assert fold("۲۰") == "20"
    assert score_page(["٢٠"], digit_tokens("مادة (۲۰)"))["correct"] == 1


def test_a_number_mixing_both_blocks_stays_one_token():
    """`مادة ( ۲٤ )` is U+06F2 then U+0664. Split, it reads as two numbers."""
    assert digit_tokens("مادة ( ۲٤ )") == ["۲٤"]
    assert fold("۲٤") == "24"


def test_a_page_missing_from_ocr_is_a_full_miss_not_a_silent_drop():
    """A ground-truth page with no matching OCR text at all must count as
    completely missed -- it must not vanish from the aggregate the way it
    does today, which lets a missing page contribute nothing to `expected`
    or `critical_failures` and hide behind a perfect recall on the pages
    that were present."""
    gt = {
        "p01": {"tokens": ["١", "٢"], "critical": {}},
        "p02": {"tokens": ["٩", "٩"], "critical": {"٩": "x"}},
    }
    result = run(gt, {"p01": "١ ٢"})
    assert result["expected"] == 4
    assert result["correct"] == 2
    assert result["recall"] == 0.5
    assert "p02" in result["pages"]
    assert result["pages"]["p02"]["missing"] == ["9", "9"]
    assert result["pages"]["p02"]["recall"] == 0.0
    assert result["pages"]["p02"]["critical"] == {"٩": False}
    assert result["critical_failures"] == ["p02:٩"]


def test_reproduces_the_missing_page_finding_recall_no_longer_hides_it():
    """docs/internal/reproductions.json's `ocr_missing_page` key recorded the
    pre-fix behaviour: an OCR file simply missing p02 (which has ground
    truth) still scored recall 1.0, because the page was skipped rather than
    scored. It must not."""
    gt = {
        "p01": {"tokens": ["12"], "critical": {"12": "x"}},
        "p02": {"tokens": ["99"], "critical": {"99": "x"}},
    }
    result = run(gt, {"p01": "12"})
    assert result["recall"] < 1.0
    assert result["pages"]["p02"]["recall"] == 0.0
    assert result["critical_failures"] == ["p02:99"]


def test_reproduces_the_spurious_digit_finding_main_now_fails(tmp_path, monkeypatch):
    """docs/internal/reproductions.json's `ocr_spurious_digits` key recorded
    two invented digit tokens (777, 888) sitting alongside an otherwise
    fully-correct page: recall was 1.0 and critical_failures was empty, so
    the pre-fix `passed` check reported PASS. An invented digit is exactly
    the silent-forever failure this gate exists to catch, so it must fail."""
    gt = {
        "p01": {"tokens": ["12"], "critical": {"12": "x"}},
        "p02": {"tokens": ["99"], "critical": {"99": "x"}},
    }
    monkeypatch.setattr(ocr_gate, "load_ground_truth", lambda: gt)
    ocr_path = tmp_path / "engine.json"
    ocr_path.write_text(
        json.dumps({"p01": "12 777 888", "p02": "99"}), encoding="utf-8"
    )
    assert ocr_gate.main([str(ocr_path)]) == 1


def test_swapped_digits_and_duplicates_still_score_as_multiset_misses():
    """Regression guard for the missing-page fix: a page that IS present
    must still go through the untouched multiset comparison in score_page --
    a token read with its digits reordered is a full miss on both sides
    (missing the real one, spurious the invented one), and a duplicated
    expected token still needs two matching reads, not one."""
    gt = {"p01": {"tokens": ["12", "4", "4"], "critical": {}}}
    result = run(gt, {"p01": "21 4"})
    r = result["pages"]["p01"]
    assert r["missing"] == ["12", "4"]
    assert r["spurious"] == ["21"]
    assert r["correct"] == 1
    assert result["recall"] == 1 / 3


def test_a_completely_empty_ocr_file_is_still_reported_as_no_match(
    tmp_path, capsys, monkeypatch
):
    """A totally empty OCR file (wrong file, or an engine that produced
    nothing at all) is a different failure than a page that is merely
    missing from otherwise-real coverage: it must still surface the
    existing 'no page matching the ground truth' message and exit 2,
    not silently score every ground-truth page as a 100% miss."""
    real_gt = load_ground_truth(REPO / "evals/ocr/digit_ground_truth.json")
    monkeypatch.setattr(ocr_gate, "load_ground_truth", lambda: real_gt)
    ocr_path = tmp_path / "empty.json"
    ocr_path.write_text("{}", encoding="utf-8")

    exit_code = ocr_gate.main([str(ocr_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "has no page matching the ground truth" in captured.out


def test_gt_flag_scores_against_the_named_file_not_the_default(tmp_path, capsys, monkeypatch):
    """`--gt` has to actually replace the ground truth, not sit beside it.

    The trap it guards is silent: if the flag were parsed but ignored, a second
    document's OCR would be scored against the FIRST document's expected
    digits, every token would miss, and the engine would look catastrophically
    worse than it is -- a wrong number that reads like a real measurement.
    `load_ground_truth` is patched to fail loudly if the default path is what
    gets read.
    """
    other = tmp_path / "other_gt.json"
    other.write_text(json.dumps({"p01": {"tokens": ["٤٧"], "critical": {}}}), encoding="utf-8")
    monkeypatch.setattr(ocr_gate, "load_ground_truth",
                        lambda path=None: load_ground_truth(path) if path is not None
                        else pytest.fail("the default ground truth was read despite --gt"))
    ocr_path = tmp_path / "engine.json"
    ocr_path.write_text(json.dumps({"p01": "مادة (٤٧)"}), encoding="utf-8")

    assert ocr_gate.main([str(ocr_path), "--gt", str(other)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_gt_flag_does_not_consume_the_engine_name(tmp_path, capsys, monkeypatch):
    """The flag is removed from argv before the positionals are read, so an
    engine name may sit on either side of it and still be the engine name --
    not `--gt`, and not the ground-truth path."""
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"p01": {"tokens": ["٤٧"], "critical": {}}}), encoding="utf-8")
    ocr_path = tmp_path / "engine.json"
    ocr_path.write_text(json.dumps({"p01": "٤٧"}), encoding="utf-8")

    ocr_gate.main([str(ocr_path), "surya 0.17.1", "--gt", str(gt)])
    assert "surya 0.17.1" in capsys.readouterr().out


def test_a_missing_gt_path_is_refused_rather_than_falling_back(tmp_path, capsys):
    """A typo in `--gt` must not quietly score against the default document.
    That would report a number for the wrong corpus, which is the one outcome
    this whole module exists to prevent."""
    ocr_path = tmp_path / "engine.json"
    ocr_path.write_text(json.dumps({"p01": "٤٧"}), encoding="utf-8")

    assert ocr_gate.main([str(ocr_path), "--gt", str(tmp_path / "nope.json")]) == 2
    assert "no such ground truth" in capsys.readouterr().out


def test_gt_flag_without_a_path_is_an_error_not_a_crash(tmp_path):
    ocr_path = tmp_path / "engine.json"
    ocr_path.write_text(json.dumps({"p01": "٤٧"}), encoding="utf-8")
    assert ocr_gate.main([str(ocr_path), "--gt"]) == 2


def test_the_law174_ground_truth_is_well_formed_and_read_by_eye():
    """A data guard on the second document's ground truth (Run 11).

    Every claim the gate makes about law 174/2025 rests on this file being
    what it says it is, and nothing else checks it. `critical` tokens must
    appear in `tokens` -- a critical token absent from the expected multiset
    can never fail, so it would be a check that silently does nothing.
    """
    path = REPO / "evals/ocr/law174_digit_ground_truth.json"
    gt = load_ground_truth(path)
    assert set(gt) == {"p00", "p09", "p49"}
    for page, entry in gt.items():
        assert entry["tokens"], f"{page} has no expected digits"
        for token in entry.get("critical", {}):
            assert token in entry["tokens"], f"{page}: critical {token} is not in tokens"
    # The comma-separated reference on p49 shares its numbers with the article
    # headers on the same page; the multiset has to carry both copies.
    assert gt["p49"]["tokens"].count("١٧٥") == 2
    assert gt["p49"]["tokens"].count("١٧٦") == 2


# --- content-only scoring (Run 12) ------------------------------------------


def test_drop_furniture_removes_one_copy_and_leaves_the_rest():
    """The header's ١٢ goes; an article numbered ١٢ on the same page stays.

    Dropping *every* match would delete the content token along with the
    furniture, which is the failure this helper exists to avoid.
    """
    assert drop_furniture(["١٢", "٤٥", "١٢", "٣١٠"], ["١٢", "٤٥"]) == ["١٢", "٣١٠"]


def test_drop_furniture_compares_folded_not_by_code_point():
    """surya emits the extended Arabic-Indic block; the header is the header."""
    assert drop_furniture(["۱۲", "٣١٠"], ["١٢"]) == ["٣١٠"]


def test_drop_furniture_tolerates_furniture_the_engine_never_read():
    assert drop_furniture(["٣١٠"], ["١٢", "٤٥"]) == ["٣١٠"]


# A page is lines, not a bag of words, and `--content` decides line by line.
# Fixtures below are shaped like a real gazette page -- bare page number, then
# the running header, then prose -- because a one-line fixture would be judged
# as a header by the same shape rule under test.
HEADER = "الجريدة الرسمية - العدد ٤٥ مكرر (د) فى ١٢ نوفمبر سنة ٢٠٢٥"
CROSS_REF = "العقوبة المقررة فى المادة ٣١٠ من قانون العقوبات"
FURNITURE = ["٢٢", "٤٥", "١٢", "٢٠٢٥"]


def page(*body, header=HEADER, number="٢٢"):
    return "\n".join([number, header, *body])


def test_content_mode_ignores_a_header_the_engine_got_wrong():
    """Both header digits reversed. Whole-page score falls; body score does not.

    This is the case `drop_furniture` alone cannot reach: ٥٤ is a token no
    ground truth lists as furniture, so subtracting by name leaves it behind
    as an invented number. The shape rule drops the line that carries it.
    """
    gt = {"p01": {"furniture": FURNITURE, "tokens": FURNITURE + ["٣١٠"]}}
    misread = {"p01": page(CROSS_REF, header=HEADER.replace("٤٥", "٥٤").replace("١٢ نوفمبر", "٢١ نوفمبر"))}
    assert run(gt, misread)["recall"] < 1.0
    body = run(gt, misread, content_only=True)
    assert body["expected"] == 1
    assert body["recall"] == 1.0
    assert not body["spurious_failures"]


def test_content_mode_does_not_reward_an_engine_for_skipping_the_header():
    """Reading the header and never reading it score the same.

    If only the expected side were filtered, an engine could raise its score
    by reading less -- the opposite of what the flag is for.
    """
    gt = {"p01": {"furniture": FURNITURE, "tokens": FURNITURE + ["٣١٠"]}}
    read_it = run(gt, {"p01": page(CROSS_REF)}, content_only=True)
    skipped_it = run(gt, {"p01": CROSS_REF}, content_only=True)
    assert read_it["recall"] == skipped_it["recall"] == 1.0
    assert not read_it["spurious_failures"]
    assert not skipped_it["spurious_failures"]


def test_content_mode_still_reports_a_number_fused_into_a_furniture_token():
    """p108's «فى البنود (١، ٢، ٣، ٤)» is the shape that breaks engines.

    An engine that swallows the Arabic comma emits ١٢ where two items belong.
    The header's own ١٢ has already left with its line, so nothing absorbs
    the invented one and it is reported.
    """
    gt = {
        "p01": {
            "furniture": FURNITURE,
            "tokens": FURNITURE + ["١", "٢", "٣", "٤", "٤١٤"],
        }
    }
    fused = run(gt, {"p01": page("فى البنود (١٢، ٣، ٤) من المادة ٤١٤")},
                content_only=True)
    assert fused["spurious_failures"] == ["p01:12"]
    assert fused["pages"]["p01"]["missing"] == ["1", "2"]


def test_content_mode_counts_a_header_the_engine_merged_into_the_prose():
    """The flag cuts both ways, and is supposed to.

    A header run together with the first line of prose is too long to match
    the shape rule, so it survives stripping here exactly as it would survive
    into the corpus -- and its digits count against the engine.
    """
    gt = {"p01": {"furniture": FURNITURE, "tokens": FURNITURE + ["٣١٠"]}}
    merged = run(gt, {"p01": "٢٢\n" + HEADER + " " + CROSS_REF}, content_only=True)
    assert sorted(merged["pages"]["p01"]["spurious"]) == ["12", "2025", "45"]


def test_content_mode_will_not_let_the_header_satisfy_a_critical_token():
    gt = {
        "p01": {
            "furniture": ["٤٥"],
            "tokens": ["٤٥", "٤٥"],
            "critical": {"٤٥": "an article that happens to share the issue number"},
        }
    }
    header_only = run(gt, {"p01": "العدد ٤٥ مكرر"}, content_only=True)
    # Raw, not folded: `critical_failures` carries the ground truth's own
    # spelling and `main` folds it at print time. `spurious` above is folded
    # because `score_page` compares on folded tokens.
    assert header_only["critical_failures"] == ["p01:٤٥"]


def test_content_mode_is_off_by_default():
    gt = {"p01": {"furniture": ["٤٥"], "tokens": ["٤٥", "٣١٠"]}}
    assert run(gt, {"p01": "٤٥ ٣١٠"})["expected"] == 2


def test_a_ground_truth_without_furniture_is_all_content():
    """The two older ground-truth files predate the key and must still score."""
    gt = {"p01": {"tokens": ["٤٧", "٣٦"]}}
    prose = {"p01": "مادة (٤٧) ويسري عليها حكم المادة ٣٦ من هذا القانون"}
    assert run(gt, prose, content_only=True)["recall"] == 1.0


def test_the_content_flag_is_not_mistaken_for_the_engine_name(tmp_path, capsys):
    ocr_path = tmp_path / "deepseek.json"
    # Prose, not a bare number: a line of nothing but digits is a page number
    # by shape, and `--content` would rightly drop it.
    ocr_path.write_text(
        json.dumps({"p01": "مادة (٤٧) من هذا القانون"}, ensure_ascii=False),
        encoding="utf-8",
    )
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps({"p01": {"furniture": [], "tokens": ["٤٧"]}}), encoding="utf-8"
    )
    assert ocr_gate.main([str(ocr_path), "--content", "--gt", str(gt_path)]) == 0
    out = capsys.readouterr().out
    assert "deepseek" in out
    assert "--content" not in out
    assert "body digits only" in out


def test_the_run12_ground_truth_is_held_out_and_meets_the_pre_registered_minimum():
    """A data guard on Run 12's ground truth, whose rule was written first.

    EVAL.md Run 12 pre-registered two sample minimums — at least 8 NEW pages
    and at least 50 body tokens — and the reason for the first: p00, p09 and
    p49 are the pages that produced the hypothesis being tested, so an engine
    judged on them is not being tested at all. Both are asserted here because
    a rule that lives only in prose is the shape ADR-013 already failed at.
    """
    gt = load_ground_truth(REPO / "evals/ocr/law174_page_roles_ground_truth.json")
    run11 = load_ground_truth(REPO / "evals/ocr/law174_digit_ground_truth.json")
    assert not set(gt) & set(run11), "Run 12 must be judged on unseen pages"
    assert len(gt) >= 8

    body = 0
    for page, entry in gt.items():
        furniture, tokens = entry["furniture"], entry["tokens"]
        assert len(drop_furniture(tokens, furniture)) == len(tokens) - len(furniture), (
            f"{page}: a furniture token is not in tokens"
        )
        content = drop_furniture(tokens, furniture)
        body += len(content)
        for token in entry.get("critical", {}):
            # A critical token that is only furniture could be satisfied by the
            # header, and one absent from `tokens` can never fail at all.
            assert token in content, f"{page}: critical {token} is not a body token"
    assert body >= 50, f"only {body} body tokens; the rule asked for 50"
