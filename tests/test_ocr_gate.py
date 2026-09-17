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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag import ocr_gate  # noqa: E402
from legalrag.ocr_gate import (  # noqa: E402
    PASS_THRESHOLD,
    digit_tokens,
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
    """docs/review/reproductions.json's `ocr_missing_page` key recorded the
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
    """docs/review/reproductions.json's `ocr_spurious_digits` key recorded
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
