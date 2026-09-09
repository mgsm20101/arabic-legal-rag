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
