"""RTL reconstruction tests (ADR-013).

Every case here is a defect that was actually observed while ingesting the
151/2020 PDF, reduced to synthetic glyphs. Glyph dicts carry only the three
keys the extractor reads, so no PDF fixture is needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.pdf_text import (  # noqa: E402
    arabic_column,
    drop_marks,
    group_lines,
    logical_line,
)


def g(text: str, x0: float, width: float = 10.0, bottom: float = 100.0) -> dict:
    return {"text": text, "x0": x0, "x1": x0 + width, "bottom": bottom}


def visual(text: str, start: float = 0.0, step: float = 10.0, bottom: float = 100.0):
    """Glyphs laid out left -> right, one per character."""
    return [g(ch, start + i * step, step, bottom) for i, ch in enumerate(text)]


def test_arabic_run_is_reversed_into_logical_order():
    # RTL text is painted with the logically-first letter on the RIGHT.
    assert logical_line(visual("ةدام")) == "مادة"


def test_lam_alef_ligature_survives_as_one_glyph():
    """The bug that string-level reversal cannot avoid.

    'الأولى' contains a لأ ligature: ONE glyph, two characters, already in
    logical order. Reversing characters would split it into 'أل'.
    """
    # visual left -> right: ى ل و [لأ] ا
    glyphs = [g("ى", 0), g("ل", 10), g("و", 20), g("لأ", 30), g("ا", 40)]
    assert logical_line(glyphs) == "الأولى"


def test_digits_keep_left_to_right_order_inside_an_rtl_line():
    """'72 ساعة' must not become '27 ساعة' — the eval set keys on that number."""
    # "خلال 72 ساعة" is painted left -> right as: ةعاس _ 72 _ للاخ
    glyphs = visual("ةعاس ", 0) + visual("72", 50) + visual(" لالخ", 70)
    assert logical_line(glyphs) == "خلال 72 ساعة"


def test_brackets_are_mirrored_back_to_logical_order():
    """An RTL renderer paints "مادة (1)" as ")1(" — reading back must mirror."""
    glyphs = visual(")1(", 0) + visual("ةدام", 30)
    assert logical_line(glyphs) == "مادة (1)"


def test_glyphs_are_ordered_by_centre_not_by_x0():
    """The ر / ا swap: 'إجراءات' came out 'إجارءات' when sorted on x0.

    ر is drawn narrow inside a wide advance, so its x0 sits left of the ا that
    visually precedes it while its centre sits right of it.
    """
    wide_ra = {"text": "ر", "x0": 100, "x1": 120, "bottom": 100}   # centre 110
    narrow_alef = {"text": "ا", "x0": 105, "x1": 109, "bottom": 100}  # centre 107
    line = group_lines([wide_ra, narrow_alef])[0]
    # centre order is ا then ر (visual), so logical order reverses to را
    assert logical_line(line) == "را"
    # x0 order would have been ر then ا, reversing to the corrupted "ار"
    assert logical_line(sorted([wide_ra, narrow_alef], key=lambda c: c["x0"])) != "را"


def test_latin_translation_is_dropped_by_default():
    """The circulating PDFs are dual-language; the English is not the corpus."""
    glyphs = visual("Article", 0) + visual("ةدام", 200)
    assert logical_line(glyphs) == "مادة"
    assert "Article" in logical_line(glyphs, keep_latin=True)


def test_a_line_with_no_arabic_yields_nothing():
    assert logical_line(visual("Article 1", 0)) == ""
    assert logical_line(visual("123", 0)) == ""


def test_group_lines_separates_baselines_and_orders_top_down():
    top = visual("ةدام", 0, bottom=100.0)
    bottom = visual("ىلولأا", 0, bottom=118.0)
    lines = group_lines(top + bottom)
    assert len(lines) == 2
    assert logical_line(lines[0]) == "مادة"


def test_group_lines_tolerates_baseline_wobble_within_one_line():
    """Glyphs on one line differ by a fraction of a point; they must not split."""
    glyphs = [g("ة", 0, bottom=100.0), g("د", 10, bottom=100.9),
              g("ا", 20, bottom=99.4), g("م", 30, bottom=100.2)]
    lines = group_lines(glyphs)
    assert len(lines) == 1
    assert logical_line(lines[0]) == "مادة"


# ------------------------------------------------ Arabic-Indic digits (ADR-013)

def test_arabic_indic_digits_are_not_reversed():
    """The bug that made مادة (٢٥) read مادة (٥٢).

    ٠-٩ sit inside the Arabic Unicode block, so a naive block test classifies
    them as letters and reverses them along with the words.
    """
    # "مادة ٢٥" painted left -> right: ٢٥ then _ةدام
    glyphs = visual("٢٥", 0) + visual(" ةدام", 30)
    assert logical_line(glyphs) == "مادة ٢٥"


def test_extended_arabic_indic_digits_are_not_reversed():
    glyphs = visual("۲۱", 0) + visual(" ةدام", 30)
    assert logical_line(glyphs) == "مادة ۲۱"


def test_arabic_comma_travels_with_the_words():
    """Punctuation is neutral: it must take the direction of its neighbours."""
    # "البيانات،" painted left -> right: ،تانايبلا
    glyphs = visual("،تانايبلا", 0)
    assert logical_line(glyphs) == "البيانات،"


# ----------------------------------------------------- column split (ADR-013)

def test_arabic_column_drops_the_english_columns_digits():
    """'مادة (1)' arrived as 'مادة (1)  )1(' — the second pair is the translation."""
    english = visual("Article (1)", 0)            # centres 5..105
    arabic = visual(")1( ةدام", 300)              # centres 305..375
    kept = arabic_column(english + arabic)
    assert all(c["x0"] >= 300 for c in kept)
    assert logical_line(group_lines(kept)[0]) == "مادة (1)"


def test_arabic_column_is_a_no_op_without_latin():
    glyphs = visual("ةدام", 0)
    assert arabic_column(glyphs) == glyphs


def test_arabic_column_handles_arabic_on_the_left():
    arabic = visual("ةدام", 0)
    english = visual("Article", 300)
    kept = arabic_column(arabic + english)
    assert logical_line(group_lines(kept)[0]) == "مادة"


def test_group_lines_does_not_split_a_line_that_straddles_a_bucket_edge():
    """The 'القانو' bug: bucketing by round(bottom/tol) split one line in two.

    Baselines 623.5 / 623.7 / 624.0 are the same line of print; dividing by 2.5
    and rounding puts the first two in bucket 249 and the third in bucket 250.
    """
    line_a = [g("ن", 0, bottom=623.5), g("و", 10, bottom=623.7),
              g("ن", 20, bottom=624.0), g("ا", 30, bottom=624.0),
              g("ق", 40, bottom=624.0), g("ل", 50, bottom=624.0),
              g("ا", 60, bottom=624.0)]
    line_b = [g("ة", 0, bottom=642.0), g("د", 10, bottom=642.0),
              g("ا", 20, bottom=641.8), g("م", 30, bottom=642.0)]
    lines = group_lines(line_a + line_b)
    assert len(lines) == 2
    assert logical_line(lines[0]) == "القانون"
    assert logical_line(lines[1]) == "مادة"


def test_line_tolerance_adapts_to_the_documents_own_spacing():
    """A displaced glyph 6.5pt above an 18pt-spaced line belongs to that line."""
    rows = []
    for i, bottom in enumerate((100.0, 118.0, 136.0, 154.0)):
        rows += visual("ةدام", 0, bottom=bottom)
    stray = [g("ن", 50, bottom=147.5)]  # 6.5pt above the 154.0 line
    lines = group_lines(rows + stray)
    assert len(lines) == 4
    assert any("ن" in logical_line(line) for line in lines)


# ------------------------------------------------- combining marks (ADR-013)

def sized(text: str, start: float = 0.0, bottom: float = 100.0, size: float = 12.0):
    out = []
    for i, ch in enumerate(text):
        c = g(ch, start + i * 10.0, 10.0, bottom)
        c["size"] = size
        out.append(c)
    return out


def test_a_mis_mapped_tanween_glyph_is_dropped():
    """'فعلاً' came back as 'فعلاا': the tanween glyph's ToUnicode says 'ا'.

    strip_tashkeel cannot remove it — it is a letter, not a diacritic — so the
    only signal left is that it is painted 6.5pt above a 12pt baseline.
    """
    line = sized("العف", 0, bottom=100.0)          # visual order for فعلا
    mark = sized("ا", 40, bottom=93.5)             # the spurious alef, 6.5 up
    assert logical_line(drop_marks(line + mark)) == "فعلا"


def test_a_correctly_mapped_tanween_glyph_is_dropped_too():
    line = sized("الوأ", 0, bottom=100.0)
    mark = sized("ً", 40, bottom=93.5)
    assert logical_line(drop_marks(line + mark)) == "أولا"


def test_a_slightly_raised_space_is_kept():
    """Space glyphs sit up to 1.7pt above the baseline — well inside the cut."""
    line = sized("ةدام", 0, bottom=100.0)
    space = sized(" ", 40, bottom=98.7)
    assert len(drop_marks(line + space)) == 5


def test_drop_marks_leaves_short_lines_alone():
    """Two glyphs give no reliable median; never guess on a short line."""
    line = sized("ام", 0, bottom=100.0)
    assert drop_marks(line) == line
