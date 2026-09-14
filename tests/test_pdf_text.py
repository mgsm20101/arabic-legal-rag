"""RTL reconstruction tests (ADR-013).

Every case here is a defect that was actually observed while ingesting the
151/2020 PDF, reduced to synthetic glyphs. Glyph dicts carry only the three
keys the extractor reads, so no PDF fixture is needed.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.pdf_text import (  # noqa: E402
    arabic_column,
    drop_marks,
    extract_pages,
    extract_text,
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


# --------------------------------- presentation forms (evals/app/policy_ar.pdf)
#
# Edge (the browser that rendered policy_ar.pdf) encodes 572 of page 1's 965
# Arabic glyphs as contextual presentation forms (U+FB50-FDFF, U+FE70-FEFF).
# The old ARABIC_LETTER range (U+0600-06FF only) treated every one of them as
# non-Arabic: never reversed, never folded, and the extractor matched zero
# headings and zero keywords on the fixture (evals/app/README.md). Code
# points below are confirmed against `unicodedata` directly, not guessed:
# isolated presentation forms of م ا د ة (U+FEE1, U+FE8D, U+FEA9, U+FE93)
# each NFKC-decompose to exactly their base letter.

def test_presentation_form_glyphs_are_read_as_arabic_and_reversed():
    """A page built entirely from presentation-form glyphs must still be
    classified as Arabic and reversed into logical order, the same as the
    base-letter case `test_arabic_run_is_reversed_into_logical_order` pins."""
    # Isolated presentation forms of teh marbuta, dal, alef, meem, laid out
    # visually left -> right (rightmost-first logical reading, same layout
    # as `visual("ةدام")` in the base-letter test above).
    glyphs = [g("ﺓ", 0), g("ﺩ", 10), g("ﺍ", 20), g("ﻡ", 30)]
    assert logical_line(glyphs) == "مادة"


def test_a_presentation_form_lam_alef_ligature_expands_in_logical_order():
    """U+FEFB is ONE glyph whose NFKC decomposition is TWO characters
    ('لا'), already in logical order — the presentation-form sibling of
    `test_lam_alef_ligature_survives_as_one_glyph`'s ToUnicode-mapped
    ligature. Folding happens per glyph, so reversing glyph ORDER (never
    the string) must not also re-reverse what NFKC expanded."""
    # visual left -> right: ى ل و [FEFB -> لا] ا
    glyphs = [g("ى", 0), g("ل", 10), g("و", 20), g("ﻻ", 30), g("ا", 40)]
    assert logical_line(glyphs) == "الاولى"


def test_persian_yeh_and_heh_from_presentation_forms_fold_to_arabic():
    """The font maps yeh/heh/kaf presentation forms so that NFKC alone
    still leaves Persian/Urdu code points (ی U+06CC, ھ U+06BE, ہ U+06C1,
    ک U+06A9) instead of Arabic ones — 'أيام' never matched 'أیام'
    downstream (evals/app/README.md). Folding must apply on top of NFKC."""
    yeh = g("ﯾ", 0)    # FARSI YEH INITIAL FORM -> NFKC ی (U+06CC)
    heh1 = g("ﮪ", 10)  # HEH DOACHASHMEE ISOLATED FORM -> NFKC ھ (U+06BE)
    heh2 = g("ﮦ", 20)  # HEH GOAL ISOLATED FORM -> NFKC ہ (U+06C1)
    kaf = g("ﮎ", 30)   # KEHEH ISOLATED FORM -> NFKC ک (U+06A9)
    assert logical_line([yeh, heh1, heh2, kaf]) == "كههي"


def test_a_base_letter_glyph_is_never_folded():
    """Folding is conditioned on the glyph being a presentation form — a
    real BASE-form Farsi yeh (U+06CC), however it got into the text, must
    survive unchanged, not be silently rewritten to U+064A."""
    assert logical_line([g("ی", 0)]) == "ی"


def test_a_raw_heh_substitute_folds_even_without_a_presentation_form():
    """evals/app/policy_ar.pdf emits U+06BE (heh doachashmee) directly —
    confirmed with a direct pdfplumber scan, not just after NFKC — for
    'القاھرة' and 'المقاھي' on page 2, with no presentation-form code
    point involved anywhere. Unlike Farsi yeh (never folded in base form,
    see above), heh-doachashmee/heh-goal/keheh have no legitimate use in
    this Arabic corpus, so a RAW occurrence is folded too."""
    assert logical_line([g("ھ", 0)]) == "ه"  # heh doachashmee (U+06BE)
    assert logical_line([g("ہ", 0)]) == "ه"  # heh goal (U+06C1)
    assert logical_line([g("ک", 0)]) == "ك"  # keheh (U+06A9)


# ------------------------------------------- column split, take 2 (ADR-013)

def test_a_single_latin_word_on_an_arabic_page_does_not_cut_the_page_into_columns():
    """A single Latin word inline in Arabic prose must never trigger a
    column split — evals/app/policy_ar.pdf page 2 has 'VPN', 'Wi-Fi' and
    'Microsoft Teams' inline in Arabic text, and the old rule (any Latin
    letter at all) cut roughly half the page's Arabic glyphs silently."""
    arabic = visual("ا" * 40, 0)   # a full line's worth of Arabic prose
    latin = visual("VPN", 500)     # one Latin word, far to the right
    assert arabic_column(arabic + latin) == arabic + latin


def test_a_real_two_column_arabic_english_page_is_still_split():
    """A genuine dual-language column (151/2020's Arabic-beside-English
    layout) must still be split under the new proportional/bleed rule —
    the case the rule must not break while fixing the single-word bug."""
    english = visual("Article one hereby applies to all", 0)  # a full column
    arabic = visual(")1( ةدام", 500)
    kept = arabic_column(english + arabic)
    assert all(c["x0"] >= 500 for c in kept)
    assert logical_line(group_lines(kept)[0]) == "مادة (1)"


# ------------------------------------------------------- extract_pages/text

def test_extract_text_is_the_join_of_extract_pages(monkeypatch):
    """`extract_text` must do nothing more than join `extract_pages`'
    output with blank pages dropped — the whole point of splitting the
    function is that the combined document text can never diverge from
    the per-page text the upload pipeline chunks on."""
    import legalrag.pdf_text as pdf_text

    monkeypatch.setattr(
        pdf_text, "extract_pages",
        lambda path, keep_latin=False, line_tol=None: ["a", "", "b"],
    )

    assert pdf_text.extract_text("ignored.pdf") == "a\nb"


def test_the_browser_rendered_fixture_keeps_every_heading_and_keyword_on_its_page():
    """evals/app/policy_ar.pdf (README.md) is the failing test the upload
    pipeline had to pass: the first extraction matched zero headings and
    zero keywords, though the page count was already right. Presentation
    forms, Persian code points and the column cut (all fixed above) are
    exactly what this fixture exercises."""
    pytest.importorskip("pdfplumber")
    from legalrag.normalize import evaluation_normalize  # noqa: E402

    root = Path(__file__).resolve().parents[1]
    pages = extract_pages(root / "evals" / "app" / "policy_ar.pdf")
    assert len(pages) == 6

    normalized = [evaluation_normalize(p) for p in pages]

    headings = {
        1: ["أولاً", "ثانياً"],
        2: ["ثالثاً", "رابعاً"],
        3: ["خامساً", "سادساً"],
        4: ["سابعاً", "ثامناً"],
        5: ["تاسعاً", "عاشراً"],
        6: ["حادي عشر", "ثاني عشر"],
    }
    for page_num, words in headings.items():
        page_text = normalized[page_num - 1]
        for word in words:
            assert evaluation_normalize(word) in page_text, (
                f"heading {word!r} missing from page {page_num}"
            )

    questions_path = root / "evals" / "app" / "questions.jsonl"
    questions = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for q in questions:
        if not q["answerable"]:
            continue
        page_text = normalized[q["expected_pages"][0] - 1]
        for kw in q["expected_keywords"]:
            assert evaluation_normalize(kw) in page_text, (
                f"question {q['id']}: keyword {kw!r} missing from page "
                f"{q['expected_pages'][0]}"
            )
