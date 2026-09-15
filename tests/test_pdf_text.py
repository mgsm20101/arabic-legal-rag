"""RTL reconstruction tests (ADR-013).

Every case here is a defect that was actually observed while ingesting the
151/2020 PDF, reduced to synthetic glyphs. Glyph dicts carry only the three
keys the extractor reads, so no PDF fixture is needed.
"""

import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.pdf_text import (  # noqa: E402
    arabic_column,
    drop_marks,
    extract_pages,
    group_lines,
    logical_line,
    mirror_pages,
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


def test_logical_line_no_longer_mirrors_brackets_itself():
    """A non-browser RTL renderer paints "مادة (1)" as ")1(" — but whether
    that needs mirroring back is a whole-document decision now
    (`mirror_pages`), never one `logical_line` makes for a single line: a
    browser-rendered PDF draws the same glyphs the other way round, and a
    per-line rule cannot tell the two apart. `logical_line` returns
    brackets exactly as the glyph codes read."""
    glyphs = visual(")1(", 0) + visual("ةدام", 30)
    assert logical_line(glyphs) == "مادة )1("


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
    # Unmirrored: logical_line no longer mirrors brackets on its own (see
    # mirror_pages for the whole-document decision that would fix this).
    assert logical_line(group_lines(kept)[0]) == "مادة )1("


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
# See the module docstring (point 4) and `ARABIC_LETTER` for what these are
# and why they matter. Code points below are confirmed against `unicodedata`
# directly, not guessed.

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


def test_a_base_letter_farsi_yeh_is_never_folded():
    """Folding is conditioned on the glyph being a presentation form — a
    real BASE-form Farsi yeh (U+06CC), however it got into the text, is
    legitimate Persian text and must survive unchanged, not be silently
    rewritten to U+064A."""
    assert logical_line([g("ی", 0)]) == "ی"


def test_a_raw_heh_doachashmee_substitute_folds_even_without_a_presentation_form():
    """See `_ALWAYS_FOLD_RAW` for what ArialMT actually emits raw and why."""
    assert logical_line([g("ھ", 0)]) == "ه"  # heh doachashmee (U+06BE)


def test_a_raw_heh_goal_or_keheh_substitute_is_never_folded():
    """Unlike heh-doachashmee, heh-goal (U+06C1) and keheh (U+06A9) were
    never observed on this fixture at all — not raw, not as a presentation
    form, in any position, on any page. They fold only when they
    demonstrably come from a presentation form, exactly like Farsi yeh —
    a raw occurrence is not assumed to be this font's bug without evidence
    for it the way U+06BE has."""
    assert logical_line([g("ہ", 0)]) == "ہ"  # heh goal (U+06C1), unchanged
    assert logical_line([g("ک", 0)]) == "ک"  # keheh (U+06A9), unchanged


def test_folding_is_applied_per_glyph_not_per_run():
    """A run holding BOTH a presentation-form glyph and a base-letter glyph
    must fold only the presentation-form one. Folding the run's ASSEMBLED
    text instead of each glyph on its own — joining raw text first, then
    checking once whether the joined string contains a presentation-form
    code point and normalising/translating all of it — would also fold
    the neighbouring base-form U+06CC, which
    `test_a_base_letter_farsi_yeh_is_never_folded` says must never happen."""
    presentation_yeh = g("ﯾ", 0)   # FARSI YEH INITIAL FORM -> folds to ي
    base_yeh = g("ی", 10)          # already U+06CC, base form: must stay ی
    assert logical_line([presentation_yeh, base_yeh]) == "یي"


def test_a_shadda_with_fatha_ligature_glyph_does_not_split_its_word():
    """U+FC60's own NFKC-injected space must not land inside the word it sits in."""
    # visual left -> right: م د [FC60] ق ي و  ->  logical: و ي ق [FC60] د م
    glyphs = [g("م", 0), g("د", 10), g("ﱠ", 20), g("ق", 30), g("ي", 40), g("و", 50)]
    assert logical_line(glyphs) == "ويقَّدم"
    assert " " not in logical_line(glyphs)


def test_a_genuine_space_next_to_a_shadda_ligature_glyph_still_survives():
    """A real word-boundary space glyph next to U+FC60 must still come through as exactly one space."""
    # visual left -> right: ج د [space] ب [FC60] ا  ->  logical: ا [FC60] ب [space] د ج
    glyphs = [g("ج", 0), g("د", 10), g(" ", 20), g("ب", 30), g("ﱠ", 40), g("ا", 50)]
    result = logical_line(glyphs)
    assert result == "اَّب دج"
    assert result.count(" ") == 1


# ------------------------------------------- column split, take 2 (ADR-013)

def test_a_single_latin_word_on_an_arabic_page_does_not_cut_the_page_into_columns():
    """A single Latin word inline in Arabic prose must never trigger a
    column split — evals/app/policy_ar.pdf page 2 has 'VPN', 'Wi-Fi' and
    'Microsoft Teams' inline in Arabic text, and the old rule (any Latin
    letter at all) cut roughly half the page's Arabic glyphs silently."""
    arabic = visual("ا" * 40, 0)   # a full line's worth of Arabic prose
    latin = visual("VPN", 500)     # one Latin word, far to the right
    assert arabic_column(arabic + latin) == arabic + latin


def test_the_latin_ratio_condition_alone_blocks_a_cleanly_separated_but_small_latin_group():
    """Isolates COLUMN_LATIN_RATIO from COLUMN_BLEED_RATIO: 3 Latin letters
    against 20 Arabic ones (15%, below the 20% floor) sit in a perfectly
    clean, zero-bleed column of their own — the bleed condition alone
    would happily split this. Only the ratio condition blocks it, so this
    fails if that condition is ever removed on its own."""
    arabic = visual("ا" * 20, 0)   # centres 5..195
    latin = visual("VPN", 1000)    # centres 1005..1025, cleanly separated
    assert arabic_column(arabic + latin) == arabic + latin


def test_the_bleed_condition_alone_blocks_a_high_ratio_but_interleaved_page():
    """Isolates COLUMN_BLEED_RATIO from COLUMN_LATIN_RATIO: 3 Latin letters
    against 10 Arabic ones (30%, comfortably past the 20% floor) — but one
    Latin letter (centre 45) sits spatially INSIDE the Arabic glyphs' own
    range (centres 5..95), so the median-midpoint boundary misclassifies
    it (1/3 = 33% bleed, far past the 5% cap). Only the bleed condition
    blocks this one, so this fails if that condition is ever removed on
    its own."""
    arabic = visual("ا" * 10, 0)                    # centres 5..95
    latin = [g("V", 40), g("P", 140), g("N", 240)]  # centres 45, 145, 245
    assert arabic_column(arabic + latin) == arabic + latin


def test_arabic_glyphs_crossing_the_divider_alone_block_the_split():
    """The bleed check has two halves; the test above only ever puts a
    LATIN glyph on the wrong side. Here one ARABIC letter sits far into
    the Latin column (1/11 = 9% arabic bleed, past the 5% cap) while every
    Latin letter stays cleanly on its own side (0% latin bleed) — only
    deleting `arabic_bleed > COLUMN_BLEED_RATIO` specifically would miss
    this and split."""
    arabic = visual("ا" * 10, 0) + [g("ا", 500)]     # centres 5..95, then 505
    latin = [g("V", 200), g("P", 210), g("N", 220)]  # centres 205, 215, 225
    assert arabic_column(arabic + latin) == arabic + latin


# --------------------------------------- mirror_pages (per document, ADR-013)
#
# Whether a document's brackets need mirroring, and separately whether its
# «» do, is decided ONCE from the whole document's raw (unmirrored) text —
# see the module docstring (point 8) and `mirror_pages` for why a per-line
# rule is unsafe: a legal parenthetical wraps across lines, so a line can
# legitimately begin with ")" and end with "(" on its own.

def test_a_majority_mirrored_bracket_document_gets_mirrored():
    pages = ["نص فيه )1( وكمان )12( في نفس الوثيقة."]
    mirrored, decision = mirror_pages(pages)
    assert decision["bracket_mirrored"] == 2 and decision["bracket_logical"] == 0
    assert decision["mirror_brackets"] is True
    assert mirrored == ["نص فيه (1) وكمان (12) في نفس الوثيقة."]


def test_a_majority_logical_bracket_document_is_left_alone():
    pages = ["نص فيه (1) وكمان (12) في نفس الوثيقة."]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_brackets"] is False
    assert mirrored == pages


def test_a_wrapped_parenthetical_does_not_flip_a_majority_logical_document():
    """The wrapped line legitimately begins with ')' and ends with '(' —
    counted as evidence for NEITHER pattern, since neither regex matches
    without a bare number immediately inside the brackets, the way a real
    citation has. The majority-logical '(3)' line elsewhere decides it."""
    pages = [
        "كذا) وفي حالة (كذا أخرى ينطبق حكم مختلف تماماً.",
        "تنص المادة على ذلك في الفقرة (3) من هذا الباب.",
    ]
    mirrored, decision = mirror_pages(pages)
    assert decision["bracket_logical"] == 1 and decision["bracket_mirrored"] == 0
    assert decision["mirror_brackets"] is False
    assert mirrored == pages


def test_a_majority_mirrored_quote_document_gets_swapped():
    """Two separate pages, not one page with both pairs: `_QUOTE_LOGICAL`'s
    middle class excludes «»`\\n` but not plain words, so two mirrored
    pairs sharing ONE line would also, harmlessly, match a spurious
    logical pair bridging the gap between them (the first pair's closing
    mark to the second's opening mark) — real evidence either way, since
    the majority (2 vs 1) still decides the same way, but avoided here so
    the counts pin exactly what this test is about."""
    pages = ["نص فيه »اقتباس أول«.", "ونص آخر فيه »اقتباس ثانٍ«."]
    mirrored, decision = mirror_pages(pages)
    assert decision["quote_mirrored"] == 2 and decision["quote_logical"] == 0
    assert decision["mirror_quotes"] is True
    assert mirrored == ["نص فيه «اقتباس أول».", "ونص آخر فيه «اقتباس ثانٍ»."]


def test_a_majority_logical_quote_document_is_kept():
    pages = ["نص فيه «اقتباس صحيح» لا يحتاج أي تعديل."]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_quotes"] is False
    assert mirrored == pages


def test_a_single_unpaired_guillemet_is_untouched():
    """Zero evidence either way — unlike brackets, quotes default to
    UNCHANGED on a tie or no evidence: «» was never mirrored before
    `mirror_pages` existed."""
    pages = ["نص فيه « بدون إغلاق على الإطلاق."]
    mirrored, decision = mirror_pages(pages)
    assert decision["quote_logical"] == 0 and decision["quote_mirrored"] == 0
    assert decision["mirror_quotes"] is False
    assert mirrored == pages


def test_brackets_and_quotes_are_decided_independently():
    """A document majority-mirrored on brackets but majority-logical on
    quotes must not have one decision leak into the other."""
    pages = ["فيه )1( وفيه أيضاً «اقتباس صحيح» في نفس السطر."]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_brackets"] is True
    assert decision["mirror_quotes"] is False
    assert mirrored == ["فيه (1) وفيه أيضاً «اقتباس صحيح» في نفس السطر."]


def test_curly_and_angle_brackets_follow_the_bracket_decision():
    """`{}`/`<>` are bound to the SAME decision as `()`/`[]` — `MIRRORED`
    has always treated all four pairs as one family."""
    pages = ["فيه )1( في مكان ما، وفيه أيضاً }غير مرتبط{ في مكان آخر."]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_brackets"] is True
    assert mirrored == ["فيه (1) في مكان ما، وفيه أيضاً {غير مرتبط} في مكان آخر."]


# ---------------------------- mirror_pages, rule (c): attached vs free -----
#
# A bracket's mirror direction depends on what it wraps, not on the whole
# document: a bracket touching a digit or a Latin letter (a citation number,
# an acronym) reads correctly regardless of how a renderer drew it, but a
# bracket around an Arabic word only reads correctly if its OWN evidence
# decides. "Legacy mode" is the escape hatch that keeps the statute's
# output byte-identical: when digit/letter-attached brackets are a clear
# mirrored majority, behave exactly as `mirror_pages` always did — mirror
# every bracket in the document via the single `MIRRORED` table. Otherwise,
# attached and free brackets each decide from their own evidence. «» are
# never split this way — see the tests below confirming that.


def test_legacy_mode_still_mirrors_every_bracket_including_a_free_one():
    """Attached brackets are almost all `)N(` — a clear mirrored majority —
    so legacy mode applies and EVERY bracket is mirrored via the single
    `MIRRORED` table, exactly as before this rule existed, including a
    free Arabic-word bracket that would decide the OPPOSITE way on its
    own evidence."""
    pages = [
        "نص فيه )1( وفيه أيضاً )2( وفيه )3( وأيضاً )4(، ثم فيه بند حر )الموظف(."
    ]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_brackets"] is True
    assert mirrored == [
        "نص فيه (1) وفيه أيضاً (2) وفيه (3) وأيضاً (4)، ثم فيه بند حر (الموظف)."
    ]


def test_mixed_document_decides_attached_and_free_independently():
    """The scenario the whole fix exists for: attached brackets read
    logically (a clear majority, not mirrored) while free brackets read
    mirrored (also a clear majority), in the SAME document. Each class
    must decide on its own evidence: the numbered citations stay exactly
    as read, and the two free Arabic-word brackets get mirrored back to
    logical order."""
    pages = [
        "المادة الأولى (1) والمادة الثانية (2) والمادة الثالثة (3)، "
        "وفيها أيضاً )الموظف( و)موظف آخر(."
    ]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_brackets"] is False
    assert decision["mirror_attached"] is False
    assert decision["mirror_free"] is True
    assert mirrored == [
        "المادة الأولى (1) والمادة الثانية (2) والمادة الثالثة (3)، "
        "وفيها أيضاً (الموظف) و(موظف آخر)."
    ]


def test_zero_evidence_document_does_not_trigger_legacy_mode():
    """A single free mirrored bracket, no attached brackets anywhere: the
    legacy gate must NOT fire on 0-vs-0 attached evidence — that was the
    old `>=` bug (see the module's `MIRRORED` docstring). The free class
    decides on its OWN 1-mirrored/0-logical evidence instead, and mirrors
    it — nothing forces "mirror everything" here, unlike the old rule."""
    pages = ["نص فيه بند حر )الموظف( لا غير."]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_brackets"] is False
    assert decision["free_mirrored"] == 1 and decision["free_logical"] == 0
    assert decision["mirror_free"] is True
    assert mirrored == ["نص فيه بند حر (الموظف) لا غير."]


def test_attached_tie_leaves_both_readings_alone():
    """Equal attached-mirrored and attached-logical counts: the strict `>`
    rule (unlike the old `>=`) means a tie changes NOTHING, in either
    direction — both stay exactly as read."""
    pages = ["نص فيه (1) وفيه أيضاً )2(."]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_brackets"] is False
    assert decision["attached_logical"] == 1 and decision["attached_mirrored"] == 1
    assert decision["mirror_attached"] is False
    assert mirrored == pages


def test_latin_attached_brackets_count_as_attached_not_free():
    """`(VPN)`/`(Wi-Fi)` are Latin-attached — not free — so they must stay
    exactly as read even though a free Arabic bracket elsewhere in the
    same document independently decides to mirror."""
    pages = [
        "يمكن استخدام (VPN) أو (Wi-Fi) للاتصال، وفيها أيضاً بند حر )الموظف(."
    ]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_attached"] is False
    assert decision["mirror_free"] is True
    assert mirrored == [
        "يمكن استخدام (VPN) أو (Wi-Fi) للاتصال، وفيها أيضاً بند حر (الموظف)."
    ]


def test_quotes_remain_one_whole_document_decision_independent_of_bracket_class():
    """«» keep exactly the same single, whole-document decision as before,
    with no attached/free split — confirmed here on a document that goes
    through the LEGACY bracket branch (attached brackets are a clear
    mirrored majority) while its quotes are majority LOGICAL and so stay
    untouched: the two decisions never leak into each other."""
    pages = ["نص فيه )1( وفيه أيضاً )2(، وفيه أيضاً «اقتباس صحيح»."]
    mirrored, decision = mirror_pages(pages)
    assert decision["mirror_brackets"] is True
    assert decision["mirror_quotes"] is False
    assert mirrored == ["نص فيه (1) وفيه أيضاً (2)، وفيه أيضاً «اقتباس صحيح»."]


def test_the_real_fixtures_guillemets_match_its_plain_text_sources_orientation(policy_ar_pages):
    """policy_ar.pdf's only «» pair (policy_ar.txt line 53) arrives
    mirror-drawn on this browser-rendered PDF — measured directly, not
    assumed: the same rendering pipeline that leaves `()` alone (there are
    none in this document) apparently does not do the same for «»."""
    text = "\n".join(p for p in policy_ar_pages if p)
    assert "« يحتاج إلى تحسين »" in text
    assert "» يحتاج إلى تحسين «" not in text


@pytest.fixture(scope="module")
def brackets_ar_pages():
    """tests/fixtures/brackets_ar.pdf, rendered from brackets_ar.html with
    Edge headless (evals/app/README.md's method) — a second, small,
    browser-rendered PDF, committed so this fixture never depends on a
    render happening again."""
    pytest.importorskip("pdfplumber")
    root = Path(__file__).resolve().parents[1]
    return extract_pages(root / "tests" / "fixtures" / "brackets_ar.pdf")


def test_the_brackets_fixture_yields_plain_numbered_brackets(brackets_ar_pages):
    """A quality reviewer rendered test pages and got «مادة )1(» — brackets
    mirrored a second time, because a browser-rendered PDF already stores
    the logical glyph. This fixture pins the fix directly: extraction must
    yield "(1)" and "(12)", never their mirrored form, and its «» (a
    separate family, decided independently) must still come out correctly
    oriented too."""
    text = "\n".join(brackets_ar_pages)
    assert "(1)" in text
    assert "(12)" in text
    assert ")1(" not in text
    assert ")12(" not in text
    assert "« مادة التعريفات »" in text
    assert "« بند الإخطار »" in text


def test_the_brackets_fixtures_decision_is_observable_via_report():
    """`report` fills with the decision `extract_pages` made, so a caller
    (or this test) can see the evidence without re-deriving it."""
    pytest.importorskip("pdfplumber")
    root = Path(__file__).resolve().parents[1]
    report: dict = {}
    extract_pages(root / "tests" / "fixtures" / "brackets_ar.pdf", report=report)
    assert report["mirror_brackets"] is False
    assert report["mirror_quotes"] is True


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


def test_extract_text_passes_keep_latin_and_line_tol_through_to_extract_pages(monkeypatch):
    """The stub above ignores both arguments — pin that a real call site
    passing non-default values actually reaches `extract_pages` with them."""
    import legalrag.pdf_text as pdf_text

    seen = {}

    def fake_extract_pages(path, keep_latin=False, line_tol=None):
        seen["keep_latin"] = keep_latin
        seen["line_tol"] = line_tol
        return []

    monkeypatch.setattr(pdf_text, "extract_pages", fake_extract_pages)

    pdf_text.extract_text("ignored.pdf", keep_latin=True, line_tol=2.5)

    assert seen == {"keep_latin": True, "line_tol": 2.5}


class _FakePage:
    def __init__(self, chars):
        self.chars = chars

    def close(self):
        pass


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_extract_pages_keeps_a_blank_page_at_its_own_index(monkeypatch):
    """Phase 4B labels chunks by page number — a blank page (no text at
    all) must still occupy its own slot as "", not be silently skipped and
    shift every later page's index down by one."""
    pytest.importorskip("pdfplumber")
    import pdfplumber

    page1 = _FakePage(visual("ةدام", 0))  # "مادة" -> non-blank
    page2 = _FakePage([])                  # blank: no chars at all
    page3 = _FakePage(visual("ةدام", 0))  # non-blank again

    monkeypatch.setattr(pdfplumber, "open", lambda path: _FakePdf([page1, page2, page3]))

    assert extract_pages("ignored.pdf") == ["مادة", "", "مادة"]


class _ClosablePage:
    """A page that records its own name when closed, and can raise on
    `.chars` access instead of returning real glyphs — see `extract_pages`
    for why every page must be closed, including one that raises."""

    def __init__(self, chars, name, closed, raises=False):
        self._chars = chars
        self.name = name
        self._closed = closed
        self._raises = raises

    @property
    def chars(self):
        if self._raises:
            raise RuntimeError(f"boom on {self.name}")
        return self._chars

    def close(self):
        self._closed.append(self.name)


def test_extract_pages_closes_every_page_even_when_a_later_page_raises(monkeypatch):
    pytest.importorskip("pdfplumber")
    import pdfplumber

    closed: list[str] = []
    page1 = _ClosablePage(visual("ةدام", 0), "p1", closed)
    page2 = _ClosablePage([], "p2", closed, raises=True)
    page3 = _ClosablePage(visual("ةدام", 0), "p3", closed)

    monkeypatch.setattr(pdfplumber, "open", lambda path: _FakePdf([page1, page2, page3]))

    with pytest.raises(RuntimeError):
        extract_pages("ignored.pdf")

    assert closed == ["p1", "p2"]  # p3 never reached; both p1 and p2 closed


@pytest.fixture(scope="module")
def policy_ar_pages():
    """`evals/app/policy_ar.pdf`, extracted once and shared by every test
    that reads it below — three separate full extractions of the same
    6-page PDF added up for no reason."""
    pytest.importorskip("pdfplumber")
    root = Path(__file__).resolve().parents[1]
    return extract_pages(root / "evals" / "app" / "policy_ar.pdf")


def test_the_browser_rendered_fixture_keeps_every_heading_and_keyword_on_its_page(
    policy_ar_pages,
):
    """evals/app/policy_ar.pdf (README.md) is the failing test the upload
    pipeline had to pass: the first extraction matched zero headings and
    zero keywords, though the page count was already right. Presentation
    forms, Persian code points and the column cut (all fixed above) are
    exactly what this fixture exercises."""
    from legalrag.normalize import evaluation_normalize  # noqa: E402

    root = Path(__file__).resolve().parents[1]
    pages = policy_ar_pages
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


def test_the_fixture_has_no_shadda_ligature_word_split_left(policy_ar_pages):
    """6 raw U+FC60 glyphs sit mid-word across 4 pages (1, 3 x2, 4 x2, 6);
    none of the 4 distinct words checked here (one per page) may come out
    split by an injected space — no expected keyword carries a shadda,
    which is why the keyword-matching check above missed this."""
    pages = policy_ar_pages

    broken = {
        1: "ويُقد َّم",
        3: "ويُسج َّل",   # page 3 also repeats "ويُقد َّم" — same pattern
        4: "يُقي َّم",     # page 4 also repeats "ويُقد َّم" — same pattern
        6: "وتوج َّه",
    }
    fixed = {
        1: "ويُقدَّم",
        3: "ويُسجَّل",
        4: "يُقيَّم",
        6: "وتوجَّه",
    }
    for page_num in (1, 3, 4, 6):
        page_text = pages[page_num - 1]
        assert broken[page_num] not in page_text, (
            f"page {page_num}: shadda ligature still split its word"
        )
        assert fixed[page_num] in page_text, (
            f"page {page_num}: expected unsplit word missing entirely"
        )


_ARABIC_LETTER = re.compile("[؀-ۿ]")
_EDGE_PUNCT = ".,،؛:؟!()[]{}«»\"'"


def _arabic_words(text: str) -> list[str]:
    """Whitespace-split `text` (already run through `evaluation_normalize`
    on the caller's side) into Arabic-letter words, edge punctuation
    stripped from each token — used only to compare the PDF's extracted
    words against the plain-text source's, where a token's leading/
    trailing punctuation can legitimately land differently (e.g. a
    sentence-final period landing as its own glyph run in the PDF)."""
    words = []
    for w in text.split():
        w = w.strip(_EDGE_PUNCT)
        if _ARABIC_LETTER.search(w):
            words.append(w)
    return words


def test_the_fixtures_extracted_words_match_its_plain_text_source_in_order(policy_ar_pages):
    """The SEQUENCE of Arabic words extracted from the PDF must equal the
    plain-text source's — a stronger guard than the keyword-substring
    checks above, and the one that catches a mutant removing the
    NFKC-leading-space fix (no expected keyword happens to carry a
    shadda, so that check alone misses it: 774 words instead of 768)."""
    from legalrag.normalize import evaluation_normalize  # noqa: E402

    root = Path(__file__).resolve().parents[1]
    # Exactly what `extract_text` itself does with `extract_pages`' output —
    # reusing the shared fixture instead of a second full extraction.
    pdf_text = "\n".join(p for p in policy_ar_pages if p)
    txt_source = (root / "evals" / "app" / "policy_ar.txt").read_text(encoding="utf-8")

    pdf_words = _arabic_words(evaluation_normalize(pdf_text))
    txt_words = _arabic_words(evaluation_normalize(txt_source))

    assert len(pdf_words) == 768
    assert pdf_words == txt_words


# ------------------------------------------------- statute corpus hash -----

def test_the_statute_corpus_hash_is_unchanged(tmp_path):
    """Regression guard on the real corpus, not just the fixture:
    COLUMN_LATIN_RATIO set to an unreasonable value (say 1.25) passes the
    rest of the suite but silently changes what `ingest.main` writes for
    the real statute PDF. Skipped when the raw PDF or the real processed
    corpus is absent (fresh clone, CI) — both are gitignored and must
    never be committed."""
    pytest.importorskip("pdfplumber")
    root = Path(__file__).resolve().parents[1]
    raw_pdf = root / "data" / "raw" / "law-151-2020-personal-data-protection.pdf"
    real_output = root / "data" / "processed" / "articles.jsonl"
    if not raw_pdf.exists() or not real_output.exists():
        pytest.skip("data/raw or data/processed corpus is absent (gitignored)")

    from legalrag import ingest  # noqa: E402

    text = ingest._read_raw(raw_pdf)
    articles = ingest.parse(text, raw_pdf.name, law_name="قانون حماية البيانات الشخصية")

    # newline="\r\n" explicitly: the real, committed corpus has CRLF on
    # all 56 lines (Windows text-mode default when `ingest.main` produced
    # it there) — pinned here rather than left to the current platform's
    # own default, so this hash is comparable on Linux/Docker too.
    scratch_output = tmp_path / "articles.jsonl"
    with scratch_output.open("w", encoding="utf-8", newline="\r\n") as fh:
        for a in articles:
            fh.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")

    assert hashlib.sha256(scratch_output.read_bytes()).hexdigest() == (
        "d11ce900418b4e9725d1fc727b2880c7062a16d225e62351181c93e7fd3dc32a"
    )
