"""Tests for the OCR -> raw text conversion (ADR-020).

Every failure these catch is silent. Markup left in stops the article-header
regex from matching and `ingest` reports "0 articles" while pointing at the
splitter; page furniture left in attaches the gazette's running header to
whichever article a page break falls inside; string-sorted page keys deliver
the law shuffled. None of them raises.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.ocr_text import (  # noqa: E402
    is_page_furniture,
    normalise_headers,
    page_order,
    strip_markup,
    to_raw_text,
)

HEADER = "الجريدة الرسمية - العدد ٢٨ مكرر (ه) فى ١٥ يولية سنة ٢٠٢٠"


def test_markup_is_stripped_or_the_header_regex_stops_matching():
    assert strip_markup("<b>مادة</b> (١) :") == "مادة (١) :"


def test_entities_are_unescaped():
    assert strip_markup("&#1605;&amp;") == "م&"


def test_the_running_header_is_dropped():
    assert is_page_furniture(HEADER)


def test_a_bare_page_number_is_dropped():
    assert is_page_furniture("٢٩")
    assert is_page_furniture("  11  ")


def test_a_real_article_line_is_kept_even_when_it_contains_furniture_words():
    """`سنة` and `فى` appear throughout the law. The rule is shape, not words."""
    line = ("يلتزم كل من المتحكم والمعالج بحسب الأحوال حال علمه بوجود خرق "
            "للبيانات الشخصية لديه بإبلاغ المركز خلال اثنتين وسبعين ساعة فى سنة")
    assert not is_page_furniture(line)


def test_an_article_header_line_is_kept():
    assert not is_page_furniture("مادة (٧) :")


def test_pages_are_ordered_numerically_not_as_strings():
    """`p10` sorts before `p2` as a string. The law would arrive shuffled."""
    assert [page_order(k) for k in ("p2", "p10", "p07")] == [2, 10, 7]
    text, _ = to_raw_text({"p10": "عاشرة", "p2": "ثانية", "p07": "سابعة"})
    assert text.splitlines() == ["ثانية", "سابعة", "عاشرة"]


def test_conversion_reports_what_it_removed():
    pages = {"p1": f"{HEADER}\n٢\nمادة (١) :\nنص المادة الأولى"}
    text, dropped = to_raw_text(pages)
    assert dropped == 2
    assert text.splitlines() == ["مادة (١) :", "نص المادة الأولى"]


def test_a_line_that_merely_mentions_the_gazette_is_not_furniture():
    """Long prose containing one furniture word stays -- half the tokens must
    be furniture, and it must be short."""
    line = ("وتنشر هذه اللائحة فى الجريدة الرسمية ويعمل بها من اليوم التالى "
            "لتاريخ نشرها ويلتزم المركز بإخطار جميع الجهات المعنية بذلك")
    assert not is_page_furniture(line)


def test_blank_lines_are_not_counted_as_furniture():
    _, dropped = to_raw_text({"p1": "مادة (١) :\n\n\nنص"})
    assert dropped == 0


def test_a_page_number_in_extended_arabic_indic_digits_is_dropped():
    """U+06F0-06F9 (۲) renders like U+0660-0669 (٢) and is a different code
    point. surya emits both; a filter that knows only one lets page numbers
    through into the corpus."""
    assert is_page_furniture("۲")
    assert is_page_furniture(" ۲۹ ")


# --- the furniture rule keyed on shape, measured on real OCR output ----------
#
# Run 12 found the rule letting a running header through. The header had not
# been misread token by token -- the model had REWRITTEN it, replacing
# «الجريدة الرسمية – العدد ٤٥» with «الجدول رقم (٥)». Four digits would have
# entered the corpus attached to article ٣٦٠, which is the exact failure this
# module's docstring says it exists to prevent.

REWRITTEN_HEADER = "الجدول رقم (٥) في ٤٥ مكرر (د) في ٢٠٢٥ نوفمبر سنة ٩٦"
NOVEMBER_HEADER = "الجريدة الرسمية – العدد ٤٥ مكرر (د) في ١٢ نوفمبر سنة ٢٠٢٥"


def test_a_running_header_the_engine_rewrote_is_still_furniture():
    """deepseek-ocr on law 174/2025 p094, verbatim (EVAL.md Run 12)."""
    assert is_page_furniture(REWRITTEN_HEADER)


def test_the_month_is_not_part_of_the_rule_whatever_the_issue_it_came_from():
    """The root cause, pinned: the same header scored 0.875 in July and 0.667
    in November, because `يولية` was in the word set and `نوفمبر` was not.

    A month is as issue-specific as the issue number, and this module's own
    comment says only invariant words belong in the rule. Keeping one month
    tuned the rule to the corpus it was written against and left every other
    issue one OCR error away from slipping through -- which is what happened.
    """
    july = "الجريدة الرسمية - العدد ٢٨ مكرر (ه) فى ١٥ يولية سنة ٢٠٢٠"
    for month in ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونية",
                  "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"):
        assert is_page_furniture(july.replace("يولية", month)), month


def test_the_issue_series_marker_is_not_counted_against_the_header():
    """«(د)» and «(ه)» are part of the gazette's own header format, and were
    scored as ordinary prose words -- evidence AGAINST the line being the
    header they appear in. A one-character token is evidence either way."""
    assert is_page_furniture(NOVEMBER_HEADER)
    assert is_page_furniture(NOVEMBER_HEADER.replace(" (د)", ""))


def test_exactly_half_furniture_words_is_not_mostly_furniture_words():
    """The issuance date of the law, from the 2020 gazette, p02 verbatim.

    Two of its four words are in the set, and a `>= 0.5` rule deleted it --
    a real date, silently, from the text of record. The comment above the
    constant says "mostly furniture words", and mostly is more than half.
    """
    assert not is_page_furniture("( الموافق ١٣ يولية سنة ٢٠٢٠م ) .")


def test_a_two_word_tail_of_a_sentence_is_not_furniture():
    """surya's p17 of the 2020 gazette, verbatim: one of its two words is
    `في`, so the old rule scored it 0.5 and dropped it."""
    assert not is_page_furniture("في شأنها .")


def test_a_line_of_ocr_noise_with_no_real_word_is_dropped():
    """surya emits stray fragments like `.N` between columns. They carry no
    word at all, so there is nothing for the ratio to be computed over."""
    assert is_page_furniture(".N")
    assert is_page_furniture(". T")


# --- markdown an engine emitted as layout ------------------------------------
#
# Running the real pipeline on law 174/2025 lost sixteen articles out of fifty
# without a word of complaint. deepseek-ocr writes some article headers as
# markdown headings -- «### مادة (٦٦)» -- and `strip_markup` removed HTML tags
# only, so `ingest`'s header regex did not match the line and the article
# simply was not there. A gap in a sequence of 800 articles is not something
# anyone notices by reading.
#
# Measured on that document, not guessed at: 14 heading lines and 8 bullet
# lines out of 285, and no bold, italics, tables, code fences or blockquotes.


def test_a_markdown_heading_does_not_hide_an_article():
    """deepseek-ocr on law 174/2025 p020, verbatim."""
    assert strip_markup("### مادة (٦٦)") == "مادة (٦٦)"


def test_every_heading_level_is_stripped():
    for hashes in ("#", "##", "###", "####", "#####", "######"):
        assert strip_markup(f"{hashes} مادة (٦٦)") == "مادة (٦٦)"


def test_a_markdown_bullet_is_not_part_of_the_article_text():
    """deepseek-ocr on p062, verbatim: the law numbers these items, the engine
    re-rendered them as bullets, and the dash is not in the law."""
    assert strip_markup("- تاريخ اليوم والشهر والسنة.") == "تاريخ اليوم والشهر والسنة."


def test_a_hash_that_is_not_a_heading_is_left_alone():
    """A heading needs whitespace after its hashes. Nothing else is markdown,
    and this module must not start editing the text of record."""
    assert strip_markup("#٦٦") == "#٦٦"
    assert strip_markup("###") == "###"


def test_a_dash_inside_a_line_is_not_a_bullet():
    line = "المادة ٦٦ - وتسري أحكامها من تاريخ النشر"
    assert strip_markup(line) == line


def test_markdown_is_stripped_per_line_not_only_at_the_start_of_the_page():
    page = "### مادة (٦٦)\nنص المادة.\n- بند أول.\n### مادة (٦٧)"
    assert strip_markup(page).splitlines() == [
        "مادة (٦٦)", "نص المادة.", "بند أول.", "مادة (٦٧)"]


def test_markdown_and_html_are_both_gone():
    """surya emits HTML, deepseek emits markdown, and one corpus may hold
    pages from either."""
    assert strip_markup("### <b>مادة</b> (٦٦)") == "مادة (٦٦)"


# The full 137-page run found what the first 50 pages did not: bold. p014
# wrote all four of its headers as «**مادة (37)**», and articles 37-40 were
# absent from the corpus -- the same silent loss, one construct later.


def test_a_bold_header_does_not_hide_an_article():
    """deepseek-ocr on law 174/2025 p014, verbatim."""
    assert strip_markup("**مادة (37)**") == "مادة (37)"


def test_a_bold_heading_loses_both_markers():
    assert strip_markup("### **مادة (٦٦)**") == "مادة (٦٦)"


def test_bold_inside_a_line_is_left_alone():
    """Only a line that is bold from end to end is layout. Two bold runs on
    one line would otherwise lose their outer markers and keep the inner ones."""
    for line in ("نص **مهم** هنا", "**أ** و **ب**"):
        assert strip_markup(line) == line


def test_asterisks_that_wrap_nothing_are_left_alone():
    for line in ("**", "****", "** **"):
        assert strip_markup(line) == line


# --- an article header whose number the engine wrote first -------------------
# Measured on law 174/2025: 27 headers on 7 pages arrived as «(143) مادة»,
# including the unbroken run 314-320. `ingest`'s header pattern requires the
# word before the number, so every one of those articles was absent from the
# corpus with no error anywhere -- the same silent-loss shape as the markdown
# headings fixed in ADR-032.


def test_a_header_written_number_first_is_put_back_in_reading_order():
    assert normalise_headers("(143) مادة") == "مادة (143)"


def test_arabic_indic_digits_survive_the_reorder():
    assert normalise_headers("(١٤٣) مادة") == "مادة (١٤٣)"


def test_a_tatweel_stretched_header_is_reordered_too():
    """Arabic justification stretches the word; the gazette does it per line."""
    assert normalise_headers("(٣١٤) مـادة") == "مـادة (٣١٤)"


def test_a_citation_inside_a_sentence_is_not_a_header():
    """A header owns its line. Without that anchor this would rewrite prose --
    the same distinction `ingest` draws between a header and a reference."""
    line = "وذلك مع مراعاة حكم (143) مادة من هذا القانون"
    assert normalise_headers(line) == line


def test_a_header_already_in_reading_order_is_untouched():
    assert normalise_headers("مادة (143)") == "مادة (143)"


def test_every_reversed_header_on_a_page_is_reordered():
    page = "(314) مادة\nنص أول.\n(315) مادة\nنص ثان."
    assert normalise_headers(page).splitlines() == [
        "مادة (314)", "نص أول.", "مادة (315)", "نص ثان."]


def test_a_four_digit_number_is_not_an_article_header():
    """Article numbers run to three digits here; «(2025) مادة» would be a year
    that happened to land on its own line."""
    line = "(2025) مادة"
    assert normalise_headers(line) == line
