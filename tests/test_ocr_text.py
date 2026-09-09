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
