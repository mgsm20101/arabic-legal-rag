"""Ingestion tests.

The regression these lock down: issuance articles (المادة الأولى) and numbered
articles (مادة (1)) were originally sliced in two independent passes, which let
the last issuance article swallow the entire numbered body of the law.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.ingest import parse, validate  # noqa: E402

SAMPLE = """
المادة الأولى
يعمل بأحكام هذا القانون المرافق في شأن حماية البيانات الشخصية.

المادة الثانية
تسري أحكام هذا القانون على المصريين والمقيمين بالجمهورية.

مادة ( 1 )
في تطبيق أحكام هذا القانون يقصد بالمصطلحات الآتية المعاني المبينة قرين كل منها.

مادة ( 2 )
لا يجوز جمع البيانات الشخصية إلا بموافقة صريحة من الشخص المعني.

مادة ( 3 )
يلتزم المتحكم بالحصول على الموافقة قبل بدء أي عملية معالجة.
"""


def _by_id(articles):
    return {a.id: a for a in articles}


def test_both_header_styles_are_found():
    arts = _by_id(parse(SAMPLE, "sample.txt"))
    assert set(arts) == {"issuance-1", "issuance-2", "law-1", "law-2", "law-3"}


def test_issuance_article_does_not_swallow_numbered_articles():
    """The bug this file exists for."""
    arts = _by_id(parse(SAMPLE, "sample.txt"))
    last_issuance = arts["issuance-2"]
    assert "المصطلحات" not in last_issuance.text
    assert last_issuance.char_len < 120


def test_article_bodies_are_verbatim_enough_to_match_keywords():
    arts = _by_id(parse(SAMPLE, "sample.txt"))
    assert "موافقة صريحة" in arts["law-2"].text


def test_validate_reports_gaps():
    text = SAMPLE + "\nمادة ( 7 )\nيلتزم المتحكم بالإبلاغ خلال ٧٢ ساعة.\n"
    problems = validate(parse(text, "sample.txt"))
    assert any("missing article numbers" in p and "4" in p for p in problems)


def test_validate_clean_corpus_has_no_problems():
    assert validate(parse(SAMPLE, "sample.txt")) == []


def test_arabic_indic_digits_survive_into_search_text():
    text = "مادة ( 7 )\nيلتزم المتحكم بالإبلاغ خلال ٧٢ ساعة من وقت العلم بالخرق.\n"
    arts = _by_id(parse(text, "sample.txt"))
    assert "72" in arts["law-7"].search_text


# ------------------------------------------- real-PDF splitter rules (ADR-013)

ISSUANCE_THEN_LAW = """
قانون بإصدار قانون حماية البيانات الشخصية

المادة (1)
يعمل بأحكام هذا القانون والقانون المرافق في شأن حماية البيانات الشخصية.

المادة (٢)
تسري أحكام هذا القانون على كل من ارتكب إحدى الجرائم المنصوص عليها فيه.

قانون حماية البيانات الشخصية

مادة (1)
في تطبيق أحكام هذا القانون يقصد بالكلمات والعبارات التالية المعنى المبين قرينها.

مادة (2)
لا يجوز جمع البيانات الشخصية إلا بموافقة صريحة من الشخص المعني بها.
"""


def test_a_cross_reference_is_not_an_article_header():
    """'المادة (21) من هذا القانون' opens a line but is a citation, not a header.

    Observed in the 151/2020 PDF: it produced a duplicate article 21 and the
    text of the real article 21 was cut in half at the citation.
    """
    text = (
        "مادة (20)\n"
        "يلتزم المتحكم باتخاذ التدابير اللازمة لحماية البيانات الشخصية من الضياع.\n"
        "المادة (21) من هذا القانون تسري على كل حائز أو متحكم أو معالج للبيانات.\n"
        "مادة (22)\n"
        "يعاقب بالغرامة كل من خالف أحكام المواد السابقة من هذا القانون المرافق.\n"
    )
    arts = _by_id(parse(text, "sample.pdf"))
    assert set(arts) == {"law-20", "law-22"}
    # the citation stays inside article 20's body rather than starting a new one
    assert "من هذا القانون تسري" in arts["law-20"].text


def test_numeric_issuance_articles_are_split_from_the_law_body():
    """This PDF numbers the issuance articles 'المادة (1)' too, not 'المادة الأولى'.

    Without a restart rule both books collide and every number 1..7 duplicates.
    """
    arts = _by_id(parse(ISSUANCE_THEN_LAW, "sample.pdf"))
    assert set(arts) == {"issuance-1", "issuance-2", "law-1", "law-2"}
    assert "والقانون المرافق" in arts["issuance-1"].text
    assert "يقصد بالكلمات" in arts["law-1"].text


def test_issuance_split_leaves_a_single_numbering_valid():
    """A law with no issuance part must not be split at all."""
    text = "مادة (1)\nالأول.\n\nمادة (2)\nالثاني.\n\nمادة (3)\nالثالث والأخير هنا.\n"
    arts = parse(text, "sample.pdf")
    assert {a.book for a in arts} == {"law"}


def test_arabic_indic_article_numbers_are_recognised():
    arts = _by_id(parse(ISSUANCE_THEN_LAW, "sample.pdf"))
    assert arts["issuance-2"].number == 2


def test_the_real_pdf_structure_validates_clean():
    assert validate(parse(ISSUANCE_THEN_LAW, "sample.pdf")) == []
