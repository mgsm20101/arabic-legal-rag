"""Lexical-overlap guard tests (ADR-009)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.evaluate import binding_problem  # noqa: E402
from legalrag.overlap import containment, leaked_terms  # noqa: E402

ARTICLE = (
    "يلتزم المتحكم بإبلاغ المركز خلال 72 ساعة من وقت علمه بوقوع خرق "
    "للبيانات الشخصية، وإخطار الشخص المعني خلال ثلاثة أيام عمل."
)


def test_echoing_the_article_scores_high():
    echoed = "خلال كام ساعة يجب إبلاغ المركز بخرق البيانات الشخصية؟"
    assert containment(echoed, ARTICLE) >= 0.45


def test_user_phrasing_scores_low():
    natural = "لو حصل اختراق للداتا عندي أعمل إيه وفي خلال قد إيه؟"
    assert containment(natural, ARTICLE) <= 0.30


def test_echoed_beats_natural():
    echoed = "خلال كام ساعة يجب إبلاغ المركز بخرق البيانات الشخصية؟"
    natural = "لو حصل اختراق للداتا عندي أعمل إيه وفي خلال قد إيه؟"
    assert containment(echoed, ARTICLE) > containment(natural, ARTICLE)


def test_stopwords_do_not_inflate_the_score():
    """A question made only of function words shares nothing meaningful."""
    assert containment("هل هذا في ما على من؟", ARTICLE) == 0.0


def test_leaked_terms_names_the_copied_words():
    echoed = "خلال كام ساعة يجب إبلاغ المركز بخرق البيانات الشخصية؟"
    terms = leaked_terms(echoed, ARTICLE)
    # terms come back clitic-stripped: "المركز" -> "مركز"
    assert "مركز" in terms and "خلال" in terms


def test_binding_blocks_a_different_law():
    meta = {"corpus_law": "حماية البيانات الشخصية"}
    assert binding_problem(meta, ["قانون المرافعات"]) is not None


def test_binding_allows_the_matching_law():
    meta = {"corpus_law": "حماية البيانات الشخصية"}
    assert binding_problem(meta, ["قانون حماية البيانات الشخصية"]) is None


def test_binding_is_silent_before_ingestion():
    assert binding_problem({"corpus_law": "أي قانون"}, []) is None
