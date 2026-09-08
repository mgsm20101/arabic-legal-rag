"""Arabic normalization tests.

These exist because Arabic orthography is where Arabic RAG quietly fails:
the same word typed two legitimate ways does not match, and an over-eager
normalizer makes a wrong answer look right at scoring time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.normalize import evaluation_normalize, search_normalize  # noqa: E402


def test_search_folds_alef_variants():
    assert search_normalize("الأسماء") == search_normalize("الاسماء")
    assert search_normalize("إتاحة") == search_normalize("اتاحة")


def test_search_folds_ta_marbuta_and_alef_maqsura():
    assert search_normalize("معالجة") == search_normalize("معالجه")
    assert search_normalize("على") == search_normalize("علي")


def test_evaluation_keeps_orthographic_distinctions():
    """The scoring normalizer must NOT fold what search folds."""
    assert evaluation_normalize("الأسماء") != evaluation_normalize("الاسماء")
    assert evaluation_normalize("معالجة") != evaluation_normalize("معالجه")


def test_both_strip_tashkeel_and_tatweel():
    assert evaluation_normalize("الْبَيَانَات") == evaluation_normalize("البيانات")
    assert evaluation_normalize("بيــانات") == evaluation_normalize("بيانات")


def test_arabic_indic_digits_are_normalized():
    assert evaluation_normalize("٧٢ ساعة") == evaluation_normalize("72 ساعة")
    assert "72" in search_normalize("خلال ٧٢ ساعة")


def test_whitespace_collapse():
    assert evaluation_normalize("مادة   7 \n\n نص") == "مادة 7 نص"


def test_clitics_are_stripped_so_attached_forms_match():
    """"البيانات" and "للبيانات" must land on the same token."""
    from legalrag.normalize import search_normalize as sn
    assert sn("البيانات") == sn("للبيانات") == sn("وبالبيانات")
    assert sn("إبلاغ") == sn("بإبلاغ") == sn("للإبلاغ")


def test_short_words_are_not_over_stripped():
    from legalrag.normalize import strip_clitics
    assert strip_clitics("لبن") == "لبن"
    assert strip_clitics("بيت") == "بيت"


def test_evaluation_normalize_does_not_strip_clitics():
    """Scoring must not fold a preposition away — that changes legal meaning."""
    from legalrag.normalize import evaluation_normalize as en
    assert en("البيانات") != en("للبيانات")
