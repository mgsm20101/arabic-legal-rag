"""Arabic normalization tests.

These exist because Arabic orthography is where Arabic RAG quietly fails:
the same word typed two legitimate ways does not match, and an over-eager
normalizer makes a wrong answer look right at scoring time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.normalize import (  # noqa: E402
    evaluation_normalize,
    orthographic_normalize,
    search_normalize,
)


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


# ---------------------------------------------------------------------------
# the middle normalizer (ADR-027)
# ---------------------------------------------------------------------------
#
# It exists to absorb the spellings a keyboard produces without turning the
# sentence into match keys. Both halves of that matter, so both are pinned:
# what it MUST fold, and what it MUST leave alone.


def test_orthographic_folds_the_forms_a_keyboard_drops():
    """Each of these is one missing modifier key, not a different word."""
    assert orthographic_normalize("الأفراد") == orthographic_normalize("الافراد")
    assert orthographic_normalize("إتاحة") == orthographic_normalize("اتاحه")
    assert orthographic_normalize("الدعائية") == orthographic_normalize("الدعائيه")
    assert orthographic_normalize("على") == orthographic_normalize("علي")
    assert orthographic_normalize("مسؤول") == orthographic_normalize("مسوول")


def test_orthographic_keeps_clitics_attached():
    """The distinction ADR-015 protected: «البيانات» and «للبيانات» are not one word."""
    assert orthographic_normalize("البيانات") != orthographic_normalize("للبيانات")


def test_orthographic_keeps_punctuation():
    """A transformer reads a sentence. `search_normalize` hands it a bag of keys."""
    assert "؟" in orthographic_normalize("ما المهلة؟")
    assert "." in orthographic_normalize("يلتزم المتحكم بالإبلاغ.")


def test_orthographic_keeps_word_boundaries_intact():
    text = "يلتزم المتحكم بإبلاغ المركز خلال اثنتين وسبعين ساعة"
    assert len(orthographic_normalize(text).split()) == len(text.split())


def test_orthographic_still_strips_diacritics_and_tatweel():
    """Inherited from evaluation_normalize, and needed for the same reason."""
    assert orthographic_normalize("مــادة") == orthographic_normalize("مادة")
    assert orthographic_normalize("مَادَة") == orthographic_normalize("مادة")


def test_orthographic_is_idempotent():
    """It runs on every passage and every query; a second pass must not drift."""
    once = orthographic_normalize("الأفراد على الهيئة؟")
    assert orthographic_normalize(once) == once


def test_search_normalize_is_the_orthographic_fold_plus_more():
    """Defined on top of it in the code, so the shared folds cannot drift apart."""
    text = "وبالبيانات الشخصية على الأفراد؟"
    folded = orthographic_normalize(text)
    assert search_normalize(text) == search_normalize(folded)
    assert search_normalize(text) != folded, "the aggressive steps stopped doing anything"


def test_the_recorded_cost_of_the_fold_is_real():
    """ADR-027 accepts that these collapse. Pinned so the cost stays visible
    rather than being rediscovered as a bug."""
    assert orthographic_normalize("إذن") == orthographic_normalize("أذن")
    assert orthographic_normalize("على") == orthographic_normalize("علي")


def test_evaluation_normalize_still_keeps_what_the_fold_collapses():
    """Scoring must not inherit the retrieval fold, or wrong answers score right."""
    assert evaluation_normalize("إذن") != evaluation_normalize("أذن")
    assert evaluation_normalize("معالجة") != evaluation_normalize("معالجه")
