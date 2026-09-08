"""Arabic text normalization.

Two DISTINCT normalizers — do not merge them:

* ``search_normalize``      — aggressive. Used only when building the retrieval
                              index and when normalizing a user query. Loses
                              information on purpose so that "الأسماء" matches
                              "الاسماء".
* ``evaluation_normalize``  — conservative. Used when comparing a system answer
                              against ground truth. Must NOT collapse
                              distinctions that a human reviewer would call a
                              real difference.

Rationale (see DECISIONS.md ADR-004): using one normalizer for both inflates
scores — an aggressive normalizer makes wrong answers look right at eval time.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------

TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"

ALEF_FORMS = "آأإٱ"  # آ أ إ ٱ
ALEF = "ا"  # ا

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EXTENDED_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
WESTERN_DIGITS = "0123456789"

_DIGIT_MAP = {
    **{ord(a): w for a, w in zip(ARABIC_INDIC_DIGITS, WESTERN_DIGITS)},
    **{ord(a): w for a, w in zip(EXTENDED_ARABIC_INDIC_DIGITS, WESTERN_DIGITS)},
}


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics."""
    return TASHKEEL.sub("", text)


def normalize_digits(text: str) -> str:
    """Arabic-Indic (٠١٢) and extended (۰۱۲) digits -> Western (012)."""
    return text.translate(_DIGIT_MAP)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evaluation_normalize(text: str) -> str:
    """Conservative normalization for scoring.

    Applied to BOTH the reference and the candidate before comparison.
    Keeps hamza forms and ta marbuta — those are real orthographic
    distinctions in a legal answer.
    """
    text = unicodedata.normalize("NFKC", text)
    text = strip_tashkeel(text)
    text = text.replace(TATWEEL, "")
    text = normalize_digits(text)
    return collapse_whitespace(text)


# Leading clitics, stripped ONCE with the longest match first.
#
# Arabic attaches conjunctions (و/ف), prepositions (ب/ك/ل) and the definite
# article (ال) to the front of a word, so "البيانات" / "للبيانات" /
# "وبالبيانات" are three different strings to any whitespace tokenizer. Left
# alone they silently cost BM25 recall and make the overlap metric under-read.
#
# Note "لل": the alef of ال elides after ل ("لـ" + "البيانات" -> "للبيانات"),
# so it needs its own entry — stripping a bare "ل" would leave "لبيانات".
#
# Applied once, never iteratively: repeated stripping eats real stems
# ("البيانات" -> "بيانات" -> "يانات"). The lookahead keeps at least three
# characters so "لبن" and "بيت" survive whole.
_CLITIC = re.compile(
    r"^(?:وبال|وكال|فبال|فكال|ولل|فلل|وال|فال|بال|كال|لل|ال|و|ف|ب|ك|ل)(?=.{3,})"
)


def strip_clitics(token: str) -> str:
    return _CLITIC.sub("", token, count=1)


def search_normalize(text: str) -> str:
    """Aggressive normalization for indexing and querying.

    On top of :func:`evaluation_normalize` it folds the orthographic variants
    that Egyptian typists mix freely, then strips leading clitics.
    """
    text = evaluation_normalize(text)
    text = re.sub(f"[{ALEF_FORMS}]", ALEF, text)
    text = text.replace("ى", "ي")  # ى -> ي
    text = text.replace("ة", "ه")  # ة -> ه
    text = text.replace("ؤ", "و")  # ؤ -> و
    text = text.replace("ئ", "ي")  # ئ -> ي
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(strip_clitics(tok) for tok in collapse_whitespace(text).split())
