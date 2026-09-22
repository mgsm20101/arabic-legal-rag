"""Arabic text normalization.

Three DISTINCT normalizers — do not merge them:

* ``search_normalize``       — aggressive. Used only when building the lexical
                               index and when normalizing a user query for it.
                               Loses information on purpose so that "الأسماء"
                               matches "الاسماء".
* ``orthographic_normalize`` — middle. Folds only the letter forms an Arabic
                               typist mixes freely; keeps words, clitics and
                               punctuation intact, so the result still reads as
                               natural Arabic. Used on both sides of the dense
                               path (ADR-027).
* ``evaluation_normalize``   — conservative. Used when comparing a system answer
                               against ground truth. Must NOT collapse
                               distinctions that a human reviewer would call a
                               real difference.

Rationale (see DECISIONS.md ADR-004): using one normalizer for both ends
inflates scores — an aggressive normalizer makes wrong answers look right at
eval time.

The middle one exists because the two ends left a gap (ADR-027). Dense retrieval
was handed the query verbatim to keep it away from the aggressive form, and the
cost was measured: typing "الافراد" for "الأفراد" — no hamza key, the way most
people type — moved the ranks of 10 of 15 dev questions. `search_normalize` was
the wrong tool for that (it strips clitics and shreds punctuation, which a
transformer reads as damage); nothing else in between existed.
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


def orthographic_normalize(text: str) -> str:
    """Fold the letter forms a typist mixes, and nothing else (ADR-027).

    On top of :func:`evaluation_normalize`, the substitutions below collapse the
    spellings that vary by keyboard habit rather than by meaning: hamza seats,
    ta marbuta, alef maqsura. Word boundaries, clitics and punctuation are left
    exactly as they were, so the output still reads as Arabic — which is the
    whole difference from :func:`search_normalize`, and the reason this one is
    safe to hand a transformer.

    Applied to BOTH the query and the passage or it does nothing: the point is
    that the two sides land on the same spelling, not that either side is
    "correct".

    It is lossy where Arabic really does distinguish these forms — «إذن» and
    «أذن» become one string, as do «على» and «علي». That cost is accepted and
    recorded in ADR-027; the measured alternative was worse.
    """
    text = evaluation_normalize(text)
    text = re.sub(f"[{ALEF_FORMS}]", ALEF, text)
    text = text.replace("ى", "ي")  # ى -> ي
    text = text.replace("ة", "ه")  # ة -> ه
    text = text.replace("ؤ", "و")  # ؤ -> و
    text = text.replace("ئ", "ي")  # ئ -> ي
    return text


def search_normalize(text: str) -> str:
    """Aggressive normalization for the lexical index and its queries.

    :func:`orthographic_normalize`, then punctuation to spaces and leading
    clitics stripped — the two steps that make the text a bag of match keys
    rather than a sentence. Defined on top of the middle normalizer so the two
    can never drift apart on the folds they share.
    """
    text = orthographic_normalize(text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(strip_clitics(tok) for tok in collapse_whitespace(text).split())
