"""Citation checks with no model in them: `audit` (bench) and `gate` (app).

A fluent answer citing an article the law does not have reads exactly like a
correct one, so every citation is checked against the corpus and against the
sources actually retrieved. Three failures are counted apart:

``fabricated``  a citation to an article or source that does not exist.
``ungrounded``  a real article that was not in the retrieved context: the model
                answered from memory, not from retrieval.
``uncited``     a claim with no citation at all.

The prompt asks for ``[مادة N]``; looser forms («المادة (٧)», «المواد ٣٦، ٣٧»)
are extracted too, so the gap between them measures instruction-following.
"""

from __future__ import annotations

import re
import unicodedata

from .normalize import TASHKEEL, TATWEEL, evaluation_normalize, normalize_digits

# Bounded, not "*": unbounded runs made matching quadratic in a run of spaces
# ("مادة" + 800 spaces took 2.4s). A test pins this value.
_WS = r"[ \t]{0,20}"

# The form the prompt requires: [مادة 7] or [المادة ٧]
STRICT_CITATION = re.compile(
    r"\[" + _WS + r"(?:ال)?مادة" + _WS + r"\(?" + _WS
    + r"([0-9٠-٩۰-۹]{1,3})" + _WS + r"\)?" + _WS + r"\]"
)

# Anything a reader would take as a reference to an article, bracketed or not,
# singular, dual or plural: مادة ٧ · المادة (٧) · المادتين ٧ و٨ · المواد ٣٦، ٣٧
_SINGULAR_HEAD = r"(?:ال)?مادة"
_DUAL_OR_PLURAL_HEAD = r"(?:ال)?(?:مادتين|مادتي|مواد|مادتان)"
LOOSE_HEAD = re.compile(f"{_SINGULAR_HEAD}|{_DUAL_OR_PLURAL_HEAD}")

_NUMBER_GROUP = (
    r"((?:[\(\[]?" + _WS + r"[0-9٠-٩۰-۹]{1,3}" + _WS
    + r"[\)\]]?" + _WS + r"[،,و]?" + _WS + r"){1,8})"
)

LOOSE_CITATION = re.compile(
    r"(?:"
    # A colon («المادة: 30») counts only after the singular head; after a plural
    # one («عدد المواد: 12») it introduces a count, not a citation.
    + _SINGULAR_HEAD + _WS + r":?" + _WS + r"(?:رقم" + _WS + r":?" + _WS + r")?"
    r"|" + _DUAL_OR_PLURAL_HEAD + _WS + r"(?:رقم)?" + _WS +
    r")" + _NUMBER_GROUP
)
NUMBER = re.compile(r"[0-9٠-٩۰-۹]{1,3}")

# A sentence ends at ؟ . ! ؛ or a newline. Arabic uses both ، and , as commas,
# neither of which ends a sentence.
SENTENCE_END = re.compile(r"[.!؟؛\n]+")

# The fixed abstention string the prompt requires; matched exactly, not fuzzily.
ABSTAIN_MARKER = "لا أستطيع الإجابة من المواد المتاحة"

# Sentences shorter than this are connectives ("وبالتالي :"), not claims.
MIN_CLAIM_CHARS = 25


def _numbers(blob: str) -> list[int]:
    return [int(normalize_digits(n)) for n in NUMBER.findall(blob)]


def strict_citations(text: str) -> list[int]:
    return [int(normalize_digits(m.group(1))) for m in STRICT_CITATION.finditer(text)]


def loose_citations(text: str) -> list[int]:
    out: list[int] = []
    for m in LOOSE_CITATION.finditer(text):
        out.extend(_numbers(m.group(1)))
    return out


def citations(text: str) -> list[int]:
    """Every article number the answer refers to, in either form."""
    seen: list[int] = []
    for n in strict_citations(text) + loose_citations(text):
        if n not in seen:
            seen.append(n)
    return seen


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def is_abstention(text: str) -> bool:
    return ABSTAIN_MARKER in text


# A run of this many characters shared with a retrieved article is copying,
# not writing. Short enough to catch a copied sentence, long enough that an
# ordinary legal phrase ("من هذا القانون") does not trip it.
COPIED_RUN = 60


def copied_from_context(sentence: str, context: list[str]) -> bool:
    """Is this sentence lifted verbatim out of one of the retrieved articles?

    Statutes cite themselves («استثناء من حكم المادة (14) من هذا القانون»), so a
    citation inside copied text is the law citing itself, not the model grounding
    an answer. The strict `[مادة N]` form cannot be produced by copying.
    """
    s = " ".join(sentence.split())
    if len(s) < COPIED_RUN:
        return False
    joined = [" ".join(c.split()) for c in context]
    step = max(1, COPIED_RUN // 2)
    for i in range(0, len(s) - COPIED_RUN + 1, step):
        window = s[i:i + COPIED_RUN]
        if any(window in c for c in joined):
            return True
    return False


def audit(
    text: str,
    corpus_numbers: set[int],
    retrieved_numbers: set[int],
    context: list[str] | None = None,
) -> dict:
    """Check one answer against the corpus and the context it was given.

    ``corpus_numbers`` / ``retrieved_numbers`` are law-article numbers.
    ``context`` is the verbatim text of the retrieved articles; given it, a
    citation sitting inside a sentence copied out of that text is reported in
    ``copied`` and excluded from ``cited``, because the law citing itself is
    not the model grounding an answer. See ``copied_from_context``.

    An abstention is not audited for citations — refusing to answer is the
    behaviour being asked for, not a failure to cite.

    Unlike `gate`, `text` is not normalised first: that would merge sentences a
    newline separates. Known gap, kept so Runs 3 and 4 still reproduce: a copied
    sentence with one stray diacritic escapes `copied_from_context`. Normalise
    per sentence, after the split, before any new text-contract run.
    """
    if is_abstention(text):
        return {
            "abstained": True, "fabricated": [], "ungrounded": [], "copied": [],
            "uncited": [], "cited": [], "strict": [], "grounded": True,
        }

    ctx = context or []
    own: list[int] = []
    copied: list[int] = []
    for s in sentences(text):
        target = copied if copied_from_context(s, ctx) else own
        for n in citations(s):
            if n not in target:
                target.append(n)

    cited = [n for n in own]
    fabricated = [n for n in cited if n not in corpus_numbers]
    ungrounded = [n for n in cited if n in corpus_numbers and n not in retrieved_numbers]

    uncited = [
        s for s in sentences(text)
        if len(s) >= MIN_CLAIM_CHARS
        and (not citations(s) or copied_from_context(s, ctx))
    ]

    return {
        "abstained": False,
        "cited": cited,
        "copied": [n for n in copied if n not in cited],
        "strict": strict_citations(text),
        "fabricated": fabricated,
        "ungrounded": ungrounded,
        "uncited": uncited,
        "grounded": not fabricated and not ungrounded and not uncited,
    }


# Invisible: category Cf or Default_Ignorable_Code_Point. None is whitespace or
# touched by evaluation_normalize, so one can hide between «مادة» and its number.
def _is_invisible(ch: str) -> bool:
    return unicodedata.category(ch) == "Cf" or _is_default_ignorable(ord(ch))


# Default_Ignorable_Code_Point ranges (DerivedCoreProperties.txt); unicodedata has no lookup.
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD),    # soft hyphen
    (0x034F, 0x034F),    # combining grapheme joiner (CGJ)
    (0x061C, 0x061C),    # Arabic letter mark (ALM) - also Cf
    (0x115F, 0x1160),    # Hangul choseong/jungseong filler
    (0x17B4, 0x17B5),    # Khmer inherent vowels AQ/AA
    (0x180B, 0x180F),    # Mongolian free variation selectors + MVS
    (0x200B, 0x200F),    # ZWSP, ZWNJ, ZWJ, LRM, RLM - also Cf
    (0x202A, 0x202E),    # bidi embeddings/overrides - also Cf
    (0x2060, 0x206F),    # word joiner and other deprecated format chars
    (0x3164, 0x3164),    # Hangul filler
    (0xFE00, 0xFE0F),    # variation selectors 1-16
    (0xFEFF, 0xFEFF),    # byte order mark - also Cf
    (0xFFA0, 0xFFA0),    # halfwidth Hangul filler
    (0xFFF0, 0xFFF8),    # reserved/unassigned, default ignorable
    (0x1BCA0, 0x1BCA3),  # shorthand format controls
    (0x1D173, 0x1D17A),  # musical symbol format controls
    (0xE0000, 0xE0FFF),  # tags, and variation selectors supplement
)

# Flattened once: set membership is ~6x faster than scanning the ranges per character.
_DEFAULT_IGNORABLE_SET = frozenset(
    cp for lo, hi in _DEFAULT_IGNORABLE_RANGES for cp in range(lo, hi + 1)
)


def _is_default_ignorable(cp: int) -> bool:
    return cp in _DEFAULT_IGNORABLE_SET


def _strip_invisible_chars(text: str) -> str:
    return "".join(ch for ch in text if not _is_invisible(ch))


_DIGITS = "0123456789\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"

# Everything removed before comparison: tashkeel, tatweel and invisible characters.
def _removed_before_comparison(ch: str) -> bool:
    return bool(TASHKEEL.match(ch)) or ch == TATWEEL or _is_invisible(ch)


# Bidi embedding, override and isolate controls (see `_has_directional_override`).
_EMBEDDING_OVERRIDE_ISOLATE = "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"


def _is_bidi_hazard(ch: str) -> bool:
    """A right-to-left or Arabic-letter character: between two digits it can
    change which one reads first (UAX#9)."""
    return unicodedata.bidirectional(ch) in ("R", "AL")


def _has_directional_override(text: str) -> bool:
    """True if `text` holds a bidi embedding, override or isolate control anywhere.

    These act on a whole range, so «مادة \u202e12\u202c» displays "21" with
    nothing unusual beside the digits. A legal claim has no use for them.
    """
    return any(ch in _EMBEDDING_OVERRIDE_ISOLATE for ch in text)


def _has_unsafe_gap_between_digits(text: str) -> bool:
    """True if two digits are separated only by characters removed before
    comparison, at least one of them a bidi hazard.

    RLM between "1" and "2" strips to "12" but displays as "21". Such a claim
    is dropped rather than guessed at. Must run before evaluation_normalize,
    which removes tatweel and small waw/yeh (both bidi AL) itself.
    """
    seen_digit = False
    gap_has_hazard = False
    for ch in text:
        if ch in _DIGITS:
            if seen_digit and gap_has_hazard:
                return True
            seen_digit = True
            gap_has_hazard = False
        elif _removed_before_comparison(ch):
            if seen_digit and _is_bidi_hazard(ch):
                gap_has_hazard = True
        else:
            seen_digit = False
            gap_has_hazard = False
    return False


def _gate_one_claim(
    claim: dict, source_numbers: list[int | None], source_texts: list[str],
    stripped_source_texts: list[str], k: int,
) -> tuple[dict | None, dict | None]:
    """One claim checked against `gate`'s rules: `(kept, None)` or `(None, dropped)`."""
    text = claim.get("text", "")
    sources = claim.get("sources") or []

    if not sources:
        return None, {"text": text, "sources": sources, "reason": "uncited"}

    if any(s < 1 or s > k for s in sources):
        return None, {"text": text, "sources": sources, "reason": "fabricated"}

    own_numbers = {source_numbers[s - 1] for s in sources if source_numbers[s - 1] is not None}

    # Sources arrive normalised but may still hold invisible characters; the caller strips them once.
    own_texts_for_comparison = [stripped_source_texts[s - 1] for s in sources]

    # A number is allowed only if a source this claim cites is that article or names it in its text.
    numbers_in_own_sources = {n for t in own_texts_for_comparison for n in citations(t)}
    allowed = own_numbers | numbers_in_own_sources

    # On the raw text: normalising first would erase the hazards these look for.
    if _has_unsafe_gap_between_digits(text) or _has_directional_override(text):
        return None, {"text": text, "sources": sources, "reason": "ungrounded"}

    # Normalised like the sources, so a diacritic or invisible character cannot hide
    # a citation from the check below. `text` itself stays raw for display.
    normalised_text = _strip_invisible_chars(evaluation_normalize(text))

    # Reported, but no longer decides what is allowed (see `gate`).
    copied = copied_from_context(normalised_text, own_texts_for_comparison)

    mentioned = citations(normalised_text)
    if any(n not in allowed for n in mentioned):
        return None, {"text": text, "sources": sources, "reason": "ungrounded"}

    return {"text": text, "sources": sources, "copied": copied}, None


def gate(
    parsed: dict | None,
    source_numbers: list[int | None],
    source_texts: list[str],
) -> dict:
    """Decide which claims of a claims-JSON answer are shown, trusting only what was retrieved.

    `source_numbers[i]` / `source_texts[i]` describe source ``i + 1`` as the model
    saw it (`claims.format_sources`); `source_numbers[i]` is None for a source with
    no article number. Rules, in order, so no claim is dropped twice:

    1. no `sources` -> ``uncited``.
    2. a source outside ``1..k`` -> ``fabricated``.
    3. a bidi hazard in the text, or an article named ("مادة N", any form) that
       none of the claim's own cited sources is or names -> ``ungrounded``.

    A kept claim is ``{"text", "sources", "copied": bool}``; `copied` is reported,
    never scored (EVAL.md Run 5 explains why a verbatim quote can be grounded).

    `status` is ``"abstained"`` when nothing is kept (including `parsed is None`),
    ``"partial"`` when some claims were dropped, else ``"answered"``. A gate that
    drops everything also shows zero fabrications, so read coverage beside it.

    Raises ValueError when `source_numbers` and `source_texts` differ in length.
    """
    if len(source_numbers) != len(source_texts):
        raise ValueError(
            f"source_numbers has {len(source_numbers)} entries but "
            f"source_texts has {len(source_texts)} — they must describe the "
            "same sources, one-to-one."
        )

    if parsed is None:
        return {
            "status": "abstained", "kept": [], "dropped": [],
            "uncited": 0, "fabricated": 0, "ungrounded": 0, "ignored_on_abstain": 0,
        }

    claims = parsed.get("claims") or []

    if parsed.get("abstain"):
        # Claims sent with an abstention are counted, not checked.
        return {
            "status": "abstained", "kept": [], "dropped": [],
            "uncited": 0, "fabricated": 0, "ungrounded": 0,
            "ignored_on_abstain": len(claims),
        }

    k = len(source_texts)
    # Stripped once, shared by every claim in this answer.
    stripped_source_texts = [_strip_invisible_chars(t) for t in source_texts]
    kept: list[dict] = []
    dropped: list[dict] = []
    for claim in claims:
        keep, drop = _gate_one_claim(claim, source_numbers, source_texts, stripped_source_texts, k)
        if keep is not None:
            kept.append(keep)
        else:
            dropped.append(drop)

    if not kept:
        status = "abstained"
    elif dropped:
        status = "partial"
    else:
        status = "answered"

    return {
        "status": status,
        "kept": kept,
        "dropped": dropped,
        "uncited": sum(1 for d in dropped if d["reason"] == "uncited"),
        "fabricated": sum(1 for d in dropped if d["reason"] == "fabricated"),
        "ungrounded": sum(1 for d in dropped if d["reason"] == "ungrounded"),
        "ignored_on_abstain": 0,
    }
