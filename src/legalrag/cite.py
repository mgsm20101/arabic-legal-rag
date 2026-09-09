"""The citation contract — PRD M2/B1, and the abstention signal for M2/B2.

PRD §1 opens on the failure this exists to stop: *"النموذج بيولّد إجابة سليمة
الصياغة حتى لو المادة اللي استند لها غلط أو مش موجودة أصلاً، والمستخدم مش قادر
يكشف ده من شكل الإجابة."* A fluent Arabic paragraph citing «المادة (٧٨)» of a
law that ends at 49 reads exactly like a correct one. No amount of prompting
makes that detectable by eye, so it is not left to the eye.

Nothing here involves a model. The check runs on generated text against the
corpus and the retrieved set, so it holds whatever produced the answer — a 3B
model on a laptop, a frontier API, or a person.

**Three failures, deliberately counted apart, because they mean different
things:**

``fabricated``  — a citation to an article number the corpus does not contain.
                  The model invented a law.
``ungrounded``  — a citation to a real article that was NOT in the retrieved
                  context. The model answered from what it memorised, and the
                  retrieval pipeline had nothing to do with it. Reads as a
                  success on any metric that only checks the number exists.
``uncited``     — a sentence making a claim with no citation at all.

**Format compliance is measured, not assumed.** The prompt asks for a fixed
form, ``[مادة N]``. Arabic legal prose has many others — «المادة (٧)»،
«المادتين ٧ و٨»، «المواد ٣٦، ٣٧». Both are extracted: the strict form is what
was asked for, the loose form is what a reader would call a citation. The gap
between them is the model's instruction-following, and it is a number rather
than an impression.
"""

from __future__ import annotations

import re

from .normalize import normalize_digits

# The form the prompt requires: [مادة 7] or [المادة ٧]
STRICT_CITATION = re.compile(r"\[[ \t]*(?:ال)?مادة[ \t]*\(?[ \t]*([0-9٠-٩۰-۹]{1,3})[ \t]*\)?[ \t]*\]")

# Anything a reader would take as a reference to an article, bracketed or not,
# singular, dual or plural: مادة ٧ · المادة (٧) · المادتين ٧ و٨ · المواد ٣٦، ٣٧
LOOSE_HEAD = re.compile(r"(?:ال)?(?:مادة|مادتين|مادتي|مواد|مادتان)")
LOOSE_CITATION = re.compile(
    LOOSE_HEAD.pattern + r"[ \t]*(?:رقم)?[ \t]*"
    r"((?:[\(\[]?[ \t]*[0-9٠-٩۰-۹]{1,3}[ \t]*[\)\]]?[ \t]*[،,و]?[ \t]*){1,8})"
)
NUMBER = re.compile(r"[0-9٠-٩۰-۹]{1,3}")

# A sentence ends at ؟ . ! ؛ or a newline. Arabic uses both ، and , as commas,
# neither of which ends a sentence.
SENTENCE_END = re.compile(r"[.!؟؛\n]+")

# The exact string the prompt requires when the retrieved articles do not
# answer the question. A fixed marker, not a judgement call about phrasing:
# "I could not find" and "the law does not say" are different claims, and
# scoring abstention on a fuzzy match would measure the matcher.
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


def audit(text: str, corpus_numbers: set[int], retrieved_numbers: set[int]) -> dict:
    """Check one answer against the corpus and the context it was given.

    ``corpus_numbers`` / ``retrieved_numbers`` are law-article numbers. An
    abstention is not audited for citations — refusing to answer is the
    behaviour being asked for, not a failure to cite.
    """
    if is_abstention(text):
        return {
            "abstained": True, "fabricated": [], "ungrounded": [],
            "uncited": [], "cited": [], "strict": [], "grounded": True,
        }

    cited = citations(text)
    fabricated = [n for n in cited if n not in corpus_numbers]
    ungrounded = [n for n in cited if n in corpus_numbers and n not in retrieved_numbers]

    uncited = [
        s for s in sentences(text)
        if len(s) >= MIN_CLAIM_CHARS and not citations(s)
    ]

    return {
        "abstained": False,
        "cited": cited,
        "strict": strict_citations(text),
        "fabricated": fabricated,
        "ungrounded": ungrounded,
        "uncited": uncited,
        "grounded": not fabricated and not ungrounded and not uncited,
    }
