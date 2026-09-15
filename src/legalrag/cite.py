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

from .normalize import evaluation_normalize, normalize_digits

# The form the prompt requires: [مادة 7] or [المادة ٧]
STRICT_CITATION = re.compile(r"\[[ \t]*(?:ال)?مادة[ \t]*\(?[ \t]*([0-9٠-٩۰-۹]{1,3})[ \t]*\)?[ \t]*\]")

# Anything a reader would take as a reference to an article, bracketed or not,
# singular, dual or plural: مادة ٧ · المادة (٧) · المادتين ٧ و٨ · المواد ٣٦، ٣٧
_SINGULAR_HEAD = r"(?:ال)?مادة"
_DUAL_OR_PLURAL_HEAD = r"(?:ال)?(?:مادتين|مادتي|مواد|مادتان)"
LOOSE_HEAD = re.compile(f"{_SINGULAR_HEAD}|{_DUAL_OR_PLURAL_HEAD}")

_NUMBER_GROUP = r"((?:[\(\[]?[ \t]*[0-9٠-٩۰-۹]{1,3}[ \t]*[\)\]]?[ \t]*[،,و]?[ \t]*){1,8})"

LOOSE_CITATION = re.compile(
    r"(?:"
    # A colon reads the same as a space to anyone fluent — «المادة: 30»،
    # «المادة :30» and «المادة رقم: 30» are unambiguously citations — but
    # ONLY after the SINGULAR head word. After a dual or plural one
    # ("المواد: 12", "عدد المواد رقم: 12") a colon is far more likely to
    # introduce a count or a list than to cite that number as an article;
    # unnarrowed, this once misread «عدد المواد: 12» ("number of subjects:
    # 12") as citing article 12.
    + _SINGULAR_HEAD + r"[ \t]*(?:رقم[ \t]*)?:?[ \t]*"
    r"|" + _DUAL_OR_PLURAL_HEAD + r"[ \t]*(?:رقم)?[ \t]*"
    r")" + _NUMBER_GROUP
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


# A run of this many characters shared with a retrieved article is copying,
# not writing. Short enough to catch a copied sentence, long enough that an
# ordinary legal phrase ("من هذا القانون") does not trip it.
COPIED_RUN = 60


def copied_from_context(sentence: str, context: list[str]) -> bool:
    """Is this sentence lifted verbatim out of one of the retrieved articles?

    This exists because of what the first real run produced. Asked a question,
    the model frequently reproduced the retrieved article instead of answering
    — and Egyptian statutes cite themselves: «استثناء من حكم المادة (14) من
    هذا القانون». The extractor read that as the model citing article 14. It
    was the law citing itself, inside text the model had copied.

    So a citation lifted from context is not evidence the model grounded
    anything, and counting it inflates exactly the number a reader most wants
    to trust. The strict `[مادة N]` form cannot be produced this way, which is
    what makes it the trustworthy signal.
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

    **Deliberately NOT given `gate`'s normalisation.** `context` reaches
    this function just as `evaluation_normalize`d as `gate`'s
    `source_texts` — both come from the same ingested corpus — so the same
    raw-candidate-vs-normalised-context asymmetry exists here in principle.
    Left alone for Runs 3/4 anyway: this function splits `text` into
    `sentences` on raw `.`/`!`/`؟`/`؛`/newline boundaries *before* checking
    any of them, and `evaluation_normalize`'s whitespace collapse turns a
    newline into a space. Normalising the whole answer up front, the way
    `gate` normalises a whole claim, would silently merge sentences a raw
    newline used to separate — a different and worse failure than the one
    being fixed here. Doing this safely would mean normalising per sentence
    *after* the split, touching every call site below rather than one, on
    the function two already-adopted, pre-registered runs are scored by.
    That is a follow-up in its own right, not a rider on this fix. Note
    that `LOOSE_CITATION`'s colon extension is a shared regex constant —
    unlike the normalisation question, it applies here regardless.

    Confirmed real, still not fixed here: a copied sentence carrying one
    stray diacritic escapes `copied_from_context` the same way it once did
    in `gate`, which can flip `grounded`. Confirmed separately: normalising
    per sentence, after the split, leaves Run 3 and Run 4 byte-identical.
    Before any new text-contract run: normalise per sentence, after
    splitting.
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


# RLM, LRM, ZWNJ, ZWJ and ALM (Unicode "format" characters, category Cf)
# have no visible glyph and are not whitespace by Python's `\s` either, so
# one sitting between a head word and its number is invisible on screen and
# untouched by `evaluation_normalize`'s NFKC/tashkeel/tatweel/whitespace
# steps. `evaluation_normalize` is not the place to fix that: it is the
# benchmark's scoring normaliser, shared with every comparison of a
# candidate answer to a reference, and is left alone here on purpose.
# Stripped instead in the gate's own comparison step below, on both sides.
_INVISIBLE_FORMAT_CHARS = str.maketrans("", "", "\u200e\u200f\u200c\u200d\u061c")


def _strip_invisible_format_chars(text: str) -> str:
    return text.translate(_INVISIBLE_FORMAT_CHARS)


def _gate_one_claim(
    claim: dict, source_numbers: list[int | None], source_texts: list[str], k: int,
) -> tuple[dict | None, dict | None]:
    """One claim from the claims contract, checked against the rules
    `gate` documents — returns `(kept, None)` or `(None, dropped)`, never
    both, so a caller can never double-count a claim."""
    text = claim.get("text", "")
    sources = claim.get("sources") or []

    if not sources:
        return None, {"text": text, "sources": sources, "reason": "uncited"}

    if any(s < 1 or s > k for s in sources):
        return None, {"text": text, "sources": sources, "reason": "fabricated"}

    own_texts = [source_texts[s - 1] for s in sources]
    own_numbers = {source_numbers[s - 1] for s in sources if source_numbers[s - 1] is not None}

    # `own_texts` reaches us already `evaluation_normalize`d — the corpus is
    # normalised at ingest time, and that is what both the app pipeline and
    # a saved eval row hand to `gate` as `source_texts` — but that leaves
    # RLM/LRM/ZWNJ/ZWJ/ALM in place (see `_strip_invisible_format_chars`),
    # and a source's own self-citation can carry one too. Stripped here,
    # locally, for the comparison below only; `own_texts` itself is never
    # part of what a caller sees.
    own_texts_for_comparison = [_strip_invisible_format_chars(t) for t in own_texts]

    # "القانون بيحيل على نفسه" licenses an article the SOURCE'S OWN TEXT
    # names — not every number a claim happens to mention while some part
    # of it is copied. Scoped to the claim's own cited sources only: a
    # number some OTHER retrieved source names does not license a claim
    # that never cited that source.
    numbers_in_own_sources = {n for t in own_texts_for_comparison for n in citations(t)}
    allowed = own_numbers | numbers_in_own_sources

    # The claim's own text never goes through `evaluation_normalize`
    # upstream; it is the model's raw output. Checking it raw against a
    # normalised world let a diacritic, a tatweel, or a no-break space
    # between "مادة" and its number hide a citation from both checks below:
    # `citations` found nothing, so the "names an article no source
    # supports" check below passed vacuously, and a claim that should have
    # been dropped was kept. Normalising here, the same way the sources
    # already are, closes that gap — and stripping the same invisible
    # format characters as above closes the gap `evaluation_normalize`
    # itself does not cover. `text` itself stays raw below, since
    # normalising is for comparison, not display.
    normalised_text = _strip_invisible_format_chars(evaluation_normalize(text))

    # `copied` is still computed and still reported (`report_claims`'s own
    # rate) — it just no longer decides what is allowed. See `gate`'s
    # docstring for the bug this replaces: a claim built from a 60+
    # character copied run plus one invented sentence used to have EVERY
    # number it mentioned exempted, including one the source never named.
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
    """Run 5's model-free gate (EVAL.md, commit ecd37f8): decide which
    claims of a claims-JSON answer survive to be shown, without trusting the
    model to have followed `sources` correctly — the claims-shaped sibling
    of `audit`, checked the same way: against what was actually retrieved,
    never against the model's word for it.

    `source_numbers[i]` / `source_texts[i]` describe source ``i + 1`` — the
    numbering `claims.format_sources` showed the model, 1-based, in
    retrieval rank order; `k = len(source_texts)`. `source_numbers[i]` is
    `None` for a source with no article number of its own (an issuance
    article). Rules, applied in this order so one claim is never dropped
    for two reasons at once:

    1. no `sources` at all -> dropped, ``uncited``.
    2. any `sources` entry outside ``1..k`` -> dropped, ``fabricated`` (the
       model pointed at a source that was never shown to it).
    3. the claim's text names an article ("مادة N", strict or loose form,
       via `citations`) that is not *allowed* -> dropped, ``ungrounded``. An
       article number N is allowed only if N is the article number of one
       of the claim's own cited sources, OR `citations` finds N in the TEXT
       of one of those same sources (a source that cites itself, e.g.
       "استثناء من حكم المادة (14) من هذا القانون" — Egyptian statutes do
       this). The scope is always the claim's own cited sources: a number
       named only by some OTHER retrieved source, one this claim did not
       cite, is not allowed either.

       The exemption follows this reason, not the wording of an earlier
       version of this rule, which instead exempted a claim outright
       whenever `copied_from_context` found it copied — checked once,
       against the whole claim, rather than per article number. That let a
       claim built from a 60+ character copied run plus one *invented*
       sentence ("...وفقاً للمادة 99") keep an article number no source
       ever named, simply because *some* other part of the same claim was
       copied. The fix cuts both ways: a copied claim that adds an article
       none of its sources name is now ``ungrounded`` (it was wrongly kept
       before), and a *paraphrase* — never copied at all — that correctly
       names an article its own cited source's text names is now kept (it
       was wrongly dropped before, since the old rule's exemption never
       triggered without a literal copy).

    A claim that survives is ``{"text", "sources", "copied": bool}`` —
    `copied` (`copied_from_context` against the claim's own cited sources)
    is still computed and still reported, it just no longer decides what is
    allowed. Kept regardless of its value, per Run 5's recorded meaning
    change: in Run 3/4 a citation sitting inside copied text was not
    counted, because the citation was part of the copied prose itself. Here
    the source is a *separate* field the model fills in alongside the
    quote, so a claim that quotes its source verbatim and correctly names
    that source counts as grounded. The rate is printed on its own in
    `answer_report.report_claims`, never folded into a pass/fail number.

    `status` is ``"abstained"`` when nothing is kept — whether because
    `abstain: true` said so, every claim was dropped, or `parsed is None`
    (two failed JSON attempts; the row's own `schema_failure` records that
    case) — ``"partial"`` when some were dropped and some kept, and
    ``"answered"`` when none were dropped. **This is the gate trap the
    pre-registration names, and `report_claims` prints beside every one of
    these numbers for exactly this reason:** a gate that drops every claim
    scores "0 fabricated, 0 ungrounded" the same way a model that never
    says anything would. Coverage and dropped-by-reason counts are what
    tell the difference from a genuinely clean run.

    Raises `ValueError` immediately when `source_numbers` and `source_texts`
    disagree in length — they must describe the same sources, position for
    position (`source_numbers[i]` is article number of `source_texts[i]`),
    and a caller that passed mismatched lists would otherwise have every
    source past the shorter list's length silently misaligned instead of
    failing loudly.
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
        # The claims that came with an explicit abstention are not run
        # through the rules above at all — only counted, so a reader can
        # see the model said "no" while still writing claims anyway.
        return {
            "status": "abstained", "kept": [], "dropped": [],
            "uncited": 0, "fabricated": 0, "ungrounded": 0,
            "ignored_on_abstain": len(claims),
        }

    k = len(source_texts)
    kept: list[dict] = []
    dropped: list[dict] = []
    for claim in claims:
        keep, drop = _gate_one_claim(claim, source_numbers, source_texts, k)
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
