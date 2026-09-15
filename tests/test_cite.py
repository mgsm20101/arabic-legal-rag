"""Citation-contract tests — PRD M2/B1.

The whole point of this module is that its failure mode is invisible in the
output: an answer citing «المادة (٧٨)» of a 49-article law is as fluent as a
correct one. These tests pin the distinctions that make it visible.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.cite import (  # noqa: E402
    ABSTAIN_MARKER,
    audit,
    citations,
    gate,
    is_abstention,
    loose_citations,
    sentences,
    strict_citations,
)
from legalrag.normalize import evaluation_normalize  # noqa: E402

CORPUS = set(range(1, 50))       # the law has 49 articles
RETRIEVED = {4, 7, 12}


def test_the_required_bracket_form_is_recognised():
    assert strict_citations("يلتزم المتحكم بالإبلاغ [مادة 7] خلال المهلة") == [7]


def test_arabic_indic_digits_in_a_citation_are_the_same_number():
    assert strict_citations("[المادة ٧]") == [7]


def test_prose_citation_forms_are_recognised_too():
    """Arabic legal writing cites in several shapes. A checker that only knows
    the one the prompt asked for would score a correct answer as uncited."""
    assert loose_citations("تنص المادة (٧) على ذلك") == [7]
    assert loose_citations("وفقا للمادتين ٧ و٨") == [7, 8]
    assert loose_citations("في المواد ٣٦، ٣٧، ٣٨") == [36, 37, 38]


def test_format_compliance_is_measurable_as_the_gap_between_the_two():
    text = "الإبلاغ خلال 72 ساعة [مادة 7] كما تنص المادة (12) على استثناء"
    assert strict_citations(text) == [7]
    assert set(citations(text)) == {7, 12}


def test_an_invented_article_is_flagged_as_fabricated():
    """The law ends at 49. A fluent sentence citing 78 reads exactly like a
    correct one — this is the failure PRD section 1 opens with."""
    r = audit("تنص [مادة 78] على غرامة مالية كبيرة على المخالف.", CORPUS, RETRIEVED)
    assert r["fabricated"] == [78]
    assert r["grounded"] is False


def test_a_real_article_that_was_not_retrieved_is_flagged_separately():
    """Article 30 exists, so any check that only asks 'does this number exist'
    passes it. But it was not in the context, so the model did not read it —
    it recalled it. Counted apart because it means something different."""
    r = audit("يلتزم المعالج بذلك [مادة 30] في جميع الأحوال دون استثناء.", CORPUS, RETRIEVED)
    assert r["fabricated"] == []
    assert r["ungrounded"] == [30]
    assert r["grounded"] is False


def test_a_claim_with_no_citation_at_all_is_flagged():
    r = audit("يجب على المتحكم إبلاغ المركز فورا بأي خرق للبيانات الشخصية.", CORPUS, RETRIEVED)
    assert len(r["uncited"]) == 1
    assert r["grounded"] is False


def test_a_short_connective_is_not_treated_as_an_uncited_claim():
    r = audit("وبالتالي: يلتزم المتحكم بالإبلاغ خلال المهلة المقررة [مادة 7].",
              CORPUS, RETRIEVED)
    assert r["uncited"] == []
    assert r["grounded"] is True


def test_a_fully_grounded_answer_passes():
    text = ("يلتزم المتحكم بإبلاغ المركز خلال المهلة المقررة قانونا [مادة 7].\n"
            "ويسري ذلك مع مراعاة أحكام الاستثناء الوارد في النص [مادة 12].")
    r = audit(text, CORPUS, RETRIEVED)
    assert r["grounded"] is True
    assert set(r["cited"]) == {7, 12}


def test_abstention_is_a_fixed_marker_not_a_phrasing_judgement():
    """«لم أجد» and «القانون لا ينص» are different claims. Scoring abstention
    on a fuzzy match would measure the matcher, not the model."""
    assert is_abstention(f"{ABSTAIN_MARKER}.")
    assert not is_abstention("لم أجد إجابة واضحة في النص المتاح أمامي الآن.")


def test_an_abstention_is_not_audited_for_citations():
    """Refusing to answer is the behaviour being asked for on out_of_corpus,
    not a failure to cite."""
    r = audit(ABSTAIN_MARKER, CORPUS, RETRIEVED)
    assert r["abstained"] is True
    assert r["grounded"] is True
    assert r["uncited"] == []


def test_sentences_split_on_arabic_terminators_but_not_on_commas():
    assert sentences("الأول، والثاني؟ الثالث. الرابع") == [
        "الأول، والثاني", "الثالث", "الرابع"]


# --- copied context (found by the first real run) -----------------------------

ARTICLE_14 = (
    "استثناء من حكم المادة (14) من هذا القانون ، يجوز في حالة الموافقة الصريحة "
    "للشخص المعني بالبيانات أو من ينوب عنه ، نقل أو مشاركة أو تداول أو معالجة "
    "البيانات الشخصية إلى دولة لا يتوفر فيها مستوى الحماية المشار إليها"
)


def test_a_citation_inside_copied_statute_text_is_not_the_model_citing():
    """Asked a question, the model reproduced the retrieved article. Egyptian
    statutes cite themselves, so the copied text contained «المادة (14)» — and
    the extractor read it as grounding. It was the law citing itself."""
    r = audit(ARTICLE_14, CORPUS, {14}, context=[ARTICLE_14])
    assert r["cited"] == []
    assert r["copied"] == [14]
    assert r["grounded"] is False


def test_copied_text_still_counts_as_an_uncited_claim():
    """Reproducing the statute is not answering, and it is certainly not
    citing. It must not buy its way out of the uncited count."""
    r = audit(ARTICLE_14, CORPUS, {14}, context=[ARTICLE_14])
    assert r["uncited"]


def test_the_models_own_citation_survives_alongside_copied_text():
    text = ARTICLE_14 + "\nوبناء عليه يلزم الحصول على موافقة صريحة قبل النقل [مادة 7]."
    r = audit(text, CORPUS, {7, 14}, context=[ARTICLE_14])
    assert r["cited"] == [7]
    assert r["copied"] == [14]


def test_without_context_the_audit_behaves_as_before():
    """`context` is optional; omitting it must not silently change a verdict."""
    r = audit("يلزم الإبلاغ خلال المهلة المقررة قانونا [مادة 7].", CORPUS, {7})
    assert r["cited"] == [7]
    assert r["grounded"] is True


def test_a_short_shared_phrase_is_not_copying():
    """«من هذا القانون» appears everywhere. Only a long shared run is copying."""
    from legalrag.cite import copied_from_context
    assert not copied_from_context("يلزم ذلك من هذا القانون.", [ARTICLE_14])


# --- Run 5's gate (claims JSON, EVAL.md commit ecd37f8) ---------------------
#
# `gate` is the model-free check applied to the claims contract's answer
# shape, the way `audit` is applied to free text — same job (do not trust a
# citation the model did not earn), a different input shape (a claim names
# its sources by position, `[1]..[k]`, instead of writing an article number
# inline). `SOURCE_NUMBERS[i]` / `SOURCE_TEXTS[i]` describe source `i + 1`,
# in retrieval rank order, the numbering `claims.format_sources` shows the
# model.

SOURCE_NUMBERS = [7, 12, None]  # source 3 is an issuance article: no number
SOURCE_TEXTS = [
    "يلتزم المتحكم بالإبلاغ خلال ستة أيام عمل من تاريخ العلم بالخرق.",
    "يجوز للمركز الإعفاء من الإخطار في حالات محددة.",
    "ينشر هذا القرار في الجريدة الرسمية ويعمل به من اليوم التالي لنشره.",
]


def test_a_claim_without_a_source_is_removed_before_display():
    parsed = {"abstain": False, "claims": [
        {"text": "الرد خلال ستة أيام عمل.", "sources": []},
    ]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["kept"] == []
    assert result["dropped"] == [
        {"text": "الرد خلال ستة أيام عمل.", "sources": [], "reason": "uncited"},
    ]
    assert result["uncited"] == 1
    assert result["status"] == "abstained"


def test_a_source_number_outside_the_retrieved_list_counts_as_fabricated():
    """`SOURCE_TEXTS` has 3 entries (k=3) — source 4 was never shown to the
    model at all."""
    parsed = {"abstain": False, "claims": [
        {"text": "الرد خلال ستة أيام عمل.", "sources": [4]},
    ]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["fabricated"] == 1
    assert result["dropped"][0]["reason"] == "fabricated"
    assert result["kept"] == []


def test_source_number_zero_is_fabricated_not_the_last_source():
    """`sources[i] - 1` indexes into `source_texts`/`source_numbers` — a
    caller that forgot the lower bound could let Python's negative-index
    wraparound turn source 0 into the LAST source instead of catching it as
    out of range."""
    parsed = {"abstain": False, "claims": [
        {"text": "الرد خلال ستة أيام عمل.", "sources": [0]},
    ]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["fabricated"] == 1
    assert result["dropped"][0]["reason"] == "fabricated"
    assert result["kept"] == []


def test_a_claim_naming_an_article_it_does_not_cite_is_ungrounded():
    """Article 30 is not among the numbers of the claim's own cited sources
    ({7}) — any check that only asks "does 30 exist somewhere" would miss
    that this citation was never earned."""
    parsed = {"abstain": False, "claims": [
        {"text": "يلزم الحفظ لمدة سنة كاملة [مادة 30].", "sources": [1]},
    ]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["kept"] == []


def test_a_claim_quoting_its_own_source_may_name_the_articles_that_source_names():
    """Egyptian statutes cite themselves — a claim that copies its cited
    source verbatim and repeats the article number the source's own text
    names is not the model inventing a citation, it is the law naming
    itself. Kept because 14 is a number `citations()` finds in the
    source's own text (`numbers_in_own_sources`), NOT merely because the
    claim happens to be a copy — `copied` is recorded here too, but it is
    not what licenses this claim; compare
    `test_a_copied_claim_that_adds_an_article_no_source_names_is_ungrounded`,
    where the claim is copied AND still dropped."""
    self_citing_text = (
        "استثناء من حكم المادة (14) من هذا القانون يجوز نقل البيانات "
        "بموافقة صريحة من صاحبها في جميع الأحوال دون استثناء يذكر."
    )
    parsed = {"abstain": False, "claims": [
        {"text": self_citing_text, "sources": [1]},
    ]}

    result = gate(parsed, [7], [self_citing_text])

    assert result["ungrounded"] == 0
    assert result["kept"] == [{"text": self_citing_text, "sources": [1], "copied": True}]


def test_a_claim_may_name_an_article_that_its_own_source_names():
    """The allowance follows the pre-registration's actual reason — "the
    law cites itself" — not the wording of the earlier rule, which only
    ever exempted a claim that was itself a verbatim copy. A PARAPHRASE
    (never copied — see the `copied: False` below) that correctly names an
    article its own cited source's text names is grounded on the same
    reasoning; it was wrongly dropped before this fix, since the old rule's
    exemption never triggered without a literal copy."""
    source_text = (
        "استثناء من حكم المادة (14) من هذا القانون، يجوز نقل البيانات "
        "بموافقة صريحة."
    )
    paraphrase = "يوجد استثناء بموجب المادة 14 يسمح بنقل البيانات بموافقة صريحة من صاحبها."
    parsed = {"abstain": False, "claims": [
        {"text": paraphrase, "sources": [1]},
    ]}

    result = gate(parsed, [7], [source_text])

    assert result["ungrounded"] == 0
    assert result["kept"] == [{"text": paraphrase, "sources": [1], "copied": False}]


def test_a_copied_claim_that_adds_an_article_no_source_names_is_ungrounded():
    """The bug the fix closes: a claim built from a 60+ character copied
    run PLUS one invented sentence ("وفقاً للمادة 99") used to have EVERY
    article number it mentioned exempted, because `copied_from_context`
    was checked once against the whole claim rather than per number. 99 is
    not the cited source's own article number, and the source's own text
    never names it either — dropped now, whether or not part of the claim
    is copied."""
    source_text = (
        "استثناء من حكم المادة (14) من هذا القانون، يجوز نقل البيانات "
        "بموافقة صريحة."
    )
    claim_text = source_text + " وفقاً للمادة 99."
    parsed = {"abstain": False, "claims": [
        {"text": claim_text, "sources": [1]},
    ]}

    result = gate(parsed, [7], [source_text])

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["kept"] == []


def test_an_article_named_only_by_a_retrieved_source_the_claim_does_not_cite_is_ungrounded():
    """The allowance is scoped to the claim's OWN cited sources — a number
    some OTHER retrieved source's text happens to name does not license a
    claim that never cited that source. Source 2's text below self-cites
    article 30, but this claim cites only source 1."""
    numbers = [7, 12]
    texts = [
        "يلتزم المتحكم بالإبلاغ خلال ستة أيام عمل من تاريخ العلم بالخرق.",
        "استثناء من حكم المادة (30) من هذا القانون، يجوز للمركز الإعفاء من الإخطار.",
    ]
    parsed = {"abstain": False, "claims": [
        {"text": "يلزم الإبلاغ خلال ستة أيام عمل [مادة 30].", "sources": [1]},
    ]}

    result = gate(parsed, numbers, texts)

    assert result["ungrounded"] == 1
    assert result["kept"] == []


def test_a_response_whose_claims_are_all_removed_becomes_an_abstention():
    parsed = {"abstain": False, "claims": [
        {"text": "الرد خلال ستة أيام عمل.", "sources": []},
        {"text": "لا يوجد أي سند لهذا الادعاء إطلاقاً.", "sources": [9]},
    ]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["status"] == "abstained"
    assert result["kept"] == []
    assert len(result["dropped"]) == 2


def test_an_explicit_abstention_is_an_abstention_even_if_claims_came_with_it():
    """`abstain: true` wins outright — the claims that came with it are not
    run through the gate at all, only counted as ignored."""
    parsed = {"abstain": True, "claims": [
        {"text": "الرد خلال ستة أيام عمل [مادة 7].", "sources": [1]},
    ]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["status"] == "abstained"
    assert result["kept"] == []
    assert result["dropped"] == []
    assert result["ignored_on_abstain"] == 1


def test_a_schema_failure_is_an_abstention():
    """Two failed JSON attempts reach the gate as `parsed=None` — there is
    nothing to ignore-count here, unlike an explicit `abstain: true`."""
    result = gate(None, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["status"] == "abstained"
    assert result["kept"] == []
    assert result["dropped"] == []
    assert result["ignored_on_abstain"] == 0


def test_a_partial_answer_keeps_its_supported_claims():
    parsed = {"abstain": False, "claims": [
        {"text": "الرد يكون خلال ستة أيام عمل من تاريخ العلم بالخرق.", "sources": [1]},
        {"text": "لا يوجد أي سند لهذا الادعاء إطلاقاً.", "sources": []},
    ]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["status"] == "partial"
    assert result["kept"] == [
        {"text": "الرد يكون خلال ستة أيام عمل من تاريخ العلم بالخرق.",
         "sources": [1], "copied": False},
    ]
    assert len(result["dropped"]) == 1
    assert result["dropped"][0]["reason"] == "uncited"


# --- Commit 2 (m2): mutation gaps in the gate's own-source scoping ---------


def test_a_claim_naming_its_own_sources_plain_article_number_is_kept_even_when_the_source_text_never_repeats_it():
    """The ordinary case: a claim cites [مادة 7] and its ONLY source IS
    article 7 — article 7's own text is plain prose that never repeats its
    own number, so this claim can only be licensed by the source's article
    NUMBER itself (`own_numbers`), not by anything `citations()` finds
    inside the source's text (`numbers_in_own_sources`, empty here). Mutant:
    `allowed = numbers_in_own_sources` (dropping `own_numbers` from the
    union) would wrongly drop the single most ordinary grounded claim
    there is, and no existing test before this one used a claim that cited
    its own source's bare number without also copying self-citing text."""
    claim_text = "يجب الرد خلال مهلة ستة أيام [مادة 7]."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 0
    assert result["kept"] == [{"text": claim_text, "sources": [1], "copied": False}]


def test_a_claim_naming_a_different_retrieved_sources_own_article_number_without_citing_it_is_ungrounded():
    """`allowed` must be scoped to the claim's OWN cited sources' article
    numbers, never every article number among ALL retrieved sources. This
    claim cites only source 1 (article 7) but its text names article 12 —
    simply another retrieved source's own plain number, not something
    source 1's text says about itself. Mutant: widening `own_numbers` to
    every retrieved source's number (not only the claim's own `sources`)
    would wrongly license this — a case the existing
    `test_an_article_named_only_by_a_retrieved_source_the_claim_does_not_cite_is_ungrounded`
    does not cover, because there the extra number comes from INSIDE
    another source's text, not from that source's own article number."""
    parsed = {"abstain": False, "claims": [
        {"text": "يلزم الإبلاغ خلال ستة أيام عمل [مادة 12].", "sources": [1]},
    ]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["kept"] == []


_PLAIN_ARTICLE_ONE = (
    "نص المادة الأولى هنا طويل بما يكفي وهو مجرد نص توضيحي عادي "
    "ولا يذكر داخله أي رقم مادة آخر إطلاقاً في أي موضع من مواضعه."
)
_PLAIN_ARTICLE_TWO = (
    "نص المادة الثانية هنا مختلف تماماً عن نص المادة الأولى ولا "
    "يشترك معه في أي جزء طويل، وهو أيضاً لا يذكر أي رقم مادة داخله."
)


def test_copied_is_checked_against_the_claims_own_cited_source_not_every_retrieved_one():
    """A claim that happens to match text from a DIFFERENT retrieved source
    it never cited is not "copying its source" — `copied` must be computed
    against the claim's own cited sources only. This claim is a verbatim
    copy of source 2's text but cites only source 1; neither text names any
    article number, so grounding is unaffected either way and only `copied`
    can tell correct code from the mutant apart. Mutant: checking
    `copied_from_context` against every retrieved source's text (`k` texts)
    instead of only the claim's own cited ones would call this claim
    copied because it happens to match SOURCE 2, which it never cited."""
    parsed = {"abstain": False, "claims": [
        {"text": _PLAIN_ARTICLE_TWO, "sources": [1]},
    ]}

    result = gate(parsed, [7, 12], [_PLAIN_ARTICLE_ONE, _PLAIN_ARTICLE_TWO])

    assert result["kept"] == [{"text": _PLAIN_ARTICLE_TWO, "sources": [1], "copied": False}]


# --- Commit 1 (m2): gate() rejects a mismatched source description ---------


def test_gate_raises_when_source_numbers_and_source_texts_disagree_in_length():
    """`source_numbers[i]` / `source_texts[i]` must describe the same source
    at the same position — a caller that passes mismatched lists would have
    every source after the shorter list's length silently misaligned
    (a number checked against the wrong text, or an index error), corrupting
    the self-citation check with no warning at all."""
    parsed = {"abstain": False, "claims": [{"text": "نص", "sources": [1]}]}

    with pytest.raises(ValueError):
        gate(parsed, [7, 12], ["نص واحد فقط"])


# --- Round 5's fix: gate must normalise the claim before checking ----------
#
# `_gate_one_claim` compared a claim's RAW text against `source_texts`,
# which arrive already `evaluation_normalize`d — the corpus is normalised at
# ingest time, and that normalised text is what both the app pipeline and a
# saved eval row hand to `gate`. A diacritic, a tatweel, or a no-break space
# the model's own generation left between "مادة" and its number broke the
# literal regex match, so `citations` on the raw claim silently found
# nothing — and the "does this claim name an article none of its sources
# support" check (rule 3) passed vacuously, keeping a claim it should have
# dropped. A second, independent gap: even already-normalised text with a
# colon between the head word and the number ("المادة: 30") was never
# recognised at all, because `LOOSE_CITATION` never allowed one.

_UNNORMALISED_ARTICLE_30_SPELLINGS = {
    "shadda": "للمادّة 30",
    "fatha": "للمَادة 30",
    "tatweel": "للمـادة 30",
    "no_break_space": "المادة 30",
}


@pytest.mark.parametrize("spelling", _UNNORMALISED_ARTICLE_30_SPELLINGS.values(),
                          ids=_UNNORMALISED_ARTICLE_30_SPELLINGS.keys())
def test_citations_misses_an_unnormalised_article_reference_on_raw_text(spelling):
    """Pins the starting bug precisely: `citations` normalises nothing on
    its own — that is the caller's job. Each of these is a real article-30
    reference that a diacritic, a tatweel, or a no-break space hides from
    the literal regex match."""
    assert citations(spelling) == []


@pytest.mark.parametrize("spelling", _UNNORMALISED_ARTICLE_30_SPELLINGS.values(),
                          ids=_UNNORMALISED_ARTICLE_30_SPELLINGS.keys())
def test_citations_finds_it_once_the_caller_normalises_first(spelling):
    """The other half of the pin: normalising first — what `_gate_one_claim`
    now does before checking a claim — is enough on its own, with no change
    to the regex, to recognise all four spellings from the bug report."""
    assert citations(evaluation_normalize(spelling)) == [30]


@pytest.mark.parametrize("spelling", _UNNORMALISED_ARTICLE_30_SPELLINGS.values(),
                          ids=_UNNORMALISED_ARTICLE_30_SPELLINGS.keys())
def test_a_claim_naming_an_uncited_article_is_ungrounded_even_when_unnormalised(spelling):
    """The bug as it actually reaches the gate: same shape as
    `test_a_claim_naming_an_article_it_does_not_cite_is_ungrounded`, but
    written the way a model actually writes — with a diacritic, a tatweel,
    or a no-break space it left in while generating. Before the fix,
    `_gate_one_claim` ran `citations` on this text raw, found nothing, and
    kept the claim, because the "names an article no source supports" check
    never saw a number to check against `allowed`."""
    claim_text = f"يلزم الحفظ لمدة سنة كاملة {spelling}."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["dropped"][0]["text"] == claim_text
    assert result["kept"] == []


@pytest.mark.parametrize("text,expected", [
    pytest.param("المادة: 30", [30], id="colon_then_space"),
    pytest.param("المادة :30", [30], id="space_then_colon"),
])
def test_a_colon_between_the_head_word_and_the_number_is_recognised(text, expected):
    """The second gap: even on already-normalised text, a colon between the
    head word and the number was not recognised at all."""
    assert citations(text) == expected


def test_an_unrelated_colon_next_to_a_number_is_not_read_as_a_citation():
    """The colon is only meaningful right after a head word like `مادة` —
    guards against a change loose enough to read any `text:number` as a
    citation."""
    assert citations("الاجتماع الساعة 5:30 مساءً، والحضور إلزامي.") == []


def test_a_claim_naming_an_uncited_article_by_colon_is_ungrounded():
    """The colon gap, at the same integration level as the diacritic tests
    above — before the regex extension, this claim's citation was invisible
    to rule 3 regardless of normalisation."""
    claim_text = "يلزم الحفظ لمدة سنة كاملة المادة: 30."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["kept"] == []


def test_copied_is_still_detected_when_the_claim_repeats_the_source_with_a_stray_diacritic():
    """`copied_from_context` has the same raw-claim-vs-normalised-source
    asymmetry as rule 3, and the same fix applies: normalise the claim
    before comparing. A single stray diacritic — the kind a model's own
    generation leaves in while quoting a source verbatim — breaks the
    literal substring match `copied_from_context` runs; before the fix,
    this claim (identical to its source but for one damma) was scored
    `copied: False`."""
    source_text = (
        "استثناء من حكم المادة (14) من هذا القانون يجوز نقل البيانات "
        "بموافقة صريحة."
    )
    claim_with_stray_diacritic = (
        "استثناء من حُكم المادة (14) من هذا القانون يجوز نقل البيانات "
        "بموافقة صريحة."
    )
    parsed = {"abstain": False, "claims": [
        {"text": claim_with_stray_diacritic, "sources": [1]},
    ]}

    result = gate(parsed, [14], [source_text])

    assert result["kept"] == [
        {"text": claim_with_stray_diacritic, "sources": [1], "copied": True},
    ]


def test_a_dropped_claims_text_is_the_original_not_the_normalised_one():
    """Normalising is for comparison only — display is the caller's
    business. A dropped claim must still show exactly what the model wrote,
    tashkeel and all, not the stripped version used internally to find the
    citation."""
    claim_text = "يلزم الحفظ لمدة سنة كاملة للمادّة 30."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["dropped"][0]["text"] == claim_text
    assert "ّ" in result["dropped"][0]["text"]


# --- Round 6's follow-ups: narrow the colon, allow it after رقم, and strip -
# --- invisible format characters in the gate's own comparison step --------
#
# Three reviewer findings on the fix above:
#
# 1. `:?` right after ANY head word over-fired: «عدد المواد: 12» ("number of
#    subjects: 12") is a count, not a citation, but `مواد` is a valid head
#    word too, so the unnarrowed regex read it as citing article 12 — a
#    false drop when a CLAIM says it, a false keep when a claim's own
#    SOURCE happens to say it. Narrowed to the singular head word only
#    (`مادة` / `المادة`): a colon reads as a citation there, not after the
#    dual or plural forms.
# 2. «المادة رقم: 30» was still missed — the colon needs to be recognised
#    after «رقم» too, with the same narrowing.
# 3. RLM (U+200F), LRM (U+200E), ZWNJ (U+200C), ZWJ (U+200D) and ALM
#    (U+061C) have no visible glyph and are not whitespace, so none of them
#    are touched by `evaluation_normalize`'s NFKC/tashkeel/tatweel/
#    whitespace-collapse steps — a claim or a source can carry one between
#    "مادة" and its number with nothing to see. Stripped locally, in the
#    gate's own comparison step, on both the claim's text and its sources'
#    text — `evaluation_normalize` itself is untouched, since it is the
#    benchmark's shared scoring normaliser.


def test_a_claim_mentioning_a_count_of_articles_with_a_colon_is_not_misread_as_citing_that_number():
    """«عدد المواد: 12» is a count, not a citation — «مواد» is plural, and
    the colon is narrowed to the singular head word precisely so a claim
    that happens to mention a count is not misread as citing article 12,
    which would otherwise cause a false 'ungrounded' drop alongside the
    claim's real, legitimate citation to article 7."""
    claim_text = ("بلغ عدد المواد: 12 مادة في هذا الباب، "
                  "ونصت المادة 7 على الإبلاغ خلال ستة أيام.")
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 0
    assert result["kept"] == [{"text": claim_text, "sources": [1], "copied": False}]


def test_a_source_mentioning_a_count_of_articles_does_not_license_a_claim_citing_that_count():
    """The same narrowing, from the other side: a claim's own cited source
    happens to mention «عدد المواد: 12» — a count, not the source citing
    article 12 as its own self-reference. Unnarrowed, `numbers_in_own_sources`
    misread that count as article 12 and wrongly licensed a claim naming
    article 12, even though its cited source never actually references it."""
    source_text = "يشتمل هذا الباب على عدد المواد: 12 مادة، تبدأ بتنظيم الإخطار."
    claim_text = "تنص المادة 12 على إجراء إضافي يجب اتباعه."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, [7], [source_text])

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["kept"] == []


@pytest.mark.parametrize("text,expected", [
    pytest.param("المادة: 30", [30], id="colon_then_space"),
    pytest.param("المادة :30", [30], id="space_then_colon"),
    pytest.param("المادة رقم: 30", [30], id="colon_after_raqm"),
])
def test_a_colon_after_the_singular_head_word_or_after_raqm_is_recognised(text, expected):
    assert citations(text) == expected


@pytest.mark.parametrize("text", [
    pytest.param("عدد المواد: 12", id="plural_bare"),
    pytest.param("عدد المواد رقم: 12", id="plural_with_raqm"),
])
def test_a_colon_after_a_dual_or_plural_head_word_is_still_not_a_citation(text):
    """The narrowing holds whether or not «رقم» sits between the head word
    and the colon."""
    assert citations(text) == []


_INVISIBLE_FORMAT_CHARS = {
    # Spelled out as explicit escapes, not literal characters — each one is
    # invisible in an editor, which is exactly the property under test.
    "rlm": "\u200f",
    "lrm": "\u200e",
    "zwnj": "\u200c",
    "zwj": "\u200d",
    "alm": "\u061c",
}


@pytest.mark.parametrize("mark", _INVISIBLE_FORMAT_CHARS.values(),
                          ids=_INVISIBLE_FORMAT_CHARS.keys())
def test_a_claim_naming_an_uncited_article_is_ungrounded_across_an_invisible_format_character(mark):
    """RLM, LRM, ZWNJ, ZWJ and ALM have no visible glyph and are not
    whitespace — on screen, a claim carrying one between "المادة" and its
    number looks identical to a plain one. `evaluation_normalize` does not
    strip them (it is the benchmark's shared scoring normaliser, not this
    gate's own comparison step); before this fix such a claim was silently
    kept regardless of whether any cited source supported it."""
    claim_text = f"يلزم الحفظ لمدة سنة كاملة المادة {mark}30."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["dropped"][0]["text"] == claim_text
    assert result["kept"] == []


def test_a_sources_self_citation_is_still_recognised_across_an_invisible_format_character():
    """The same stripping applies to the source side of the comparison: a
    source that cites itself with an invisible mark between the head word
    and its number must still license a claim that repeats that same, real
    self-citation — the paraphrase case from
    `test_a_claim_may_name_an_article_that_its_own_source_names`, with an
    RLM the model's own generation of the SOURCE left in (sources are the
    ingested corpus text, not something this gate controls the byte
    content of)."""
    source_text = ("استثناء من حكم المادة \u200f(14) من هذا القانون، "
                   "يجوز نقل البيانات بموافقة صريحة.")
    claim_text = "يوجد استثناء بموجب المادة 14 يسمح بنقل البيانات بموافقة صريحة من صاحبها."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, [7], [source_text])

    assert result["ungrounded"] == 0
    assert result["kept"] == [{"text": claim_text, "sources": [1], "copied": False}]


# --- Further follow-ups: digit-adjacent format characters, every Cf -------
# --- character (not just five), and the colon before رقم -----------------
#
# A reviewer found one more escape and two more gaps in the fix above.
#
# 1. Stripping a format character can JOIN two digits it used to sit
#    between. "المادة 1<RLM>2" (RLM between "1" and "2") reads, after
#    stripping, as article 12 -- but UAX#9 says a bidi-aware renderer
#    DISPLAYS that same text as "21": the mark changes which digit a
#    reader sees first. Neither "12" nor "21" is a number the claim
#    unambiguously named, so a claim with this anywhere is dropped
#    outright, before any number is read from it -- never joined, never
#    split. Between a letter and a digit, or between two letters, nothing
#    changes: the character is stripped as before.
# 2. The five characters from the previous fix are all Unicode category
#    Cf ("format") -- the fix now strips every Cf character, tested
#    against `unicodedata.category` directly rather than a fixed list, so
#    it also covers ZWSP (U+200B), word joiner (U+2060), the BOM
#    (U+FEFF), soft hyphen (U+00AD), the bidi embeddings/overrides
#    (U+202A-202E) and isolates (U+2066-2069) -- and anything added to
#    the category later. `evaluation_normalize` itself is untouched.
# 3. "المادة: رقم 30" -- the colon BEFORE رقم -- was still missed; only
#    the colon AFTER رقم ("المادة رقم: 30") was recognised.

_FORMAT_CHARS_BETWEEN_DIGITS = {
    "rlm": "\u200f",
    "alm": "\u061c",
}


@pytest.mark.parametrize("mark", _FORMAT_CHARS_BETWEEN_DIGITS.values(),
                          ids=_FORMAT_CHARS_BETWEEN_DIGITS.keys())
def test_a_format_character_between_two_digits_drops_the_claim(mark):
    """The exact scenario the reviewer found: the claim's article number
    is only readable as "12" because stripping joined "1" and "2" across
    an RLM (or ALM) -- a bidi-aware renderer would show "21" for the same
    text. The source DOES name article 12, so before this fix the claim
    was wrongly kept; now it is dropped regardless of what its sources
    name, because the number itself is never safely readable."""
    claim_text = f"يلزم الحفظ بموجب المادة 1{mark}2 لمدة سنة كاملة."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, [12], ["نص يذكر المادة 12 بوضوح تام."])

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["dropped"][0]["text"] == claim_text
    assert result["kept"] == []


_FORMAT_CHARS_BETWEEN_WORD_AND_NUMBER = {
    "lrm": "\u200e",
    "zwsp": "\u200b",
}


@pytest.mark.parametrize("mark", _FORMAT_CHARS_BETWEEN_WORD_AND_NUMBER.values(),
                          ids=_FORMAT_CHARS_BETWEEN_WORD_AND_NUMBER.keys())
def test_a_claim_naming_an_uncited_article_is_ungrounded_across_any_cf_character(mark):
    """Between a word and a number, not between two digits -- the claim is
    still read and correctly dropped as ungrounded, exactly like the five
    originally-named characters. LRM was already one of the five; ZWSP
    (U+200B) was not, and pins that the general Cf-category strip covers
    it too, not just the characters named when the fix first shipped."""
    claim_text = f"يلزم الحفظ لمدة سنة كاملة المادة {mark}30."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["kept"] == []


def test_a_colon_before_raqm_is_recognised_too():
    """"المادة: رقم 30" -- colon BEFORE رقم. Round 6 recognised the colon
    AFTER رقم ("المادة رقم: 30"); this position was still missed."""
    assert citations("المادة: رقم 30") == [30]


def test_a_colon_before_raqm_following_a_plural_head_word_is_still_not_a_citation():
    """The narrowing to the singular head word holds at this colon
    position too."""
    assert citations("عدد المواد: رقم 12") == []


def test_a_claim_naming_an_uncited_article_by_colon_before_raqm_is_ungrounded():
    claim_text = "يلزم الحفظ لمدة سنة كاملة المادة: رقم 30."
    parsed = {"abstain": False, "claims": [{"text": claim_text, "sources": [1]}]}

    result = gate(parsed, SOURCE_NUMBERS, SOURCE_TEXTS)

    assert result["ungrounded"] == 1
    assert result["dropped"][0]["reason"] == "ungrounded"
    assert result["kept"] == []
