"""Citation-contract tests — PRD M2/B1.

The whole point of this module is that its failure mode is invisible in the
output: an answer citing «المادة (٧٨)» of a 49-article law is as fluent as a
correct one. These tests pin the distinctions that make it visible.
"""

import sys
from pathlib import Path

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
    source verbatim is not the model inventing a citation, it is the law
    naming itself (the same exemption `audit` already gives free text via
    `copied_from_context`). This is Run 5's recorded meaning change: the
    claim is KEPT, with `copied` set so the rate is visible on its own."""
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
