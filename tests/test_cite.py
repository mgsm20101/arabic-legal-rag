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
