"""Broken-word detector tests (ADR-018).

The defect these catch is silent by construction: `عشر ين` reads as ordinary
text, and BM25 indexes it as two tokens, so a query for `عشرين` matches
neither half. Nothing errors; the lexical baseline is just quietly lower than
it should be. ADR-013 declared this class "0 remaining" and it was not.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.broken_words import find_broken_words, token_counts  # noqa: E402


def _doc(doc_id: str, text: str) -> dict:
    return {"id": doc_id, "number": 1, "law_name": "س", "text": text, "search_text": text}


def test_flags_a_split_word_when_the_joined_form_exists_elsewhere():
    docs = [
        _doc("law-2", "بما لا يجاوز عشر ين ألف جنيه"),
        _doc("law-9", "غرامة لا تقل عن عشرين ألف جنيه"),
    ]
    hits = find_broken_words(docs)
    assert [h["broken"] for h in hits] == ["عشر ين"]
    assert hits[0]["joined"] == "عشرين"
    assert hits[0]["article"] == "law-2"


def test_does_not_flag_when_the_joined_form_never_occurs():
    """The corpus is the lexicon. No evidence, no claim."""
    docs = [_doc("law-1", "يلتزم المتحكم بإبلاغ المركز فورا")]
    assert find_broken_words(docs) == []


def test_a_long_second_fragment_is_not_treated_as_a_tail():
    """`max_tail` keeps two genuine words from being fused on a coincidence."""
    docs = [
        _doc("law-1", "البيانات الشخصية محمية"),
        _doc("law-2", "البياناتالشخصية"),  # implausible join, long tail
    ]
    assert find_broken_words(docs, max_tail=3) == []


def test_reports_every_occurrence_not_just_the_first():
    docs = [
        _doc("law-1", "أو نشر ها أو محوها"),
        _doc("law-4", "ثم نشر ها مرة أخرى"),
        _doc("law-9", "عند نشرها للجمهور"),
    ]
    hits = find_broken_words(docs)
    assert len(hits) == 2
    assert {h["article"] for h in hits} == {"law-1", "law-4"}


def test_punctuation_does_not_hide_a_split():
    docs = [
        _doc("law-1", "أو نشر ها، أو محوها"),
        _doc("law-9", "عند نشرها للجمهور"),
    ]
    assert [h["joined"] for h in find_broken_words(docs)] == ["نشرها"]


def test_single_letter_head_is_ignored():
    """A one-letter first token is a preposition far more often than a stem."""
    docs = [_doc("law-1", "و صف الحال"), _doc("law-2", "وصف الحال")]
    assert find_broken_words(docs) == []


def test_token_counts_strips_punctuation():
    counts = token_counts([_doc("law-1", "المركز، المركز . المركز")])
    assert counts["المركز"] == 3


def test_the_shipped_corpus_is_checked_by_this_rule_not_by_a_claim():
    """ADR-013 said 0 remaining; that was a claim, not a measurement.

    This test does not assert a count — the corpus is replaced when a faithful
    text is ingested. It asserts the detector runs on the real corpus shape and
    returns something answerable.
    """
    from legalrag.broken_words import load_corpus

    docs = load_corpus()
    if not docs:  # fresh clone, nothing ingested — nothing to check
        return
    hits = find_broken_words(docs)
    assert isinstance(hits, list)
    for h in hits:
        assert h["joined"] == h["broken"].replace(" ", "")
