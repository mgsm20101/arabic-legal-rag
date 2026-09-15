"""Upload chunking — the P1 demo layer (ADR-023), Phase 4B task 1.

An uploaded document is one of two kinds, decided by the statute splitter
the benchmark already trusts rather than by a second guess here:

* a statute — `ingest.parse` finds at least three articles and
  `ingest.validate` finds no problem — gets one chunk per article, on the
  page its header sits on;
* everything else gets chunks built from a page's lines that never cross a
  page, so a displayed source can always name one page to open.

Pure text in, chunks out: no model, no network. The fixture tests read the
real `evals/app/policy_ar.pdf` the way the library does
(`extract_pages(..., keep_latin=True)`).
"""

import json
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.chunking import (  # noqa: E402
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    chunk_document,
    tidy,
)
from legalrag.normalize import evaluation_normalize  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "evals" / "app"
DOC = "0123456789ab"

# Every body clears `ingest.validate`'s 40-character floor, or a valid
# statute would be rejected as a bad split.
BODIES = [
    "يلتزم المتحكم بالحصول على موافقة صريحة من الشخص المعني قبل جمع بياناته الشخصية",
    "يجب إخطار المركز بأي خرق للبيانات الشخصية خلال اثنتين وسبعين ساعة من العلم به",
    "يحق للشخص المعني طلب تصحيح بياناته الشخصية أو محوها متى كانت غير صحيحة",
    "يعاقب بغرامة لا تقل عن مائة ألف جنيه كل من جمع بيانات شخصية دون موافقة أصحابها",
]
CONTINUATION = "ويسري هذا الحكم على المعالج أيضا"


def _statute_pages() -> list[str]:
    """Four law articles over three pages. Article 2 runs onto page 2, and
    article 4's header opens page 3 — so its regex match starts on the
    newline that separates pages 2 and 3, the offset a naive mapping puts
    on the wrong page."""
    return [
        "قانون تجريبي لحماية البيانات\nمادة 1\n" + BODIES[0] + "\nمادة 2\n" + BODIES[1],
        CONTINUATION + "\nمادة (3)\n" + BODIES[2],
        "مادة 4\n" + BODIES[3],
    ]


# --------------------------------------------------------------- statutes --


def test_a_statute_with_three_valid_articles_gets_one_chunk_per_article_on_its_header_page():
    kind, chunks = chunk_document(DOC, _statute_pages(), title="قانون تجريبي")

    assert kind == "statute"
    assert [c.article for c in chunks] == [1, 2, 3, 4]
    assert [c.label for c in chunks] == ["مادة 1", "مادة 2", "مادة 3", "مادة 4"]
    assert [c.page for c in chunks] == [1, 1, 2, 3]
    assert [c.number for c in chunks] == [1, 2, 3, 4]
    assert [c.id for c in chunks] == [f"{DOC}:{n}" for n in (1, 2, 3, 4)]
    assert all(c.doc_id == DOC for c in chunks)
    # The whole article, including the part that ran onto the next page.
    assert chunks[1].text == tidy(evaluation_normalize(BODIES[1] + "\n" + CONTINUATION))


def test_issuance_articles_are_labelled_as_issuance_and_carry_no_article_number():
    """`source_numbers` in `cite.gate` is the LAW article number; an
    issuance article has none of its own there, exactly as in answer_eval."""
    pages = [
        "المادة الأولى\n" + BODIES[0] + "\nالمادة الثانية\n" + BODIES[1],
        "مادة 1\n" + BODIES[2] + "\nمادة 2\n" + BODIES[3] + "\nمادة 3\n" + BODIES[0],
    ]

    kind, chunks = chunk_document(DOC, pages)

    assert kind == "statute"
    assert [c.label for c in chunks] == [
        "مادة 1 (إصدار)", "مادة 2 (إصدار)", "مادة 1", "مادة 2", "مادة 3",
    ]
    assert [c.article for c in chunks] == [None, None, 1, 2, 3]
    assert [c.page for c in chunks] == [1, 1, 2, 2, 2]


def test_articles_that_fail_validation_fall_back_to_page_chunks():
    missing_article_3 = ["مادة 1\n" + BODIES[0] + "\nمادة 2\n" + BODIES[1], "مادة 4\n" + BODIES[3]]
    kind, chunks = chunk_document(DOC, missing_article_3)
    assert kind == "generic"
    assert [c.label for c in chunks] == ["ص 1", "ص 2"]

    only_two_articles = ["مادة 1\n" + BODIES[0] + "\nمادة 2\n" + BODIES[1]]
    kind, _ = chunk_document(DOC, only_two_articles)
    assert kind == "generic"


def test_two_documents_with_the_same_article_numbers_get_different_chunk_ids():
    _, first = chunk_document("aaaaaaaaaaaa", _statute_pages())
    _, second = chunk_document("bbbbbbbbbbbb", _statute_pages())

    assert [c.article for c in first] == [c.article for c in second]
    ids = [c.id for c in first + second]
    assert len(set(ids)) == len(ids)


def test_a_statute_saved_with_windows_line_endings_is_still_a_statute():
    """ingest's header patterns end a header line at "\\n", so "مادة 1\\r\\n"
    matched nothing and a Notepad-saved statute came out generic."""
    pages = [p.replace("\n", "\r\n") for p in _statute_pages()]

    kind, chunks = chunk_document(DOC, pages)

    assert kind == "statute"
    assert [c.page for c in chunks] == [1, 1, 2, 3]


def test_statute_and_page_chunks_can_each_be_asked_for_on_their_own():
    """The library decides which extraction of a PDF each kind is chunked
    from, so both halves of chunk_document are public."""
    import legalrag.chunking as chunking_mod

    statute = _statute_pages()
    assert [c.label for c in chunking_mod.statute_chunks(DOC, statute)] == ["مادة 1", "مادة 2", "مادة 3", "مادة 4"]
    assert chunking_mod.statute_chunks(DOC, ["نص عادي بلا أي مادة مرقمة فيه على الإطلاق"]) is None
    assert [c.label for c in chunking_mod.page_chunks(DOC, statute)] == ["ص 1", "ص 2", "ص 3"]
    assert chunking_mod.may_be_statute(statute) is True
    assert chunking_mod.may_be_statute(["مادة 1\nنص", "مادة 2\nنص"]) is False  # two headers are a quotation


# ---------------------------------------------------------- generic pages --


def test_a_document_without_article_headers_is_chunked_by_page_not_rejected():
    pages = [
        "سياسة العمل عن بعد\nتسري هذه السياسة على جميع الموظفين الدائمين.",
        "   \n\n",
        "ـــ",  # nothing but tatweel: normalizes to nothing, so no chunk
        "يلتزم الموظف بساعات التواصل الإلزامية.",
    ]

    kind, chunks = chunk_document(DOC, pages, title="سياسة")

    assert kind == "generic"
    assert [(c.id, c.number, c.page, c.label, c.article) for c in chunks] == [
        (f"{DOC}:1", 1, 1, "ص 1", None),
        (f"{DOC}:2", 2, 4, "ص 4", None),
    ]
    assert chunks[0].text == "سياسة العمل عن بعد تسري هذه السياسة على جميع الموظفين الدائمين."


def test_a_generic_chunk_never_carries_an_article_number():
    """The text names articles, but they fail validation: no chunk may claim
    one, or `cite.gate` would accept «مادة 7» on the strength of a label."""
    pages = [
        "مادة 7\n" + BODIES[0],
        "مادة 9\n" + BODIES[1] + "\nوفقا لأحكام المادة 7 من هذا القانون.",
    ]

    kind, chunks = chunk_document(DOC, pages)

    assert kind == "generic"
    assert chunks
    assert all(c.article is None for c in chunks)
    assert all(c.label == f"ص {c.page}" for c in chunks)


PAGE_ONE_WORDS = ["البيانات", "الشخصية", "المتحكم", "المعالج"]
PAGE_TWO_WORDS = ["الترخيص", "الموافقة", "الخرق", "المركز"]


def _lines(words: list[str], count: int, per_line: int) -> list[str]:
    return [
        " ".join(words[(i + j) % len(words)] for j in range(per_line))
        for i in range(count)
    ]


def test_a_page_chunk_never_crosses_a_page_or_exceeds_the_cap():
    one_long_line = " ".join(PAGE_ONE_WORDS * 250)            # ~7,500 chars, one line
    one_huge_word = "ب" * (MAX_CHUNK_CHARS * 2 + 300)          # longer than the cap alone
    pages = [
        "\n".join(_lines(PAGE_ONE_WORDS, 60, 10) + [one_long_line, one_huge_word]
                  + _lines(PAGE_ONE_WORDS, 4, 10)),
        "\n".join(_lines(PAGE_TWO_WORDS, 40, 9)),
    ]

    kind, chunks = chunk_document(DOC, pages)

    assert kind == "generic"
    for page_number in (1, 2):
        on_page = [c for c in chunks if c.page == page_number]
        assert len(on_page) > 1
        page_text = evaluation_normalize(pages[page_number - 1])
        for c in on_page:
            assert c.text in page_text, "a chunk holds text from another page"
        # Only a page's last chunk may carry a merged short tail.
        assert all(len(c.text) <= MAX_CHUNK_CHARS for c in on_page[:-1])
        assert len(on_page[-1].text) <= MAX_CHUNK_CHARS + MIN_CHUNK_CHARS
    assert "ب" * MAX_CHUNK_CHARS in [c.text for c in chunks], "the huge word was not hard-split"
    assert [c.number for c in chunks] == list(range(1, len(chunks) + 1))


def test_a_short_last_piece_is_merged_into_the_previous_chunk_on_its_page():
    ten_lines = ["س" * 98] * 10    # joined: 989 chars
    short = "قصير" * 10            # 40 chars, under MIN_CHUNK_CHARS
    pages = ["\n".join(ten_lines + [short]), short]

    kind, chunks = chunk_document(DOC, pages)

    assert kind == "generic"
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].text.endswith(" " + short)
    assert len(chunks[0].text) == 989 + 1 + 40     # past the cap, by less than MIN
    assert chunks[1].text == short                 # alone on its page: nothing to merge into


def test_a_full_page_piece_is_closed_after_its_last_sentence_end_not_mid_sentence():
    """Page lines break where the page edge falls, not where sentences end:
    a cut mid-sentence splits a phrase a question needs across two chunks."""
    l1 = "أ" * 299 + "."
    l2 = "ب" * 399 + "."
    l3 = "ج" * 250               # this sentence carries on ...
    l4 = "د" * 399 + "."         # ... and ends here

    _, chunks = chunk_document(DOC, ["\n".join([l1, l2, l3, l4])])

    # The plain greedy cut would be [l1 l2 l3] | [l4].
    assert [c.text for c in chunks] == [f"{l1} {l2}", f"{l3} {l4}"]


def test_the_plain_cut_stays_when_no_sentence_end_leaves_min_chunk_chars_behind():
    m1 = "ه" * 99 + "."          # a sentence end, but only 100 chars would stay
    m2 = "و" * 700
    m3 = "ز" * 150
    m4 = "ح" * 300

    _, chunks = chunk_document(DOC, ["\n".join([m1, m2, m3, m4])])

    assert [c.text for c in chunks] == [f"{m1} {m2} {m3}", m4]


def test_a_short_tail_still_merges_after_a_sentence_end_cut():
    a = "ا" * 499 + "."
    b = "ب" * 400                # carried past the cut after `a` ...
    c = "ت" * 449 + "."          # ... to end its sentence here
    d = "ث" * 159 + "."          # the page's last piece, under MIN_CHUNK_CHARS

    _, chunks = chunk_document(DOC, ["\n".join([a, b, c, d])])

    # Plain greedy: [a b] | [c d]. Sentence-end cut: [a] | [b c] | [d], and d merges back.
    assert [x.text for x in chunks] == [a, f"{b} {c} {d}"]
    assert MAX_CHUNK_CHARS < len(chunks[1].text) <= MAX_CHUNK_CHARS + MIN_CHUNK_CHARS


def test_no_piece_exceeds_the_cap_when_sentence_end_cuts_are_in_play():
    rng = random.Random(4)
    lines = [
        "".join(rng.choice("سشصضطظعغ") for _ in range(rng.randint(1, 400)))
        + ("." if rng.random() < 0.3 else "")
        for _ in range(400)
    ]
    page = "\n".join(lines)

    _, chunks = chunk_document(DOC, [page])

    assert len(chunks) > 20
    assert sum(c.text.endswith(".") for c in chunks[:-1]) > len(chunks) // 2, "no sentence-end cuts happened"
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks[:-1])
    assert len(chunks[-1].text) <= MAX_CHUNK_CHARS + MIN_CHUNK_CHARS
    # Nothing lost, repeated or reordered by moving lines between pieces.
    assert " ".join(c.text for c in chunks) == evaluation_normalize(page)


def test_a_piece_of_exactly_the_cap_stays_whole_and_one_character_more_is_cut():
    a, b = "أ" * 500, "ب" * 499
    _, chunks = chunk_document(DOC, [f"{a}\n{b}"])          # joined: exactly MAX_CHUNK_CHARS
    assert [c.text for c in chunks] == [f"{a} {b}"]
    assert len(chunks[0].text) == MAX_CHUNK_CHARS
    _, chunks = chunk_document(DOC, [f"{a}\n{b}ب"])         # one more: two pieces, both long enough to stand
    assert [c.text for c in chunks] == [a, b + "ب"]

    # A single line of exactly the cap is never split; a longer one is split
    # on whitespace, and a part may fill to exactly the cap.
    line = "ت" * 750 + " " + "ث" * 249
    _, chunks = chunk_document(DOC, [line])
    assert [c.text for c in chunks] == [line]
    _, chunks = chunk_document(DOC, [line + " " + "ج" * 300])
    assert [c.text for c in chunks] == [line, "ج" * 300]


def test_a_last_piece_of_exactly_min_chunk_chars_stands_alone_and_one_character_less_merges():
    full = "\n".join(["س" * 98] * 10)                         # 989 chars: no further line fits
    _, chunks = chunk_document(DOC, [f"{full}\n{'ق' * MIN_CHUNK_CHARS}"])
    assert [len(c.text) for c in chunks] == [989, MIN_CHUNK_CHARS]
    _, chunks = chunk_document(DOC, [f"{full}\n{'ق' * (MIN_CHUNK_CHARS - 1)}"])
    assert [len(c.text) for c in chunks] == [989 + 1 + MIN_CHUNK_CHARS - 1]

    # The sentence-end cut needs exactly MIN_CHUNK_CHARS left behind, no more.
    ends = "أ" * (MIN_CHUNK_CHARS - 1) + "."
    carried, rest = "ب" * 700, "ج" * 150
    _, chunks = chunk_document(DOC, ["\n".join([ends, carried, rest])])
    assert [c.text for c in chunks] == [ends, f"{carried} {rest}"]


def test_no_page_chunk_exceeds_the_cap_once_normalization_has_expanded_it():
    """NFKC turns one ligature into a whole phrase (U+FDFA comes out as 18
    characters), so a piece measured before normalizing can end up many times
    the cap."""
    ligature = "\ufdfa"
    assert len(evaluation_normalize(ligature)) > 10
    line = " ".join([ligature] * 45)          # 89 characters as extracted
    page = "\n".join([line] * 11)             # 989: one piece, measured before normalizing

    _, chunks = chunk_document(DOC, [page])

    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks[:-1])
    assert len(chunks[-1].text) <= MAX_CHUNK_CHARS + MIN_CHUNK_CHARS
    assert " ".join(c.text for c in chunks) == tidy(evaluation_normalize(page))


# ------------------------------------------------------------------ tidy --


def test_tidy_removes_space_before_punctuation_and_rejoins_a_wrapped_latin_word_without_touching_letters():
    cases = {
        "بموجب هذه السياسة .": "بموجب هذه السياسة.",
        "أولا : نطاق التطبيق": "أولا: نطاق التطبيق",
        "يناير 2026 ، وتحل": "يناير 2026، وتحل",
        "هل يجوز ذلك ؟ نعم !": "هل يجوز ذلك؟ نعم!",
        "ثلاثة ؛ أربعة": "ثلاثة؛ أربعة",
        "الفقرة ( أ ) والبند [ 2 ]": "الفقرة (أ) والبند [2]",
        "تقدير « يحتاج إلى تحسين » في": "تقدير «يحتاج إلى تحسين» في",
        "شبكات Wi- Fi العامة": "شبكات Wi-Fi العامة",
        "شبكات Wi-\nFi العامة": "شبكات Wi-Fi العامة",
    }
    for raw, expected in cases.items():
        assert tidy(raw) == expected, raw
        # Spacing only: the same characters in the same order, less whitespace.
        assert re.sub(r"\s", "", tidy(raw)) == re.sub(r"\s", "", raw)

    # Only a Latin word wrapped at its hyphen is rejoined.
    assert tidy("الشبكة - الشركة") == "الشبكة - الشركة"


def test_tidy_never_removes_the_space_before_a_mark_glued_to_the_next_token():
    """policy_ar.pdf extracts «يناير 2026. وثيقة» as «يناير .2026 وثيقة»:
    pdf_text's bidi ordering puts the period before the number. That period
    closes nothing, and removing the space would weld «يناير.2026» into one
    word — which is also why the PDF and TXT words could not be equal."""
    assert tidy("يناير .2026 وثيقة") == "يناير .2026 وثيقة"
    assert tidy("في مارس .2023") == "في مارس .2023"
    # The same guard mirrored: an opening mark glued to the previous token.
    assert tidy("مادة( 1") == "مادة( 1"


def test_tidy_looks_past_a_whole_run_of_marks_before_removing_a_space():
    """A run of marks is one token edge: it loses the space before it only
    when the whole run ends at whitespace or at the end of the text."""
    assert tidy("يناير ..2026 وثيقة") == "يناير ..2026 وثيقة"
    assert tidy("هل يجوز ذلك ؟!نعم") == "هل يجوز ذلك ؟!نعم"
    assert tidy("يناير .2026 وثيقة") == "يناير .2026 وثيقة"   # the single mark, as before
    # A run that does end its token still closes up.
    assert tidy("هل يجوز ذلك ؟! نعم") == "هل يجوز ذلك؟! نعم"
    assert tidy("انتهى البند ..") == "انتهى البند.."
    # The mirror: a run of opening marks glued to the previous token opens nothing.
    assert tidy("مادة(( 1") == "مادة(( 1"
    assert tidy("البند (( أ") == "البند ((أ"


# ------------------------------------------------- the app eval fixture --


@pytest.fixture(scope="module")
def policy_pdf_pages() -> list[str]:
    pytest.importorskip("pdfplumber")
    from legalrag.pdf_text import extract_pages

    return extract_pages(APP / "policy_ar.pdf", keep_latin=True)


@pytest.fixture(scope="module")
def policy_txt_pages() -> list[str]:
    """Read the way the library reads an uploaded .txt: the decoded text,
    split into pages on form feeds."""
    return (APP / "policy_ar.txt").read_bytes().decode("utf-8-sig").split("\f")


def _answerable() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (APP / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r["answerable"]]


def _holds_every_keyword(text: str, keywords: list[str]) -> bool:
    text = evaluation_normalize(text)
    return all(evaluation_normalize(k) in text for k in keywords)


def test_the_policy_pdf_is_generic_and_every_expected_keyword_lands_in_a_chunk_on_its_expected_page(
    policy_pdf_pages, policy_txt_pages,
):
    kind, chunks = chunk_document(DOC, policy_pdf_pages)
    assert kind == "generic"
    assert len(policy_pdf_pages) == 6

    for q in _answerable():
        pages = {c.page for c in chunks if _holds_every_keyword(c.text, q["expected_keywords"])}
        assert pages & set(q["expected_pages"]), (q["id"], q["expected_keywords"], sorted(pages))

    # No form feed in the TXT: one page, so there is no page to check.
    assert len(policy_txt_pages) == 1
    kind, txt_chunks = chunk_document(DOC, policy_txt_pages)
    assert kind == "generic"
    for q in _answerable():
        assert any(_holds_every_keyword(c.text, q["expected_keywords"]) for c in txt_chunks), q["id"]


_EDGE_PUNCTUATION = re.compile(r"^\W+|\W+$")


def _words(chunks) -> list[str]:
    """Every word of `chunks` in order, compared after `tidy`: a token that
    is only punctuation is dropped, and punctuation at a word's edges is
    stripped (the PDF puts a sentence-final period on the wrong side of a
    number, «.2026»)."""
    words = []
    for chunk in chunks:
        for token in tidy(chunk.text).split():
            word = _EDGE_PUNCTUATION.sub("", token)
            if word:
                words.append(word)
    return words


def test_the_policy_pdf_and_txt_chunk_to_the_same_words(policy_pdf_pages, policy_txt_pages):
    _, pdf_chunks = chunk_document(DOC, policy_pdf_pages)
    _, txt_chunks = chunk_document(DOC, policy_txt_pages)

    pdf_words = _words(pdf_chunks)
    for term in ("Microsoft", "Teams", "VPN", "Wi-Fi"):
        assert term in pdf_words, term
    assert pdf_words == _words(txt_chunks)
