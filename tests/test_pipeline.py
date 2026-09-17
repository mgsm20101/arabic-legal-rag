"""The ask round trip — P1 demo layer (ADR-023, ADR-025), Phase 4B task 3.

retrieve -> relevance -> claims -> gate, ending in a ChatResult an HTTP
layer can return as it is. What must hold whichever model answers: only
claims that survived `cite.gate` are shown, nothing the gate kept out of
view leaks into the result, an empty answer always says why, and the
timings are this answer's own. No model runs here: `ScriptedChat` replies
from a script, the way test_claims.py's stub does.
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import legalrag.pipeline as pipeline_mod  # noqa: E402
from legalrag.chunking import Chunk  # noqa: E402
from legalrag.cite import citations, gate  # noqa: E402
from legalrag.claims import ClaimsGenerator, final_abstain_reason  # noqa: E402
from legalrag.library import Library, LibraryHit, StorageError  # noqa: E402
from legalrag.pipeline import (  # noqa: E402
    MAX_QUESTION_CHARS,
    MIN_QUESTION_CHARS,
    TOP_K,
    ChatResult,
    Pipeline,
    Source,
)
from stubs import (  # noqa: E402
    ANSWERS_NO,
    ANSWERS_YES,
    KeywordEncoder,
    ScriptedChat,
    claims_json,
    text_document,
)

POLICY_ID = "aaaaaaaaaaaa"
STATUTE_ID = "bbbbbbbbbbbb"
POLICY_TEXT = "يلتزم المدير بالرد على طلب العمل عن بعد خلال خمسة أيام عمل من تاريخ استلام الطلب."
STATUTE_TEXT = "يصرف بدل إنترنت شهري قدره 400 جنيه لكل موظف يعمل عن بعد يومين أو أكثر."
POLICY_CHUNK = Chunk(id=f"{POLICY_ID}:1", doc_id=POLICY_ID, number=1, article=None,
                     label="ص 1", page=1, text=POLICY_TEXT)
STATUTE_CHUNK = Chunk(id=f"{STATUTE_ID}:7", doc_id=STATUTE_ID, number=7, article=7,
                      label="مادة 7", page=3, text=STATUTE_TEXT)
ONE_GOOD_CLAIM = claims_json(("يرد المدير على الطلب خلال خمسة أيام عمل.", [1]))


class _FakeLibrary:
    """Canned hits, in order; records every search it is asked."""

    def __init__(self, *hits: LibraryHit):
        self.hits = list(hits)
        self.searches: list[tuple] = []

    def search(self, query, k=5, doc_ids=None):
        self.searches.append((query, k, doc_ids))
        return self.hits[:k]


def _two_hits() -> tuple[LibraryHit, LibraryHit]:
    return (LibraryHit(POLICY_CHUNK, "سياسة العمل", 0.91),
            LibraryHit(STATUTE_CHUNK, "لائحة البدلات", 0.87))


def _gated(relevance_replies, claims_replies):
    relevance = ScriptedChat(relevance_replies)
    claims = ScriptedChat(claims_replies)
    return ClaimsGenerator(model=claims, relevance_model=relevance), relevance, claims


def test_chat_returns_only_claims_that_survived_the_gate():
    """One claims-JSON reply may hold at most 3 claims (T13, finding F13:
    SYSTEM_CLAIMS's own "ثلاث claims على الأكثر" is now enforced by
    `parse_claims`) — the 5 drop-reason examples this test needs (2 kept,
    1 of each of uncited/fabricated/ungrounded) are split across two `ask`
    calls on the same pipeline/library instead of one 5-claim reply, so the
    gate's per-claim classification is still exercised end to end for every
    reason, just no longer inside a reply that violates the very contract
    this task tightens."""
    library = _FakeLibrary(*_two_hits())
    kept_plain = "يرد المدير على طلب العمل عن بعد خلال خمسة أيام عمل."
    kept_article = "يصرف بدل الإنترنت الشهري وفقا للمادة 7."
    generator, _, _ = _gated([ANSWERS_YES, ANSWERS_YES], [
        claims_json(
            (kept_plain, [1]),
            ("بدل الإنترنت الشهري 400 جنيه.", []),     # uncited
            ("المهلة خمسة أيام عمل.", [3]),            # fabricated: two sources were shown
        ),
        claims_json(
            ("يصرف البدل وفقا للمادة 9.", [2]),        # ungrounded: source 2 is article 7, naming no 9
            (kept_article, [2]),
        ),
    ])
    pipeline = Pipeline(library, generator)

    first = pipeline.ask("  كم مهلة رد المدير؟  ")
    second = pipeline.ask("  كم مهلة رد المدير؟  ")

    assert isinstance(first, ChatResult) and isinstance(second, ChatResult)
    assert library.searches == [("كم مهلة رد المدير؟", TOP_K, None)] * 2
    assert (first.status, first.abstain_reason) == ("partial", None)
    assert (second.status, second.abstain_reason) == ("partial", None)
    assert first.claims == [{"text": kept_plain, "sources": [1]}]
    assert second.claims == [{"text": kept_article, "sources": [2]}]
    assert first.dropped == {"uncited": 1, "fabricated": 1, "ungrounded": 0}
    assert second.dropped == {"uncited": 0, "fabricated": 0, "ungrounded": 1}
    expected_sources = [
        Source(n=1, chunk_id=POLICY_CHUNK.id, doc_id=POLICY_ID, doc_title="سياسة العمل",
               label="ص 1", page=1, article=None, text=POLICY_TEXT),
        Source(n=2, chunk_id=STATUTE_CHUNK.id, doc_id=STATUTE_ID, doc_title="لائحة البدلات",
               label="مادة 7", page=3, article=7, text=STATUTE_TEXT),
    ]
    assert first.sources == expected_sources
    assert second.sources == expected_sources


def test_chat_with_no_documents_abstains_without_calling_the_model(tmp_path):
    generator, relevance, claims = _gated([], [])
    pipeline = Pipeline(Library(tmp_path, encoder=KeywordEncoder()), generator)

    for doc_ids in (None, []):
        result = pipeline.ask("كم يوم إجازة سنوية؟", doc_ids=doc_ids)

        assert (result.status, result.abstain_reason) == ("abstained", "no_sources")
        assert (result.claims, result.sources) == ([], [])
        assert result.dropped == {"uncited": 0, "fabricated": 0, "ungrounded": 0}
        assert (result.timings_ms["relevance"], result.timings_ms["claims"]) == (0, 0)
    assert relevance.seen == [] and claims.seen == []


def test_a_page_source_never_grounds_a_claim_that_names_an_article_its_text_does_not(tmp_path):
    page = ("يلتزم الموظف بإبلاغ فريق أمن المعلومات خلال 24 ساعة من اكتشاف الواقعة، "
            "وتحدد المادة 12 من لائحة الجزاءات عقوبة التأخير.")
    library = Library(tmp_path, encoder=KeywordEncoder())
    library.add(text_document(page), "policy.txt")
    named_by_the_page = "تحدد المادة 12 من لائحة الجزاءات عقوبة التأخير."
    named_by_nothing = "يجب الإبلاغ عن فقدان الجهاز خلال 24 ساعة وفقا للمادة 30."
    generator, _, _ = _gated([ANSWERS_YES], [claims_json((named_by_nothing, [1]), (named_by_the_page, [1]))])

    result = Pipeline(library, generator).ask("متى يجب الإبلاغ عن فقدان الجهاز؟")

    # A page has no article number of its own: only a number its text names can ground a claim.
    assert [(s.label, s.article) for s in result.sources] == [("ص 1", None)]
    assert result.claims == [{"text": named_by_the_page, "sources": [1]}]
    assert result.dropped == {"uncited": 0, "fabricated": 0, "ungrounded": 1}
    assert result.status == "partial"


def test_a_page_sources_position_page_or_ordinal_never_becomes_an_article_number():
    """A page chunk has no article number. Gating with its position in the
    prompt, its page or its ordinal instead would let «وفقاً للمادة 3» ground
    itself on source [3] of a document that has no article 3 at all."""
    texts = [
        "تسري هذه السياسة على الموظفين الدائمين الذين أتموا فترة الاختبار.",
        "يحق للموظف طلب العمل عن بعد بعد مرور ستة أشهر على تعيينه.",
        "يلتزم الموظف بإبلاغ فريق أمن المعلومات خلال 24 ساعة من فقدان الجهاز.",
    ]
    pages = [Chunk(id=f"{POLICY_ID}:{n}", doc_id=POLICY_ID, number=n, article=None,
                   label=f"ص {n}", page=n, text=text) for n, text in enumerate(texts, start=1)]
    library = _FakeLibrary(*(LibraryHit(chunk, "سياسة العمل", 0.9) for chunk in pages))
    invented = "وفقاً للمادة 3 يبلغ الموظف فريق أمن المعلومات خلال 24 ساعة."
    plain = "يبلغ الموظف فريق أمن المعلومات خلال 24 ساعة من فقدان الجهاز."
    generator, _, _ = _gated([ANSWERS_YES], [claims_json((invented, [3]), (plain, [3]))])

    result = Pipeline(library, generator).ask("متى يجب الإبلاغ عن فقدان الجهاز؟")

    assert citations(invented) == [3] and citations(texts[2]) == []
    assert [(s.n, s.page, s.article) for s in result.sources][2] == (3, 3, None)
    assert result.claims == [{"text": plain, "sources": [3]}]
    assert result.dropped == {"uncited": 0, "fabricated": 0, "ungrounded": 1}


def test_an_article_respelled_with_tashkeel_tatweel_or_a_colon_is_checked_against_its_sources_all_the_same():
    """The gate reads each claim after evaluation_normalize, so a shadda, a
    fatha or a tatweel inside «مادة» hides nothing, and «المادة: 30» names
    article 30 too. None may rest on a page whose text names no article 30;
    the plain spelling, citing a page that does name it, is kept.

    One claims-JSON reply may hold at most 3 claims (T13, finding F13:
    SYSTEM_CLAIMS's own "ثلاث claims على الأكثر" is now enforced by
    `parse_claims`), so the 4 respelled variants plus the grounded claim are
    split across two `ask` calls instead of one 5-claim reply — every
    spelling is still checked against its sources exactly as before."""
    silent = "يلتزم الموظف بإبلاغ فريق أمن المعلومات خلال 24 ساعة من فقدان الجهاز."
    naming = "وتحدد المادة 30 من لائحة الجزاءات عقوبة التأخير في الإبلاغ عن فقدان الجهاز."
    pages = [Chunk(id=f"{POLICY_ID}:{n}", doc_id=POLICY_ID, number=n, article=None,
                   label=f"ص {n}", page=n, text=text) for n, text in enumerate((silent, naming), start=1)]
    library = _FakeLibrary(*(LibraryHit(chunk, "سياسة العمل", 0.9) for chunk in pages))
    penalty = "يعاقب الموظف على التأخير في الإبلاغ"
    respelled = [
        f"{penalty} وفقاً للمادّة 30.",   # shadda on the dal: للمادّة
        f"{penalty} وفقاً للمَادة 30.",   # fatha on the meem: للمَادة
        f"{penalty} وفقاً للمـادة 30.",   # tatweel: للمـادة
        f"{penalty} بحسب المادة: 30.",
    ]
    grounded = f"{penalty} وفقاً للمادة 30."
    generator, _, _ = _gated([ANSWERS_YES, ANSWERS_YES], [
        claims_json(*((text, [1]) for text in respelled[:3])),
        claims_json((respelled[3], [1]), (grounded, [2])),
    ])
    pipeline = Pipeline(library, generator)

    first = pipeline.ask("ما عقوبة التأخير في الإبلاغ عن فقدان الجهاز؟")
    second = pipeline.ask("ما عقوبة التأخير في الإبلاغ عن فقدان الجهاز؟")

    assert citations(silent) == [] and citations(naming) == [30]
    assert first.claims == []
    assert first.dropped == {"uncited": 0, "fabricated": 0, "ungrounded": 3}
    assert first.status == "abstained"
    assert second.claims == [{"text": grounded, "sources": [2]}]
    assert second.dropped == {"uncited": 0, "fabricated": 0, "ungrounded": 1}
    assert second.status == "partial"


def test_the_result_never_carries_raw_model_output_or_dropped_claim_text():
    copied_claim = POLICY_TEXT  # verbatim, so the gate marks it copied
    dropped_claim = "جملة محذوفة لا يجوز أن تصل إلى المستخدم أبدا."
    claims_raw = claims_json((copied_claim, [1]), (dropped_claim, []))
    generator, _, _ = _gated([ANSWERS_YES], [claims_raw])

    result = Pipeline(_FakeLibrary(*_two_hits()), generator).ask("كم مهلة رد المدير؟")
    as_dict = result.to_dict()
    blob = json.dumps(as_dict, ensure_ascii=False)

    assert json.loads(blob) == as_dict  # JSON-ready as it comes
    assert set(as_dict) == {"status", "abstain_reason", "claims", "sources", "dropped", "timings_ms"}
    assert as_dict["claims"] == [{"text": copied_claim, "sources": [1]}]
    assert as_dict["dropped"] == {"uncited": 1, "fabricated": 0, "ungrounded": 0}
    assert dropped_claim not in blob
    assert claims_raw not in blob and ANSWERS_YES not in blob
    assert '"copied"' not in blob and '"raw"' not in blob


def test_an_answer_that_ends_empty_says_why():
    generator, _, claims = _gated(
        [ANSWERS_YES, ANSWERS_NO],
        [claims_json(("يرد المدير خلال خمسة أيام عمل وفقا للمادة 44.", [1]), ("بلا مصدر.", []))],
    )
    pipeline = Pipeline(_FakeLibrary(*_two_hits()), generator)

    every_claim_dropped = pipeline.ask("كم مهلة رد المدير؟")
    assert (every_claim_dropped.status, every_claim_dropped.abstain_reason) == ("abstained", "all_dropped")
    assert every_claim_dropped.claims == []
    assert every_claim_dropped.dropped == {"uncited": 1, "fabricated": 0, "ungrounded": 1}

    relevance_said_no = pipeline.ask("كم عدد موظفي الشركة؟")
    assert (relevance_said_no.status, relevance_said_no.abstain_reason) == ("abstained", "relevance_no")
    assert len(claims.seen) == 1, "the claims model was called after relevance said no"


def test_final_abstain_reason_names_every_way_an_answer_ends_empty():
    shown = ["يلتزم المدير بالرد خلال خمسة أيام عمل."]
    kept = {"text": "يرد المدير خلال خمسة أيام عمل.", "sources": [1]}
    uncited = {"text": "يرد المدير خلال خمسة أيام عمل.", "sources": []}
    nothing = {"abstain": True, "claims": []}

    def reason(generator_reason, parsed, numbers=(None,), texts=tuple(shown)):
        result = gate(parsed, list(numbers), list(texts))
        return final_abstain_reason(generator_reason, parsed, result), result["status"]

    assert reason("no_sources", nothing, numbers=(), texts=()) == ("no_sources", "abstained")
    assert reason("relevance_no", nothing) == ("relevance_no", "abstained")
    assert reason("relevance_failure", nothing) == ("relevance_failure", "abstained")
    assert reason(None, None) == ("schema_failure", "abstained")
    assert reason(None, {"abstain": True, "claims": [kept]}) == ("model_abstained", "abstained")
    assert reason(None, {"abstain": False, "claims": [uncited, uncited]}) == ("all_dropped", "abstained")
    assert reason(None, {"abstain": False, "claims": []}) == ("no_claims", "abstained")
    assert reason(None, {"abstain": False, "claims": [kept]}) == (None, "answered")
    assert reason(None, {"abstain": False, "claims": [kept, uncited]}) == (None, "partial")
    # The generator's own reason comes first, whatever else is true.
    assert reason("relevance_failure", None)[0] == "relevance_failure"


class _ClockedChat(ScriptedChat):
    """Advances a shared fake clock by its own `total_s` on every call, the
    way a real request's wall time would."""

    def __init__(self, replies, total_s, clock):
        super().__init__(replies, total_s=total_s)
        self._clock = clock

    def __call__(self, messages):
        self._clock["now"] += self.total_s
        return super().__call__(messages)


def test_timings_come_from_this_answers_own_calls(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(pipeline_mod, "perf_counter", lambda: clock["now"])
    relevance = _ClockedChat([ANSWERS_YES], total_s=0.25, clock=clock)
    claims = _ClockedChat(["{not json", ONE_GOOD_CLAIM], total_s=1.5, clock=clock)
    # Earlier answers' calls are still on the models: none of them is this answer's.
    relevance.calls.append({"total_s": 90.0, "stage": "relevance"})
    claims.calls.extend([{"total_s": 90.0}, {"total_s": 90.0}])

    class _SlowLibrary(_FakeLibrary):
        def search(self, query, k=5, doc_ids=None):
            clock["now"] += 0.12
            return super().search(query, k, doc_ids)

    pipeline = Pipeline(_SlowLibrary(*_two_hits()), ClaimsGenerator(model=claims, relevance_model=relevance))
    result = pipeline.ask("كم مهلة رد المدير؟")

    assert result.status == "answered"
    # One relevance call; an invalid claims reply plus its retry, summed.
    assert result.timings_ms == {"retrieval": 120, "relevance": 250, "claims": 3000, "total": 3370}


def test_a_question_outside_the_length_limits_is_rejected_before_retrieval():
    library = _FakeLibrary(*_two_hits())
    generator, relevance, claims = _gated([], [])
    pipeline = Pipeline(library, generator)

    too_short = "أ" * (MIN_QUESTION_CHARS - 1)
    for question in ("", "   ", too_short, f"   {too_short}   ", "س" * (MAX_QUESTION_CHARS + 1)):
        with pytest.raises(pipeline_mod.QuestionRejected) as rejected:
            pipeline.ask(question)
        assert rejected.value.code == "question_length"
    # Its own type, so a caller never confuses it with another ValueError — but still one.
    assert issubclass(pipeline_mod.QuestionRejected, ValueError)
    assert library.searches == [] and relevance.seen == [] and claims.seen == []

    # The limits themselves are allowed, measured after stripping.
    nothing_found = Pipeline(_FakeLibrary(), generator)
    for question in (f"  {'أ' * MIN_QUESTION_CHARS}  ", "س" * MAX_QUESTION_CHARS):
        assert nothing_found.ask(question).abstain_reason == "no_sources"


def test_ask_searches_with_the_pipelines_own_k_and_exactly_the_doc_ids_it_was_given():
    library = _FakeLibrary(*_two_hits())
    generator, _, _ = _gated([ANSWERS_YES, ANSWERS_YES], [ONE_GOOD_CLAIM, ONE_GOOD_CLAIM])
    pipeline = Pipeline(library, generator, k=3)

    pipeline.ask("كم مهلة رد المدير؟", [POLICY_ID])
    pipeline.ask("كم مهلة رد المدير؟")

    assert library.searches == [("كم مهلة رد المدير؟", 3, [POLICY_ID]), ("كم مهلة رد المدير؟", 3, None)]
    with pytest.raises(ValueError):
        Pipeline(library, generator, k=0)  # refused when built, never as a raw error out of ask


def test_a_damaged_document_reaches_the_caller_typed_never_as_a_raw_value_error(tmp_path):
    """A chunks file that no longer parses raises json's own ValueError deep in
    the library; out of `ask` it would read as a rejected question."""
    library = Library(tmp_path, encoder=KeywordEncoder())
    meta = library.add(text_document(POLICY_TEXT), "policy.txt")
    (tmp_path / "docs" / meta.doc_id / "chunks.jsonl").write_text("{not json\n", encoding="utf-8")
    generator, relevance, claims = _gated([], [])
    pipeline = Pipeline(library, generator)

    with pytest.raises(StorageError):
        pipeline.ask("كم مهلة رد المدير؟", [meta.doc_id])
    assert pipeline.ask("كم مهلة رد المدير؟").abstain_reason == "no_sources"
    assert relevance.seen == [] and claims.seen == []


def test_a_result_holding_a_lone_surrogate_still_encodes_as_utf8_json():
    """json.loads accepts a model's escaped lone surrogate ("\\ud800") and UTF-8
    cannot encode one, so a response carrying it would fail to send."""
    reply = '{"abstain": false, "claims": [{"text": "يرد المدير خلال خمسة أيام \\ud800 عمل.", "sources": [1]}]}'
    generator, _, _ = _gated([ANSWERS_YES], [reply])
    library = _FakeLibrary(LibraryHit(POLICY_CHUNK, "سياسة \udc80 العمل", 0.91))

    result = Pipeline(library, generator).ask("كم مهلة رد المدير؟")

    body = json.dumps(result.to_dict(), ensure_ascii=False).encode("utf-8")
    assert result.claims == [{"text": "يرد المدير خلال خمسة أيام \ufffd عمل.", "sources": [1]}]
    assert result.sources[0].doc_title == "سياسة \ufffd العمل"
    assert "\ufffd".encode("utf-8") in body


class _OverlapChat(ScriptedChat):
    """Counts calls that started while another call on this model was running."""

    def __init__(self, replies):
        super().__init__(replies)
        self.overlaps = 0
        self._running = 0
        self._guard = threading.Lock()

    def __call__(self, messages):
        with self._guard:
            self._running += 1
            if self._running > 1:
                self.overlaps += 1
        try:
            time.sleep(0.05)
            return super().__call__(messages)
        finally:
            with self._guard:
                self._running -= 1


def test_two_questions_at_once_never_share_a_model_or_its_call_stats():
    """`claims._stage_calls` reads a model's `calls` list by position, so two
    answers running at once on one model would each count the other's calls."""
    relevance = _OverlapChat([ANSWERS_YES, ANSWERS_YES])
    claims = _OverlapChat([ONE_GOOD_CLAIM, ONE_GOOD_CLAIM])
    pipeline = Pipeline(_FakeLibrary(*_two_hits()), ClaimsGenerator(model=claims, relevance_model=relevance))
    start = threading.Barrier(2, timeout=10)
    results = []

    def ask():
        start.wait()
        results.append(pipeline.ask("كم مهلة رد المدير؟"))

    threads = [threading.Thread(target=ask) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert (relevance.overlaps, claims.overlaps) == (0, 0)
    assert [r.status for r in results] == ["answered", "answered"]
    assert [(r.timings_ms["relevance"], r.timings_ms["claims"]) for r in results] == [(500, 500), (500, 500)]
