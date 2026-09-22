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
    TooBusy,
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


class _BlockingChat(ScriptedChat):
    """Blocks in `__call__` until `release` is set, so a test can hold
    `Pipeline._generating` open for exactly as long as it needs — `entered`
    (if given) is set the instant the call starts, proving the caller
    already holds the lock at that point."""

    def __init__(self, replies, release: threading.Event, entered: threading.Event | None = None):
        super().__init__(replies)
        self._release = release
        self._entered = entered

    def __call__(self, messages):
        if self._entered is not None:
            self._entered.set()
        self._release.wait(timeout=10)
        return super().__call__(messages)


def test_a_caller_past_the_admission_bound_is_refused_immediately_not_queued():
    """T11 finding: the generation lock had no admission bound at all — any
    number of callers could pile up waiting on it forever. At most
    `max_concurrent_generations` may now wait for (or hold) it; the next one
    must be refused right away, not after joining a queue."""
    release = threading.Event()
    entered = threading.Event()
    chat = _BlockingChat([ONE_GOOD_CLAIM, ONE_GOOD_CLAIM], release=release, entered=entered)
    pipeline = Pipeline(_FakeLibrary(*_two_hits()), ClaimsGenerator(model=chat),
                        max_concurrent_generations=2, max_wait_seconds=10.0)
    results: list = []
    errors: list = []

    def ask():
        try:
            results.append(pipeline.ask("كم مهلة رد المدير؟"))
        except TooBusy as e:
            errors.append(e)

    first = threading.Thread(target=ask)
    first.start()
    assert entered.wait(timeout=5), "the first caller must reach the model"  # now holds _generating

    second = threading.Thread(target=ask)
    second.start()
    time.sleep(0.05)  # the second caller has time to be admitted and start waiting on the lock

    before = time.perf_counter()
    with pytest.raises(TooBusy) as excinfo:
        pipeline.ask("كم مهلة رد المدير؟")
    elapsed = time.perf_counter() - before

    assert elapsed < 0.5, "the (N+1)th caller must be refused immediately, never by waiting for the lock"
    assert excinfo.value.code == "rate_limited"
    assert excinfo.value.retry_after_s > 0

    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert [r.status for r in results] == ["answered", "answered"]
    assert errors == []


def test_a_caller_that_already_waited_past_the_deadline_never_reaches_the_model(monkeypatch):
    """T11 finding: nothing expired a caller who had already been waiting a
    long time once its turn at the lock finally came — an expensive
    generation call would start anyway for someone who may well be gone."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(pipeline_mod, "perf_counter", lambda: clock["now"])

    class _SlowLibrary(_FakeLibrary):
        def search(self, query, k=5, doc_ids=None):
            clock["now"] += 999.0  # far past any max_wait_seconds used below
            return super().search(query, k, doc_ids)

    chat = ScriptedChat([ONE_GOOD_CLAIM])
    pipeline = Pipeline(_SlowLibrary(*_two_hits()), ClaimsGenerator(model=chat), max_wait_seconds=10.0)

    with pytest.raises(TooBusy) as excinfo:
        pipeline.ask("كم مهلة رد المدير؟")

    assert excinfo.value.code == "rate_limited"
    assert chat.seen == [], "the model must never be called once the deadline has already passed"

    # The admission slot and the lock were both released immediately, not
    # leaked: a fresh, fast request right after must still go through.
    clock["now"] = 2000.0
    pipeline.library = _FakeLibrary(*_two_hits())
    assert pipeline.ask("كم مهلة رد المدير؟").status == "answered"


class _DistinctOverlapChat(_OverlapChat):
    """Like `_OverlapChat`, but each call in the script gets its own
    `total_s` instead of one shared value — so if two overlapping answers'
    calls were ever mixed up, the resulting timing would belong to neither
    answer, instead of two identical numbers hiding the mix-up."""

    def __init__(self, replies, total_s_by_call):
        super().__init__(replies)
        self._total_s_by_call = list(total_s_by_call)

    def __call__(self, messages):
        self.total_s = self._total_s_by_call.pop(0)
        return super().__call__(messages)


def test_call_history_is_cleared_before_the_generation_lock_is_released_not_after():
    """T11's critical ordering constraint: `_reset_call_history` must run
    while `_generating` is still held, never after it is released — if it
    ran after, a second, already-waiting request could start appending to
    `model.calls` before the first request's clear ran, and that clear would
    then wipe out the second request's own in-flight calls (corrupting the
    `timings_ms` `claims._stage_calls` derives from `model.calls` by
    *position*).

    A genuine two-thread race is not a reliable way to prove this: the reset
    is one line of pure-Python bytecode right after the lock's release, with
    no I/O in between, so a buggy "release, then reset" ordering essentially
    never actually loses the OS scheduling race in practice (verified by
    hand: 30/30 runs of a barrier-started two-thread version of this test
    still passed against a deliberately reintroduced "reset after release"
    bug). This instead records the real order `_generate` performs "reset"
    and "release" in, which fails the instant that order is ever reversed,
    regardless of scheduling luck."""
    order: list[str] = []
    real_lock = threading.Lock()

    class _OrderRecordingLock:
        def __enter__(self):
            real_lock.acquire()
            return self

        def __exit__(self, *exc_info):
            real_lock.release()
            order.append("release")

    generator = ClaimsGenerator(model=ScriptedChat([ONE_GOOD_CLAIM]))
    pipeline = Pipeline(_FakeLibrary(*_two_hits()), generator)
    pipeline._generating = _OrderRecordingLock()
    real_reset = pipeline._reset_call_history

    def recording_reset():
        order.append("reset")
        real_reset()

    pipeline._reset_call_history = recording_reset

    result = pipeline.ask("كم مهلة رد المدير؟")

    assert result.status == "answered"
    assert order == ["reset", "release"], (
        "the calls-history reset must be recorded before the lock's release, never after"
    )


def test_clearing_a_finished_answers_calls_never_corrupts_an_overlapping_answers_timings():
    """A supplementary, best-effort concurrency check alongside the
    deterministic ordering test above: two real overlapping `ask()` calls
    must still never share a model's calls, and `OllamaChat.calls` (T11
    finding: it grew for the app's whole lifetime) is bounded back to empty
    after each one, not left to accumulate. Two distinct `total_s` values
    make any mix-up between the two answers show up as a wrong number,
    instead of two coincidentally-equal ones hiding it."""
    claims = _DistinctOverlapChat([ONE_GOOD_CLAIM, ONE_GOOD_CLAIM], total_s_by_call=[0.05, 0.09])
    pipeline = Pipeline(_FakeLibrary(*_two_hits()), ClaimsGenerator(model=claims))
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

    assert claims.overlaps == 0, "the two answers' model calls must still never overlap"
    assert [r.status for r in results] == ["answered", "answered"]
    assert sorted(r.timings_ms["claims"] for r in results) == [50, 90], (
        "each answer must keep exactly its own call's timing, never the other's, and never zero"
    )
    assert claims.calls == [], "T11: bounded back to empty after the second answer too, not left to accumulate"


# --- the one-time encoder load is not this request's cost --------------------
#
# Observed in the container's first production run: `/api/chat` answered 429
# after 244.9s, saying «waited 244.9s for a generation slot, past the 120s
# limit». No caller was queued and the lock was free the whole time. The 244.9s
# was `LazyEncoder` downloading the 1.1 GB e5 encoder on its first `encode`,
# inside retrieval, inside the budget that starts when `ask` does.
#
# T11 asked for an end-to-end budget and that is what stays. What changes is
# what the budget covers: loading a model once for the life of the process is
# the process starting up, not the request queueing, and a request that happens
# to be the first one must not be billed for it. `ask` now waits for the
# encoder to be ready BEFORE starting its clock, and the app warms it off the
# request path at startup so that wait is normally zero.


class _ColdLibrary(_FakeLibrary):
    """A library whose encoder is not loaded yet, and takes `load_s` to load."""

    def __init__(self, *hits, clock, load_s):
        super().__init__(*hits)
        self.clock, self.load_s, self.warmed = clock, load_s, 0

    def warm_encoder(self):
        self.warmed += 1
        if self.warmed == 1:
            self.clock["now"] += self.load_s


def test_a_cold_encoder_load_is_not_charged_to_the_generation_budget(monkeypatch):
    """The reported bug, reproduced: nothing is queued, the lock is free, and
    the only slow thing is a one-time model load."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(pipeline_mod, "perf_counter", lambda: clock["now"])
    chat = ScriptedChat([ONE_GOOD_CLAIM])
    library = _ColdLibrary(*_two_hits(), clock=clock, load_s=244.9)
    pipeline = Pipeline(library, ClaimsGenerator(model=chat), max_wait_seconds=120.0)

    result = pipeline.ask("كم مهلة رد المدير؟")

    assert result.status == "answered"
    assert chat.seen, "the model must be reached: no one was queued and the lock was free"
    assert library.warmed == 1


def test_the_encoder_is_waited_for_on_every_ask_not_only_the_first(monkeypatch):
    """Cheap once warm, and the only thing that makes the first ask's wait
    zero in production is that startup already did it — so `ask` must keep
    asking rather than trusting a flag it set itself."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(pipeline_mod, "perf_counter", lambda: clock["now"])
    library = _ColdLibrary(*_two_hits(), clock=clock, load_s=1.0)
    pipeline = Pipeline(library, ClaimsGenerator(model=ScriptedChat([ONE_GOOD_CLAIM, ONE_GOOD_CLAIM])))

    pipeline.ask("كم مهلة رد المدير؟")
    pipeline.ask("كم مهلة رد المدير؟")

    assert library.warmed == 2


def test_a_library_that_cannot_be_warmed_is_left_alone(monkeypatch):
    """Injected fakes and any other library without the method must still
    work -- the same `getattr` shape this codebase already uses for `.close`
    and `.calls`."""
    pipeline = Pipeline(_FakeLibrary(*_two_hits()), ClaimsGenerator(model=ScriptedChat([ONE_GOOD_CLAIM])))
    assert pipeline.ask("كم مهلة رد المدير؟").status == "answered"


def test_slow_retrieval_still_spends_the_budget(monkeypatch):
    """The fix must not turn the budget off. Retrieval is this request's own
    work, however slow, and T11's end-to-end budget still covers it -- which
    is what `test_a_caller_that_already_waited_past_the_deadline...` above
    asserts through `_SlowLibrary`. Restated here next to its exception so
    the boundary between them is visible in one place."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(pipeline_mod, "perf_counter", lambda: clock["now"])

    class _SlowSearch(_ColdLibrary):
        def search(self, query, k=5, doc_ids=None):
            self.clock["now"] += 999.0
            return super().search(query, k, doc_ids)

    chat = ScriptedChat([ONE_GOOD_CLAIM])
    library = _SlowSearch(*_two_hits(), clock=clock, load_s=0.0)
    pipeline = Pipeline(library, ClaimsGenerator(model=chat), max_wait_seconds=10.0)

    with pytest.raises(TooBusy):
        pipeline.ask("كم مهلة رد المدير؟")
    assert chat.seen == []
