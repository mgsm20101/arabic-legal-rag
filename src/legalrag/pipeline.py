"""The app's question round trip.

    Library.search -> ClaimsGenerator.answer (relevance, then claims) -> cite.gate

`Pipeline.ask` returns a `ChatResult` ready to serialize (`to_dict`). It holds
only the claims that survived the gate, each citing its sources by the numbers
the model saw; raw model text and dropped claims never reach it. An empty
answer always says why (`claims.final_abstain_reason`).
"""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass
from time import perf_counter

from . import cite
from .claims import ClaimsGenerator, final_abstain_reason

TOP_K = 5
MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 500

# One generation in flight plus two callers queued: enough for a retried tab.
MAX_CONCURRENT_GENERATIONS = 3

# Covers waiting out one cold generation (~90s measured) ahead in the queue.
MAX_WAIT_SECONDS = 120.0

RETRY_AFTER_S = 5.0  # a fixed Retry-After hint, not a queue-depth estimate

_DROP_REASONS = ("uncited", "fabricated", "ungrounded")

# The only code points UTF-8 cannot encode. json.loads accepts one escaped in
# a model's reply, and a result holding it could not be sent as a response.
_SURROGATE = re.compile("[\ud800-\udfff]")


class QuestionRejected(ValueError):
    """A question outside MIN_QUESTION_CHARS..MAX_QUESTION_CHARS once stripped —
    its own type, so a caller never mistakes another ValueError for it."""

    code = "question_length"


class TooBusy(RuntimeError):
    """Too many callers already queued for generation, or this one waited past
    `max_wait_seconds`. Reported with web_guard's existing "rate_limited" code."""

    code = "rate_limited"

    def __init__(self, message: str, retry_after_s: float = RETRY_AFTER_S) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class Source:
    n: int               # 1-based: the number the model saw
    chunk_id: str
    doc_id: str
    doc_title: str
    label: str
    page: int
    article: int | None
    text: str


@dataclass(frozen=True)
class ChatResult:
    status: str                  # "answered" | "partial" | "abstained", from the gate
    abstain_reason: str | None
    claims: list[dict]           # [{"text": str, "sources": [int]}]: kept claims only
    sources: list[Source]        # every source shown to the model, in prompt order
    dropped: dict[str, int]      # {"uncited": n, "fabricated": n, "ungrounded": n}
    timings_ms: dict[str, int]   # {"retrieval", "relevance", "claims", "total"}

    def to_dict(self) -> dict:
        """JSON-ready: nothing but dicts, lists, strings, ints and None."""
        return asdict(self)


class Pipeline:
    """Retrieve, answer as claims, gate. `library` is anything with
    `Library.search`'s signature; `generator` is normally
    `claims.build_generators(spec, "gated")`.

    One Pipeline per generator. The lock that makes answers take turns belongs
    to the Pipeline, but the call stats it protects live on the generator's
    models: two Pipelines sharing one generator would mix them again.
    """

    def __init__(
        self, library, generator: ClaimsGenerator, k: int = TOP_K, *,
        max_concurrent_generations: int = MAX_CONCURRENT_GENERATIONS,
        max_wait_seconds: float = MAX_WAIT_SECONDS,
    ):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if max_concurrent_generations < 1:
            raise ValueError(
                f"max_concurrent_generations must be at least 1, got {max_concurrent_generations}"
            )
        self.library = library
        self.generator = generator
        self.k = k
        self.max_concurrent_generations = max_concurrent_generations
        self.max_wait_seconds = max_wait_seconds
        # Generation takes turns: `claims._stage_calls` reads an answer's calls by
        # position in the model's shared `calls` list.
        self._generating = threading.Lock()
        # Caps callers waiting for or holding `_generating`; the next one gets TooBusy.
        self._admission = threading.BoundedSemaphore(max_concurrent_generations)

    def ask(self, question: str, doc_ids: list[str] | None = None) -> ChatResult:
        """Answer `question` from every live document, or from exactly `doc_ids`.

        Every anticipated failure arrives as one of these, never as a raw
        library, parser or HTTP error:
        - QuestionRejected, before anything is retrieved, when the stripped
          question is outside MIN_QUESTION_CHARS..MAX_QUESTION_CHARS;
        - `library.DocumentNotFound` for a malformed, unknown or deleted
          document in `doc_ids`, `library.StorageError` for a damaged one, or
          another `library.LibraryError`;
        - `library.EncoderUnavailable` when the embedding model cannot load;
        - `ollama.GeneratorUnavailable` when the model server cannot be
          reached or refuses the request;
        - `TooBusy` when `max_concurrent_generations` callers are already
          waiting for (or holding) the generation lock, or when this caller
          waited past `max_wait_seconds` for its own turn at it.
        """
        # Before the clock starts: a one-time model load is not this request's
        # wait (see DECISIONS.md ADR-031). Normally a no-op; the app warms at startup.
        warm = getattr(self.library, "warm_encoder", None)
        if callable(warm):
            warm()

        started = perf_counter()
        question = question.strip()
        if not MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS:
            raise QuestionRejected(
                f"a question must be {MIN_QUESTION_CHARS} to {MAX_QUESTION_CHARS} "
                f"characters long, not {len(question)}"
            )

        searching = perf_counter()
        hits = self.library.search(question, self.k, doc_ids)
        retrieval_s = perf_counter() - searching
        sources = [
            Source(n=n, chunk_id=hit.chunk.id, doc_id=hit.chunk.doc_id, doc_title=_encodable(hit.doc_title),
                   label=_encodable(hit.chunk.label), page=hit.chunk.page, article=hit.chunk.article,
                   text=_encodable(hit.chunk.text))
            for n, hit in enumerate(hits, start=1)
        ]
        texts = [s.text for s in sources]

        answer = self._generate(question, texts, started)

        # A page chunk's article is None; a claim naming «مادة N» its sources do
        # not name is dropped as ungrounded, as for a statute.
        verdict = cite.gate(answer.parsed, [s.article for s in sources], texts)
        return ChatResult(
            status=verdict["status"],
            abstain_reason=final_abstain_reason(answer.abstain_reason, answer.parsed, verdict),
            claims=[{"text": _encodable(c["text"]), "sources": list(c["sources"])} for c in verdict["kept"]],
            sources=sources,
            dropped={reason: verdict[reason] for reason in _DROP_REASONS},
            timings_ms={
                "retrieval": _ms(retrieval_s),
                "relevance": _ms(_stage_seconds(answer.calls, "relevance")),
                "claims": _ms(_stage_seconds(answer.calls, "claims")),
                "total": _ms(perf_counter() - started),
            },
        )

    def _generate(self, question: str, texts: list[str], started: float):
        """`self.generator.answer` under `_generating`, or `TooBusy` before any call when
        the queue is full or the caller has waited past `max_wait_seconds` since `started`."""
        if not self._admission.acquire(blocking=False):
            raise TooBusy(
                f"{self.max_concurrent_generations} callers are already waiting to "
                "generate an answer; try again shortly"
            )
        try:
            with self._generating:
                waited_s = perf_counter() - started
                if waited_s > self.max_wait_seconds:
                    raise TooBusy(
                        f"waited {waited_s:.1f}s for a generation slot, past the "
                        f"{self.max_wait_seconds:.0f}s limit; try again shortly"
                    )
                answer = self.generator.answer(question, texts)
                # Keeps `calls` to one request's worth. Must run inside the lock, or it
                # could wipe the calls of the next request already generating.
                self._reset_call_history()
                return answer
        finally:
            self._admission.release()

    def _reset_call_history(self) -> None:
        """Clear `.calls` on the claims and relevance models; a model without one is left alone."""
        models = (self.generator.model, getattr(self.generator, "relevance_model", None))
        for model in models:
            if getattr(model, "calls", None) is not None:
                model.calls = []


def _stage_seconds(calls: list[dict], stage: str) -> float:
    """Seconds this answer's own `stage` calls took (`total_s`; missing counts as zero)."""
    return sum(c.get("total_s") or 0.0 for c in calls if c.get("stage") == stage)


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _encodable(text: str) -> str:
    """`text` with every lone surrogate replaced by U+FFFD, so any JSON encoder can send it."""
    return _SURROGATE.sub("\ufffd", text)
