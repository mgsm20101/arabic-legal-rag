"""Ask a question of the uploaded documents: the P1 demo layer's round trip
(ADR-023, ADR-025).

    Library.search -> ClaimsGenerator.answer (relevance, then claims) -> cite.gate

`Pipeline.ask` returns a `ChatResult` an HTTP layer can serialize as it is
(`to_dict`). It holds only the claims that survived the gate, each pointing
at its sources by the numbers the model saw. What the gate or the contract
kept out of view never gets in: the raw model text, the relevance reply, a
dropped claim's text, or whether a kept claim was copied. An empty answer
always says why (`claims.final_abstain_reason`).
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

_DROP_REASONS = ("uncited", "fabricated", "ungrounded")

# The only code points UTF-8 cannot encode. json.loads accepts one escaped in
# a model's reply, and a result holding it could not be sent as a response.
_SURROGATE = re.compile("[\ud800-\udfff]")


class QuestionRejected(ValueError):
    """A question outside MIN_QUESTION_CHARS..MAX_QUESTION_CHARS once stripped —
    its own type, so a caller never mistakes another ValueError for it."""

    code = "question_length"


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

    def __init__(self, library, generator: ClaimsGenerator, k: int = TOP_K):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.library = library
        self.generator = generator
        self.k = k
        # `claims._stage_calls` finds an answer's calls by their position in
        # the model's `calls` list, so two answers generated at once on one
        # model would each count the other's. Generation takes turns.
        self._generating = threading.Lock()

    def ask(self, question: str, doc_ids: list[str] | None = None) -> ChatResult:
        """Answer `question` from every live document, or from exactly `doc_ids`.

        Every anticipated failure arrives as one of these, never as a raw
        library, parser or HTTP error:
        - QuestionRejected, before anything is retrieved, when the stripped
          question is outside MIN_QUESTION_CHARS..MAX_QUESTION_CHARS;
        - `library.DocumentNotFound` for a malformed, unknown, deleted or
          unreadable document in `doc_ids`, or another `library.LibraryError`;
        - `library.EncoderUnavailable` when the embedding model cannot load;
        - `ollama.GeneratorUnavailable` when the model server cannot be
          reached or refuses the request.
        """
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

        with self._generating:
            answer = self.generator.answer(question, texts)

        # A page chunk's article is None, so a claim naming «مادة N» that none
        # of its cited sources' text names is dropped as ungrounded — exactly
        # as it would be for a statute's article.
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


def _stage_seconds(calls: list[dict], stage: str) -> float:
    """Time this answer spent in `stage`: its own stage-tagged calls'
    `total_s` (Ollama's whole-request duration), never the model's running
    `calls` list, which holds every earlier answer's calls too. A call that
    reports no duration counts as zero."""
    return sum(c.get("total_s") or 0.0 for c in calls if c.get("stage") == stage)


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _encodable(text: str) -> str:
    """`text` with every lone surrogate replaced by U+FFFD, so any JSON encoder can send it."""
    return _SURROGATE.sub("\ufffd", text)
