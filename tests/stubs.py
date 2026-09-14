"""Test doubles shared by the upload-pipeline tests (Phase 4B).

`tests/` is not a package, so a test file puts this directory on `sys.path`
itself, beside the `src` insertion every test file already makes, and
imports `stubs` as a top-level module. `StubEncoder` (test_dense.py) and
`StubModel` (test_generate.py) stay where they are.

Nothing here imports `legalrag`: plain doubles, so the order of the two
`sys.path` insertions never matters.
"""

from __future__ import annotations

import json
import threading

DEFAULT_KEYWORDS = ("العمل", "الموظف", "الشركة", "الإنترنت", "الإجازة", "الأمن")

# Padding for `text_document`: contains none of the keywords any test uses.
FILLER = " ".join(["نص"] * 110)


class KeywordEncoder:
    """A sentence-transformers stand-in whose vectors can be worked out by
    hand: one dimension per keyword, holding how many times that keyword
    occurs in the text. Texts sharing keywords score higher, so a search
    order means something — with no model and no network.

    Records every text it was asked to encode, in order, across threads.
    """

    def __init__(self, keywords=DEFAULT_KEYWORDS):
        self.keywords = tuple(keywords)
        self.seen: list[str] = []
        self._lock = threading.Lock()

    def encode(self, texts, **kwargs):
        import numpy as np

        texts = list(texts)
        with self._lock:
            self.seen.extend(texts)
        vectors = np.zeros((len(texts), len(self.keywords)), dtype="float32")
        for row, text in enumerate(texts):
            for column, keyword in enumerate(self.keywords):
                vectors[row, column] = text.count(keyword)
        return vectors

    def passages(self) -> list[str]:
        """Every text embedded as a document passage, not as a query."""
        with self._lock:
            return [t for t in self.seen if t.startswith("passage: ")]


class ScriptedChat:
    """A `messages -> str` chat model that replies from a script, in order,
    and keeps one Ollama-shaped stats dict per call in `calls` — the list
    `claims.ask_json` stage-tags. A script entry that is an exception is
    raised instead, before any stats are kept, the way `OllamaChat` fails."""

    def __init__(self, replies, total_s: float = 0.5):
        self._replies = list(replies)
        self.total_s = total_s
        self.calls: list[dict] = []
        self.seen: list[list[dict]] = []

    def __call__(self, messages):
        self.seen.append(messages)
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        self.calls.append({"output_tokens": 1, "total_s": self.total_s})
        return reply


ANSWERS_YES = json.dumps({"answers": True})
ANSWERS_NO = json.dumps({"answers": False})


def claims_json(*claims: tuple[str, list[int]], abstain: bool = False) -> str:
    """A claims-contract reply: `claims_json(("نص الجملة", [1]), ...)`."""
    return json.dumps(
        {"abstain": abstain, "claims": [{"text": t, "sources": s} for t, s in claims]},
        ensure_ascii=False,
    )


def text_document(*pages: str) -> bytes:
    """The bytes of a UTF-8 .txt upload: `pages` joined by form feeds, each
    padded with `FILLER` so any document clears the library's
    minimum-text floor and stays one chunk per page."""
    return "\f".join(f"{page} {FILLER}" for page in pages).encode("utf-8")
