"""Verdict tests for the M2 report — PRD M2/B1 and M2/B2.

Both criteria have a way of being satisfied by a system that does nothing
useful, and the first run of this eval fell into one of them: it printed
"B1: PASS" over 12 of 14 answers that carried no citation at all.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.answer_eval import (  # noqa: E402
    DEFAULT_MODEL,
    ROWS_PATH,
    _stats_for,
    _weights_line,
    report,
    rows_path,
    run,
)
from legalrag.evaluate import Question  # noqa: E402
from legalrag.generate import Generator  # noqa: E402
from legalrag.retrieve import Hit  # noqa: E402


def _row(qid, *, answerable=True, abstained=False, cited=(), strict=(),
         fabricated=(), ungrounded=(), uncited=(), expected=(), stats=None):
    row = {
        "id": qid, "category": "direct", "answerable": answerable,
        "abstained": abstained, "cited": list(cited), "strict": list(strict),
        "fabricated": list(fabricated), "ungrounded": list(ungrounded),
        "uncited": list(uncited), "expected": list(expected),
        "cited_expected": bool(set(expected) & set(cited)),
        "grounded": not (fabricated or ungrounded or uncited),
        "seconds": 1.0, "text": "", "retrieved": [7],
    }
    if stats is not None:
        row["stats"] = stats
    return row


def _run(rows):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = report(rows)
    return code, buf.getvalue()


OOC = [_row(f"Q{i}", answerable=False, abstained=True) for i in range(16, 21)]


def test_an_answer_with_no_citation_fails_b1():
    """Not citing is not the same as not lying. A model that never cites has
    zero fabricated and zero ungrounded citations, and grounds nothing."""
    rows = [_row("Q1", uncited=["ادعاء بلا سند"])] + OOC
    code, out = _run(rows)
    assert "B1: FAIL" in out
    assert code == 1


def test_a_fabricated_citation_fails_b1():
    rows = [_row("Q1", cited=[78], fabricated=[78])] + OOC
    assert "B1: FAIL" in _run(rows)[1]


def test_a_real_but_unretrieved_citation_fails_b1():
    rows = [_row("Q1", cited=[30], ungrounded=[30])] + OOC
    assert "B1: FAIL" in _run(rows)[1]


def test_a_cited_grounded_answer_passes_b1():
    rows = [_row("Q1", cited=[7], strict=[7], expected=[7])] + OOC
    code, out = _run(rows)
    assert "B1: PASS" in out
    assert code == 0


def test_abstaining_on_everything_is_visible_next_to_b2():
    """B2 alone reads 100%. The line under it is what makes that readable."""
    rows = [_row(f"Q{i}", abstained=True) for i in range(1, 6)] + OOC
    _, out = _run(rows)
    assert "5/5 = 100.0%" in out          # B2 looks perfect
    assert "5/5 = 100.0%" in out.split("abstained on answerable")[1][:40]


def test_b2_fails_below_the_threshold():
    ooc = [_row("Q16", answerable=False, abstained=True),
           _row("Q17", answerable=False, abstained=False, cited=[7])]
    rows = [_row("Q1", cited=[7], strict=[7])] + ooc
    assert "B2: FAIL" in _run(rows)[1]     # 1/2 = 50% < 80%


def test_each_model_writes_its_own_rows_file_and_the_default_keeps_run_3s():
    """Run 4 changes only the model, one candidate at a time — if every
    candidate wrote to `runs/answer_eval.json` the way Run 3 did, the second
    model's run would silently overwrite the first's saved answers, and
    `--report-only` could no longer reproduce Run 3 at all."""
    assert rows_path("hf:" + DEFAULT_MODEL) == ROWS_PATH

    assert rows_path("ollama:qwen3:4b") == Path("runs/answer_eval-ollama-qwen3-4b.json")
    assert rows_path("ollama:qwen2.5:1.5b-instruct") == Path(
        "runs/answer_eval-ollama-qwen2.5-1.5b-instruct.json"
    )
    # Every character outside [A-Za-z0-9._-] becomes '-' — only the ':' and
    # '/' here qualify, so the dots in a version-like name pass through
    # untouched. A non-default `hf:` spec gets its own file too: the default
    # exemption is for one exact string, not the whole "hf:" prefix.
    assert rows_path("hf:Qwen/Qwen2.5-3B-Instruct") == Path(
        "runs/answer_eval-hf-Qwen-Qwen2.5-3B-Instruct.json"
    )


def test_a_question_answered_without_calling_the_model_carries_no_stats():
    """A question the model was not called for must report `[]`, never a
    previous question's numbers — and a runtime with no `calls` list at all
    (the hf: runtime is a plain function) must report None, not crash.

    `before` is a snapshot of `len(model.calls)` taken just before the
    question was asked (see `run`); `_stats_for` no longer looks at
    `Generator` or `articles` at all, so it stays correct once a single
    question can make more than one call (Run 5's retry)."""
    class NoCallsModel:
        """The hf: runtime's shape: a plain callable, no `calls` list."""

    class CallTrackingModel:
        def __init__(self):
            self.calls = [{"output_tokens": 99, "output_s": 1.0, "cut": True}]

    assert _stats_for(NoCallsModel(), before=0) is None

    tracked = CallTrackingModel()
    # Nothing was appended after this snapshot -> no calls for this question.
    assert _stats_for(tracked, before=len(tracked.calls)) == []
    # Everything from the start -> the one call already recorded.
    assert _stats_for(tracked, before=0) == [
        {"output_tokens": 99, "output_s": 1.0, "cut": True},
    ]


def test_an_hf_spec_with_no_repo_name_prints_no_weights_line():
    """`--model hf:` names no repo — `resolve_model` rejects it, but only
    later, inside the timed "loading ..." block in main(). Printing a
    weights line for it first (from calling `model_source` on an empty or
    whitespace-only name) would show a blank `weights    :` line — or worse,
    since `Path("").is_dir()` can be True — before that real error ever
    appears."""
    assert _weights_line("hf:") is None
    assert _weights_line("hf:   ") is None
    assert _weights_line("ollama:qwen3:4b") is None  # not an hf: spec at all


def test_an_hf_spec_with_a_repo_name_prints_a_weights_line():
    """The ordinary case still gets its provenance line, whichever branch of
    `model_source` ends up resolving it on this machine."""
    line = _weights_line("hf:Qwen/Qwen2.5-1.5B-Instruct")
    assert line is not None
    assert line.startswith("weights    : ")


def _question(qid, question, *, answerable=True, category="direct"):
    return Question(
        id=qid, category=category, question=question,
        expected_articles=[], expected_keywords=[],
        answerable=answerable, ref_status="verified",
    )


class _StubIndex:
    """Returns fixed hits keyed by question text. A question mapped to `[]`
    means nothing was retrieved for it at all — not even an issuance
    article — so `run` builds an empty `articles` list for it."""

    def __init__(self, hits_by_question):
        self._hits_by_question = hits_by_question

    def search(self, query, k):
        return self._hits_by_question.get(query, [])


class _StubCallModel:
    """A model whose `calls` list grows by one caller-supplied stats dict
    per call, consumed in order — the shape `_stats_for`/`run` rely on."""

    def __init__(self, replies):
        self._replies = list(replies)  # [(reply_text, stats_dict), ...]
        self.calls: list[dict] = []

    def __call__(self, messages):
        text, stats = self._replies.pop(0)
        self.calls.append(stats)
        return text


def test_run_attaches_each_questions_own_model_calls_to_its_row():
    """Each row's stats must be exactly the calls made answering THAT
    question — not the model's cumulative state, and not another question's
    numbers. Q3 retrieves no law article at all, so the model is never
    called for it and its stats must be `[]`, not stale or missing."""
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    hit7 = Hit(id="law-7", law_name="", number=7, score=1.0, snippet="")

    index = _StubIndex({"س1": [hit7], "س2": [hit7], "س3": []})
    model = _StubCallModel([
        ("جواب1 [مادة 7].", {"output_tokens": 10, "output_s": 1.0}),
        ("جواب2 [مادة 7].", {"output_tokens": 20, "output_s": 2.0}),
    ])
    generator = Generator(model=model)
    questions = [
        _question("Q1", "س1"),
        _question("Q2", "س2"),
        _question("Q3", "س3", answerable=False, category="out_of_corpus"),
    ]

    rows = run(questions, docs, generator, "ollama:stub-test-model", index=index)

    assert rows[0]["stats"] == [{"output_tokens": 10, "output_s": 1.0}]
    assert rows[1]["stats"] == [{"output_tokens": 20, "output_s": 2.0}]
    assert rows[2]["stats"] == []


def test_report_prints_the_runtime_lines_from_per_call_stats():
    """Aggregation is per call, not per row: a cut call and a truncated call
    in two different rows must both count, and the tokens/s line is the sum
    of output over the sum of seconds across every call."""
    rows = [
        _row("Q1", cited=[7], strict=[7], stats=[
            {"output_tokens": 10, "output_s": 2.0, "truncated": True, "cut": False},
        ]),
        _row("Q2", cited=[7], strict=[7], stats=[
            {"output_tokens": 30, "output_s": 3.0, "truncated": False, "cut": True},
        ]),
    ] + OOC
    _, out = _run(rows)

    assert "  mean output tokens/s               : 8.0" in out  # (10+30)/(2+3)
    assert "  prompts truncated by num_ctx       : 1" in out
    assert "  answers cut by the token cap       : 1" in out
    assert "  model calls                        : 2" in out
    assert "  retries (more than 1 call)         : 0" in out


def test_report_counts_retries_as_calls_beyond_the_first_per_question():
    rows = [
        _row("Q1", cited=[7], strict=[7], stats=[
            {"output_tokens": 5, "output_s": 1.0, "done_reason": "length"},
            {"output_tokens": 15, "output_s": 1.0, "done_reason": "stop"},
        ]),
    ] + OOC
    _, out = _run(rows)

    assert "  model calls                        : 2" in out
    assert "  retries (more than 1 call)         : 1" in out


def test_report_still_reads_run_4_rows_whose_stats_are_a_dict():
    """Run 4's saved rows carry `stats` as a single dict, not a list —
    `report()` must keep reading those exactly as before this feature."""
    rows = [
        _row("Q1", cited=[7], strict=[7], stats={
            "output_tokens": 67, "output_s": 5.0, "truncated": False, "cut": False,
        }),
    ] + OOC
    _, out = _run(rows)

    assert "  mean output tokens/s               : 13.4" in out
    assert "  model calls                        : 1" in out
    assert "  retries (more than 1 call)         : 0" in out
