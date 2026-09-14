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
)
from legalrag.generate import Generator  # noqa: E402


def _row(qid, *, answerable=True, abstained=False, cited=(), strict=(),
         fabricated=(), ungrounded=(), uncited=(), expected=()):
    return {
        "id": qid, "category": "direct", "answerable": answerable,
        "abstained": abstained, "cited": list(cited), "strict": list(strict),
        "fabricated": list(fabricated), "ungrounded": list(ungrounded),
        "uncited": list(uncited), "expected": list(expected),
        "cited_expected": bool(set(expected) & set(cited)),
        "grounded": not (fabricated or ungrounded or uncited),
        "seconds": 1.0, "text": "", "retrieved": [7],
    }


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
    """`Generator.answer` returns early on empty `articles` (the abstention
    rule) without ever touching the model, so `last_stats` on the model
    object is stale — left over from whatever question was last actually
    answered. Reading it unconditionally would silently mislabel this row
    with someone else's tokens/s, truncation and cut flags."""
    class StaleStatsModel:
        """A model whose `last_stats` predates this call — it is never
        invoked here, exactly like the real runtime on an empty-articles
        question."""
        last_stats = {"output_tokens": 99, "output_s": 1.0, "cut": True}

    g = Generator(model=StaleStatsModel())

    assert _stats_for(g, []) is None
    assert _stats_for(g, [{"number": 7, "text": "نص المادة"}]) == {
        "output_tokens": 99, "output_s": 1.0, "cut": True,
    }


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
