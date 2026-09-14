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

from legalrag.answer_report import report  # noqa: E402


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


def _run(rows, meta=None):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = report(rows, meta=meta)
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
    assert "  calls truncated by num_ctx         : 1" in out
    assert "  calls cut by the token cap         : 1" in out
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


def test_the_report_prints_seconds_excluding_model_load_when_load_s_is_present():
    """`load_s` is the one-time weight-load cost Ollama folds into a call's
    own timing. Left in, it inflates "mean seconds per answer" for any
    run that loads the model mid-eval; this line is what a reader actually
    compares against the hf: runtime's number, which never pays a load cost
    partway through a run. The line must only appear once some call in the
    run actually carries a `load_s` — see the dict-shaped-stats test above,
    which has none and prints no such line at all."""
    rows = [
        _row("Q1", cited=[7], strict=[7], stats=[
            {"output_tokens": 10, "output_s": 1.0, "load_s": 2.0},
        ]),
    ] + OOC
    rows[0]["seconds"] = 9.0  # 9.0 - 2.0 load = 7.0; OOC rows contribute 1.0 each (5 of them)
    _, out = _run(rows)

    assert "  mean seconds excl. load            : 2.0" in out  # (7.0 + 5*1.0) / 6


def test_the_report_prints_gpu_share_quantization_and_digest_from_meta():
    """The meta lines come from the `meta` argument now, not from `report`
    reading `runs/` itself — passing a plain dict must be enough to see all
    three lines, with no file on disk at all."""
    rows = [_row("Q1", cited=[7], strict=[7])] + OOC
    meta = {"gpu_share": 0.55, "quantization": "Q4_K_M", "digest": "abcdef1234567890"}

    _, out = _run(rows, meta=meta)

    assert "  GPU share                          : 55%" in out
    assert "  quantization                       : Q4_K_M" in out
    assert "  digest                             : abcdef123456" in out  # first 12 chars
