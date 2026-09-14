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

from legalrag.answer_report import _cites_expected, report, report_claims  # noqa: E402


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


# --- Run 5's claims report (--contract json), EVAL.md commit ecd37f8 -------


def _claims_row(qid, *, answerable=True, expected=(), source_numbers=(),
                 kept=(), dropped=(), fabricated=0, uncited=0, ungrounded=0,
                 ignored_on_abstain=0, status=None, schema_failure=False,
                 attempts=1, seconds=1.0):
    kept = list(kept)
    dropped = list(dropped)
    if status is None:
        # Mirrors `cite.gate`'s own status rule: nothing kept is an
        # abstention regardless of why; kept and dropped together is
        # partial; kept with nothing dropped is answered.
        status = "abstained" if not kept else ("partial" if dropped else "answered")
    return {
        "id": qid, "answerable": answerable, "expected": list(expected),
        "source_numbers": list(source_numbers), "model": "ollama:gemma3:4b",
        "contract": "json", "attempts": attempts, "schema_failure": schema_failure,
        "seconds": seconds,
        "gate": {
            "status": status, "kept": kept, "dropped": dropped,
            "uncited": uncited, "fabricated": fabricated, "ungrounded": ungrounded,
            "ignored_on_abstain": ignored_on_abstain,
        },
    }


def _claims_run(rows, meta=None, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = report_claims(rows, meta=meta, **kwargs)
    return code, buf.getvalue()


def _answerable_claims_rows(n_hit: int, n_total: int = 15) -> list[dict]:
    """`n_hit` answerable rows keep one claim citing the expected article
    (source 1 -> article 7); the rest keep nothing at all."""
    hits = [
        _claims_row(f"Q{i}", expected=[7], source_numbers=[7, 12],
                    kept=[{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}])
        for i in range(1, n_hit + 1)
    ]
    misses = [
        _claims_row(f"Q{i}", expected=[7], source_numbers=[7, 12])
        for i in range(n_hit + 1, n_total + 1)
    ]
    return hits + misses


def _abstained_ooc_rows(n: int = 5, start: int = 16) -> list[dict]:
    return [
        _claims_row(f"Q{i}", answerable=False, source_numbers=[3, 4])
        for i in range(start, start + n)
    ]


def test_the_claims_report_prints_coverage_beside_every_grounding_number():
    """The gate trap (EVAL.md Run 5): a gate that drops every claim scores
    "0 fabricated" the same way a genuinely clean run does. Coverage and the
    dropped-by-reason counts must always be printed, not only on failure."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()

    _, out = _claims_run(rows)

    assert "coverage" in out
    assert "10/15" in out            # 10 of 15 answerable kept >= 1 claim
    assert "fabricated" in out
    assert "dropped" in out or "uncited" in out
    assert "schema failures" in out
    assert "copy their source" in out or "copied" in out


def test_adoption_requires_zero_fabricated_sources():
    rows = _answerable_claims_rows(9) + _abstained_ooc_rows()
    # `_answerable_claims_rows(9)` already returns 15 rows (Q1..Q15): 9 hits
    # and 6 misses. Promote the first miss (Q10) into a hit that ALSO
    # carries one fabricated claim, instead of appending a 16th row under a
    # colliding id — that used to work only because nothing checked the
    # split size; now that the verdict is pinned to exactly 15/5 (Commit 2),
    # a 16th answerable row would make the verdict "n/a" instead of "NO".
    rows[9]["gate"]["kept"] = [{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}]
    rows[9]["gate"]["dropped"] = [{"text": "بلا سند", "sources": [9], "reason": "fabricated"}]
    rows[9]["gate"]["fabricated"] = 1
    rows[9]["gate"]["status"] = "partial"

    code, out = _claims_run(rows)

    assert "ADOPT for the app: NO" in out
    assert "fabricated" in out.split("ADOPT for the app:")[1]
    assert code == 1
    assert code == 1


def test_adoption_requires_b2_of_at_least_80_percent():
    """Fabricated stays 0 and 10/15 still cite the expected article — only
    B2 (3/5, below the 80% floor) should sink the verdict. With only 5
    out_of_corpus questions, B2 can only ever be a multiple of 20%, so 3/5
    (60%) — one step below the 4/5 (80%) floor — IS the pinned boundary;
    see test_adoption_at_exactly_the_pinned_thresholds_is_adopted for the
    4/5 side of it."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows(3, start=16) + [
        _claims_row(f"Q{i}", answerable=False, source_numbers=[3, 4],
                    kept=[{"text": "نص غير ذي صلة بالسؤال.", "sources": [1], "copied": False}])
        for i in range(19, 21)
    ]

    code, out = _claims_run(rows)

    assert "ADOPT for the app: NO" in out
    assert "B2" in out.split("ADOPT for the app:")[1]
    assert code == 1


def test_adoption_requires_the_expected_article_on_at_least_10_answerable_questions():
    """fabricated=0 and B2=5/5 both pass; only 9 of 15 cite article 7."""
    rows = _answerable_claims_rows(9) + _abstained_ooc_rows()

    code, out = _claims_run(rows)

    reason = out.split("ADOPT for the app:")[1]
    assert "ADOPT for the app: NO" in out
    assert "9" in reason and "10" in reason
    assert code == 1


def test_a_gate_that_drops_everything_is_not_adopted():
    """A gate that drops every claim looks clean on fabricated/ungrounded
    alone (both zero) and even B2 is trivially satisfied — every answerable
    question ends up abstained too. Coverage (0/15) is what shows the
    difference, and the adoption rule itself must not be fooled: nothing
    ever cites the expected article, so it is still NO."""
    rows = [
        _claims_row(f"Q{i}", expected=[7], source_numbers=[7, 12],
                    dropped=[{"text": "نص", "sources": [], "reason": "uncited"}],
                    uncited=1)
        for i in range(1, 16)
    ] + _abstained_ooc_rows()

    code, out = _claims_run(rows)

    assert "0/15" in out
    assert "ADOPT for the app: NO" in out
    assert code == 1


def _boundary_rows(n_hit: int, n_b2: int, fabricate_last_hit: bool = False) -> list[dict]:
    """15 answerable rows (Q1..Q15) — `n_hit` keep a claim citing the
    expected article, the rest keep nothing — plus 5 out_of_corpus rows
    (Q16..Q20), `n_b2` of which correctly abstain and the rest wrongly
    answer. If `fabricate_last_hit`, the LAST hit row also carries one
    fabricated-dropped claim, so cited-expected and B2 stay exactly at
    `n_hit`/`n_b2` while fabricated goes from 0 to 1 — the one condition
    a caller wants to flip in isolation.
    """
    answerable = []
    for i in range(1, 16):
        if i > n_hit:
            answerable.append(_claims_row(f"Q{i}", expected=[7], source_numbers=[7, 12]))
            continue
        kept = [{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}]
        dropped, fabricated = [], 0
        if fabricate_last_hit and i == n_hit:
            dropped = [{"text": "بلا سند", "sources": [9], "reason": "fabricated"}]
            fabricated = 1
        answerable.append(_claims_row(
            f"Q{i}", expected=[7], source_numbers=[7, 12],
            kept=kept, dropped=dropped, fabricated=fabricated,
        ))

    ooc = []
    for j, i in enumerate(range(16, 21)):
        if j < n_b2:
            ooc.append(_claims_row(f"Q{i}", answerable=False, source_numbers=[3, 4]))
        else:
            ooc.append(_claims_row(
                f"Q{i}", answerable=False, source_numbers=[3, 4],
                kept=[{"text": "نص غير ذي صلة بالسؤال.", "sources": [1], "copied": False}],
            ))
    return answerable + ooc


def test_adoption_at_exactly_the_pinned_thresholds_is_adopted():
    """10/15 cited-expected, 4/5 B2, 0 fabricated are the exact conditions
    EVAL.md's pre-registration (ecd37f8) fixes — "at least 10", "B2 >= 80%".
    A report that always prints NO, or that used `<=` where the
    pre-registration means `<` (or the reverse, on the fabricated side),
    would pass every other adoption test here and fail only this one."""
    rows = _boundary_rows(n_hit=10, n_b2=4)

    code, out = _claims_run(rows)

    assert "ADOPT for the app: YES" in out
    assert code == 0


def test_adoption_fails_one_short_of_the_cited_expected_threshold():
    """9/15, one below the pinned 10 — B2 and fabricated stay at their
    passing values so only this condition can be responsible for the NO."""
    rows = _boundary_rows(n_hit=9, n_b2=4)

    code, out = _claims_run(rows)

    assert "ADOPT for the app: NO" in out
    reason = out.split("ADOPT for the app:")[1]
    assert "9" in reason and "10" in reason
    assert code == 1


def test_adoption_fails_with_one_fabricated_claim_even_at_the_other_thresholds():
    """10/15 cited-expected and 4/5 B2 both sit exactly at their pinned,
    passing values — only the single fabricated claim should sink this."""
    rows = _boundary_rows(n_hit=10, n_b2=4, fabricate_last_hit=True)

    code, out = _claims_run(rows)

    assert "ADOPT for the app: NO" in out
    assert "fabricated" in out.split("ADOPT for the app:")[1]
    assert code == 1


def test_the_claims_report_prints_claims_ignored_alongside_an_explicit_abstention():
    """The pre-run smoke test's Q017 risk (EVAL.md, ecd37f8): a model that
    sets `abstain: true` but still hands over claims anyway is a different
    failure than simply not abstaining, and `report()` (the text contract)
    has no equivalent count — it must be visible here on its own, not
    folded into `dropped` or `false abstention`."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()
    rows[0]["gate"]["ignored_on_abstain"] = 2

    _, out = _claims_run(rows)

    assert "ignored on an abstain=true   : 2" in out


# --- Commit 2 (m2): mutation gaps in the claims report ----------------------


def test_cites_expected_ignores_a_dropped_claim_that_would_otherwise_match():
    """A DROPPED claim was never shown to a reader — it must not make a
    question look like it cited the expected article. Mutant: iterating
    `gate["dropped"]` (or every claim regardless of kept/dropped) instead of
    only `gate["kept"]` would count this row as citing article 7, since the
    dropped claim's source 1 IS article 7."""
    row = _claims_row(
        "Q1", expected=[7], source_numbers=[7],
        kept=[],
        dropped=[{"text": "نص", "sources": [1], "reason": "ungrounded"}],
    )

    assert _cites_expected(row) is False


def test_report_claims_prints_exact_labelled_lines_not_placeholder_constants():
    """A handful of mutants each hard-code one of these numbers to a
    constant (`covered = []`, `false_abstain = []`, `schema_failures = 0`,
    ...) and still pass a report whose test only checks that a bare number
    like "10/15" appears SOMEWHERE in the output — because a different,
    unaffected line (here, "cited the expected article") often prints that
    exact same number too. This fixture is built so every value below
    differs from every other and from both 0 and the full count, and every
    assertion pins the WHOLE labelled line, not a bare number."""
    q1 = _claims_row(
        "Q1", expected=[7], source_numbers=[7, 12],
        kept=[{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}],
    )
    q1["stats"] = [
        {"output_tokens": 5, "output_s": 1.0},
        {"output_tokens": 10, "output_s": 1.0},
    ]  # two calls for ONE question -> exactly one retry
    q2 = _claims_row(
        "Q2", expected=[7], source_numbers=[7, 12], kept=[],
        dropped=[{"text": "بلا سند", "sources": [], "reason": "uncited"}],
        uncited=1,
    )
    q3 = _claims_row(
        "Q3", expected=[7], source_numbers=[7, 12],
        kept=[{"text": "نص آخر [مادة 12].", "sources": [2], "copied": False}],
        dropped=[
            {"text": "ادعاء غير مسند", "sources": [9], "reason": "fabricated"},
            {"text": "ادعاء آخر", "sources": [1], "reason": "ungrounded"},
        ],
        fabricated=1, ungrounded=1,
    )
    q4 = _claims_row(
        "Q4", expected=[7], source_numbers=[7, 12],
        kept=[], dropped=[], schema_failure=True,
    )
    q5 = _claims_row("Q5", answerable=False, source_numbers=[3, 4], kept=[])
    q6 = _claims_row(
        "Q6", answerable=False, source_numbers=[3, 4],
        kept=[{"text": "نص غير متعلق بالسؤال.", "sources": [1], "copied": False}],
    )

    _, out = _claims_run([q1, q2, q3, q4, q5, q6])

    assert "  coverage (>=1 kept claim)          : 2/4 = 50.0%" in out
    assert "  false abstention on answerable     : 2/4 = 50.0%" in out
    assert "  B2 - abstained on out_of_corpus     : 1/2 = 50.0%    (criterion: >= 80%)" in out
    assert "  cited the expected article         : 1/4 = 25.0%    (criterion: >= 10 of 4)" in out
    assert "  fabricated claims                  : 1    (criterion: 0)" in out
    assert "  dropped - uncited / ungrounded      : 1 / 1" in out
    assert "  schema failures                     : 1" in out
    assert "  retries (more than 1 call)         : 1" in out


def test_the_report_counts_kept_claims_shorter_than_a_real_claim():
    """An empty `{"text": "", "sources": [1]}` is KEPT by today's gate (Run
    6 pre-registers the gate itself as unchanged — this is a report-only
    addition) and silently counts toward both coverage and cited-expected.
    This line is the only place that surfaces it. (`_answerable_claims_rows`'s
    own default claim text is itself only 15 characters — shorter than
    `MIN_CLAIM_CHARS` — so this test builds its own claim text long enough
    to isolate the ONE genuinely short claim from the other nine.)"""
    long_claim_text = "الرد يكون خلال مهلة قدرها ستة أيام عمل من تاريخ العلم بالخرق."  # 61 chars
    rows = [
        _claims_row(f"Q{i}", expected=[7], source_numbers=[7, 12],
                    kept=[{"text": long_claim_text, "sources": [1], "copied": False}])
        for i in range(1, 10)
    ] + [
        _claims_row("Q10", expected=[7], source_numbers=[7, 12],
                    kept=[{"text": "", "sources": [1], "copied": False}]),
    ] + _abstained_ooc_rows()

    _, out = _claims_run(rows)

    assert "1/10" in out.split("kept claims shorter than a real claim")[1][:20]


def test_kept_claims_shorter_than_a_real_claim_is_judged_on_stripped_length():
    """Padding a short claim with whitespace must not let it escape this
    line: `len(text)` (raw) counts the padding, `len(text.strip())` (the
    fix) does not. The padding is LEADING-only (20 spaces, none
    trailing) so this also kills a `.strip()` -> `.rstrip()` mutant:
    `.rstrip()` would leave the leading padding untouched (27 chars, not
    short) where `.strip()` correctly removes it too (7 chars, short) —
    symmetric padding on both sides could not tell the two apart."""
    padded_short = "                    " + "نص قصير"  # 20 leading spaces only
    assert len(padded_short) >= 25          # looks long enough unstripped
    assert len(padded_short.strip()) < 25   # is not, once fully stripped
    assert len(padded_short.rstrip()) >= 25  # rstrip alone would miss it
    rows = [
        _claims_row("Q1", expected=[7], source_numbers=[7, 12],
                    kept=[{"text": padded_short, "sources": [1], "copied": False}]),
    ] + _abstained_ooc_rows()

    _, out = _claims_run(rows)

    # The whole labelled line, not a bare "1/1" that "1/12" or "11/15"
    # would also satisfy.
    assert "  kept claims shorter than a real claim : 1/1 = 100.0%" in out


def test_the_report_header_defaults_to_run_5_and_its_pre_registration_commit():
    """Existing callers (every current test, and both call sites in
    `answer_eval.py`) pass no contract name at all — the header they have
    always seen must not change."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()

    _, out = _claims_run(rows)

    assert "Run 5 contract - claims JSON + gate (EVAL.md, ecd37f8)" in out


def test_the_report_header_names_the_given_contract_and_pre_registration_commit():
    """Run 6 reuses this same report with its own contract name and its own
    pre-registration commit — the header must not stay hard-coded to
    Run 5's."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()

    _, out = _claims_run(rows, contract_name="Run 6", pre_registration_commit="d3f39c3")

    assert "Run 6 contract - claims JSON + gate (EVAL.md, d3f39c3)" in out
    assert "Run 5 contract" not in out


def test_the_verdict_is_not_applicable_when_the_split_is_not_exactly_15_and_5():
    """The three adoption thresholds (B2 >= 80%, >= 10 of 15 cited,
    fabricated = 0) were pinned for exactly 15 answerable + 5 out_of_corpus
    questions (EVAL.md) — evaluating them against a differently-sized split
    would silently compare against thresholds that were never set for it."""
    rows = _answerable_claims_rows(10, n_total=14) + _abstained_ooc_rows()  # 14, not 15

    code, out = _claims_run(rows)

    assert "ADOPT for the app: n/a" in out
    reason = out.split("ADOPT for the app:")[1]
    assert "15" in reason and "5" in reason
    assert code == 1


def test_the_verdict_still_applies_normally_at_exactly_15_and_5():
    """A guard against the split size must not also break the ordinary
    15/5 case it is meant to leave alone: 10/15 cited-expected, 0
    fabricated and 5/5 B2 clear all three pinned thresholds."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()

    code, out = _claims_run(rows)

    assert "ADOPT for the app: n/a" not in out
    assert "ADOPT for the app: YES" in out
    assert code == 0


# --- Commit 3 (m2): Run 6 — stage-tagged calls and the relevance lines -----


def test_retries_come_from_attempts_per_stage_not_the_total_number_of_calls():
    """A row that reaches two DIFFERENT stages once each (Run 6: a
    relevance call that says "yes", then a claims call) makes two calls
    but retried neither — `len(calls) - 1` would wrongly count every
    question that simply passed the relevance step as a retry."""
    row = _claims_row(
        "Q1", expected=[7], source_numbers=[7, 12],
        kept=[{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}],
    )
    row["relevance"] = {"answers": True, "attempts": 1, "failure": False, "raw": ["r"]}
    row["calls"] = [
        {"output_tokens": 2, "output_s": 1.0, "stage": "relevance"},
        {"output_tokens": 9, "output_s": 1.0, "stage": "claims"},
    ]
    rows = [row] + _abstained_ooc_rows()

    _, out = _claims_run(rows)

    assert "  model calls                        : 2" in out
    assert "  retries (more than 1 call)         : 0" in out


def test_a_real_retry_within_one_stage_still_counts_as_a_retry():
    """The fix above must not also hide a GENUINE retry — two calls tagged
    with the SAME stage is still exactly one retry."""
    row = _claims_row(
        "Q1", expected=[7], source_numbers=[7, 12],
        kept=[{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}],
    )
    row["calls"] = [
        {"output_tokens": 2, "output_s": 1.0, "stage": "claims"},
        {"output_tokens": 9, "output_s": 1.0, "stage": "claims"},
    ]
    rows = [row] + _abstained_ooc_rows()

    _, out = _claims_run(rows)

    assert "  model calls                        : 2" in out
    assert "  retries (more than 1 call)         : 1" in out


def test_retries_print_broken_down_by_stage_alongside_the_existing_total():
    """The existing total mixes stages together and is not comparable to
    Run 5's claims-only retry count — EVAL.md asks for the breakdown beside
    it, not instead of it. Row 1 retries only on relevance (2 calls), row 2
    retries only on claims (2 calls): total retries = 2, but relevance's own
    retries = 1 and claims' own = 1 — a total-only line could not show
    that split."""
    row1 = _claims_row(
        "Q1", expected=[7], source_numbers=[7, 12],
        kept=[{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}],
    )
    row1["calls"] = [
        {"output_tokens": 2, "output_s": 1.0, "stage": "relevance"},
        {"output_tokens": 2, "output_s": 1.0, "stage": "relevance"},
        {"output_tokens": 9, "output_s": 1.0, "stage": "claims"},
    ]
    row2 = _claims_row(
        "Q2", expected=[7], source_numbers=[7, 12],
        kept=[{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}],
    )
    row2["calls"] = [
        {"output_tokens": 2, "output_s": 1.0, "stage": "relevance"},
        {"output_tokens": 4, "output_s": 1.0, "stage": "claims"},
        {"output_tokens": 9, "output_s": 1.0, "stage": "claims"},
    ]
    rows = [row1, row2] + _abstained_ooc_rows()

    _, out = _claims_run(rows)
    lines = out.splitlines()

    assert "  model calls                        : 6" in lines
    assert "  retries (more than 1 call)         : 2" in lines
    # Whole line, matched against `splitlines()`: `in out` alone would
    # also match as a substring of a differently-numbered longer line.
    assert "  retries - relevance / claims       : 1 / 1" in lines


def test_retries_print_even_when_only_the_relevance_stage_ever_ran():
    """A gated run where relevance said "no" (or failed) on every single
    question never makes one claims call — only ONE stage ever appears
    ("relevance"), not two. `len(stage_names) > 1` would hide the
    breakdown for exactly this run; the fix (`any(s is not None ...)`)
    still shows it, since Run 3/4/5's untagged calls are the only case
    meant to stay hidden."""
    row1 = _claims_row("Q1", expected=[7], source_numbers=[7, 12])
    row1["calls"] = [{"output_tokens": 2, "output_s": 1.0, "stage": "relevance"}]
    row2 = _claims_row("Q2", expected=[7], source_numbers=[7, 12])
    row2["calls"] = [
        {"output_tokens": 2, "output_s": 1.0, "stage": "relevance"},
        {"output_tokens": 2, "output_s": 1.0, "stage": "relevance"},
    ]
    rows = [row1, row2] + _abstained_ooc_rows()

    _, out = _claims_run(rows)
    lines = out.splitlines()

    assert "  model calls                        : 3" in lines
    assert "  retries (more than 1 call)         : 1" in lines
    assert "  retries - relevance                : 1" in lines


def test_old_stats_shaped_rows_still_print_the_unlabelled_runtime_lines():
    """Run 3/4/5's saved rows carry `stats`, never `calls` — the per-stage
    grouping must degenerate back to exactly the old, unlabelled lines for
    them (a single implicit stage), or their saved numbers would no
    longer print the way this project already measured and quoted them."""
    rows = [
        _claims_row("Q1", expected=[7], source_numbers=[7, 12],
                    kept=[{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}]),
    ] + _abstained_ooc_rows()
    rows[0]["stats"] = [{"output_tokens": 10, "output_s": 2.0}]

    _, out = _claims_run(rows)

    assert "  mean output tokens/s               : 5.0" in out
    assert "relevance mean output tokens/s" not in out
    assert "claims mean output tokens/s" not in out
    assert "retries -" not in out  # only ever printed when >1 stage exists


def test_the_gated_report_prints_the_relevance_lines_beside_the_verdict():
    """EVAL.md's Run 6 section names these four numbers as what to print
    beside coverage/B2: how often the step said no on an answerable
    question, how often it said yes on an out_of_corpus one, relevance
    failures, and abstain reasons by count."""
    answerable = []
    for i in range(1, 16):
        said_yes = i > 3  # 3 of 15 wrongly say "no" (a known pre-registered risk)
        kept = ([{"text": "الرد سبعة أيام.", "sources": [1], "copied": False}]
                if said_yes else [])
        row = _claims_row(f"Q{i}", expected=[7], source_numbers=[7, 12], kept=kept)
        row["relevance"] = {"answers": said_yes, "attempts": 1, "failure": False, "raw": ["r"]}
        row["abstain_reason"] = None if said_yes else "relevance_no"
        answerable.append(row)

    out_of_corpus = []
    for j, i in enumerate(range(16, 21)):
        said_yes = j == 0  # 1 of 5 wrongly says "yes"
        kept = ([{"text": "نص غير ذي صلة بالسؤال.", "sources": [1], "copied": False}]
                if said_yes else [])
        row = _claims_row(f"Q{i}", answerable=False, source_numbers=[3, 4], kept=kept)
        row["relevance"] = {"answers": said_yes, "attempts": 1, "failure": False, "raw": ["r"]}
        row["abstain_reason"] = None if said_yes else "relevance_no"
        out_of_corpus.append(row)

    _, out = _claims_run(answerable + out_of_corpus)

    assert "  relevance said no on answerable    : 3/15 = 20.0%" in out
    assert "  relevance said yes on out_of_corpus : 1/5 = 20.0%" in out
    assert "  relevance_failures                  : 0" in out
    # "abstain reasons" counts every row abstained by the relevance step,
    # correct or not: 3 wrong "no"s on answerable questions PLUS the 4
    # out_of_corpus questions that correctly got a "no" too (only 1 of the
    # 5 out_of_corpus rows said "yes") = 7, not just the 3 false ones.
    assert "relevance_no=7" in out.split("abstain reasons")[1][:60]


def test_the_relevance_lines_do_not_print_for_the_plain_json_contract():
    """A Run 5 ("json") row never carries `relevance` at all — nothing
    about Run 6's own diagnostics belongs in a report that never ran its
    relevance step."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()

    _, out = _claims_run(rows)

    assert "relevance said no on answerable" not in out
    assert "relevance_failures" not in out


def test_report_claims_refuses_the_verdict_when_expect_relevance_finds_no_relevance_data():
    """`_report_gated` passes `expect_relevance=True`. If every row's
    `relevance` is None — Run 5's own rows, or a code regression that
    stopped attaching relevance data — the verdict must be refused instead
    of silently printing Run 5's numbers under a Run 6 header. This split
    (10/15, 5/5, 0 fabricated) would otherwise print "ADOPT for the app:
    YES" (see test_the_verdict_still_applies_normally_at_exactly_15_and_5)."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()

    code, out = _claims_run(
        rows, contract_name="Run 6", pre_registration_commit="d3f39c3",
        expect_relevance=True,
    )

    assert "ADOPT for the app: n/a — no relevance data in gated rows" in out
    assert "ADOPT for the app: YES" not in out
    assert code == 1


def test_expect_relevance_does_not_refuse_when_rows_do_carry_relevance_data():
    """The new refusal must not also block the ordinary case it is not
    about: gated rows that DO carry relevance data verdict normally."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()
    for r in rows:
        r["relevance"] = {"answers": True, "attempts": 1, "failure": False, "raw": ["r"]}

    code, out = _claims_run(
        rows, contract_name="Run 6", pre_registration_commit="d3f39c3",
        expect_relevance=True,
    )

    assert "no relevance data in gated rows" not in out
    assert "ADOPT for the app: YES" in out
    assert code == 0


def test_relevance_failures_are_counted_separately_from_a_claims_schema_failure():
    row = _claims_row("Q1", expected=[7], source_numbers=[7, 12], kept=[])
    row["relevance"] = {"answers": None, "attempts": 2, "failure": True, "raw": ["x", "y"]}
    row["abstain_reason"] = "relevance_failure"
    rows = [row] + _abstained_ooc_rows()

    _, out = _claims_run(rows)

    assert "  relevance_failures                  : 1" in out
    assert "relevance_failure=1" in out.split("abstain reasons")[1][:60]


def test_a_no_sources_row_reaches_abstain_reasons_even_with_no_relevance_data():
    """A no-sources question (`claims.ClaimsGenerator.answer`'s
    `no_sources` branch, which takes priority over the relevance step)
    gets `relevance=None` AND `abstain_reason="no_sources"` under EITHER
    contract — "no relevance data" is not exclusively Run 5's signature.
    This row contributes to none of the first three relevance-specific
    counts, but must still show up in `abstain reasons`."""
    rows = _answerable_claims_rows(10) + _abstained_ooc_rows()
    for r in rows:
        r["relevance"] = {"answers": True, "attempts": 1, "failure": False, "raw": ["r"]}
    no_sources_row = _claims_row("Q_NS", expected=[7], source_numbers=[])
    no_sources_row["relevance"] = None
    no_sources_row["abstain_reason"] = "no_sources"
    rows.append(no_sources_row)

    _, out = _claims_run(
        rows, contract_name="Run 6", pre_registration_commit="d3f39c3",
        expect_relevance=True,
    )

    assert "no_sources=1" in out.split("abstain reasons")[1][:60]
