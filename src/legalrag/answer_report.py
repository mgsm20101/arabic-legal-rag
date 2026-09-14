"""Console reports for the M2 answer evaluation — PRD M2/B1 and M2/B2.

Split out of `answer_eval.py` (task 2.0): that module (518 lines at the time)
was past the ~500-line threshold the code review set for it, and Run 5 adds a
second report (`report_claims`) alongside this one — two reports belong in
their own module, not bolted onto the file that does retrieval and generation
wiring.

`report()` takes already-loaded `rows` and an optional `meta` dict — the
sibling `.meta.json` an `ollama:` run wrote next to its rows. Reading `runs/`
directly from here would duplicate `answer_eval.rows_path`'s file-naming
logic in a module that has no other reason to know about it; `answer_eval.main`
resolves the meta file and passes its contents in, the same way it already
resolves and passes in `rows`.

**B2 has a trap, and this is designed against it.** "Abstention accuracy on
``out_of_corpus`` >= 80%" is satisfied trivially by a model that abstains on
everything: it scores 100% and answers nothing. So the false-abstention rate on
*answerable* questions is printed next to it, always, and neither number is
quotable alone. The pair is the metric.

**B1 counts three failures apart** — fabricated, ungrounded, uncited — because
they say different things about where the system broke. See ``cite``.
"""

from __future__ import annotations

from collections import Counter

from .cite import MIN_CLAIM_CHARS
from .generate import DEFAULT_MODEL

# Mirrors `answer_eval.DEFAULT_SPEC` exactly (both derive from the same
# `generate.DEFAULT_MODEL`) without importing it from there: `answer_eval`
# imports `report` from this module, so the reverse import would be a cycle.
# Only used as the printed "model" line's fallback when `rows` predates the
# "model" field entirely (Run 3's saved shape) — real saved runs always
# carry it.
DEFAULT_SPEC = "hf:" + DEFAULT_MODEL


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.1%}" if d else f"{n}/0 = -"


def _calls_of(row: dict) -> list[dict]:
    """A row's per-call stats, normalized to a flat list of dicts.

    `"calls"` (Run 6 onward: every call, stage-tagged "relevance"/"claims")
    is read first when present; otherwise `"stats"` (Run 4's single dict, or
    the untagged list Run 5 wrote) — every report here must keep reading
    those saved files exactly as before this field existed. A question
    answered without calling the model (no articles retrieved) normalizes
    to `[]` either way, which is falsy; the rest have at least one call.
    """
    calls = row.get("calls")
    if calls is not None:
        return calls
    s = row.get("stats")
    if s is None:
        return []
    return [s] if isinstance(s, dict) else s


def _stage_groups(calls: list[dict]) -> list[tuple[str | None, list[dict]]]:
    """Group `calls` by their `"stage"` tag, preserving first-seen order.

    A call with no `"stage"` key at all — every Run 3/4/5 call, since the
    field did not exist yet — groups under `None`: a single implicit stage.
    That is what keeps those runs' numbers identical to before stages
    existed, since grouping by "the one stage present" computes the exact
    same aggregate as not grouping at all.
    """
    order: list[str | None] = []
    groups: dict[str | None, list[dict]] = {}
    for c in calls:
        stage = c.get("stage")
        if stage not in groups:
            groups[stage] = []
            order.append(stage)
        groups[stage].append(c)
    return [(stage, groups[stage]) for stage in order]


def _row_retries(row: dict) -> int:
    """Calls beyond the first, counted PER STAGE within this row and
    summed — not `len(calls) - 1` over the whole row regardless of stage.

    A question that reaches two DIFFERENT stages once each (Run 6: one
    relevance call that says "yes", then one claims call) makes two calls
    but retried neither; only more than one call within the SAME stage is
    a retry. Scoring `len(calls) - 1` here would count every question that
    simply passed the relevance step as a retry, which it is not.
    """
    counts: dict[str | None, int] = {}
    for c in _calls_of(row):
        stage = c.get("stage")
        counts[stage] = counts.get(stage, 0) + 1
    return sum(max(0, n - 1) for n in counts.values())


def _print_stage_calls(stage: str | None, calls: list[dict]) -> None:
    """One stage's slice of the throughput/truncation/cut lines.

    Unlabelled when `stage` is `None` (the single-implicit-stage case, see
    `_stage_groups`) — printed character for character as Run 3/4/5 always
    have been. Labelled per stage otherwise (Run 6's relevance/claims
    split), so a reader can tell a slow relevance step from a slow claims
    one instead of only ever seeing their sum.
    """
    label = "" if stage is None else f"{stage} "
    with_rate = [c for c in calls if c.get("output_tokens") is not None and c.get("output_s")]
    if with_rate:
        tokens = sum(c["output_tokens"] for c in with_rate)
        secs = sum(c["output_s"] for c in with_rate)
        print(f"  {label}mean output tokens/s               : {tokens / secs:.1f}")
    n_truncated = sum(1 for c in calls if c.get("truncated"))
    n_cut = sum(1 for c in calls if c.get("cut"))
    # Named for what they count: calls, not prompts or answers — a question
    # that retries (Run 5 retries once on invalid JSON) makes more than one
    # call, and the old names ("prompts truncated", "answers cut") would
    # silently under-describe that.
    print(f"  {label}calls truncated by num_ctx         : {n_truncated}")
    print(f"  {label}calls cut by the token cap         : {n_cut}")


def _print_runtime_and_meta(rows: list[dict], meta: dict | None) -> None:
    """The per-call timing/throughput lines and the `.meta.json` lines,
    shared verbatim between `report` and `report_claims` — Run 5 changes the
    answer shape, not how a call's wall-clock cost or the server's own
    provenance are measured; Run 6 adds a second model/stage, not a second
    way of measuring one.

    Only an `ollama:` run ever carries per-call stats at all (the hf:
    runtime exposes no such attribute — see `answer_eval._stats_for`).
    Within one such run, a question answered without calling the model (no
    articles retrieved) has no calls, which is falsy; the rest have at
    least one. `_calls_of` normalizes every saved shape (Run 4's single
    dict, Run 5's untagged list, Run 6's stage-tagged list) into the same
    flat shape, so aggregation below does not need to know which one a
    given row was saved in.
    """
    mean_s = sum(r["seconds"] for r in rows) / max(len(rows), 1)
    print(f"  mean seconds per answer            : {mean_s:.1f}")

    stat_rows = [r for r in rows if r.get("stats") or r.get("calls")]
    all_calls = [c for r in stat_rows for c in _calls_of(r)]
    if all_calls:
        if any(c.get("load_s") is not None for c in all_calls):
            adjusted = [
                r["seconds"] - sum(c.get("load_s") or 0 for c in _calls_of(r))
                for r in rows
            ]
            mean_adj = sum(adjusted) / max(len(adjusted), 1)
            print(f"  mean seconds excl. load            : {mean_adj:.1f}")

        for stage, stage_calls in _stage_groups(all_calls):
            _print_stage_calls(stage, stage_calls)

        # More than one call for a question is a retry — worth surfacing on
        # its own, since a model that retries often is slower than its mean
        # seconds alone would suggest.
        retries = sum(_row_retries(r) for r in stat_rows)
        print(f"  model calls                        : {len(all_calls)}")
        print(f"  retries (more than 1 call)         : {retries}")

    if meta is not None:
        if meta.get("gpu_share") is not None:
            print(f"  GPU share                          : {meta['gpu_share']:.0%}")
        if meta.get("quantization"):
            print(f"  quantization                       : {meta['quantization']}")
        if meta.get("digest"):
            print(f"  digest                             : {meta['digest'][:12]}")


def report(rows: list[dict], meta: dict | None = None) -> int:
    answerable = [r for r in rows if r["answerable"]]
    out_of_corpus = [r for r in rows if not r["answerable"]]

    print("\n" + "=" * 68)
    print("  B1 - grounding: no answer may cite an article it was not given")
    print("=" * 68)
    scored = [r for r in answerable if not r["abstained"]]
    fabricated = [r for r in scored if r["fabricated"]]
    ungrounded = [r for r in scored if r["ungrounded"]]
    uncited = [r for r in scored if r["uncited"]]
    clean = [r for r in scored if r["grounded"]]

    print(f"  answered (of {len(answerable)} answerable)  : {len(scored)}")
    print(f"  fully grounded                     : {_pct(len(clean), len(scored))}")
    line = f"  cited an article not in the law    : {len(fabricated)}"
    if fabricated:
        line += "   " + str([(r["id"], r["fabricated"]) for r in fabricated])
    print(line)
    line = f"  cited a real article NOT retrieved : {len(ungrounded)}"
    if ungrounded:
        line += "   " + str([(r["id"], r["ungrounded"]) for r in ungrounded])
    print(line)
    line = f"  made a claim with no citation      : {len(uncited)}"
    if uncited:
        line += "   " + str([r["id"] for r in uncited])
    print(line)
    copied = [r for r in scored if r.get("copied")]
    line = f"  citation was inside copied statute : {len(copied)}"
    if copied:
        line += "   " + str([(r["id"], r["copied"]) for r in copied])
    print(line)
    if copied:
        print("      (the law citing itself in text the model reproduced -")
        print("       not the model grounding an answer; excluded from `cited`)")

    # PRD B1 reads «صفر إجابة **بلا استشهاد** بمادة موجودة فعلا في المتن» — zero
    # answers *without* a citation to an article that really exists. An answer
    # carrying no citation at all is a violation, not a neutral outcome.
    #
    # Getting this wrong is the same species of error as the B2 trap one
    # section below: a verdict of `not fabricated and not ungrounded` is passed
    # trivially by a model that never cites anything, and the first run of this
    # eval printed exactly that — PASS, on 12 of 14 answers with no citation.
    # Silence is not grounding.
    b1 = not fabricated and not ungrounded and not uncited
    verdict = "PASS" if b1 else "FAIL"
    print(f"\n  B1: {verdict} - every answer must carry a citation, it must resolve to a")
    print("      real article, and that article must have been retrieved. An answer")
    print("      with no citation fails: not citing is not the same as not lying.")

    print("\n" + "=" * 68)
    print("  B2 - abstention, with the false-abstention rate beside it")
    print("=" * 68)
    correct_abstain = [r for r in out_of_corpus if r["abstained"]]
    false_abstain = [r for r in answerable if r["abstained"]]
    print(f"  abstained on out_of_corpus   : {_pct(len(correct_abstain), len(out_of_corpus))}"
          "    (criterion: >= 80%)")
    print(f"  abstained on answerable      : {_pct(len(false_abstain), len(answerable))}"
          "    (abstaining on everything scores 100% above)")
    b2 = bool(out_of_corpus) and len(correct_abstain) / len(out_of_corpus) >= 0.80
    verdict = "PASS" if b2 else "FAIL"
    print(f"\n  B2: {verdict} - unreadable without the line above it.")

    print("\n" + "=" * 68)
    print("  beyond the criteria")
    print("=" * 68)
    model_spec = rows[0]["model"] if rows and "model" in rows[0] else DEFAULT_SPEC
    print(f"  model                              : {model_spec}")
    hit = [r for r in scored if r["cited_expected"]]
    strict = [r for r in scored if r["strict"]]
    print(f"  cited the expected article         : {_pct(len(hit), len(scored))}"
          "    (bounded above by retrieval)")
    print(f"  used the requested bracket form    : {_pct(len(strict), len(scored))}")
    _print_runtime_and_meta(rows, meta)

    print("\nSmall sample. Read the caveats in EVAL.md before quoting any of this,")
    print("starting with the one that matters most: the corpus is not the gazette text.")
    return 0 if (b1 and b2) else 1


# --- Run 5 — claims JSON + gate (EVAL.md, commit ecd37f8) ------------------
#
# Fixed here, not computed after seeing a run's numbers — the pre-
# registration commits to these three conditions before any real Run 5
# number existed, specifically so a threshold cannot be chosen after seeing
# which one the run happens to clear.
FABRICATED_MAX = 0
B2_MIN = 0.80
# gemma3:4b's own Run 4 count, free-text contract (EVAL.md) — the bar Run
# 5's JSON shape has to clear, not beat by an arbitrary margin: citing the
# expected article on fewer of the 15 answerable questions than free text
# already did would be a regression in the one thing retrieval bounds, not
# a tradeoff worth taking for a stricter answer shape.
CITED_EXPECTED_MIN = 10

# The dev split's exact shape when these thresholds were pinned (EVAL.md).
# B2 as a fraction and "10 of 15" as a count are both meaningless against a
# differently-sized split — a future dev set with, say, 8 out_of_corpus
# questions would silently compare its B2 against a floor that was never
# set for it. `report_claims` refuses the verdict, not the rest of the
# report, when a saved run's rows do not match this shape exactly.
ANSWERABLE_SPLIT = 15
OUT_OF_CORPUS_SPLIT = 5


def _cites_expected(row: dict) -> bool:
    """Does any claim `row` *kept* cite a source whose article number is
    one of `row["expected"]`? Only a kept claim counts — a dropped one was
    never shown to a reader, so it cannot be the thing that made this
    question's answer correct."""
    expected = set(row["expected"])
    if not expected:
        return False
    source_numbers = row["source_numbers"]
    for claim in row["gate"]["kept"]:
        for s in claim["sources"]:
            if 1 <= s <= len(source_numbers) and source_numbers[s - 1] in expected:
                return True
    return False


def _print_relevance_lines(
    rows: list[dict], answerable: list[dict], out_of_corpus: list[dict],
) -> None:
    """Run 6's own pre-registered lines (EVAL.md, "Run 6"), printed before
    everything Run 5 already prints: coverage and B2 alone cannot tell a
    relevance-driven abstention apart from a claims-gate one, and these are
    the four numbers EVAL.md names for reading that risk —

    - how often the step said "no" on an ANSWERABLE question (of 15): the
      known risk pre-registered before this ever ran (colloquial and
      multi-article questions, EVAL.md).
    - how often it said "yes" on an out_of_corpus question (of 5): the
      relevance step's own false-negative-on-abstention rate.
    - `relevance_failures`: two invalid-JSON attempts on the relevance step
      itself, counted apart from a claims `schema_failure` because this
      abstention never reached the claims call at all.
    - every `abstain_reason` this run actually produced, by count.

    A row with `relevance is None` (Run 5's own contract, or a `gated` row
    where the relevance step itself failed before ever answering true/false)
    contributes to none of the first two counts — only to `abstain reasons`
    and (if it failed) `relevance_failures`.
    """
    said_no = [r for r in answerable
               if r.get("relevance") and r["relevance"]["answers"] is False]
    said_yes = [r for r in out_of_corpus
                if r.get("relevance") and r["relevance"]["answers"] is True]
    failures = sum(1 for r in rows if r.get("relevance") and r["relevance"]["failure"])
    reasons = Counter(r["abstain_reason"] for r in rows if r.get("abstain_reason"))
    reasons_text = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "none"

    print(f"  relevance said no on answerable    : {_pct(len(said_no), len(answerable))}")
    print(f"  relevance said yes on out_of_corpus : {_pct(len(said_yes), len(out_of_corpus))}")
    print(f"  relevance_failures                  : {failures}")
    print(f"  abstain reasons                     : {reasons_text}")


def report_claims(
    rows: list[dict],
    meta: dict | None = None,
    *,
    contract_name: str = "Run 5",
    pre_registration_commit: str = "ecd37f8",
) -> int:
    """The claims-JSON contract and its model-free gate, scored against the
    same three fixed quantities `report` uses — 15 answerable questions, 5
    out_of_corpus — but read through `row["gate"]`, not the free-text
    contract's `row["cited"]`/`"fabricated"`/... (a claims row carries no
    such fields).

    Introduced for Run 5 (EVAL.md, commit ecd37f8, the defaults below) and
    reused as-is for Run 6: `contract_name`/`pre_registration_commit` only
    change the printed header, never the scoring — Run 6 pre-registers
    this exact gate and these exact numbers as unchanged from Run 5.

    **The gate trap, printed unconditionally (EVAL.md's own name for it):**
    a gate that drops every claim would score zero fabricated and zero
    ungrounded the same way a genuinely clean run does — coverage and the
    dropped-by-reason counts are what tell the two apart, so they are
    printed beside the adoption numbers every time, not only when adoption
    fails.
    """
    answerable = [r for r in rows if r["answerable"]]
    out_of_corpus = [r for r in rows if not r["answerable"]]

    print("\n" + "=" * 68)
    print(f"  {contract_name} contract - claims JSON + gate "
          f"(EVAL.md, {pre_registration_commit})")
    print("=" * 68)

    # Detected from the data, not a separate flag: a "gated" row (Run 6)
    # carries `relevance` on every row (or `None` when that step itself
    # failed before ever answering); a "json" row (Run 5) never does. This
    # is what lets a saved run answer its own question about which contract
    # produced it, the same way `contract_name`/`pre_registration_commit`
    # only change the header rather than needing a `is_gated` parameter.
    if any(r.get("relevance") is not None for r in rows):
        _print_relevance_lines(rows, answerable, out_of_corpus)

    covered = [r for r in answerable if r["gate"]["kept"]]
    false_abstain = [r for r in answerable if r["gate"]["status"] == "abstained"]
    correct_abstain = [r for r in out_of_corpus if r["gate"]["status"] == "abstained"]
    hit = [r for r in answerable if _cites_expected(r)]

    print(f"  coverage (>=1 kept claim)          : {_pct(len(covered), len(answerable))}")
    print(f"  false abstention on answerable     : {_pct(len(false_abstain), len(answerable))}")
    # The pre-run smoke test's Q017 risk (EVAL.md, ecd37f8): a model that
    # sets abstain=true but hands over claims anyway is a different failure
    # from simply not abstaining — worth its own line, since `report()`
    # (the free-text contract) has no equivalent count to fold it into.
    ignored_on_abstain = sum(r["gate"]["ignored_on_abstain"] for r in rows)
    print(f"  claims ignored on an abstain=true   : {ignored_on_abstain}")
    b2 = (len(correct_abstain) / len(out_of_corpus)) if out_of_corpus else 0.0
    print(f"  B2 - abstained on out_of_corpus     : "
          f"{_pct(len(correct_abstain), len(out_of_corpus))}    (criterion: >= 80%)")
    print(f"  cited the expected article         : {_pct(len(hit), len(answerable))}"
          f"    (criterion: >= {CITED_EXPECTED_MIN} of {len(answerable)})")

    fabricated = sum(r["gate"]["fabricated"] for r in rows)
    uncited = sum(r["gate"]["uncited"] for r in rows)
    ungrounded = sum(r["gate"]["ungrounded"] for r in rows)
    print(f"  fabricated claims                  : {fabricated}    (criterion: 0)")
    print(f"  dropped - uncited / ungrounded      : {uncited} / {ungrounded}")

    schema_failures = sum(1 for r in rows if r["schema_failure"])
    print(f"  schema failures                     : {schema_failures}")

    kept_total = [c for r in rows for c in r["gate"]["kept"]]
    copied = [c for c in kept_total if c.get("copied")]
    print(f"  kept claims that copy their source : {_pct(len(copied), len(kept_total))}")
    # The gate keeps a claim as soon as it is sourced and grounded — it says
    # nothing about whether the text itself is long enough to be a claim at
    # all. An empty `{"text": "", "sources": [1]}` passes the gate today
    # (Run 6 pre-registers the gate as unchanged) and silently counts toward
    # both coverage and cited-expected; this line is the only place it is
    # visible, without changing what the gate keeps.
    short = [c for c in kept_total if len(c["text"]) < MIN_CLAIM_CHARS]
    print(f"  kept claims shorter than a real claim : {_pct(len(short), len(kept_total))}")

    n_answered = sum(1 for r in rows if r["gate"]["status"] == "answered")
    n_partial = sum(1 for r in rows if r["gate"]["status"] == "partial")
    n_abstained = sum(1 for r in rows if r["gate"]["status"] == "abstained")
    print(f"  status - answered / partial / abstained : {n_answered} / {n_partial} / "
          f"{n_abstained}  (of {len(rows)})")

    _print_runtime_and_meta(rows, meta)

    # The three thresholds below were set FOR this exact split (EVAL.md) —
    # B2 as a fraction of 5 and "10 of 15" both stop meaning what the
    # pre-registration says the moment the split is a different size, so the
    # verdict itself (not the diagnostics above it) is refused rather than
    # silently compared against thresholds nobody set for this data.
    if len(answerable) != ANSWERABLE_SPLIT or len(out_of_corpus) != OUT_OF_CORPUS_SPLIT:
        print(f"\n  ADOPT for the app: n/a — the thresholds were set for "
              f"{ANSWERABLE_SPLIT} answerable / {OUT_OF_CORPUS_SPLIT} "
              "out_of_corpus questions")
        print("\nSmall sample. Read the caveats in EVAL.md before quoting any of this.")
        return 1

    # Same three conditions as the pre-registration, applied exactly: no
    # partial credit, no rounding a near-miss up.
    failed = []
    if fabricated != FABRICATED_MAX:
        failed.append(f"fabricated {fabricated} != {FABRICATED_MAX}")
    if b2 < B2_MIN:
        failed.append(f"B2 {b2:.0%} < {B2_MIN:.0%}")
    if len(hit) < CITED_EXPECTED_MIN:
        failed.append(f"cited expected {len(hit)} < {CITED_EXPECTED_MIN}")

    adopted = not failed
    if adopted:
        print("\n  ADOPT for the app: YES")
    else:
        print(f"\n  ADOPT for the app: NO — {'; '.join(failed)}")

    print("\nSmall sample. Read the caveats in EVAL.md before quoting any of this.")
    return 0 if adopted else 1
