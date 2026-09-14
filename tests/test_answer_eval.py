"""Wiring tests for the M2 end-to-end pipeline (PRD M2/B1 + M2/B2) —
retrieval-to-row plumbing, rows-file naming and collisions, and
`--report-only` re-scoring.

The report's own verdict tests (B1/B2 pass/fail, the runtime and meta lines)
live in `test_answer_report.py` next to the `report()` they exercise. What
is left here is getting the right rows to the right file without silently
overwriting a different run's saved answers, and `main()`'s exit codes for
the mistakes that are cheap to catch before any model loads.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.answer_eval import (  # noqa: E402
    DEFAULT_MODEL,
    ROWS_PATH,
    _check_rows_collision,
    _reaudit,
    _regate,
    _rows_model_mismatch,
    _stats_for,
    _weights_line,
    main,
    rows_path,
    run,
    run_claims,
)
from legalrag.claims import ClaimsGenerator  # noqa: E402
from legalrag.dense import corpus_fingerprint  # noqa: E402
from legalrag.evaluate import Question  # noqa: E402
from legalrag.generate import Generator  # noqa: E402
from legalrag.retrieve import Hit  # noqa: E402


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


def test_an_hf_spec_with_a_repo_name_prints_a_weights_line(monkeypatch, tmp_path):
    """`MODELS_DIR` is monkeypatched to an empty `tmp_path` so this is
    deterministic: unpatched, the exact line would depend on whether a real
    `models/Qwen/Qwen2.5-1.5B-Instruct` happens to exist on this machine."""
    import legalrag.generate as generate_mod
    monkeypatch.setattr(generate_mod, "MODELS_DIR", tmp_path)

    line = _weights_line("hf:Qwen/Qwen2.5-1.5B-Instruct")

    assert line == "weights    : Qwen/Qwen2.5-1.5B-Instruct"


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

    # The collision guard (`_check_rows_collision`) and the meta lookup
    # (`_meta_for_rows`) both key off this field — a row silently missing it,
    # or holding the wrong spec, would let either check pass while actually
    # comparing nothing, or the wrong thing.
    assert rows[0]["model"] == "ollama:stub-test-model"
    assert rows[0]["stats"] == [{"output_tokens": 10, "output_s": 1.0}]
    assert rows[1]["stats"] == [{"output_tokens": 20, "output_s": 2.0}]
    assert rows[2]["stats"] == []


def test_hf_runs_record_the_resolved_weights_path_in_each_row(monkeypatch):
    """An hf: run's provenance (disk path vs hub id) belongs on the row
    itself, not only in the console header — a saved run should be able to
    answer "what did this actually load?" without also keeping its log."""
    import legalrag.answer_eval as ae

    monkeypatch.setattr(ae, "model_source", lambda name: f"/resolved/{name}")

    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    hit7 = Hit(id="law-7", law_name="", number=7, score=1.0, snippet="")
    index = _StubIndex({"س1": [hit7]})
    generator = Generator(model=lambda messages: "جواب [مادة 7].")

    rows = run(
        [_question("Q1", "س1")], docs, generator,
        "hf:Qwen/Qwen2.5-1.5B-Instruct", index=index,
    )

    assert rows[0]["weights"] == "/resolved/Qwen/Qwen2.5-1.5B-Instruct"


def test_ollama_runs_do_not_record_a_weights_field():
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    hit7 = Hit(id="law-7", law_name="", number=7, score=1.0, snippet="")
    index = _StubIndex({"س1": [hit7]})
    model = _StubCallModel([("جواب [مادة 7].", {"output_tokens": 1, "output_s": 1.0})])
    generator = Generator(model=model)

    rows = run(
        [_question("Q1", "س1")], docs, generator, "ollama:some-model", index=index,
    )

    assert "weights" not in rows[0]


def test_a_rows_file_collision_from_a_different_model_is_refused(tmp_path):
    """Two different --model specs can slugify to the same file name (every
    character outside [A-Za-z0-9._-] becomes '-') — writing over another
    model's saved answers without noticing would corrupt that run's
    history."""
    rows_file = tmp_path / "answer_eval-collide.json"
    rows_file.write_text(
        json.dumps([{"model": "ollama:other-model", "id": "Q1"}]), encoding="utf-8"
    )

    message = _check_rows_collision(rows_file, "ollama:my-model")

    assert message is not None
    assert "ollama:other-model" in message
    assert "ollama:my-model" in message


def test_a_rows_file_for_the_same_model_is_not_a_collision(tmp_path):
    rows_file = tmp_path / "answer_eval-x.json"
    rows_file.write_text(json.dumps([{"model": "ollama:x", "id": "Q1"}]), encoding="utf-8")

    assert _check_rows_collision(rows_file, "ollama:x") is None


def test_a_missing_rows_file_is_never_a_collision(tmp_path):
    assert _check_rows_collision(tmp_path / "does-not-exist.json", "ollama:x") is None


def test_rows_predating_the_model_field_are_never_a_collision(tmp_path):
    """Run 3's saved rows have no "model" key at all — that must not be
    mistaken for a mismatch against some other spec."""
    rows_file = tmp_path / "answer_eval.json"
    rows_file.write_text(json.dumps([{"id": "Q1"}]), encoding="utf-8")

    assert _check_rows_collision(rows_file, "hf:something") is None


def test_a_full_run_refuses_to_overwrite_a_different_models_rows(tmp_path, monkeypatch, capsys):
    """Refused before load_questions even runs — a slug collision is
    detectable from the spec and the existing file alone, so the refusal
    should not cost a corpus load or a wasted generation run."""
    import legalrag.answer_eval as ae

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(
        json.dumps([{"id": "Q1", "model": "ollama:other"}]), encoding="utf-8"
    )
    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="text": fake_rows_file)

    def boom():
        raise AssertionError("load_questions must not run after a rows collision")
    monkeypatch.setattr(ae, "load_questions", boom)

    code = main(["--model", "ollama:mine"])
    out = capsys.readouterr().out

    assert code == 2
    assert "ollama:other" in out
    assert "ollama:mine" in out


def test_report_only_warns_on_a_rows_model_mismatch_but_still_reports(tmp_path, monkeypatch, capsys):
    """A mismatch is only ever a warning under --report-only, never a
    refusal: nothing is written to disk in this path, so there is nothing
    to protect by blocking it — but silently reporting another model's
    answers as if they were the one asked for would be misleading."""
    import legalrag.answer_eval as ae

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(json.dumps([
        {"id": "Q1", "category": "direct", "answerable": True, "model": "ollama:other",
         "text": "", "seconds": 1.0, "retrieved": [], "expected": []},
    ]), encoding="utf-8")

    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="text": fake_rows_file)
    # A non-empty stub corpus: this test is about the model-mismatch warning,
    # not about the (separately tested) empty-corpus refusal, and the row's
    # own `retrieved` is `[]` so nothing here needs to resolve to real text.
    monkeypatch.setattr(ae, "load_docs", lambda path: [{"id": "law-1", "text": "نص"}])

    code = main(["--model", "ollama:mine", "--report-only"])
    out = capsys.readouterr().out

    assert "ollama:other" in out
    assert "ollama:mine" in out
    assert "beyond the criteria" in out
    assert code in (0, 1)  # a report was produced, not a refusal


def test_a_bad_model_spec_exits_early_without_touching_the_corpus(monkeypatch, capsys):
    """The spec is validated before load_questions/load_docs/verify-refs
    even run — a typo in --model should not cost a corpus load, and must
    not silently fall through to run() with a broken model."""
    import legalrag.answer_eval as ae

    def boom():
        raise AssertionError("load_questions must not be called for a bad spec")
    monkeypatch.setattr(ae, "load_questions", boom)

    code = main(["--model", "bogus"])
    out = capsys.readouterr().out

    assert code == 2
    assert "bogus" in out


def test_reaudit_recomputes_verdicts_from_saved_text_not_from_stored_flags():
    """The audit itself has been wrong twice before (B1's wording, citations
    lifted from copied statute text) — replaying stored verdicts would keep
    reproducing a fixed bug's old, wrong answer forever."""
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    rows = [{
        "id": "Q1", "text": "جواب [مادة 7].", "retrieved": [7], "expected": [7],
        # Deliberately wrong stored verdict — _reaudit must overwrite it.
        "grounded": False, "cited": [], "fabricated": [], "ungrounded": [],
        "uncited": [], "abstained": False, "strict": [], "cited_expected": False,
    }]

    reaudited = _reaudit(rows, docs)

    assert reaudited[0]["cited"] == [7]
    assert reaudited[0]["grounded"] is True
    assert reaudited[0]["cited_expected"] is True


# --- Run 5's claims contract (--contract json) -----------------------------


def test_the_json_contract_writes_its_own_rows_file_beside_run_4s():
    """`--contract json` must never collide with Run 4's saved free-text
    answers for the same model — same slug, with a `-json` suffix, so both
    can be read back independently."""
    assert rows_path("ollama:gemma3:4b", "json") == Path(
        "runs/answer_eval-ollama-gemma3-4b-json.json"
    )
    assert rows_path("ollama:gemma3:4b") == Path("runs/answer_eval-ollama-gemma3-4b.json")
    assert rows_path("ollama:gemma3:4b", "text") == rows_path("ollama:gemma3:4b")


def test_run_claims_keeps_sources_in_rank_order_and_every_attempts_raw_text():
    """Same retrieval contract as `run()`: rank order preserved, never
    re-sorted by article number. Both attempts' raw text must survive even
    though only the second one parsed — `report_claims` needs the first to
    count retries and schema failures honestly."""
    docs = [
        {"id": "law-12", "text": "نص المادة الثانية عشرة"},
        {"id": "law-7", "text": "نص المادة السابعة"},
    ]
    # The index hands back article 12 before article 7 — out of numeric
    # order on purpose, so a bug that re-sorts by article number would show.
    hit12 = Hit(id="law-12", law_name="", number=12, score=0.9, snippet="")
    hit7 = Hit(id="law-7", law_name="", number=7, score=0.8, snippet="")
    index = _StubIndex({"س1": [hit12, hit7]})

    good = '{"abstain": false, "claims": [{"text": "الرد سبعة أيام.", "sources": [2]}]}'
    model = _StubCallModel([
        ("not json", {"output_tokens": 4, "output_s": 0.5}),
        (good, {"output_tokens": 9, "output_s": 1.0}),
    ])
    generator = ClaimsGenerator(model=model)

    rows = run_claims([_question("Q1", "س1")], docs, generator, "ollama:stub", index=index)

    assert rows[0]["source_numbers"] == [12, 7]
    assert rows[0]["raw"] == ["not json", good]
    assert rows[0]["attempts"] == 2
    assert rows[0]["schema_failure"] is False
    assert rows[0]["model"] == "ollama:stub"
    assert rows[0]["contract"] == "json"
    assert rows[0]["stats"] == [
        {"output_tokens": 4, "output_s": 0.5},
        {"output_tokens": 9, "output_s": 1.0},
    ]
    assert rows[0]["gate"]["kept"] == [
        {"text": "الرد سبعة أيام.", "sources": [2], "copied": False},
    ]


def test_report_only_reapplies_the_gate_to_saved_claims():
    """The gate is model-free by design specifically so a fixed gate never
    needs Ollama again — recomputed from the saved `parsed` answer and
    `source_numbers`, the same argument `_reaudit` makes for the free-text
    contract."""
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    rows = [{
        "id": "Q1", "answerable": True, "source_numbers": [7], "expected": [7],
        "parsed": {"abstain": False, "claims": [
            {"text": "الرد سبعة أيام.", "sources": [1]},
        ]},
        # Deliberately wrong stored verdict — _regate must overwrite it.
        "gate": {"status": "abstained", "kept": [], "dropped": [],
                 "uncited": 0, "fabricated": 0, "ungrounded": 0,
                 "ignored_on_abstain": 0},
    }]

    regated = _regate(rows, docs)

    assert regated[0]["gate"]["status"] == "answered"
    assert regated[0]["gate"]["kept"] == [
        {"text": "الرد سبعة أيام.", "sources": [1], "copied": False},
    ]


def test_the_json_contract_with_an_hf_model_exits_2_before_loading_anything(monkeypatch, capsys):
    """Schema-constrained decoding is not available on the transformers
    path here — refused the same way a bad --model spec is, before
    load_questions/load_docs/verify-refs cost anything."""
    import legalrag.answer_eval as ae

    def boom():
        raise AssertionError("load_questions must not run for an incompatible contract")
    monkeypatch.setattr(ae, "load_questions", boom)

    code = main(["--model", "hf:" + DEFAULT_MODEL, "--contract", "json"])
    out = capsys.readouterr().out

    assert code == 2
    assert "json" in out
    assert "hf:" + DEFAULT_MODEL in out


def test_an_unknown_contract_exits_2_before_loading_anything(monkeypatch, capsys):
    import legalrag.answer_eval as ae

    def boom():
        raise AssertionError("load_questions must not run for an unknown contract")
    monkeypatch.setattr(ae, "load_questions", boom)

    code = main(["--contract", "yaml"])
    out = capsys.readouterr().out

    assert code == 2
    assert "yaml" in out


def test_a_rows_file_saved_for_the_other_contract_is_refused_both_directions(
    tmp_path, monkeypatch, capsys
):
    """`contract="json"` only changes a rows file's name by a fixed `-json`
    suffix (`rows_path`) — a `text` run of a model literally named
    `...-json` can land on the exact same path as a `json` run of a
    differently-named model. Both directions must be refused, not just a
    plain model-name mismatch."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    collide_path = tmp_path / "runs" / "answer_eval-ollama-foo-json.json"

    # Direction 1: "ollama:foo-json" under "text" already saved there;
    # "ollama:foo" under "json" computes the exact same path.
    collide_path.write_text(
        json.dumps([{"id": "Q1", "model": "ollama:foo-json", "contract": "text"}]),
        encoding="utf-8",
    )
    code = main(["--model", "ollama:foo", "--contract", "json"])
    out = capsys.readouterr().out
    assert code == 2
    assert "ollama:foo-json" in out
    assert "text" in out

    # Direction 2: the same path now holds "ollama:foo" under "json";
    # "ollama:foo-json" under "text" computes the exact same path back.
    collide_path.write_text(
        json.dumps([{"id": "Q1", "model": "ollama:foo", "contract": "json"}]),
        encoding="utf-8",
    )
    code = main(["--model", "ollama:foo-json", "--contract", "text"])
    out = capsys.readouterr().out
    assert code == 2
    assert "json" in out


def test_report_only_json_through_main_re_applies_the_gate(tmp_path, monkeypatch, capsys):
    """The CLI path, not only the `_regate` unit itself: `--report-only
    --contract json` must actually reach `_regate` and print through
    `report_claims` — a deliberately wrong saved verdict must come back
    corrected, the same guarantee `_reaudit` already gives the text
    contract."""
    import legalrag.answer_eval as ae

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(ae, "load_docs", lambda path: [{"id": "law-7", "text": "نص المادة السابعة"}])

    rows_file = tmp_path / "runs" / "answer_eval-ollama-stub-json.json"
    rows_file.write_text(json.dumps([{
        "id": "Q1", "answerable": True, "category": "direct",
        "model": "ollama:stub", "contract": "json",
        "source_numbers": [7], "expected": [7],
        "parsed": {"abstain": False, "claims": [
            {"text": "الرد سبعة أيام.", "sources": [1]},
        ]},
        "raw": ['{"abstain": false, "claims": [{"text": "الرد سبعة أيام.", "sources": [1]}]}'],
        "attempts": 1, "schema_failure": False, "seconds": 1.0,
        # Deliberately wrong stored verdict — the CLI path must overwrite
        # it via _regate, not replay it.
        "gate": {"status": "abstained", "kept": [], "dropped": [],
                 "uncited": 0, "fabricated": 0, "ungrounded": 0,
                 "ignored_on_abstain": 0},
    }]), encoding="utf-8")

    code = main(["--model", "ollama:stub", "--contract", "json", "--report-only"])
    out = capsys.readouterr().out

    assert "re-gated" in out
    assert "coverage" in out
    assert "1/1" in out  # the one answerable question now has a kept claim
    assert code in (0, 1)


# --- Commit 1 (m2): argparse-based CLI parsing ------------------------------


def test_an_unrecognized_flag_like_a_report_only_typo_exits_2_without_a_full_run(
    monkeypatch, capsys
):
    """`--report_only` (underscore) is a plausible typo for `--report-only`
    but not a flag this CLI defines. The hand-rolled parser it replaces
    matched flags with `in argv`, so an unrecognized argument was simply
    never seen — the typo silently fell through to a full run instead of
    only reporting."""
    import legalrag.answer_eval as ae

    def boom():
        raise AssertionError("an unrecognized flag must not reach load_questions")
    monkeypatch.setattr(ae, "load_questions", boom)

    code = main(["--report_only"])
    out = capsys.readouterr().out

    assert code == 2
    assert "report_only" in out


def test_a_model_flag_with_no_value_exits_2_instead_of_crashing(monkeypatch, capsys):
    import legalrag.answer_eval as ae

    def boom():
        raise AssertionError("a malformed --model must not reach load_questions")
    monkeypatch.setattr(ae, "load_questions", boom)

    code = main(["--model"])
    out = capsys.readouterr().out

    assert code == 2
    assert "--model" in out
    assert "usage" in out.lower()  # proves argparse itself produced this, not a hand check


def test_main_catches_argparses_systemexit_and_returns_an_int(capsys):
    """`main` must stay a plain testable function — argparse raises
    `SystemExit` on any parse error, and that must never escape `main`."""
    try:
        code = main(["--contract", "not-a-real-contract"])
    except SystemExit:
        pytest.fail("main() must catch argparse's SystemExit and return its code")

    out = capsys.readouterr().out
    assert code == 2
    assert "usage" in out.lower()  # proves argparse itself produced this, not a hand check


# --- Commit 1 (m2): refusing to overwrite a saved run -----------------------


def test_a_full_run_without_overwrite_refuses_an_existing_rows_file_even_for_the_same_model(
    tmp_path, monkeypatch, capsys
):
    """Before this fix, a full run of the SAME model/contract silently
    overwrote its own previously saved rows — `_check_rows_collision` only
    ever refused a MISMATCHED model or contract. The rows in `runs/` are the
    only record of a measurement, so any existing file now blocks a full run
    unless `--overwrite` says the replacement is intentional."""
    import legalrag.answer_eval as ae

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(
        json.dumps([{"id": "Q1", "model": "ollama:mine"}]), encoding="utf-8"
    )
    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="text": fake_rows_file)

    def boom():
        raise AssertionError("load_questions must not run before the overwrite refusal")
    monkeypatch.setattr(ae, "load_questions", boom)

    code = main(["--model", "ollama:mine"])
    out = capsys.readouterr().out

    assert code == 2
    assert str(fake_rows_file) in out
    assert "runs" in out
    assert "overwrite" in out


def test_overwrite_lets_a_full_run_proceed_past_its_own_existing_rows_file(
    tmp_path, monkeypatch
):
    """`--overwrite` on the SAME model/contract that already has saved rows
    must let the run proceed — Commit 1 only refuses the silent,
    unconfirmed replacement, not every rerun."""
    import legalrag.answer_eval as ae

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(
        json.dumps([{"id": "Q1", "model": "ollama:mine"}]), encoding="utf-8"
    )
    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="text": fake_rows_file)

    reached = {"called": False}

    def stop_here():
        reached["called"] = True
        return [], ["stop the test here on purpose"]
    monkeypatch.setattr(ae, "load_questions", stop_here)

    code = main(["--model", "ollama:mine", "--overwrite"])

    assert reached["called"] is True
    assert code == 1  # main's own "errors from load_questions" path


def test_overwrite_does_not_bypass_a_genuine_collision_with_a_different_models_rows(
    tmp_path, monkeypatch, capsys
):
    """`--overwrite` means "replace MY OWN previous measurement" — it must
    not also license silently clobbering a DIFFERENT model's saved answers
    that happen to slugify to the same file name."""
    import legalrag.answer_eval as ae

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(
        json.dumps([{"id": "Q1", "model": "ollama:other"}]), encoding="utf-8"
    )
    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="text": fake_rows_file)

    def boom():
        raise AssertionError("load_questions must not run on a genuine collision")
    monkeypatch.setattr(ae, "load_questions", boom)

    code = main(["--model", "ollama:mine", "--overwrite"])
    out = capsys.readouterr().out

    assert code == 2
    assert "ollama:other" in out
    assert "ollama:mine" in out


# --- Commit 1 (m2): report-only re-scores only against the corpus it saw ---


def test_report_only_refuses_when_the_corpus_is_empty_or_missing(
    tmp_path, monkeypatch, capsys
):
    """Today `by_number.get(n, "")` silently substitutes empty text for a
    missing corpus, which is how re-gating Run 5 with no corpus changed
    coverage 15 -> 14 with no error at all. An empty/missing corpus must be
    refused outright, before any re-scoring runs."""
    import legalrag.answer_eval as ae

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(json.dumps([
        {"id": "Q1", "category": "direct", "answerable": True, "model": "ollama:mine",
         "text": "جواب [مادة 7].", "seconds": 1.0, "retrieved": [7], "expected": [7]},
    ]), encoding="utf-8")
    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="text": fake_rows_file)
    monkeypatch.setattr(ae, "load_docs", lambda path: [])

    code = main(["--model", "ollama:mine", "--report-only"])
    out = capsys.readouterr().out

    assert code == 2
    assert "corpus" in out


def test_reaudit_raises_when_a_retrieved_article_is_missing_from_the_loaded_corpus():
    """A retrieved article number the current corpus does not contain means
    this corpus does not match the one the run saw — that must be a clear
    error, not silently-empty context that could misclassify a real
    citation as fabricated."""
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]  # article 12 is missing
    rows = [{
        "id": "Q1", "text": "جواب [مادة 12].", "retrieved": [12], "expected": [12],
        "grounded": False, "cited": [], "fabricated": [], "ungrounded": [],
        "uncited": [], "abstained": False, "strict": [], "cited_expected": False,
    }]

    with pytest.raises(ValueError):
        _reaudit(rows, docs)


def test_regate_raises_when_a_sources_article_number_is_missing_from_the_loaded_corpus():
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]  # article 12 is missing
    rows = [{
        "id": "Q1", "answerable": True, "source_numbers": [12], "expected": [12],
        "parsed": {"abstain": False, "claims": [{"text": "نص", "sources": [1]}]},
        "gate": {"status": "abstained", "kept": [], "dropped": [],
                 "uncited": 0, "fabricated": 0, "ungrounded": 0, "ignored_on_abstain": 0},
    }]

    with pytest.raises(ValueError):
        _regate(rows, docs)


def test_reaudit_returns_new_row_dicts_without_mutating_the_ones_passed_in():
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    original = {
        "id": "Q1", "text": "جواب [مادة 7].", "retrieved": [7], "expected": [7],
        "grounded": False, "cited": [], "fabricated": [], "ungrounded": [],
        "uncited": [], "abstained": False, "strict": [], "cited_expected": False,
    }
    rows = [original]

    reaudited = _reaudit(rows, docs)

    assert reaudited[0]["grounded"] is True
    assert original["grounded"] is False  # the input dict must be untouched
    assert reaudited[0] is not original


def test_regate_returns_new_row_dicts_without_mutating_the_ones_passed_in():
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    original = {
        "id": "Q1", "answerable": True, "source_numbers": [7], "expected": [7],
        "parsed": {"abstain": False, "claims": [
            {"text": "الرد سبعة أيام.", "sources": [1]},
        ]},
        "gate": {"status": "abstained", "kept": [], "dropped": [],
                 "uncited": 0, "fabricated": 0, "ungrounded": 0, "ignored_on_abstain": 0},
    }
    rows = [original]

    regated = _regate(rows, docs)

    assert regated[0]["gate"]["status"] == "answered"
    assert original["gate"]["status"] == "abstained"  # the input dict must be untouched
    assert regated[0] is not original


def test_report_only_refuses_when_the_saved_corpus_fingerprint_does_not_match(
    tmp_path, monkeypatch, capsys
):
    import legalrag.answer_eval as ae

    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    monkeypatch.setattr(ae, "load_docs", lambda path: docs)

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(json.dumps([
        {"id": "Q1", "category": "direct", "answerable": True, "model": "ollama:mine",
         "text": "جواب [مادة 7].", "seconds": 1.0, "retrieved": [7], "expected": [7],
         "corpus_fingerprint": "not-" + corpus_fingerprint(docs)},
    ]), encoding="utf-8")
    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="text": fake_rows_file)

    code = main(["--model", "ollama:mine", "--report-only"])
    out = capsys.readouterr().out

    assert code == 2
    assert "corpus" in out.lower() or "fingerprint" in out.lower()


def test_report_only_warns_but_proceeds_when_saved_rows_predate_corpus_fingerprints(
    tmp_path, monkeypatch, capsys
):
    """Run 3/4/5's saved rows were written before `corpus_fingerprint`
    existed — they must still be reportable, with a warning that the corpus
    could not be verified, not refused outright, or none of the saved
    invariant runs this project pins numbers to could be reported any more."""
    import legalrag.answer_eval as ae

    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    monkeypatch.setattr(ae, "load_docs", lambda path: docs)

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(json.dumps([
        {"id": "Q1", "category": "direct", "answerable": True, "model": "ollama:mine",
         "text": "جواب [مادة 7].", "seconds": 1.0, "retrieved": [7], "expected": [7]},
    ]), encoding="utf-8")
    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="text": fake_rows_file)

    code = main(["--model", "ollama:mine", "--report-only"])
    out = capsys.readouterr().out

    assert code in (0, 1)
    assert "fingerprint" in out.lower()  # the specific warning, not just boilerplate


def test_report_only_refuses_clearly_when_a_json_rows_file_actually_holds_text_rows(
    tmp_path, monkeypatch, capsys
):
    """A `-json` rows file that actually holds text-contract rows (no
    `parsed`/`source_numbers` fields) must be reported as a clear mismatch,
    not crash `_regate` with a `KeyError` on a field text rows never had."""
    import legalrag.answer_eval as ae

    fake_rows_file = tmp_path / "fake_rows.json"
    fake_rows_file.write_text(json.dumps([
        {"id": "Q1", "category": "direct", "answerable": True, "model": "ollama:mine",
         "contract": "text", "text": "جواب [مادة 7].", "seconds": 1.0,
         "retrieved": [7], "expected": [7]},
    ]), encoding="utf-8")
    monkeypatch.setattr(ae, "rows_path", lambda spec, contract="json": fake_rows_file)

    code = main(["--model", "ollama:mine", "--contract", "json", "--report-only"])
    out = capsys.readouterr().out

    assert code == 2
    assert "text" in out


# --- Commit 1 (m2): saved rows carry their own corpus provenance ------------


def test_run_rows_record_the_corpus_fingerprint_and_the_retrieved_source_ids():
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    hit7 = Hit(id="law-7", law_name="", number=7, score=1.0, snippet="")
    index = _StubIndex({"س1": [hit7]})
    generator = Generator(model=lambda messages: "جواب [مادة 7].")

    rows = run([_question("Q1", "س1")], docs, generator, "ollama:some-model", index=index)

    assert rows[0]["corpus_fingerprint"] == corpus_fingerprint(docs)
    assert rows[0]["retrieved_ids"] == ["law-7"]


def test_run_claims_rows_record_the_corpus_fingerprint_and_the_source_ids():
    docs = [{"id": "law-7", "text": "نص المادة السابعة"}]
    hit7 = Hit(id="law-7", law_name="", number=7, score=1.0, snippet="")
    index = _StubIndex({"س1": [hit7]})
    good = '{"abstain": false, "claims": [{"text": "الرد سبعة أيام.", "sources": [1]}]}'
    model = _StubCallModel([(good, {"output_tokens": 9, "output_s": 1.0})])
    generator = ClaimsGenerator(model=model)

    rows = run_claims([_question("Q1", "س1")], docs, generator, "ollama:stub", index=index)

    assert rows[0]["corpus_fingerprint"] == corpus_fingerprint(docs)
    assert rows[0]["source_ids"] == ["law-7"]


# --- Commit 2 (m2): the CLI path must actually reach _regate ----------------


def test_report_only_json_through_main_prints_the_exact_regated_coverage_line_not_the_saved_one(
    tmp_path, monkeypatch, capsys
):
    """A weak "1/1 appears somewhere in the output" check could pass by
    accident if some OTHER unrelated line happened to contain that same
    substring. This pins the EXACT labelled coverage line `_regate` must
    produce, confirms the WRONG saved verdict's line is gone, and checks
    the exit code — a mutant that skips `_regate` (using the saved, wrong
    `gate` as-is: `status: abstained, kept: []`) would keep printing the
    saved 0/1 line and fail every assertion here."""
    import legalrag.answer_eval as ae

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(ae, "load_docs", lambda path: [{"id": "law-7", "text": "نص المادة السابعة"}])

    rows_file = tmp_path / "runs" / "answer_eval-ollama-stub-json.json"
    rows_file.write_text(json.dumps([{
        "id": "Q1", "answerable": True, "category": "direct",
        "model": "ollama:stub", "contract": "json",
        "source_numbers": [7], "expected": [7],
        "parsed": {"abstain": False, "claims": [
            {"text": "الرد سبعة أيام.", "sources": [1]},
        ]},
        "raw": ['{"abstain": false, "claims": [{"text": "الرد سبعة أيام.", "sources": [1]}]}'],
        "attempts": 1, "schema_failure": False, "seconds": 1.0,
        # Deliberately wrong stored verdict — the CLI path must overwrite
        # it via _regate, not replay it.
        "gate": {"status": "abstained", "kept": [], "dropped": [],
                 "uncited": 0, "fabricated": 0, "ungrounded": 0,
                 "ignored_on_abstain": 0},
    }]), encoding="utf-8")

    code = main(["--model", "ollama:stub", "--contract", "json", "--report-only"])
    out = capsys.readouterr().out

    assert "  coverage (>=1 kept claim)          : 1/1 = 100.0%" in out
    assert "  coverage (>=1 kept claim)          : 0/1 = 0.0%" not in out
    assert code == 1  # no out_of_corpus rows here -> B2 = 0% < 80%, not adopted
