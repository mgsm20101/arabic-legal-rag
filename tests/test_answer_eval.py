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
    monkeypatch.setattr(ae, "load_docs", lambda path: [])

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
