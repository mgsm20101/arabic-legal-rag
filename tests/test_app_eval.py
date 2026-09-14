"""App eval — the upload pipeline on the app-dev split (ADR-023), Phase 4B task 4.

`tasks.py app-eval` scores retrieval in every mode and, without
--retrieval-only, runs the whole ask pipeline over all 15 questions. That
full run calls a model, and its adoption rule is not pre-registered yet, so
nothing here runs one: the library gets a keyword encoder, the generator is
scripted, and --retrieval-only is shown never to build a generator at all.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")

import legalrag.app_eval as app_eval  # noqa: E402
from legalrag.claims import ClaimsGenerator  # noqa: E402
from legalrag.library import Library  # noqa: E402
from legalrag.ollama import GeneratorUnavailable  # noqa: E402
from stubs import ANSWERS_NO, ANSWERS_YES, KeywordEncoder, ScriptedChat, claims_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROW_FIELDS = {"id", "category", "answerable", "question", "status", "abstain_reason", "claims",
              "source_chunk_ids", "source_pages", "dropped", "timings_ms"}
CLAIM = "جملة من المصدر الأول."


@pytest.fixture
def offline(monkeypatch, tmp_path) -> Path:
    """The committed fixture, a keyword encoder instead of e5, and runs/ inside
    tmp_path: no model is loaded and the repository's runs/ is never touched."""
    monkeypatch.setattr(app_eval, "APP_DIR", ROOT / "evals" / "app")
    monkeypatch.setattr(app_eval, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(app_eval, "Library", lambda root: Library(root, encoder=KeywordEncoder()))
    monkeypatch.delenv("LEGALRAG_MODEL", raising=False)
    return tmp_path / "runs"


def _scripted_generator(fail_on_question: int | None = None) -> ClaimsGenerator:
    """Relevance says yes to the 10 answerable questions and no to the 5
    out_of_doc ones (the file's order); every claims reply cites source 1.
    `fail_on_question` makes that question's relevance call raise, the way an
    Ollama server that went away does."""
    relevance = [ANSWERS_YES] * 10 + [ANSWERS_NO] * 5
    if fail_on_question is not None:
        relevance[fail_on_question - 1] = GeneratorUnavailable("Ollama went away")
    return ClaimsGenerator(model=ScriptedChat([claims_json((CLAIM, [1]))] * 10),
                           relevance_model=ScriptedChat(relevance))


def test_a_retrieval_hit_needs_every_expected_keyword_after_normalization():
    # Arabic-Indic digits in the chunk, Western ones in the keyword: the same number.
    assert app_eval.retrieval_hit(["تصرف الشركة بدل إنترنت قدره ٤٠٠ جنيه."], ["400 جنيه"])
    # Tatweel and diacritics hide nothing either.
    assert app_eval.retrieval_hit(["خلال خمـسة أيّام عمل"], ["خمسة أيام عمل"])
    # EVERY keyword, and all of them in one chunk.
    both = ["إصابة العمل", "وأربعين ساعة"]
    assert app_eval.retrieval_hit(["تعامل معاملة إصابة العمل بشرط الإبلاغ خلال ثمان وأربعين ساعة"], both)
    assert not app_eval.retrieval_hit(["تعامل معاملة إصابة العمل", "خلال ثمان وأربعين ساعة"], both)
    assert not app_eval.retrieval_hit(["يرد خلال ثلاثة أيام عمل"], ["خمسة أيام عمل"])
    assert not app_eval.retrieval_hit([], ["400 جنيه"])
    assert not app_eval.retrieval_hit(["أي نص"], [])  # no keyword is no evidence


def test_retrieval_only_never_builds_a_generator(offline, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("--retrieval-only built a generator")

    monkeypatch.setattr(app_eval, "build_generators", refuse)

    code = app_eval.main(["--retrieval-only", "--doc", "txt", "--model", "ollama:never-used"])

    out = capsys.readouterr().out
    assert code == 0
    assert "generic" in out and "1 page" in out
    assert all(f"A{n:02d}" in out for n in range(1, 11))
    assert "O01" not in out  # an out_of_doc question has nothing to retrieve
    assert re.search(r"hit@5\s*:\s*\d+/10", out)
    assert re.search(r"page-hit@5\s*:\s*n/a", out)  # one page: no page to hit
    assert not offline.exists(), "a retrieval-only run wrote to runs/"


def test_a_saved_app_eval_run_is_not_overwritten_without_the_flag(offline, monkeypatch, capsys):
    saved = offline / "app_eval-ollama-gemma3-4b-txt.json"
    saved.parent.mkdir(parents=True)
    saved.write_text('[{"id": "A01", "saved": true}]', encoding="utf-8")
    built, opened = [], []
    monkeypatch.setattr(app_eval, "build_generators",
                        lambda spec, contract: built.append((spec, contract)) or _scripted_generator())
    monkeypatch.setattr(app_eval, "Library",
                        lambda root: opened.append(root) or Library(root, encoder=KeywordEncoder()))

    assert app_eval.main(["--doc", "txt"]) == 2
    assert saved.read_text(encoding="utf-8") == '[{"id": "A01", "saved": true}]'
    assert (built, opened) == ([], []), "a refused run still started"
    assert "--overwrite" in capsys.readouterr().out

    assert app_eval.main(["--doc", "txt", "--overwrite"]) == 0
    assert built == [("ollama:gemma3:4b", "gated")]
    rows = json.loads(saved.read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == [q["id"] for q in app_eval.load_questions()]


def test_rows_are_saved_after_every_question(offline, monkeypatch, capsys):
    monkeypatch.setattr(app_eval, "build_generators",
                        lambda spec, contract: _scripted_generator(fail_on_question=2))

    code = app_eval.main(["--doc", "txt", "--model", "ollama:gemma3:4b"])

    assert code == 5
    assert "Ollama went away" in capsys.readouterr().out
    rows = json.loads((offline / "app_eval-ollama-gemma3-4b-txt.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == ["A01"]
    assert set(rows[0]) == ROW_FIELDS
    assert (rows[0]["status"], rows[0]["claims"]) == ("answered", [{"text": CLAIM, "sources": [1]}])
    assert len(rows[0]["source_chunk_ids"]) == len(rows[0]["source_pages"]) == 5
    assert [p.name for p in offline.iterdir()] == ["app_eval-ollama-gemma3-4b-txt.json"]


def test_the_model_defaults_to_legalrag_model(offline, monkeypatch):
    built = []
    monkeypatch.setattr(app_eval, "build_generators",
                        lambda spec, contract: built.append(spec) or _scripted_generator())
    monkeypatch.setenv("LEGALRAG_MODEL", "ollama:qwen2.5:7b-instruct")

    assert app_eval.main(["--doc", "txt"]) == 0

    assert built == ["ollama:qwen2.5:7b-instruct"]
    assert (offline / "app_eval-ollama-qwen2.5-7b-instruct-txt.json").exists()


def test_the_summary_counts_each_number_over_the_questions_it_is_about():
    questions = [
        {"id": "A1", "answerable": True, "expected_pages": [2]},
        {"id": "A2", "answerable": True, "expected_pages": [5]},
        {"id": "A3", "answerable": True, "expected_pages": [1]},
        {"id": "O1", "answerable": False, "expected_pages": []},
        {"id": "O2", "answerable": False, "expected_pages": []},
    ]

    def row(qid, status, claims, pages, fabricated=0, total_ms=1000):
        return {"id": qid, "status": status, "claims": claims, "source_pages": pages,
                "dropped": {"uncited": 0, "fabricated": fabricated, "ungrounded": 0},
                "timings_ms": {"total": total_ms}}

    rows = [
        row("A1", "answered", [{"text": "x", "sources": [2]}], [1, 2]),            # grounded: page 2
        row("A2", "partial", [{"text": "y", "sources": [1]}], [3, 5], fabricated=1),  # cites page 3, not 5
        row("A3", "abstained", [], [1]),                                            # a false abstention
        row("O1", "abstained", [], [4], total_ms=3000),                             # abstained, as it should
        row("O2", "answered", [{"text": "z", "sources": [1]}], [2]),                # answered out of doc
    ]

    assert app_eval.summarize(rows, questions) == {
        "coverage": (2, 3),
        "page_grounded": (1, 3),
        "out_of_doc_abstentions": (1, 2),
        "false_abstentions": (1, 3),
        "fabricated": 1,
        "mean_seconds": 1.4,
    }


def test_tasks_py_offers_app_eval():
    """tasks.py's docstring IS the usage `python tasks.py` prints."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("tasks_under_test", ROOT / "tasks.py")
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)

    assert "app-eval" in tasks.TASKS
    assert "app-eval [--retrieval-only] [--doc pdf|txt] [--model SPEC] [--overwrite]" in tasks.__doc__
