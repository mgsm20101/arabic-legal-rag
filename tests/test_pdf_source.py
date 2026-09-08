"""PDF/TXT source tests (ADR-012 — the corpus comes from a PDF again).

Two regressions these lock down:

1. The old PDF branch left ``law_name`` empty, which silently disarmed the
   ADR-010 corpus/question binding guard: ``binding_problem`` returns None on
   an empty law list, so 151/2020 questions scored against the wrong law would
   have looked like weak retrieval. ``--law`` is now mandatory.
2. Arabic PDF extraction fails in ways that produce *plausible* output — a
   scanned page yields nothing, a bad embedded font yields Latin gibberish,
   and RTL run-order damage yields text with no recognisable article headers.
   All three previously ended as "ingested 0 articles" with no explanation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag import ingest  # noqa: E402

GOOD = """
مادة ( 1 )
في تطبيق أحكام هذا القانون يقصد بالمصطلحات الآتية المعاني المبينة قرين كل منها،
وذلك ما لم يقتض سياق النص خلاف ذلك، ويصدر بتحديدها قرار من الوزير المختص.

مادة ( 2 )
لا يجوز جمع البيانات الشخصية إلا بموافقة صريحة من الشخص المعني، ولا يجوز
معالجتها أو الإفصاح عنها أو إتاحتها للغير بأي وسيلة إلا في الأحوال المقررة قانونا.
"""


# ---------------------------------------------------------------- sources()

def test_sources_finds_pdf_and_txt(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    (tmp_path / "law.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "law.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in ingest.sources()] == ["law.pdf", "law.txt"]


def test_sources_ignores_leftover_corpus_directory(tmp_path, monkeypatch):
    """The 23 MB parquet from ADR-007 may still sit in data/raw/ — it is not a source."""
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    nested = tmp_path / "egypt-legal-corpus" / "data"
    nested.mkdir(parents=True)
    (nested / "train-00000-of-00001.parquet").write_bytes(b"PAR1")
    (tmp_path / "SOURCE.txt").write_text("provenance card", encoding="utf-8")
    (tmp_path / "law.pdf").write_bytes(b"%PDF-1.4")
    assert [p.name for p in ingest.sources()] == ["law.pdf"]


# ------------------------------------------------------ extraction_problems

def test_extraction_flags_a_page_with_no_text_layer():
    problems = ingest.extraction_problems("", "scan.pdf")
    assert any("no text" in p.lower() for p in problems)


def test_extraction_flags_latin_gibberish_from_a_broken_font():
    garbled = "Cjhg jklm nopq rstu vwxy zabc defg hijk lmno pqrs tuvw xyza bcde fghi jklm nopq" * 4
    problems = ingest.extraction_problems(garbled, "law.pdf")
    assert any("arabic" in p.lower() for p in problems)


def test_extraction_flags_arabic_text_with_no_article_headers():
    prose = "هذا نص عربي سليم تماما لكنه لا يحتوي على أي ترويسة مادة على الإطلاق. " * 8
    problems = ingest.extraction_problems(prose, "law.pdf")
    assert any("header" in p.lower() for p in problems)


def test_extraction_accepts_a_clean_arabic_law():
    assert ingest.extraction_problems(GOOD, "law.pdf") == []


# ------------------------------------------------------------- main() / --law

def test_ingest_refuses_without_law(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(ingest, "OUT_PATH", tmp_path / "out" / "articles.jsonl")
    (tmp_path / "law.txt").write_text(GOOD, encoding="utf-8")
    assert ingest.main([]) == 1
    assert "--law" in capsys.readouterr().out


def test_ingest_stamps_law_name_so_the_binding_guard_stays_armed(tmp_path, monkeypatch):
    import json

    from legalrag.evaluate import binding_problem, corpus_laws

    out = tmp_path / "out" / "articles.jsonl"
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(ingest, "OUT_PATH", out)
    (tmp_path / "law.txt").write_text(GOOD, encoding="utf-8")

    assert ingest.main(["--law", "قانون حماية البيانات الشخصية"]) == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["law_name"] for r in rows} == {"قانون حماية البيانات الشخصية"}

    laws = corpus_laws(out)
    assert binding_problem({"corpus_law": "حماية البيانات الشخصية"}, laws) is None
    assert binding_problem({"corpus_law": "المرافعات"}, laws) is not None


def test_ingest_fails_loudly_on_an_unreadable_pdf(tmp_path, monkeypatch, capsys):
    """Extraction damage must stop the run, not produce a corpus of 0 articles."""
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(ingest, "OUT_PATH", tmp_path / "out" / "articles.jsonl")
    (tmp_path / "law.txt").write_text("Cjhg jklm nopq rstu vwxy zabc defg hijk" * 6, encoding="utf-8")
    assert ingest.main(["--law", "أي قانون"]) == 2
    assert "law.txt" in capsys.readouterr().out


def test_unsupported_extension_is_rejected(tmp_path):
    p = tmp_path / "corpus.parquet"
    p.write_bytes(b"PAR1")
    with pytest.raises(SystemExit):
        ingest._read_raw(p)
