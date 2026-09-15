"""Extracting a stored PDF upload — P1 demo layer (ADR-023).

`test_library` uploads through `Library.add`; these tests call `docextract` on
its own, with `pdf_text.extract_pages` replaced, for what an extraction decides
and what it may log.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import legalrag.docextract as docextract  # noqa: E402
from legalrag import pdf_text  # noqa: E402

DOC_ID = "0123456789ab"
DOCUMENT_TEXT = "يلتزم المتحكم بالحصول على موافقة صريحة من الشخص المعني"
REPORT_TEXT = "«مقتطف من السطر الذي حُسم عليه القرار»"


def _extractor(pages: dict[bool, list[str]], report_items: dict, calls: list):
    """`extract_pages` over a PDF whose pages are `pages[keep_latin]`, filling a report with `report_items`."""
    def fake(path, keep_latin=False, line_tol=None, *, max_pages=None, deadline=None, report=None):
        calls.append(keep_latin)
        if report is not None:
            report.update(report_items)
        return list(pages[keep_latin])

    return fake


def test_each_extraction_logs_its_mirroring_counts_and_flags_but_no_string_and_no_document_text(
    tmp_path, monkeypatch, caplog,
):
    """The report's keys are pdf_text's to change, so every count and flag in it is logged,
    and nothing else: a string there could carry the document's own text."""
    pages = {True: [f"مادة 1\n{DOCUMENT_TEXT} Article 1"], False: [f"مادة 1\n{DOCUMENT_TEXT}"]}
    report = {"bracket_logical": 3, "mirror_brackets": True, "quote_mirrored": 0,
              "evidence": REPORT_TEXT, "lines": [DOCUMENT_TEXT]}
    calls: list[bool] = []
    monkeypatch.setattr(pdf_text, "extract_pages", _extractor(pages, report, calls))

    with caplog.at_level(logging.DEBUG):
        docextract.extract_pdf(DOC_ID, tmp_path / "source.pdf", "law.pdf")

    assert calls == [True, False]
    for keep_latin in (True, False):
        [line] = [r.getMessage() for r in caplog.records
                  if r.levelno == logging.INFO and f"keep_latin={keep_latin}" in r.getMessage()]
        assert DOC_ID in line
        assert all(item in line for item in ("'bracket_logical': 3", "'mirror_brackets': True", "'quote_mirrored': 0"))
    logged = caplog.text + " ".join(repr(r.args) for r in caplog.records)
    assert REPORT_TEXT not in logged and "evidence" not in logged
    assert DOCUMENT_TEXT not in logged and "Article" not in logged
