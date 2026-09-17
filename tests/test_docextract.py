"""Extracting a stored PDF upload — P1 demo layer (ADR-023), now isolated in a subprocess (T04).

`test_library` uploads through `Library.add`; these tests call `docextract` on its own.

Most tests here monkeypatch `docextract._extract_pages_isolated` rather than
`pdf_text.extract_pages`: extraction now runs through `isolate.run_isolated`, which pickles its
`target` BY REFERENCE and hands it to a freshly spawned subprocess — that child re-imports
`pdf_text` from disk, so a monkeypatch on the *parent* process's `pdf_text.extract_pages` has no
effect on it at all. `_extract_pages_isolated` is the seam that stays fast and in-process:
replacing it bypasses `run_isolated`/multiprocessing entirely, which is what lets these tests use
ordinary closures (call-tracking lists and all) the way they always have. A couple of tests
further down deliberately do NOT replace that seam, to prove the real subprocess pipeline still
behaves — see their docstrings.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from time import monotonic

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import legalrag.docextract as docextract  # noqa: E402
from legalrag import pdf_text  # noqa: E402
from legalrag.docerrors import PdfTooLarge, StorageError, UnsupportedFile  # noqa: E402
from legalrag.isolate import IsolationCrash  # noqa: E402

DOC_ID = "0123456789ab"
DOCUMENT_TEXT = "يلتزم المتحكم بالحصول على موافقة صريحة من الشخص المعني"
REPORT_TEXT = "«مقتطف من السطر الذي حُسم عليه القرار»"
STATUTE_LIKE_PAGE = "مادة 1\nنص المادة الأولى مصحوب بترجمة Article 1 مواكبة لها"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _extractor(pages: dict[bool, list[str]], report_items: dict, calls: list):
    """`_extract_pages_isolated` over a PDF whose pages are `pages[keep_latin]`, returning
    `report_items` as the mirroring report — see the module docstring for why this seam, and
    not `pdf_text.extract_pages` or `docextract._extract_pages_worker`, is the one to replace."""
    def fake(path, keep_latin, max_pages, deadline):
        calls.append(keep_latin)
        return list(pages[keep_latin]), dict(report_items)

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
    monkeypatch.setattr(docextract, "_extract_pages_isolated", _extractor(pages, report, calls))

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


def test_a_non_deadline_failure_on_the_arabic_only_pass_is_a_storage_error_never_the_clients_fault(
    tmp_path, monkeypatch,
):
    """F1: the first pass already proved the file readable, so a bug on the second, Arabic-only
    pass is never the client's fault the way a damaged PDF (raised as UnsupportedFile) is —
    only ExtractionLimitExceeded there means "ran out of time", everything else is a bug here."""
    calls: list[bool] = []

    def fake(path, keep_latin, max_pages, deadline):
        calls.append(keep_latin)
        if not keep_latin:
            raise RuntimeError("a pdfminer bug, not a bad file")
        return [STATUTE_LIKE_PAGE], {}

    monkeypatch.setattr(docextract, "_extract_pages_isolated", fake)

    with pytest.raises(StorageError) as failed:
        docextract.extract_pdf(DOC_ID, tmp_path / "source.pdf", "law.pdf")

    assert failed.value.code == "internal"
    assert calls == [True, False]


def test_a_first_pass_worker_crash_becomes_unsupported_file(tmp_path, monkeypatch):
    """A crash — as opposed to an ExtractionLimitExceeded — on the FIRST pass means the file
    itself is presumably bad: it must map the same way a directly-raised exception always did."""
    def fake(path, keep_latin, max_pages, deadline):
        raise IsolationCrash("the worker exited unexpectedly")

    monkeypatch.setattr(docextract, "_extract_pages_isolated", fake)

    with pytest.raises(UnsupportedFile):
        docextract.extract_pdf(DOC_ID, tmp_path / "source.pdf", "law.pdf")


def test_a_second_pass_worker_crash_becomes_storage_error(tmp_path, monkeypatch):
    """Same crash, but on the Arabic-only pass: the first pass already proved this exact file
    readable, so this is StorageError (this side's fault), never UnsupportedFile."""
    calls: list[bool] = []

    def fake(path, keep_latin, max_pages, deadline):
        calls.append(keep_latin)
        if keep_latin:
            return [STATUTE_LIKE_PAGE], {}
        raise IsolationCrash("the worker exited unexpectedly")

    monkeypatch.setattr(docextract, "_extract_pages_isolated", fake)

    with pytest.raises(StorageError):
        docextract.extract_pdf(DOC_ID, tmp_path / "source.pdf", "law.pdf")

    assert calls == [True, False]


def test_extraction_over_the_character_cap_raises_pdf_too_large(tmp_path, monkeypatch):
    """MAX_PDF_PAGES only caps page COUNT: a single pathologically dense page must still be
    refused, via the separate MAX_EXTRACTED_CHARS ceiling."""
    huge_page = "x" * (docextract.MAX_EXTRACTED_CHARS + 1)

    def fake(path, keep_latin, max_pages, deadline):
        return [huge_page], {}

    monkeypatch.setattr(docextract, "_extract_pages_isolated", fake)

    with pytest.raises(PdfTooLarge):
        docextract.extract_pdf(DOC_ID, tmp_path / "source.pdf", "law.pdf")


def test_the_semaphore_limits_concurrent_extractions(monkeypatch):
    """MAX_CONCURRENT_EXTRACTIONS bounds how many isolated extractions run at once, so a burst
    of uploads cannot spawn an unbounded number of subprocesses. Uses a short-sleeping fake in
    place of `run_isolated` itself (not `_extract_pages_isolated`, which is where the semaphore
    lives) so the semaphore's own gating is what's under test, not the seam that bypasses it."""
    limit = docextract.MAX_CONCURRENT_EXTRACTIONS
    lock = threading.Lock()
    current = 0
    peak = 0

    def fake_run_isolated(target, args, timeout_seconds):
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.2)
        with lock:
            current -= 1
        return [], {}

    monkeypatch.setattr(docextract, "run_isolated", fake_run_isolated)

    threads = [
        threading.Thread(
            target=docextract._extract_pages_isolated,
            args=(Path("x.pdf"), True, docextract.MAX_PDF_PAGES, monotonic() + 10),
        )
        for _ in range(limit + 3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert peak <= limit, f"concurrency exceeded the {limit}-slot limit: saw {peak} at once"
    assert peak == limit, f"expected concurrency to reach the {limit}-slot limit; only saw {peak}"


def test_a_first_pass_exceeding_its_deadline_is_isolated_terminated_and_becomes_pdf_too_large(monkeypatch):
    """Genuine subprocess isolation, no monkeypatching of the isolation seam: with the
    extraction budget already exhausted before the worker can even finish spawning,
    run_isolated has to actually kill the OS process, and extract_pdf must still map that into
    PdfTooLarge — matching current behaviour for the old in-loop DeadlineExceeded, now enforced
    preemptively instead of cooperatively."""
    monkeypatch.setattr(docextract, "EXTRACTION_BUDGET_SECONDS", -5.0)

    with pytest.raises(PdfTooLarge):
        docextract.extract_pdf(DOC_ID, FIXTURES / "brackets_ar.pdf", "law.pdf")


def test_a_real_small_pdf_extracts_correctly_through_the_full_isolation_pipeline():
    """No monkeypatching at all: the real _extract_pages_worker runs in a real subprocess
    against a real, small PDF fixture already used elsewhere in this suite (tests/test_pdf_text.py),
    and extract_pdf gets back exactly what the direct, in-process pdf_text.extract_pages call
    would have produced. brackets_ar.pdf has no "مادة" headers (may_be_statute is False for it),
    so extract_pdf only runs the one, keep_latin=True pass — matching the direct call below."""
    direct = pdf_text.extract_pages(FIXTURES / "brackets_ar.pdf", keep_latin=True, max_pages=docextract.MAX_PDF_PAGES)

    result = docextract.extract_pdf(DOC_ID, FIXTURES / "brackets_ar.pdf", "brackets.pdf")

    assert result.pages == direct
    assert result.kind == "generic"
