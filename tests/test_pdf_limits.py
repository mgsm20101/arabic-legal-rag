"""Extraction limits for a PDF nobody vetted — security review, MEDIUM.

`extract_pages` had neither a page cap nor a time limit: 6,000 blank pages
took 8.7 s to extract, and a 20 MiB upload can hold 100,000 pages or more.
Two optional limits, both off by default so every existing caller extracts
exactly as before: `max_pages` refuses a longer document before any page is
extracted, and `deadline` is checked between pages. The PDFs are written by
hand (`stubs.blank_pdf`); the clock is injected, never slept on.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("pdfplumber")

from legalrag import pdf_text  # noqa: E402
from legalrag.pdf_text import extract_pages  # noqa: E402
from stubs import blank_pdf  # noqa: E402


@pytest.fixture
def extracted(monkeypatch) -> list[int]:
    """One entry per page whose glyphs were grouped into lines — every page
    `extract_pages` extracts passes through `group_lines` exactly once."""
    calls: list[int] = []
    real_group_lines = pdf_text.group_lines

    def counting(chars, tol=None):
        calls.append(len(chars))
        return real_group_lines(chars, tol)

    monkeypatch.setattr(pdf_text, "group_lines", counting)
    return calls


def _write(tmp_path: Path, page_count: int) -> Path:
    path = tmp_path / f"blank-{page_count}.pdf"
    path.write_bytes(blank_pdf(page_count))
    return path


def test_without_limits_or_within_them_every_page_is_extracted(tmp_path, extracted):
    pdf = _write(tmp_path, 3)

    assert extract_pages(pdf) == ["", "", ""]
    assert extract_pages(pdf, keep_latin=True, max_pages=3, deadline=pdf_text.monotonic() + 3600) == ["", "", ""]
    assert len(extracted) == 6


def test_a_pdf_over_max_pages_is_refused_before_any_page_is_extracted(tmp_path, extracted):
    pdf = _write(tmp_path, 5)

    with pytest.raises(pdf_text.TooManyPages) as refused:
        extract_pages(pdf, keep_latin=True, max_pages=4)

    assert extracted == []
    assert refused.value.max_pages == 4
    assert isinstance(refused.value, pdf_text.ExtractionLimitExceeded)


def test_refusing_a_pdf_far_over_the_cap_reads_no_more_of_its_page_tree_than_the_cap(tmp_path, monkeypatch):
    """pdfplumber's `pdf.pages` builds every page before it can be counted —
    for a 20 MiB upload that is 100,000 page objects spent on a refusal."""
    from pdfminer.pdfpage import PDFPage

    pdf = _write(tmp_path, 2000)
    created: list[int] = []
    real_init = PDFPage.__init__

    def counting_init(self, *args, **kwargs):
        created.append(1)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(PDFPage, "__init__", counting_init)

    with pytest.raises(pdf_text.TooManyPages):
        extract_pages(pdf, max_pages=10)

    assert len(created) <= 11


def test_a_passed_deadline_stops_extraction_after_the_page_in_progress(tmp_path, extracted):
    pdf = _write(tmp_path, 4)

    with pytest.raises(pdf_text.DeadlineExceeded) as stopped:
        extract_pages(pdf, deadline=pdf_text.monotonic() - 1)

    assert len(extracted) == 1  # the first page runs; the check comes before the next one
    assert (stopped.value.pages_done, stopped.value.pages) == (1, 4)
    assert isinstance(stopped.value, pdf_text.ExtractionLimitExceeded)


def test_a_deadline_passing_mid_document_lets_that_page_finish_and_starts_no_other(tmp_path, monkeypatch, extracted):
    pdf = _write(tmp_path, 4)
    readings = iter([5.0, 11.0])  # the clock as page 2, then page 3, is about to start
    monkeypatch.setattr(pdf_text, "monotonic", lambda: next(readings))

    with pytest.raises(pdf_text.DeadlineExceeded) as stopped:
        extract_pages(pdf, deadline=10.0)

    assert len(extracted) == 2
    assert (stopped.value.pages_done, stopped.value.pages) == (2, 4)
