"""A stored PDF upload's pages and chunks.

At most two extractions, under one budget. The first keeps Latin, which suits
an Arabic document's inline terms (VPN, Wi-Fi) but welds a bilingual statute's
translation column into its Arabic lines: Law 151/2020 shows 4 of its 56
articles that way. So pages showing any article header are extracted again,
Arabic-only, and chunked as a statute if that validates. Each extraction logs
its bracket and « » mirroring decision, as counts and flags only.

Each extraction runs in a killable subprocess (`isolate.run_isolated`):
pdf_text's own deadline check runs only between pages, so a pathological page
could otherwise run forever.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from time import monotonic
from typing import NamedTuple

from . import pdf_text
from .chunking import Chunk, may_be_statute, page_chunks, statute_chunks
from .docerrors import PdfTooLarge, StorageError, UnsupportedFile
from .isolate import IsolationCrash, IsolationTimeout, run_isolated

logger = logging.getLogger(__name__)

# A PDF past either limit is refused. Extraction was measured at 0.8-1.26 s a page, each pass,
# so the budget follows the cap: budget ≈ cap × 1 s × 1.2. A statute's second, Arabic-only pass
# shares the budget, and one that runs out leaves page chunks, not a refusal.
MAX_PDF_PAGES = 250
EXTRACTION_BUDGET_SECONDS = 300

# A second ceiling, on text rather than pages: 20,000 characters a page is several
# times a dense legal page, so only a pathological PDF reaches it.
MAX_EXTRACTED_CHARS = MAX_PDF_PAGES * 20_000  # 5,000,000 characters

# Each extraction is a subprocess; two at once is plenty for a single-user app.
MAX_CONCURRENT_EXTRACTIONS = 2
_extraction_slots = threading.BoundedSemaphore(MAX_CONCURRENT_EXTRACTIONS)

# POSIX-only address-space cap per worker; pdfplumber used ~1.03 GB at 600 pages.
WORKER_MEMORY_LIMIT_BYTES = 1 * 1024 * 1024 * 1024


class PdfChunks(NamedTuple):
    pages: list[str]     # the first extraction's, Latin kept: what the document's pages and characters count
    kind: str            # "statute" | "generic"
    chunks: list[Chunk]
    # True only when this looked like a statute but its Arabic-only pass hit the deadline (the
    # page limit cannot fire here: see docstore.DocMeta.fell_back_to_pages) and so was never
    # checked: same bytes, a slower run, "generic" instead of "statute".
    fell_back_to_pages: bool


def extract_pdf(doc_id: str, source: Path, title: str) -> PdfChunks:
    """The PDF at `source` as pages and chunks: PdfTooLarge when its first extraction passes a limit,
    UnsupportedFile when that cannot read it. Pages showing any article header are extracted again
    Arabic-only, under the same deadline, and chunked as a statute if that validates. Otherwise they
    stay page chunks, and so they do when that second pass runs out of time: they are a usable document already."""
    deadline = monotonic() + EXTRACTION_BUDGET_SECONDS  # one budget for both extractions
    try:
        pages = _pages(doc_id, source, True, deadline)
    except pdf_text.ExtractionLimitExceeded as e:
        raise PdfTooLarge(str(e)) from e
    except Exception as e:  # a damaged PDF raises whatever pdfminer hits first
        raise UnsupportedFile("the PDF could not be read") from e
    fell_back_to_pages = False
    if may_be_statute(pages):
        try:
            arabic = _pages(doc_id, source, False, deadline)
        except pdf_text.ExtractionLimitExceeded as e:  # the deadline: these pages already passed the page cap
            logger.warning("document %s, %d pages: its Arabic-only extraction stopped (%s), so it is "
                           "chunked by page and not checked as a statute", doc_id, len(pages), e)
            fell_back_to_pages = True
        except Exception as e:
            # The first pass already proved this exact file readable, so a failure here is never
            # the client's fault (UnsupportedFile) — it is a fault on this side.
            raise StorageError("the document's second extraction pass failed unexpectedly") from e
        else:
            statute = statute_chunks(doc_id, arabic, title)
            if statute is not None:
                return PdfChunks(pages, "statute", statute, False)
    return PdfChunks(pages, "generic", page_chunks(doc_id, pages), fell_back_to_pages)


def _pages(doc_id: str, path: Path, keep_latin: bool, deadline: float) -> list[str]:
    """One extraction within MAX_PDF_PAGES, MAX_EXTRACTED_CHARS and `deadline`. keep_latin=True keeps an
    uploaded Arabic document's inline terms (VPN, Wi-Fi, Microsoft Teams), which the default drops. Its
    mirroring report is logged as counts and flags only: the report's keys are pdf_text's to change, and a
    string could carry document text."""
    pages, report = _extract_pages_isolated(path, keep_latin, MAX_PDF_PAGES, deadline)
    counts = {k: v for k, v in report.items() if isinstance(v, (bool, int))}
    logger.info("document %s, keep_latin=%s: mirroring %s", doc_id, keep_latin, counts)
    total_chars = sum(len(p) for p in pages)
    if total_chars > MAX_EXTRACTED_CHARS:
        raise pdf_text.ExtractionLimitExceeded(
            f"extraction produced {total_chars} characters, over the {MAX_EXTRACTED_CHARS} limit"
        )
    return pages


def _extract_pages_worker(path: Path, keep_latin: bool, max_pages: int, deadline: float) -> tuple[list[str], dict]:
    """The subprocess target: `pdf_text.extract_pages`, returning `(pages, report)`.

    Module-level so the spawned child can import it by name; the report is
    returned, since the child's memory is not the parent's.
    """
    if sys.platform != "win32":
        # Windows has no stdlib equivalent; there the timeout and the caps bound it.
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_LIMIT_BYTES, WORKER_MEMORY_LIMIT_BYTES))
    report: dict = {}
    pages = pdf_text.extract_pages(path, keep_latin=keep_latin, max_pages=max_pages,
                                   deadline=deadline, report=report)
    return pages, report


def _extract_pages_isolated(
    path: Path, keep_latin: bool, max_pages: int, deadline: float,
) -> tuple[list[str], dict]:
    """`_extract_pages_worker` in a killable subprocess, within `deadline` and MAX_CONCURRENT_EXTRACTIONS.

    Tests replace this function, not the worker: the child re-imports the worker
    from disk. A timeout becomes `pdf_text.ExtractionLimitExceeded` (the base
    class: a killed process cannot say how many pages it finished). A crash
    caused by a pdf_text limit re-raises that exact exception; any other crash
    stays `IsolationCrash`.
    """
    timeout_seconds = max(0.0, deadline - monotonic())
    with _extraction_slots:
        try:
            return run_isolated(_extract_pages_worker, (path, keep_latin, max_pages, deadline), timeout_seconds)
        except IsolationTimeout as e:
            raise pdf_text.ExtractionLimitExceeded(str(e)) from e
        except IsolationCrash as e:
            if isinstance(e.__cause__, pdf_text.ExtractionLimitExceeded):
                raise e.__cause__ from e
            raise
