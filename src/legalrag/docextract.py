"""A stored PDF upload's pages and chunks — the P1 demo layer (ADR-023).

At most two extractions, under one budget. The first keeps Latin, which suits
an Arabic document's inline terms (VPN, Wi-Fi) but welds a bilingual statute's
translation column into its Arabic lines: Law 151/2020 shows 4 of its 56
articles that way. So pages showing any article header are extracted again,
Arabic-only, and chunked as a statute if that validates. Each extraction logs
its bracket and « » mirroring decision, as counts and flags only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import monotonic
from typing import NamedTuple

from . import pdf_text
from .chunking import Chunk, may_be_statute, page_chunks, statute_chunks
from .docerrors import PdfTooLarge, StorageError, UnsupportedFile

logger = logging.getLogger(__name__)

# A PDF past either limit is refused. Extraction was measured at 0.8-1.26 s a page, each pass,
# so the budget follows the cap: budget ≈ cap × 1 s × 1.2. A statute's second, Arabic-only pass
# shares the budget, and one that runs out leaves page chunks, not a refusal.
MAX_PDF_PAGES = 250
EXTRACTION_BUDGET_SECONDS = 300


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
    """One extraction within MAX_PDF_PAGES and `deadline`. keep_latin=True keeps an uploaded Arabic document's
    inline terms (VPN, Wi-Fi, Microsoft Teams), which the default drops. Its mirroring report is logged as
    counts and flags only: the report's keys are pdf_text's to change, and a string could carry document text."""
    report: dict = {}
    pages = pdf_text.extract_pages(path, keep_latin=keep_latin, max_pages=MAX_PDF_PAGES,
                                   deadline=deadline, report=report)
    counts = {k: v for k, v in report.items() if isinstance(v, (bool, int))}
    logger.info("document %s, keep_latin=%s: mirroring %s", doc_id, keep_latin, counts)
    return pages
