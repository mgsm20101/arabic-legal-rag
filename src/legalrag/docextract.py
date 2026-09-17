"""A stored PDF upload's pages and chunks — the P1 demo layer (ADR-023).

At most two extractions, under one budget. The first keeps Latin, which suits
an Arabic document's inline terms (VPN, Wi-Fi) but welds a bilingual statute's
translation column into its Arabic lines: Law 151/2020 shows 4 of its 56
articles that way. So pages showing any article header are extracted again,
Arabic-only, and chunked as a statute if that validates. Each extraction logs
its bracket and « » mirroring decision, as counts and flags only.

Each extraction runs in its own subprocess (T04): `pdf_text.extract_pages`'s
own deadline check only runs between pages, after pdfplumber has already
opened the file and the first page has already started — a malformed or
pathological PDF can hang or run long before that check ever gets a chance to
fire, and nothing can stop it once it is running. `_extract_pages_isolated`
wraps the call through `isolate.run_isolated` instead, which can actually
terminate the OS process on a timeout. `pdf_text.py` itself is untouched
(beyond `TooManyPages`/`DeadlineExceeded` gaining a `__reduce__`, so they
survive being pickled back across that subprocess boundary): the isolation
is entirely an orchestration concern that lives here.
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

# MAX_PDF_PAGES caps page COUNT, not text density: a handful of pathologically dense pages
# (degenerate tiny-font text packed edge to edge) could still produce an enormous amount of
# text/memory without ever tripping the page cap. This is a second, independent ceiling on
# total extracted characters, checked after extraction returns. A generous per-page estimate
# for an actually dense real legal page is on the order of a few thousand characters; budgeting
# 20,000 chars/page (5-10x that) x MAX_PDF_PAGES gives a limit that a legitimate document should
# never reach, while still catching a pathological one before it threatens memory.
MAX_EXTRACTED_CHARS = MAX_PDF_PAGES * 20_000  # 5,000,000 characters

# Each extraction is now an isolated subprocess (its own Python interpreter plus pdfplumber's
# C extensions), so an unbounded burst of uploads could otherwise spawn an unbounded number of
# them. This is a single-user local demo (ADR-023), not a multi-tenant service: 2 lets one
# extraction proceed while another is starting or finishing, instead of serializing every
# upload to one at a time, while still keeping worst-case concurrent subprocess/memory use
# small on a laptop-class machine. It never gates `/api/health` (webapp.py), which does not
# call into this module at all.
MAX_CONCURRENT_EXTRACTIONS = 2
_extraction_slots = threading.BoundedSemaphore(MAX_CONCURRENT_EXTRACTIONS)

# Best-effort per-worker address-space cap, POSIX only — see `_extract_pages_worker`.
# 1 GiB: pdfplumber was measured at ~1.03 GB RSS at 600 pages (pdf_text.extract_pages's own
# docstring), and MAX_PDF_PAGES caps well under that (250), so 1 GiB leaves headroom for a
# legitimate page-heavy document while still bounding a pathological one.
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
    """The picklable target `run_isolated` spawns a subprocess for. Just calls
    `pdf_text.extract_pages` and returns `(pages, report)` — not a `report` dict mutated in
    place, because the child's memory is separate from the parent's: an in-place mutation here
    would never be visible once the child exits, so the report has to travel back as part of
    the return value instead.

    A module-level function, not a closure or a method: `multiprocessing`'s spawn start method
    (used on every platform — see isolate.py) re-imports this module fresh in the child and
    looks `_extract_pages_worker` up by name; a closure has no name to look up.
    """
    if sys.platform != "win32":
        # Best-effort only. Windows has no stdlib equivalent — proper enforcement needs Win32
        # Job Objects, which needs `pywin32` or raw `ctypes` calls, a new dependency clearly
        # disproportionate to this task's scope. On Windows the worker is bounded only by the
        # wall-clock timeout (below) and by MAX_PDF_PAGES/MAX_EXTRACTED_CHARS, not by memory —
        # an accepted, documented gap, not something worked around by adding new tooling.
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_LIMIT_BYTES, WORKER_MEMORY_LIMIT_BYTES))
    report: dict = {}
    pages = pdf_text.extract_pages(path, keep_latin=keep_latin, max_pages=max_pages,
                                   deadline=deadline, report=report)
    return pages, report


def _extract_pages_isolated(
    path: Path, keep_latin: bool, max_pages: int, deadline: float,
) -> tuple[list[str], dict]:
    """Runs `_extract_pages_worker` in a killable subprocess, bounded by whatever is left of
    `deadline` and by MAX_CONCURRENT_EXTRACTIONS.

    This — not `_extract_pages_worker` — is the seam orchestration tests replace (see
    tests/test_docextract.py and tests/test_library.py): monkeypatching `_extract_pages_worker`
    itself would not help a fast, in-process fake, because `run_isolated` pickles `target` BY
    REFERENCE (module + qualified name) to hand to the spawned child, which re-imports the real
    module from disk and gets back the original, un-monkeypatched function — never whatever a
    parent-process test temporarily assigned to that name.

    A timeout here is re-raised as `pdf_text.ExtractionLimitExceeded` (the same type
    `extract_pages`'s own in-loop deadline check used to raise) so that `extract_pdf`'s existing
    `except pdf_text.ExtractionLimitExceeded` / `except Exception` clauses keep working
    unchanged — a preemptive kill is functionally the same "this took too long" case, just no
    longer dependent on a per-page loop checking in time. It is deliberately the *base* class,
    not `DeadlineExceeded`: a killed subprocess cannot honestly report how many pages it
    finished, and `DeadlineExceeded` requires exactly that.

    A worker exception that pdf_text itself raised synchronously and well within the timeout
    (`TooManyPages`, or a `DeadlineExceeded` from pdf_text's own in-loop check) arrives here as
    an `IsolationCrash` whose `__cause__` is that *exact* original exception object — `run_isolated`
    puts the worker's exception on its result queue as-is, and pickling it (see pdf_text's
    `TooManyPages.__reduce__`/`DeadlineExceeded.__reduce__`) preserves both its subclass and its
    attributes. Unwrapping that `__cause__` when it is already a `pdf_text.ExtractionLimitExceeded`
    keeps that exact subclass and its attributes intact for `extract_pdf` and its callers. Any
    other crash (a genuinely broken PDF, an interpreter-level fault, ...) is left as
    `IsolationCrash`, which is not an `ExtractionLimitExceeded` and so already falls through
    those same `except` clauses into `except Exception` — exactly the "this file/pass is bad"
    branch a crash should hit.
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
