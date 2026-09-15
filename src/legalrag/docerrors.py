"""Every error the document library raises on purpose (ADR-023).

Each carries a stable `code` for an HTTP layer to map. They live apart from
`library` so that `docextract`, which `library` imports, can raise them too;
`library` re-exports every one, and callers keep importing them from there.
"""


class LibraryError(Exception):
    """Raised on purpose, with a stable `code` for an HTTP layer to map."""
    code = "library_error"


class UnsupportedFile(LibraryError):
    """A bad suffix, no %PDF- magic, a .txt with NUL bytes or undecodable text, an unreadable PDF."""
    code = "unsupported_file"


class FileTooLarge(LibraryError):
    code = "file_too_large"


class PdfTooLarge(LibraryError):
    """A PDF over docextract.MAX_PDF_PAGES pages, or still on its first extraction after EXTRACTION_BUDGET_SECONDS."""
    code = "pdf_too_large"


class NoTextLayer(LibraryError):
    """Under MIN_TEXT_CHARS of extracted text: most likely a scanned PDF. OCR is out of scope."""
    code = "no_text"


class DocumentNotFound(LibraryError):
    """A malformed id, an unknown one, or a soft-deleted document."""
    code = "not_found"


class StorageError(LibraryError):
    """The library's own files could not be read, or are not what it wrote: staging that lost a file, an
    index that will not load, a directory not skipped at open, a full sha256 that differs. Never the client's fault."""
    code = "internal"


class DocumentDamaged(StorageError):
    """Index files that are not what the library wrote: uploading the document's bytes again replaces them."""
