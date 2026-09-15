"""The upload document library — P1 demo layer (ADR-023), Phase 4B task 2.

On trial is everything an HTTP layer will lean on without looking: an
upload is refused before anything touches disk, the client's filename never
becomes a path, the same bytes are one document, a soft-deleted document is
never searched and never destroyed, and a reopened library trusts its saved
embeddings instead of running the model again. A keyword-count encoder
stands in for e5 throughout: no model, no network.
"""

import builtins
import errno
import hashlib
import io
import itertools
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")

import legalrag.docextract as docextract_mod  # noqa: E402
import legalrag.docindex as docindex_mod  # noqa: E402
import legalrag.library as library_mod  # noqa: E402
from legalrag import dense, pdf_text  # noqa: E402
from legalrag.library import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    STALE_STAGING_SECONDS,
    DocumentNotFound,
    EncoderUnavailable,
    FileTooLarge,
    Library,
    LibraryError,
    LibraryHit,
    NoTextLayer,
    StorageError,
    UnsupportedFile,
)
from legalrag.normalize import evaluation_normalize  # noqa: E402
from stubs import KeywordEncoder, blank_pdf, text_document  # noqa: E402

POLICY = text_document("العمل عن بعد يحتاج موافقة المدير", "بدل الإنترنت الشهري للموظف")
STORED_TXT = ["chunks.jsonl", "embeddings.npz", "meta.json", "source.txt"]
STORED_PDF = ["chunks.jsonl", "embeddings.npz", "meta.json", "source.pdf"]


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _refuse_new_directories(monkeypatch) -> None:
    """Staging starts with a directory; a rejected upload must never get
    that far — not just be cleaned up after."""
    def refuse(*args, **kwargs):
        raise AssertionError("the upload reached the disk before it was rejected")

    monkeypatch.setattr(os, "mkdir", refuse)


def _fake_extract_pages(*pages: str, seen: list | None = None):
    def fake(path, keep_latin=False, line_tol=None, *, max_pages=None, deadline=None, report=None):
        if seen is not None:
            seen.append({"path": Path(path), "keep_latin": keep_latin, "max_pages": max_pages, "deadline": deadline})
        return list(pages)

    return fake


# ---------------------------------------------------------------- identity --


def test_uploading_the_same_bytes_twice_creates_one_document(tmp_path):
    encoder = KeywordEncoder()
    library = Library(tmp_path, encoder=encoder)

    first = library.add(POLICY, "policy.txt")
    embedded = len(encoder.passages())
    second = library.add(POLICY, "another name.txt")

    assert second == first
    assert first.doc_id == hashlib.sha256(POLICY).hexdigest()[:12]
    assert library.documents() == [first]
    assert len(encoder.passages()) == embedded, "the duplicate upload was embedded again"
    assert _names(tmp_path / "docs") == [first.doc_id]


def test_an_upload_is_stored_as_its_source_meta_chunks_and_embeddings(tmp_path):
    library = Library(tmp_path, encoder=KeywordEncoder())

    meta = library.add(POLICY, "سياسة العمل.txt")

    doc_dir = tmp_path / "docs" / meta.doc_id
    assert _names(doc_dir) == STORED_TXT
    assert (doc_dir / "source.txt").read_bytes() == POLICY
    assert json.loads((doc_dir / "meta.json").read_text(encoding="utf-8")) == asdict(meta)
    chunk_lines = (doc_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(chunk_lines) == meta.chunks == 2
    assert (meta.title, meta.kind, meta.suffix, meta.pages) == ("سياسة العمل.txt", "generic", ".txt", 2)
    assert (meta.size_bytes, meta.sha256) == (len(POLICY), hashlib.sha256(POLICY).hexdigest())
    assert meta.chars == len(re.sub(r"\s", "", POLICY.decode("utf-8")))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", meta.created_at)
    assert (meta.deleted, meta.deleted_at) == (False, None)
    assert list((tmp_path / "tmp").iterdir()) == []


def test_a_stored_document_whose_full_sha256_differs_is_never_returned_for_these_bytes(tmp_path):
    """doc_id keeps 48 bits of the sha256: the stored document is these bytes
    only if the FULL hash agrees."""
    library = Library(tmp_path, encoder=KeywordEncoder())
    meta = library.add(POLICY, "policy.txt")
    meta_path = tmp_path / "docs" / meta.doc_id / "meta.json"
    forged = {**json.loads(meta_path.read_text(encoding="utf-8")), "sha256": meta.doc_id + "0" * 52}
    assert forged["sha256"] != meta.sha256  # the same 12-hex id, other bytes
    meta_path.write_text(json.dumps(forged), encoding="utf-8")
    reopened = Library(tmp_path, encoder=KeywordEncoder())

    with pytest.raises(StorageError) as mismatch:
        reopened.add(POLICY, "policy.txt")

    assert mismatch.value.code == "internal"
    assert json.loads(meta_path.read_text(encoding="utf-8")) == forged
    assert list((tmp_path / "tmp").iterdir()) == []


# ------------------------------------------------------------- soft delete --


def test_a_soft_deleted_document_is_never_searched(tmp_path, monkeypatch):
    library = Library(tmp_path, encoder=KeywordEncoder(["الإجازة", "الإنترنت"]))
    kept = library.add(text_document("رصيد الإجازة السنوية"), "kept.txt")
    gone = library.add(text_document("بدل الإنترنت ورصيد الإجازة"), "gone.txt")
    assert {h.chunk.doc_id for h in library.search("الإجازة", k=5)} == {kept.doc_id, gone.doc_id}

    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src, dst):
        replaced.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)

    deleted = library.soft_delete(gone.doc_id)

    assert deleted.deleted is True
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", deleted.deleted_at)
    assert {h.chunk.doc_id for h in library.search("الإنترنت", k=5)} == {kept.doc_id}
    assert library.documents() == [kept]
    assert gone.doc_id not in library._indexes, "the deleted document's index is still cached"
    with pytest.raises(DocumentNotFound):
        library.get(gone.doc_id)
    with pytest.raises(DocumentNotFound):
        library.search("الإنترنت", doc_ids=[gone.doc_id])
    with pytest.raises(DocumentNotFound):
        library.soft_delete(gone.doc_id)

    # Nothing destroyed, and the one write was an atomic replace of meta.json.
    doc_dir = tmp_path / "docs" / gone.doc_id
    assert _names(doc_dir) == STORED_TXT
    assert json.loads((doc_dir / "meta.json").read_text(encoding="utf-8"))["deleted"] is True
    assert [(src.parent, dst) for src, dst in replaced] == [(doc_dir, doc_dir / "meta.json")]


def test_re_uploading_a_soft_deleted_document_restores_it(tmp_path):
    encoder = KeywordEncoder(["الإنترنت"])
    library = Library(tmp_path, encoder=encoder)
    original = library.add(POLICY, "policy.txt")
    library.soft_delete(original.doc_id)
    embedded = len(encoder.passages())

    restored = library.add(POLICY, "policy (1).txt")

    assert restored == original  # same id, title and created_at; deleted=False, deleted_at=None
    assert library.documents() == [original]
    assert [h.chunk.doc_id for h in library.search("الإنترنت", k=1)] == [original.doc_id]
    assert len(encoder.passages()) == embedded, "restoring embedded the document again"
    on_disk = json.loads((tmp_path / "docs" / original.doc_id / "meta.json").read_text(encoding="utf-8"))
    assert (on_disk["deleted"], on_disk["deleted_at"]) == (False, None)


def _fail_writes_to(monkeypatch, target: Path) -> None:
    """Every atomic replace onto `target` fails as a full disk would."""
    real_replace = os.replace

    def disk_full(src, dst):
        if Path(dst) == target:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", disk_full)


def test_a_failed_meta_write_during_soft_delete_changes_nothing_in_memory_or_on_disk(tmp_path, monkeypatch):
    library = Library(tmp_path, encoder=KeywordEncoder(["الإنترنت"]))
    meta = library.add(text_document("بدل الإنترنت الشهري"), "internet.txt")
    library.search("الإنترنت", k=1)  # its index is cached now
    meta_path = tmp_path / "docs" / meta.doc_id / "meta.json"
    on_disk = meta_path.read_bytes()

    _fail_writes_to(monkeypatch, meta_path)
    with pytest.raises(OSError):
        library.soft_delete(meta.doc_id)
    monkeypatch.undo()

    assert meta_path.read_bytes() == on_disk
    assert library.get(meta.doc_id) == meta
    assert library.documents() == [meta]
    assert meta.doc_id in library._indexes
    assert [h.chunk.doc_id for h in library.search("الإنترنت", k=1)] == [meta.doc_id]
    assert not [p.name for p in meta_path.parent.iterdir() if p.name.endswith(".tmp")]


def test_a_failed_meta_write_while_restoring_changes_nothing_in_memory_or_on_disk(tmp_path, monkeypatch):
    internet = text_document("بدل الإنترنت الشهري")
    library = Library(tmp_path, encoder=KeywordEncoder(["الإنترنت"]))
    meta = library.add(internet, "internet.txt")
    library.soft_delete(meta.doc_id)
    meta_path = tmp_path / "docs" / meta.doc_id / "meta.json"
    on_disk = meta_path.read_bytes()

    _fail_writes_to(monkeypatch, meta_path)
    with pytest.raises(OSError):
        library.add(internet, "internet.txt")
    monkeypatch.undo()

    assert meta_path.read_bytes() == on_disk and json.loads(on_disk)["deleted"] is True
    with pytest.raises(DocumentNotFound):
        library.get(meta.doc_id)
    assert library.documents() == []
    assert library.add(internet, "internet.txt") == meta  # once the disk takes the write, it restores


def test_a_document_deleted_while_its_index_loads_serves_that_search_and_caches_nothing(tmp_path, monkeypatch):
    library = Library(tmp_path, encoder=KeywordEncoder(["الإنترنت"]))
    meta = library.add(text_document("بدل الإنترنت الشهري"), "internet.txt")
    real_load_index = library_mod.load_index

    def load_then_delete(*args, **kwargs):
        loaded = real_load_index(*args, **kwargs)
        library.soft_delete(meta.doc_id)  # another request deletes it mid-load
        return loaded

    monkeypatch.setattr(library_mod, "load_index", load_then_delete)

    hits = library.search("الإنترنت", k=1)

    assert [h.chunk.doc_id for h in hits] == [meta.doc_id]  # the search already under way is served
    assert meta.doc_id not in library._indexes               # a deleted document is never cached
    assert library.search("الإنترنت", k=1) == []


# ------------------------------------------------- refused before the disk --


def test_a_pdf_extension_without_pdf_magic_bytes_is_rejected_before_anything_is_written(tmp_path, monkeypatch):
    library = Library(tmp_path, encoder=KeywordEncoder())
    before = _tree(tmp_path)
    _refuse_new_directories(monkeypatch)

    with pytest.raises(UnsupportedFile) as rejected:
        library.add(b"<html><body>" + "مستند".encode("utf-8") * 100 + b"</body></html>", "report.pdf")

    assert rejected.value.code == "unsupported_file"
    assert _tree(tmp_path) == before
    assert library.documents() == []


def test_an_unsupported_suffix_and_an_oversized_upload_are_rejected_before_anything_is_written(tmp_path, monkeypatch):
    library = Library(tmp_path, encoder=KeywordEncoder())
    before = _tree(tmp_path)
    _refuse_new_directories(monkeypatch)

    cases = [
        (POLICY, "policy.docx", UnsupportedFile, "unsupported_file"),
        (POLICY, "policy", UnsupportedFile, "unsupported_file"),
        (POLICY, "policy.txt.exe", UnsupportedFile, "unsupported_file"),
        (b"", "empty.txt", UnsupportedFile, "unsupported_file"),
        # Size is checked first: too large is FileTooLarge whatever the name.
        (b"%PDF-" + b"0" * MAX_UPLOAD_BYTES, "big.pdf", FileTooLarge, "file_too_large"),
        (b"0" * (MAX_UPLOAD_BYTES + 1), "big.docx", FileTooLarge, "file_too_large"),
    ]
    for data, filename, error, code in cases:
        with pytest.raises(error) as rejected:
            library.add(data, filename)
        assert isinstance(rejected.value, LibraryError)
        assert rejected.value.code == code, filename

    assert MAX_UPLOAD_BYTES == 20 * 1024 * 1024
    assert _tree(tmp_path) == before


def test_the_client_filename_never_becomes_a_path(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_text, "extract_pages", _fake_extract_pages("نص مستخرج من ملف PDF " * 30))
    root = tmp_path / "library"
    library = Library(root, encoder=KeywordEncoder())

    uploads = {
        "../../evil.pdf": ("evil.pdf", b"%PDF-1.7 evil"),
        "..\\..\\evil.txt": ("evil.txt", text_document("الأولى")),
        "C:\\Windows\\System32\\evil.txt": ("evil.txt", text_document("الثانية")),
        "/etc/cron.d/evil.txt": ("evil.txt", text_document("الثالثة")),
        "\u202etxt.exe\u202c\u2067  تقرير\u200f\t.txt": ("txt.exe تقرير.txt", text_document("الرابعة")),
        "a" * 300 + ".txt": ("a" * 120, text_document("الخامسة")),
    }
    for filename, (title, data) in uploads.items():
        assert library.add(data, filename).title == title, repr(filename)

    assert _names(tmp_path) == ["library"]
    assert _names(root) == ["docs", "tmp"]
    assert list((root / "tmp").iterdir()) == []
    for doc_dir in (root / "docs").iterdir():
        assert re.fullmatch(r"[0-9a-f]{12}", doc_dir.name)
        assert _names(doc_dir) in (STORED_PDF, STORED_TXT)
    for escaped in (root / "evil.pdf", tmp_path / "evil.pdf", tmp_path.parent / "evil.pdf",
                    tmp_path / "evil.txt", tmp_path.parent / "evil.txt"):
        assert not escaped.exists(), escaped
    assert not list(tmp_path.rglob("*evil*"))
    # A name that sanitizes to nothing still gets a title.
    assert library_mod._display_title("\u200f\u202e\t", ".pdf") == "document.pdf"


def test_a_pdf_without_a_text_layer_raises_no_text_layer_and_leaves_nothing_behind(tmp_path, monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(pdf_text, "extract_pages", _fake_extract_pages("", "   ", "\n", seen=seen))
    library = Library(tmp_path, encoder=KeywordEncoder())
    before = _tree(tmp_path)

    with pytest.raises(NoTextLayer) as rejected:
        library.add(b"%PDF-1.4\n% scanned pages only\n", "scan.pdf")

    assert rejected.value.code == "no_text"
    assert seen and seen[0]["keep_latin"] is True
    assert library.documents() == []
    assert _tree(tmp_path) == before


def test_a_pdf_the_extractor_cannot_read_is_unsupported_and_leaves_nothing_behind(tmp_path, monkeypatch):
    def broken(path, keep_latin=False, line_tol=None, **limits):
        raise ValueError("corrupt cross-reference table")

    monkeypatch.setattr(pdf_text, "extract_pages", broken)
    library = Library(tmp_path, encoder=KeywordEncoder())
    before = _tree(tmp_path)

    with pytest.raises(UnsupportedFile) as rejected:
        library.add(b"%PDF-1.7\nnot really a pdf", "broken.pdf")

    assert isinstance(rejected.value.__cause__, ValueError)
    assert _tree(tmp_path) == before


def test_a_pdf_over_the_page_limit_is_pdf_too_large_and_leaves_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(docextract_mod, "MAX_PDF_PAGES", 3)
    library = Library(tmp_path, encoder=KeywordEncoder())
    before = _tree(tmp_path)

    with pytest.raises(library_mod.PdfTooLarge) as refused:
        library.add(blank_pdf(4), "long.pdf")

    assert refused.value.code == "pdf_too_large"
    assert isinstance(refused.value, LibraryError)
    assert isinstance(refused.value.__cause__, pdf_text.TooManyPages)
    assert _tree(tmp_path) == before
    assert library.documents() == []


def test_a_pdf_whose_extraction_outruns_its_budget_is_pdf_too_large_and_leaves_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(docextract_mod, "EXTRACTION_BUDGET_SECONDS", -1)  # the deadline has passed before page 2
    library = Library(tmp_path, encoder=KeywordEncoder())
    before = _tree(tmp_path)

    with pytest.raises(library_mod.PdfTooLarge) as refused:
        library.add(blank_pdf(2), "slow.pdf")

    assert refused.value.code == "pdf_too_large"
    assert isinstance(refused.value.__cause__, pdf_text.DeadlineExceeded)
    assert _tree(tmp_path) == before
    assert library.documents() == []


def test_both_pdf_limits_reach_extract_pages_from_add(tmp_path, monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(pdf_text, "extract_pages", _fake_extract_pages("نص مستخرج من ملف PDF " * 30, seen=seen))
    monkeypatch.setattr(docextract_mod, "monotonic", lambda: 1000.0)
    library = Library(tmp_path, encoder=KeywordEncoder())

    library.add(b"%PDF-1.7 within the limits", "limits.pdf")

    assert (docextract_mod.MAX_PDF_PAGES, docextract_mod.EXTRACTION_BUDGET_SECONDS) == (250, 300)
    assert [(s["keep_latin"], s["max_pages"], s["deadline"]) for s in seen] == [(True, 250, 1300.0)]


_BODIES = [
    "يلتزم المتحكم بالحصول على موافقة صريحة من الشخص المعني قبل جمع بياناته الشخصية",
    "يجب إخطار المركز بأي خرق للبيانات الشخصية خلال اثنتين وسبعين ساعة من العلم به",
    "يحق للشخص المعني طلب تصحيح بياناته الشخصية أو محوها متى كانت غير صحيحة",
]
# The same statute as extracted with Latin kept and without: a bilingual page
# welds the translation into the Arabic text (and here a stray fourth header),
# so only the Arabic-only extraction validates as a statute.
STATUTE_ARABIC = [f"مادة {n}\n{body}" for n, body in enumerate(_BODIES, start=1)]
STATUTE_LATIN = [
    f"مادة {n}\nArticle {n} of the translation, set beside the Arabic text\n{body}"
    for n, body in enumerate(_BODIES, start=1)
]
STATUTE_LATIN[-1] += "\nمادة 7\nArticle 7 continues in English only"
RAW_STATUTE = Path(__file__).resolve().parents[1] / "data" / "raw" / "law-151-2020-personal-data-protection.pdf"


def _two_pass_extractor(latin: list[str], arabic: list[str], calls: list, second_pass_error=None):
    def fake(path, keep_latin=False, line_tol=None, *, max_pages=None, deadline=None, report=None):
        calls.append((keep_latin, max_pages, deadline))
        if not keep_latin and second_pass_error is not None:
            raise second_pass_error
        return list(latin if keep_latin else arabic)

    return fake


def _stored_chunks(root: Path, meta) -> list[dict]:
    lines = (root / "docs" / meta.doc_id / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_a_statute_like_pdf_is_chunked_as_a_statute_from_a_second_arabic_only_extraction(tmp_path, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(pdf_text, "extract_pages", _two_pass_extractor(STATUTE_LATIN, STATUTE_ARABIC, calls))
    ticks = itertools.count(1000.0, 60.0)  # each reading of the clock is a minute after the last
    monkeypatch.setattr(docextract_mod, "monotonic", lambda: next(ticks))
    library = Library(tmp_path, encoder=KeywordEncoder())

    meta = library.add(b"%PDF-1.7 bilingual statute", "law.pdf")

    assert calls[1][2] == calls[0][2], "the Arabic-only pass was given a budget of its own"
    assert calls == [(True, 250, 1300.0), (False, 250, 1300.0)]
    assert (meta.kind, meta.pages, meta.chunks) == ("statute", 3, 3)
    chunks = _stored_chunks(tmp_path, meta)
    assert [c["label"] for c in chunks] == ["مادة 1", "مادة 2", "مادة 3"]
    assert not any(re.search("[A-Za-z]", c["text"]) for c in chunks)


def test_one_article_header_left_in_the_first_extraction_still_gets_the_arabic_only_extraction(
    tmp_path, monkeypatch,
):
    """Law 151/2020 shows 4 of its 56 headers with Latin kept, and a small shift in that
    extraction would show fewer: finding the statute must not hang on how many survive."""
    calls: list[tuple] = []
    one_header_left = [STATUTE_LATIN[0], *(
        f"مادة {n} Article {n} of the translation, welded onto the header line\n{body}"
        for n, body in enumerate(_BODIES[1:], start=2)
    )]
    monkeypatch.setattr(pdf_text, "extract_pages", _two_pass_extractor(one_header_left, STATUTE_ARABIC, calls))
    library = Library(tmp_path, encoder=KeywordEncoder())

    meta = library.add(b"%PDF-1.7 statute with one header left", "law.pdf")

    assert [keep_latin for keep_latin, _, _ in calls] == [True, False]
    assert (meta.kind, meta.chunks) == ("statute", 3)
    assert [c["label"] for c in _stored_chunks(tmp_path, meta)] == ["مادة 1", "مادة 2", "مادة 3"]


def test_headers_whose_arabic_only_text_is_no_valid_statute_fall_back_to_the_first_extractions_pages(
    tmp_path, monkeypatch,
):
    calls: list[tuple] = []
    not_a_statute = [*STATUTE_ARABIC[:2], STATUTE_ARABIC[2].replace("مادة 3", "مادة 5")]  # 3 and 4 missing
    monkeypatch.setattr(pdf_text, "extract_pages", _two_pass_extractor(STATUTE_LATIN, not_a_statute, calls))
    library = Library(tmp_path, encoder=KeywordEncoder())

    meta = library.add(b"%PDF-1.7 almost a statute", "almost.pdf")

    assert [keep_latin for keep_latin, _, _ in calls] == [True, False]
    assert meta.kind == "generic"
    chunks = _stored_chunks(tmp_path, meta)
    assert [c["label"] for c in chunks] == ["ص 1", "ص 2", "ص 3"]
    assert all("Article" in c["text"] for c in chunks)  # the pages kept their Latin terms


def test_a_pdf_without_article_headers_is_extracted_once(tmp_path, monkeypatch):
    calls: list[tuple] = []
    policy = ["سياسة العمل عن بعد تسري على الموظفين الدائمين عبر شبكة VPN الخاصة " * 5]
    monkeypatch.setattr(pdf_text, "extract_pages", _two_pass_extractor(policy, ["must never be read"], calls))
    library = Library(tmp_path, encoder=KeywordEncoder())

    meta = library.add(b"%PDF-1.7 policy", "policy.pdf")

    assert [keep_latin for keep_latin, _, _ in calls] == [True]
    assert meta.kind == "generic"
    assert "VPN" in _stored_chunks(tmp_path, meta)[0]["text"]


def test_a_deadline_hit_during_the_arabic_only_extraction_keeps_the_upload_as_the_first_extractions_page_chunks(
    tmp_path, monkeypatch, caplog,
):
    """A statute too long for both passes in one budget already has usable pages from the
    first: refusing it after that wait would throw them away."""
    calls: list[tuple] = []
    passed = pdf_text.DeadlineExceeded(1, len(STATUTE_ARABIC))
    monkeypatch.setattr(pdf_text, "extract_pages",
                        _two_pass_extractor(STATUTE_LATIN, STATUTE_ARABIC, calls, second_pass_error=passed))
    library = Library(tmp_path, encoder=KeywordEncoder())

    with caplog.at_level(logging.WARNING, logger="legalrag.library"):
        meta = library.add(b"%PDF-1.7 slow statute", "slow.pdf")

    assert [keep_latin for keep_latin, _, _ in calls] == [True, False]
    assert (meta.kind, meta.pages, meta.chunks) == ("generic", 3, 3)
    chunks = _stored_chunks(tmp_path, meta)
    assert [c["label"] for c in chunks] == ["ص 1", "ص 2", "ص 3"]
    assert all("Article" in c["text"] for c in chunks)  # the first extraction's pages, Latin kept
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(meta.doc_id in w and "3 pages" in w for w in warnings), warnings
    assert library.documents() == [meta]
    assert _names(tmp_path / "docs" / meta.doc_id) == STORED_PDF
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.skipif(not RAW_STATUTE.exists(), reason="the statute PDF is gitignored third-party text")
def test_the_real_statute_uploads_as_a_statute_of_its_56_articles(tmp_path):
    pytest.importorskip("pdfplumber")
    library = Library(tmp_path, encoder=KeywordEncoder())

    meta = library.add(RAW_STATUTE.read_bytes(), RAW_STATUTE.name)

    chunks = _stored_chunks(tmp_path, meta)
    assert (meta.kind, meta.pages, meta.chunks) == ("statute", 33, 56)
    assert sum(c["article"] is not None for c in chunks) == 49  # the law; the other 7 are the issuance law
    assert not any(re.search("[A-Za-z]", c["text"]) for c in chunks)


def test_a_text_file_in_cp1256_or_with_a_utf8_bom_is_decoded(tmp_path):
    text = "تسري هذه السياسة على الموظفين الدائمين، ويلتزم المدير بالرد خلال خمسة أيام عمل. " * 4
    library = Library(tmp_path, encoder=KeywordEncoder())

    with_bom = library.add(b"\xef\xbb\xbf" + text.encode("utf-8"), "bom.txt")
    legacy = library.add(text.encode("cp1256"), "LEGACY.TXT")

    for meta in (with_bom, legacy):
        lines = (tmp_path / "docs" / meta.doc_id / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["text"] for line in lines] == [evaluation_normalize(text)], meta.title
    assert legacy.suffix == ".txt"

    with pytest.raises(UnsupportedFile):
        library.add(text.encode("utf-8") + b"\x00", "binary.txt")


def test_a_text_file_that_fails_decoding_is_rejected_before_anything_is_written(tmp_path, monkeypatch):
    library = Library(tmp_path, encoder=KeywordEncoder())
    before = _tree(tmp_path)
    _refuse_new_directories(monkeypatch)
    arabic = "تسري هذه السياسة على جميع الموظفين الدائمين. " * 10

    with pytest.raises(UnsupportedFile):
        library.add(arabic.encode("utf-8") + b"\x00", "nul.txt")
    # cp1256 maps every byte value, so text is undecodable only under a stricter list.
    monkeypatch.setattr(library_mod, "TEXT_ENCODINGS", ("utf-8-sig",))
    with pytest.raises(UnsupportedFile):
        library.add(arabic.encode("cp1256"), "legacy.txt")

    assert _tree(tmp_path) == before
    assert library.documents() == []


# ------------------------------------------------------------------ lookup --


def test_a_malformed_or_unknown_doc_id_is_not_found(tmp_path):
    library = Library(tmp_path, encoder=KeywordEncoder())
    meta = library.add(POLICY, "policy.txt")
    unknown = "0" * 12 if meta.doc_id != "0" * 12 else "1" * 12

    bad_ids = ["", "..", "../../etc/passwd", f"{meta.doc_id}\n", meta.doc_id[:-1],
               meta.doc_id + "0", unknown, None, 42]
    if meta.doc_id.upper() != meta.doc_id:
        bad_ids.append(meta.doc_id.upper())
    for bad in bad_ids:
        with pytest.raises(DocumentNotFound) as missing:
            library.get(bad)
        assert missing.value.code == "not_found"
        with pytest.raises(DocumentNotFound):
            library.soft_delete(bad)
        with pytest.raises(DocumentNotFound):
            library.search("العمل", doc_ids=[meta.doc_id, bad])

    assert library.get(meta.doc_id) == meta  # none of that touched the real one


def test_search_merges_documents_by_score_and_honours_doc_ids(tmp_path, monkeypatch):
    stamps = iter(["2026-09-01T10:00:00+00:00", "2026-09-01T10:00:05+00:00"])
    monkeypatch.setattr(library_mod, "_now", lambda: next(stamps))
    encoder = KeywordEncoder(["قطط", "كلاب", "طيور"])
    library = Library(tmp_path, encoder=encoder)
    older = library.add(text_document("قطط", "قطط كلاب"), "older.txt")
    newer = library.add(text_document("كلاب", "طيور", "طيور"), "newer.txt")

    hits = library.search("قطط كلاب", k=3)

    assert all(isinstance(h, LibraryHit) for h in hits)
    # 1.0 first; then a 0.7071 tie across documents, broken by created_at.
    assert [(h.chunk.id, h.doc_title, h.score) for h in hits] == [
        (f"{older.doc_id}:2", "older.txt", 1.0),
        (f"{older.doc_id}:1", "older.txt", 0.7071),
        (f"{newer.doc_id}:1", "newer.txt", 0.7071),
    ]
    # A tie inside one document goes by chunk number; zero-score chunks
    # still fill k, oldest document first.
    assert [h.chunk.id for h in library.search("طيور", k=5)] == [
        f"{newer.doc_id}:2", f"{newer.doc_id}:3",
        f"{older.doc_id}:1", f"{older.doc_id}:2", f"{newer.doc_id}:1",
    ]

    scoped = library.search("قطط كلاب", k=5, doc_ids=[newer.doc_id])
    assert [h.chunk.id for h in scoped] == [f"{newer.doc_id}:1", f"{newer.doc_id}:2", f"{newer.doc_id}:3"]

    asked = len(encoder.seen)
    assert library.search("قطط", k=5, doc_ids=[]) == []
    assert len(encoder.seen) == asked, "an empty scope still encoded the query"
    with pytest.raises(ValueError):
        library.search("قطط", k=0)


def test_a_library_reopened_on_the_same_root_sees_its_documents_without_re_embedding(tmp_path):
    keywords = ["الإنترنت", "الإجازة"]
    library = Library(tmp_path, encoder=KeywordEncoder(keywords))
    kept = library.add(text_document("بدل الإنترنت الشهري", "رصيد الإجازة"), "kept.txt")
    gone = library.add(text_document("الإجازة المرضية"), "gone.txt")
    library.soft_delete(gone.doc_id)
    before = library.search("الإنترنت", k=2)
    crashed_upload = tmp_path / "tmp" / "0123abcd" / "source.pdf"
    crashed_upload.parent.mkdir(parents=True)
    crashed_upload.write_bytes(b"%PDF-1.7")
    long_ago = time.time() - STALE_STAGING_SECONDS - 60
    for entry in (crashed_upload, crashed_upload.parent):
        os.utime(entry, (long_ago, long_ago))

    encoder = KeywordEncoder(keywords)
    reopened = Library(tmp_path, encoder=encoder)

    assert reopened.documents() == [kept]
    with pytest.raises(DocumentNotFound):
        reopened.get(gone.doc_id)
    assert reopened.search("الإنترنت", k=2) == before
    assert encoder.passages() == [], "a document with saved embeddings was embedded again"
    assert list((tmp_path / "tmp").iterdir()) == [], "stale staging survived a restart"


def test_a_document_whose_meta_json_is_missing_or_unreadable_is_skipped_and_a_re_upload_replaces_it(tmp_path):
    library = Library(tmp_path, encoder=KeywordEncoder())
    good = library.add(text_document("مستند سليم"), "good.txt")
    unreadable = library.add(POLICY, "policy.txt")
    missing = library.add(text_document("مستند بلا بيانات"), "missing.txt")
    (tmp_path / "docs" / unreadable.doc_id / "meta.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "docs" / missing.doc_id / "meta.json").unlink()

    reopened = Library(tmp_path, encoder=KeywordEncoder())

    assert reopened.documents() == [good]
    with pytest.raises(DocumentNotFound):
        reopened.get(unreadable.doc_id)
    restored = reopened.add(POLICY, "policy.txt")
    assert restored.doc_id == unreadable.doc_id
    assert reopened.get(restored.doc_id) == restored
    assert _names(tmp_path / "docs" / restored.doc_id) == STORED_TXT
    assert list((tmp_path / "tmp").iterdir()) == []


# ------------------------------------------------------------ damaged disk --


def test_replacing_a_skipped_directory_puts_it_back_when_the_final_move_fails(tmp_path, monkeypatch):
    library = Library(tmp_path, encoder=KeywordEncoder())
    meta = library.add(POLICY, "policy.txt")
    doc_dir = tmp_path / "docs" / meta.doc_id
    (doc_dir / "meta.json").write_text("{not json", encoding="utf-8")
    skipped_files = {p.name: p.read_bytes() for p in doc_dir.iterdir()}
    reopened = Library(tmp_path, encoder=KeywordEncoder())
    real_replace = os.replace
    failed: list[Path] = []

    def fail_the_install(src, dst):
        if Path(dst) == doc_dir and not failed:
            failed.append(Path(src))
            raise OSError(errno.EIO, "simulated failure moving the upload into place")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_the_install)
    with pytest.raises(OSError, match="simulated failure"):
        reopened.add(POLICY, "policy.txt")
    monkeypatch.undo()

    assert failed and failed[0].parent == tmp_path / "tmp"
    assert {p.name: p.read_bytes() for p in doc_dir.iterdir()} == skipped_files
    assert list((tmp_path / "tmp").iterdir()) == []
    assert reopened.documents() == []
    # Nothing was lost: the next upload of these bytes replaces the directory.
    assert reopened.add(POLICY, "policy.txt") == reopened.get(meta.doc_id)
    assert _names(doc_dir) == STORED_TXT


def test_a_directory_that_appeared_after_the_library_opened_is_never_replaced(tmp_path):
    library = Library(tmp_path, encoder=KeywordEncoder())
    stranger = tmp_path / "docs" / hashlib.sha256(POLICY).hexdigest()[:12]
    stranger.mkdir()
    (stranger / "meta.json").write_text("written by something else", encoding="utf-8")

    with pytest.raises(StorageError) as refused:
        library.add(POLICY, "policy.txt")

    assert refused.value.code == "internal"
    assert _names(stranger) == ["meta.json"]
    assert (stranger / "meta.json").read_text(encoding="utf-8") == "written by something else"
    assert list((tmp_path / "tmp").iterdir()) == []
    assert library.documents() == []


class _StagingWiper(KeywordEncoder):
    """Wipes every staging directory while passages are being embedded."""

    def __init__(self, tmp: Path):
        super().__init__()
        self.tmp = tmp

    def encode(self, texts, **kwargs):
        texts = list(texts)
        if any(t.startswith("passage: ") for t in texts):
            for staging in self.tmp.iterdir():
                shutil.rmtree(staging)
        return super().encode(texts, **kwargs)


def test_an_upload_whose_staging_lost_files_is_refused_not_registered(tmp_path):
    """Saving the embeddings recreates a wiped staging directory, so without
    a check the upload "succeeded" holding only embeddings.npz and meta.json."""
    library = Library(tmp_path, encoder=_StagingWiper(tmp_path / "tmp"))

    with pytest.raises(StorageError) as refused:
        library.add(POLICY, "policy.txt")

    assert refused.value.code == "internal"
    assert library.documents() == []
    assert list((tmp_path / "docs").iterdir()) == []
    assert list((tmp_path / "tmp").iterdir()) == []


class _OpensAnotherLibrary(KeywordEncoder):
    """Opens a second Library on the same root while passages are being embedded."""

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.opened = 0

    def encode(self, texts, **kwargs):
        texts = list(texts)
        if any(t.startswith("passage: ") for t in texts) and not self.opened:
            self.opened += 1
            Library(self.root, encoder=KeywordEncoder())
        return super().encode(texts, **kwargs)


def test_a_library_opened_during_an_upload_leaves_that_upload_alone(tmp_path):
    encoder = _OpensAnotherLibrary(tmp_path)
    library = Library(tmp_path, encoder=encoder)

    meta = library.add(POLICY, "policy.txt")

    assert encoder.opened == 1
    assert library.documents() == [meta]
    assert _names(tmp_path / "docs" / meta.doc_id) == STORED_TXT


def test_opening_a_library_clears_only_staging_older_than_the_stale_limit(tmp_path):
    tmp = tmp_path / "tmp"
    stale, fresh = tmp / ("a" * 32), tmp / ("b" * 32)
    for staging in (stale, fresh):
        staging.mkdir(parents=True)
        (staging / "source.pdf").write_bytes(b"%PDF-1.7")
    stray_file = tmp / ".meta.json.tmp"
    stray_file.write_bytes(b"{")
    long_ago = time.time() - STALE_STAGING_SECONDS - 60
    for entry in (stale / "source.pdf", stale, stray_file):
        os.utime(entry, (long_ago, long_ago))

    Library(tmp_path, encoder=KeywordEncoder())

    assert [p.name for p in tmp.iterdir()] == [fresh.name]
    assert _names(fresh) == ["source.pdf"]


def test_a_stored_document_missing_its_chunks_or_embeddings_is_skipped_with_a_warning_and_a_re_upload_replaces_it(
    tmp_path, caplog,
):
    vectors_doc = text_document("مستند بلا متجهات")
    library = Library(tmp_path, encoder=KeywordEncoder())
    good = library.add(text_document("مستند سليم"), "good.txt")
    no_chunks = library.add(POLICY, "policy.txt")
    no_embeddings = library.add(vectors_doc, "vectors.txt")
    (tmp_path / "docs" / no_chunks.doc_id / "chunks.jsonl").unlink()
    (tmp_path / "docs" / no_embeddings.doc_id / "embeddings.npz").unlink()

    with caplog.at_level(logging.WARNING, logger="legalrag.library"):
        reopened = Library(tmp_path, encoder=KeywordEncoder())

    assert reopened.documents() == [good]
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert no_chunks.doc_id in warned and "chunks.jsonl" in warned
    assert no_embeddings.doc_id in warned and "embeddings.npz" in warned
    assert _names(tmp_path / "docs" / no_chunks.doc_id) == ["embeddings.npz", "meta.json", "source.txt"]
    assert [h.chunk.doc_id for h in reopened.search("مستند", k=5)] == [good.doc_id]

    for data, filename, broken in ((POLICY, "policy.txt", no_chunks), (vectors_doc, "vectors.txt", no_embeddings)):
        assert reopened.add(data, filename).doc_id == broken.doc_id
        assert _names(tmp_path / "docs" / broken.doc_id) == STORED_TXT
    assert len(reopened.documents()) == 3
    assert list((tmp_path / "tmp").iterdir()) == []


def test_an_index_that_fails_to_load_is_skipped_unscoped_and_a_storage_error_scoped(
    tmp_path, caplog,
):
    library = Library(tmp_path, encoder=KeywordEncoder(["الإنترنت"]))
    good = library.add(text_document("بدل الإنترنت الشهري"), "good.txt")
    missing = library.add(text_document("بدل الإنترنت للموظف"), "missing.txt")
    corrupt = library.add(text_document("الإنترنت في المنزل"), "corrupt.txt")
    # Damaged after the library opened, before any search loaded their indexes.
    (tmp_path / "docs" / missing.doc_id / "chunks.jsonl").unlink()
    (tmp_path / "docs" / corrupt.doc_id / "chunks.jsonl").write_text("{not json\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="legalrag.library"):
        hits = library.search("الإنترنت", k=5)

    assert [h.chunk.doc_id for h in hits] == [good.doc_id]
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert missing.doc_id in warned and corrupt.doc_id in warned
    for broken in (missing, corrupt):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="legalrag.library"), pytest.raises(StorageError) as damaged:
            library.search("الإنترنت", doc_ids=[broken.doc_id])
        assert damaged.value.code == "internal"
        assert broken.doc_id in " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert library.get(broken.doc_id) == broken  # the document still exists; its index is what is damaged
    assert [h.chunk.doc_id for h in library.search("الإنترنت", doc_ids=[good.doc_id])] == [good.doc_id]


@pytest.mark.parametrize("damage", ["zero_bytes", "not_a_zip", "truncated_zip"])
def test_damaged_embeddings_are_reported_as_damage_never_silently_re_embedded(tmp_path, caplog, damage):
    encoder = KeywordEncoder(["الإنترنت"])
    library = Library(tmp_path, encoder=encoder)
    good = library.add(text_document("بدل الإنترنت الشهري"), "good.txt")
    broken = library.add(text_document("الإنترنت في المنزل"), "broken.txt")
    npz = tmp_path / "docs" / broken.doc_id / "embeddings.npz"
    saved = npz.read_bytes()
    npz.write_bytes({"zero_bytes": b"", "not_a_zip": b"these bytes are not an npz archive",
                     "truncated_zip": saved[: len(saved) // 2]}[damage])
    embedded = len(encoder.passages())

    with caplog.at_level(logging.WARNING, logger="legalrag.library"):
        hits = library.search("الإنترنت", k=5)
        with pytest.raises(StorageError):
            library.search("الإنترنت", doc_ids=[broken.doc_id])

    assert [h.chunk.doc_id for h in hits] == [good.doc_id]
    assert broken.doc_id in " ".join(r.getMessage() for r in caplog.records)
    assert len(encoder.passages()) == embedded, "a damaged document's passages were embedded again"


@pytest.mark.parametrize("kept_lines", [0, 1], ids=["zero_bytes", "short"])
def test_a_chunks_file_that_lost_chunks_is_reported_never_served_as_fewer_chunks(tmp_path, kept_lines):
    encoder = KeywordEncoder(["العمل"])
    library = Library(tmp_path, encoder=encoder)
    meta = library.add(text_document("العمل الأول", "العمل الثاني", "العمل الثالث"), "three.txt")
    chunks_file = tmp_path / "docs" / meta.doc_id / "chunks.jsonl"
    lines = chunks_file.read_text(encoding="utf-8").splitlines(keepends=True)
    chunks_file.write_text("".join(lines[:kept_lines]), encoding="utf-8")
    embedded = len(encoder.passages())

    assert meta.chunks == 3
    assert library.search("العمل", k=5) == []
    with pytest.raises(StorageError):
        library.search("العمل", doc_ids=[meta.doc_id])
    assert len(encoder.passages()) == embedded


def test_a_bug_while_building_an_index_surfaces_as_itself_never_as_a_skipped_document(tmp_path, monkeypatch, caplog):
    library = Library(tmp_path, encoder=KeywordEncoder())
    library.add(POLICY, "policy.txt")

    def buggy(*args, **kwargs):
        raise TypeError("DenseIndex() got an unexpected keyword argument 'cache'")

    monkeypatch.setattr(docindex_mod, "DenseIndex", buggy)
    with caplog.at_level(logging.WARNING, logger="legalrag.library"), pytest.raises(TypeError, match="unexpected"):
        library.search("العمل")
    assert not caplog.records


class _BrokenQueryEncoder(KeywordEncoder):
    """Embeds passages, then fails on every query the way a programming error would."""

    def encode(self, texts, **kwargs):
        texts = list(texts)
        if any(t.startswith("query: ") for t in texts):
            raise TypeError("encode() got an unexpected keyword argument 'batch'")
        return super().encode(texts, **kwargs)


def test_an_encoder_bug_on_a_query_surfaces_as_itself_never_as_a_skipped_document(tmp_path, caplog):
    library = Library(tmp_path, encoder=_BrokenQueryEncoder())
    library.add(POLICY, "policy.txt")

    with caplog.at_level(logging.WARNING, logger="legalrag.library"), pytest.raises(TypeError, match="unexpected"):
        library.search("العمل")
    assert not caplog.records


def test_re_uploading_a_damaged_documents_bytes_replaces_its_files(tmp_path):
    internet = text_document("بدل الإنترنت الشهري")
    library = Library(tmp_path, encoder=KeywordEncoder(["الإنترنت"]))
    meta = library.add(internet, "internet.txt")
    npz = tmp_path / "docs" / meta.doc_id / "embeddings.npz"
    npz.write_bytes(b"")
    with pytest.raises(StorageError):
        library.search("الإنترنت", doc_ids=[meta.doc_id])

    repaired = library.add(internet, "internet.txt")

    assert repaired.doc_id == meta.doc_id
    assert npz.stat().st_size > 0
    assert [h.chunk.doc_id for h in library.search("الإنترنت", doc_ids=[meta.doc_id])] == [meta.doc_id]
    assert library.documents() == [repaired]
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.parametrize("deleted", [False, True], ids=["live", "soft_deleted"])
def test_replacing_a_damaged_document_keeps_its_title_and_created_at(tmp_path, monkeypatch, deleted):
    internet = text_document("بدل الإنترنت الشهري")
    library = Library(tmp_path, encoder=KeywordEncoder(["الإنترنت"]))
    readings = iter(["2026-01-01T09:00:00+00:00", "2026-09-15T18:00:00+00:00", "2026-09-15T18:30:00+00:00"])
    monkeypatch.setattr(library_mod, "_now", lambda: next(readings))
    meta = library.add(internet, "internet.txt")
    if deleted:
        library.soft_delete(meta.doc_id)
    doc_dir = tmp_path / "docs" / meta.doc_id
    (doc_dir / "embeddings.npz").write_bytes(b"")

    repaired = library.add(internet, "a new name.txt")

    assert repaired == meta  # its own title and created_at, and live again
    assert json.loads((doc_dir / "meta.json").read_text(encoding="utf-8")) == asdict(meta)
    assert (doc_dir / "embeddings.npz").stat().st_size > 0
    assert library.documents() == [meta]


def _unreadable(monkeypatch, path: Path) -> None:
    """Every open of `path` fails as a file another process holds open fails on Windows:
    through io.open, which pathlib reads with, and the builtin open, which np.load uses."""
    real_open = io.open

    def refuse(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == path:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", refuse)
    monkeypatch.setattr(builtins, "open", refuse)


@pytest.mark.parametrize("name", ["chunks.jsonl", "embeddings.npz"])
def test_a_file_that_cannot_be_read_just_now_is_not_damage_and_a_re_upload_replaces_nothing(
    tmp_path, monkeypatch, name,
):
    """A PermissionError says nothing about what a file holds. Taken for damage, it had a
    re-upload replace a healthy document, under a new title and created_at."""
    internet = text_document("بدل الإنترنت الشهري")
    encoder = KeywordEncoder(["الإنترنت"])
    library = Library(tmp_path, encoder=encoder)
    meta = library.add(internet, "internet.txt")
    doc_dir = tmp_path / "docs" / meta.doc_id
    stored = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in doc_dir.iterdir()}
    embedded = len(encoder.passages())

    _unreadable(monkeypatch, doc_dir / name)
    with pytest.raises(StorageError) as unreadable:
        library.add(internet, "a new name.txt")
    with pytest.raises(StorageError):
        library.search("الإنترنت", doc_ids=[meta.doc_id])
    monkeypatch.undo()

    assert unreadable.value.code == "internal"
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in doc_dir.iterdir()} == stored
    assert list((tmp_path / "tmp").iterdir()) == []
    assert len(encoder.passages()) == embedded
    assert library.add(internet, "a new name.txt") == meta  # readable again: the same document
    assert [h.chunk.doc_id for h in library.search("الإنترنت", doc_ids=[meta.doc_id])] == [meta.doc_id]


def test_a_damaged_document_stored_for_other_bytes_is_never_replaced_by_these(tmp_path):
    """doc_id keeps 48 bits of the sha256, and only a damaged document's own bytes may replace
    it: the full hash is checked before its index is loaded, not only on the way to a restore."""
    library = Library(tmp_path, encoder=KeywordEncoder())
    meta = library.add(POLICY, "policy.txt")
    doc_dir = tmp_path / "docs" / meta.doc_id
    meta_path = doc_dir / "meta.json"
    other_bytes = {**json.loads(meta_path.read_text(encoding="utf-8")), "sha256": meta.doc_id + "0" * 52}
    meta_path.write_text(json.dumps(other_bytes), encoding="utf-8")
    (doc_dir / "embeddings.npz").write_bytes(b"")  # damaged, so ITS bytes would replace it
    reopened = Library(tmp_path, encoder=KeywordEncoder())
    stored = {p.name: p.read_bytes() for p in doc_dir.iterdir()}

    with pytest.raises(StorageError) as refused:
        reopened.add(POLICY, "policy.txt")

    assert refused.value.code == "internal"
    assert {p.name: p.read_bytes() for p in doc_dir.iterdir()} == stored
    assert list((tmp_path / "tmp").iterdir()) == []
    assert [m.sha256 for m in reopened.documents()] == [other_bytes["sha256"]]


# ------------------------------------------------------ concurrency, model --


class _MeetingEncoder(KeywordEncoder):
    """Holds every passage embedding until `parties` uploads are embedding
    at once — which can only happen when embedding runs outside the
    library's lock."""

    def __init__(self, parties: int):
        super().__init__()
        self._barrier = threading.Barrier(parties, timeout=10)

    def encode(self, texts, **kwargs):
        texts = list(texts)
        if any(t.startswith("passage: ") for t in texts):
            self._barrier.wait()
        return super().encode(texts, **kwargs)


def test_two_concurrent_identical_uploads_end_with_one_document_and_no_staging_left(tmp_path, monkeypatch):
    library = Library(tmp_path, encoder=_MeetingEncoder(parties=2))
    docs = tmp_path / "docs"
    moves: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src, dst):
        moves.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    results, errors = [], []

    def upload():
        try:
            results.append(library.add(POLICY, "policy.txt"))
        except BaseException as e:  # surfaced by the assertion below
            errors.append(e)

    threads = [threading.Thread(target=upload) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == []
    assert len(results) == 2 and results[0] == results[1]
    assert library.documents() == [results[0]]
    assert _names(docs) == [results[0].doc_id]
    assert list((tmp_path / "tmp").iterdir()) == []
    installs = [src for src, dst in moves if dst.parent == docs]
    assert len(installs) == 1 and installs[0].parent == tmp_path / "tmp", "the upload was installed twice"
    assert all(src != docs / results[0].doc_id for src, _ in moves), "the first upload's directory was replaced"


def test_a_missing_sentence_transformers_raises_encoder_unavailable_not_system_exit(tmp_path, monkeypatch):
    """`dense.load_model` answers a missing package with SystemExit, which
    would take a server down with it. The library checks first — and a
    missing model is the server's problem, not the client's."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    library = Library(tmp_path)  # no injected encoder: the real one, loaded lazily

    assert library.search("العمل") == []  # an empty library never needs the model
    with pytest.raises(EncoderUnavailable):
        library.add(POLICY, "policy.txt")

    assert not issubclass(EncoderUnavailable, LibraryError)
    assert library.documents() == []
    assert list((tmp_path / "tmp").iterdir()) == []
    assert list((tmp_path / "docs").iterdir()) == []


@pytest.mark.parametrize("failure", [
    SystemExit("sentence-transformers is not installed"),   # dense.load_model's answer to a failed import
    ImportError("No module named 'torch'"),
    OSError("intfloat/multilingual-e5-base is not in the local cache"),
], ids=["system_exit", "import_error", "os_error"])
def test_every_way_the_model_fails_to_load_is_encoder_unavailable_and_never_ends_the_process(
    tmp_path, monkeypatch, failure,
):
    def load_model(name=dense.DEFAULT_MODEL):
        raise failure

    monkeypatch.setattr(docindex_mod, "find_spec", lambda name: object())  # the package looks installed
    monkeypatch.setattr(dense, "load_model", load_model)
    library = Library(tmp_path)

    with pytest.raises(EncoderUnavailable) as unavailable:
        library.add(POLICY, "policy.txt")

    assert unavailable.value.__cause__ is failure
    assert library.documents() == []
    assert list((tmp_path / "tmp").iterdir()) == []
    assert list((tmp_path / "docs").iterdir()) == []
