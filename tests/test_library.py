"""The upload document library — P1 demo layer (ADR-023), Phase 4B task 2.

On trial is everything an HTTP layer will lean on without looking: an
upload is refused before anything touches disk, the client's filename never
becomes a path, the same bytes are one document, a soft-deleted document is
never searched and never destroyed, and a reopened library trusts its saved
embeddings instead of running the model again. A keyword-count encoder
stands in for e5 throughout: no model, no network.
"""

import hashlib
import json
import os
import re
import sys
import threading
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")

import legalrag.library as library_mod  # noqa: E402
from legalrag import pdf_text  # noqa: E402
from legalrag.library import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    DocumentNotFound,
    EncoderUnavailable,
    FileTooLarge,
    Library,
    LibraryError,
    LibraryHit,
    NoTextLayer,
    UnsupportedFile,
)
from legalrag.normalize import evaluation_normalize  # noqa: E402
from stubs import KeywordEncoder, text_document  # noqa: E402

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
    def fake(path, keep_latin=False, line_tol=None):
        if seen is not None:
            seen.append({"path": Path(path), "keep_latin": keep_latin})
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
    def broken(path, keep_latin=False, line_tol=None):
        raise ValueError("corrupt cross-reference table")

    monkeypatch.setattr(pdf_text, "extract_pages", broken)
    library = Library(tmp_path, encoder=KeywordEncoder())
    before = _tree(tmp_path)

    with pytest.raises(UnsupportedFile) as rejected:
        library.add(b"%PDF-1.7\nnot really a pdf", "broken.pdf")

    assert isinstance(rejected.value.__cause__, ValueError)
    assert _tree(tmp_path) == before


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

    encoder = KeywordEncoder(keywords)
    reopened = Library(tmp_path, encoder=encoder)

    assert reopened.documents() == [kept]
    with pytest.raises(DocumentNotFound):
        reopened.get(gone.doc_id)
    assert reopened.search("الإنترنت", k=2) == before
    assert encoder.passages() == [], "a document with saved embeddings was embedded again"
    assert list((tmp_path / "tmp").iterdir()) == [], "leftover staging survived a restart"


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


def test_two_concurrent_identical_uploads_end_with_one_document_and_no_staging_left(tmp_path):
    library = Library(tmp_path, encoder=_MeetingEncoder(parties=2))
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
    assert _names(tmp_path / "docs") == [results[0].doc_id]
    assert list((tmp_path / "tmp").iterdir()) == []


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
