"""`reindex`, the repair a user can run after the index format changes.

`docindex.rebuild_index` does the work and `tests/test_docindex.py` pins it.
What is tested here is the part a person actually meets: which documents it
decides to touch, which it refuses to touch, and whether `--dry-run` means it.

The encoder is a stub, so nothing here loads a model.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

np = pytest.importorskip("numpy")

from legalrag import dense, reindex  # noqa: E402
from legalrag.chunking import Chunk  # noqa: E402
from legalrag.docindex import index_docs  # noqa: E402
from legalrag.docstore import CHUNKS, EMBEDDINGS, META, DocMeta, write_chunks, write_meta  # noqa: E402
from stubs import KeywordEncoder  # noqa: E402

TITLE = "سياسة تجريبية"
DOC_A = "0123456789ab"
DOC_B = "fedcba987654"


def _seed(root: Path, doc_id: str, encoder) -> Path:
    """A stored document as the library leaves one, under a real 12-hex id."""
    doc_dir = root / "docs" / doc_id
    doc_dir.mkdir(parents=True)
    chunks = [
        Chunk(id=f"{doc_id}:1", doc_id=doc_id, number=1, article=1, label="مادة 1",
              page=1, text="نص المادة الأولى من السياسة التجريبية."),
        Chunk(id=f"{doc_id}:2", doc_id=doc_id, number=2, article=2, label="مادة 2",
              page=1, text="نص المادة الثانية من السياسة التجريبية."),
    ]
    write_chunks(doc_dir / CHUNKS, chunks)
    index = dense.DenseIndex(index_docs(chunks, TITLE), encoder=encoder, cache_path=doc_dir / EMBEDDINGS)
    index.save()
    write_meta(doc_dir / META, DocMeta(
        doc_id=doc_id, title=TITLE, sha256="0" * 64, suffix=".txt", kind="generic",
        pages=1, chunks=len(chunks), chars=400, size_bytes=100,
        created_at="2026-01-01T00:00:00+00:00",
    ))
    return doc_dir


def _make_stale(doc_dir: Path) -> None:
    """Strip the keys a cache written before ADR-026/ADR-027 never had."""
    path = doc_dir / EMBEDDINGS
    with np.load(path, allow_pickle=False) as z:
        embeddings, meta = z["embeddings"], json.loads(str(z["meta"]))
    for key in ("window_overlap", "windows", "fold"):
        meta.pop(key, None)
    np.savez(path, embeddings=embeddings, meta=np.array(json.dumps(meta, ensure_ascii=False)))


@pytest.fixture
def stub_encoder(monkeypatch):
    """`reindex.main` builds its own LazyEncoder; give it one that needs no model."""
    monkeypatch.setattr(reindex, "LazyEncoder", lambda model_name: KeywordEncoder())


def test_a_library_with_nothing_stale_rebuilds_nothing(tmp_path, stub_encoder, capsys):
    _seed(tmp_path, DOC_A, KeywordEncoder())

    assert reindex.main(["--dir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "rebuilt 0" in out and "already current 1" in out


def test_a_stale_document_is_rebuilt_and_loads_afterwards(tmp_path, stub_encoder, capsys):
    doc_dir = _seed(tmp_path, DOC_A, KeywordEncoder())
    _make_stale(doc_dir)

    assert reindex.main(["--dir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "REBUILT" in out and "rebuilt 1" in out
    with np.load(doc_dir / EMBEDDINGS, allow_pickle=False) as z:
        assert "rows" in z.files, "the rebuilt cache is still in the old format"
        assert json.loads(str(z["meta"]))["fold"] == "verbatim"


def test_only_the_stale_document_is_touched(tmp_path, stub_encoder, capsys):
    healthy = _seed(tmp_path, DOC_A, KeywordEncoder())
    stale = _seed(tmp_path, DOC_B, KeywordEncoder())
    _make_stale(stale)
    untouched = (healthy / EMBEDDINGS).read_bytes()

    reindex.main(["--dir", str(tmp_path)])

    assert (healthy / EMBEDDINGS).read_bytes() == untouched, "a healthy index was rewritten"
    out = capsys.readouterr().out
    assert "rebuilt 1" in out and "already current 1" in out


def test_all_recomputes_a_healthy_document_too(tmp_path, stub_encoder, capsys):
    """For vectors that load but were produced by something since found wrong — a truncating
    encoder, say. The format is intact there, so nothing else would notice."""
    _seed(tmp_path, DOC_A, KeywordEncoder())

    reindex.main(["--dir", str(tmp_path), "--all"])

    assert "rebuilt 1" in capsys.readouterr().out


def test_dry_run_changes_nothing_on_disk(tmp_path, stub_encoder, capsys):
    doc_dir = _seed(tmp_path, DOC_A, KeywordEncoder())
    _make_stale(doc_dir)
    before = (doc_dir / EMBEDDINGS).read_bytes()

    reindex.main(["--dir", str(tmp_path), "--dry-run"])

    assert (doc_dir / EMBEDDINGS).read_bytes() == before
    out = capsys.readouterr().out
    assert "WOULD" in out and "would rebuild 1" in out
    assert "restart the app" not in out, "nothing was rebuilt, so nothing needs restarting"


def test_damaged_chunks_are_reported_not_rebuilt(tmp_path, stub_encoder, capsys):
    """The one case a rebuild cannot fix, and the exit code says so."""
    doc_dir = _seed(tmp_path, DOC_A, KeywordEncoder())
    _make_stale(doc_dir)
    (doc_dir / CHUNKS).write_text("{not json\n", encoding="utf-8")

    assert reindex.main(["--dir", str(tmp_path)]) == 1

    out = capsys.readouterr().out
    assert "BROKEN" in out and "re-upload" in out


def test_a_directory_that_is_not_a_library_says_so(tmp_path, stub_encoder, capsys):
    assert reindex.main(["--dir", str(tmp_path / "nowhere")]) == 2
    assert "no library" in capsys.readouterr().out


def test_an_empty_library_is_not_an_error(tmp_path, stub_encoder, capsys):
    (tmp_path / "docs").mkdir(parents=True)

    assert reindex.main(["--dir", str(tmp_path)]) == 0
    assert "no documents" in capsys.readouterr().out


def test_a_directory_that_is_not_a_document_id_is_ignored(tmp_path, stub_encoder, capsys):
    """Staging, a stray folder, anything an operator left there. Only 12-hex names are documents."""
    _seed(tmp_path, DOC_A, KeywordEncoder())
    (tmp_path / "docs" / "notes").mkdir()

    assert reindex.main(["--dir", str(tmp_path)]) == 0
    assert "notes" not in capsys.readouterr().out
