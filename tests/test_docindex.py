"""`docindex.load_index`'s contract for a corrupted `embeddings.npz` (F09/T09).

`dense.DenseIndex._load_cache` rejects a structurally invalid embeddings array
the same way it already rejects a stale-fingerprint one: by returning `None`.
For a *stored document*, `load_index` wraps the shared encoder in
`_QueriesOnly`, which refuses to embed a passage — so the `None` return sends
`DenseIndex.__init__` down its no-cache path, which tries to embed a passage
through `_QueriesOnly`, which raises `IndexDamaged`. This file is the one
test that walks that whole chain end to end instead of trusting the trace.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

np = pytest.importorskip("numpy")

from legalrag import dense  # noqa: E402
from legalrag.chunking import Chunk  # noqa: E402
from legalrag.docindex import IndexDamaged, index_docs, load_index  # noqa: E402
from legalrag.docstore import CHUNKS, EMBEDDINGS, write_chunks  # noqa: E402
from stubs import KeywordEncoder  # noqa: E402

TITLE = "قانون تجريبي"


def _make_chunks(doc_id: str) -> list[Chunk]:
    return [
        Chunk(id=f"{doc_id}:1", doc_id=doc_id, number=1, article=1, label="مادة 1",
              page=1, text="نص المادة الأولى من القانون التجريبي لأغراض الاختبار."),
        Chunk(id=f"{doc_id}:2", doc_id=doc_id, number=2, article=2, label="مادة 2",
              page=1, text="نص المادة الثانية من القانون التجريبي لأغراض الاختبار."),
    ]


def _seed_document(doc_dir: Path, encoder) -> list[Chunk]:
    """A stored document exactly as the library would leave one: `chunks.jsonl` plus a real,
    saved `embeddings.npz` built from those same chunks with a live encoder — the state a
    document is in right after upload, before anything has had a chance to corrupt it."""
    doc_dir.mkdir(parents=True)
    chunks = _make_chunks(doc_dir.name)
    write_chunks(doc_dir / CHUNKS, chunks)
    index = dense.DenseIndex(index_docs(chunks, TITLE), encoder=encoder, cache_path=doc_dir / EMBEDDINGS)
    index.save()
    return chunks


def _corrupt_row_count(embeddings_path: Path) -> None:
    """Overwrite a saved cache's embeddings with the wrong number of rows, keeping its metadata
    (fingerprint, model, passage_prefix) intact — a file that loads without raising, for a
    corpus it no longer describes, same as the review's repro."""
    with np.load(embeddings_path, allow_pickle=False) as z:
        meta = str(z["meta"])
    np.savez(embeddings_path, embeddings=np.zeros((0, 0), dtype="float32"), meta=np.array(meta))


def test_a_row_count_mismatch_in_a_stored_documents_cache_raises_index_damaged(tmp_path):
    doc_dir = tmp_path / "doc-1"
    chunks = _seed_document(doc_dir, KeywordEncoder())

    _corrupt_row_count(doc_dir / EMBEDDINGS)

    with pytest.raises(IndexDamaged):
        load_index(doc_dir, TITLE, len(chunks), KeywordEncoder(), dense.DEFAULT_MODEL)


def test_an_intact_stored_document_still_loads_without_touching_the_model(tmp_path):
    """Regression guard: the new validation must not reject a legitimate, undamaged cache."""
    doc_dir = tmp_path / "doc-2"
    chunks = _seed_document(doc_dir, KeywordEncoder())

    encoder = KeywordEncoder()
    result = load_index(doc_dir, TITLE, len(chunks), encoder, dense.DEFAULT_MODEL)

    assert not encoder.seen, "an intact cache should never need the encoder at all"
    assert set(result.chunks) == {c.id for c in chunks}
