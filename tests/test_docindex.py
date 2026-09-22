"""`docindex.load_index`'s contract for a corrupted `embeddings.npz` (F09/T09).

`dense.DenseIndex._load_cache` rejects a structurally invalid embeddings array
the same way it already rejects a stale-fingerprint one: by returning `None`.
For a *stored document*, `load_index` wraps the shared encoder in
`_QueriesOnly`, which refuses to embed a passage — so the `None` return sends
`DenseIndex.__init__` down its no-cache path, which tries to embed a passage
through `_QueriesOnly`, which raises `IndexDamaged`. This file is the one
test that walks that whole chain end to end instead of trusting the trace.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

np = pytest.importorskip("numpy")

from legalrag import dense  # noqa: E402
from legalrag.chunking import Chunk  # noqa: E402
from legalrag.docindex import IndexDamaged, index_docs, load_index, rebuild_index  # noqa: E402
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


# ---------------------------------------------------------------------------
# rebuild_index: the repair for a cache the format moved past
# ---------------------------------------------------------------------------
#
# `load_index` may not embed a passage, so a change to the cache format leaves
# every document stored before it unopenable — and indistinguishable, to the
# library, from a file the user broke. These pin the way back.


def _downgrade_to_pre_window_format(embeddings_path: Path) -> None:
    """A cache exactly as it was written before ADR-026: no row map, no fold name.

    Not a corruption — this is what a real document uploaded in an earlier version has on
    disk, and the shape that sent a working library to "damaged" after an upgrade.
    """
    with np.load(embeddings_path, allow_pickle=False) as z:
        embeddings, meta = z["embeddings"], json.loads(str(z["meta"]))
    for key in ("window_overlap", "windows", "fold"):
        meta.pop(key, None)
    np.savez(
        embeddings_path,
        embeddings=embeddings,
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )


def test_a_cache_from_before_the_row_map_reads_as_damaged(tmp_path):
    """The defect this repair exists for, stated first: nothing is wrong with the document."""
    doc_dir = tmp_path / "doc-3"
    chunks = _seed_document(doc_dir, KeywordEncoder())
    _downgrade_to_pre_window_format(doc_dir / EMBEDDINGS)

    with pytest.raises(IndexDamaged):
        load_index(doc_dir, TITLE, len(chunks), KeywordEncoder(), dense.DEFAULT_MODEL)


def test_rebuild_index_makes_such_a_document_load_again(tmp_path):
    doc_dir = tmp_path / "doc-4"
    chunks = _seed_document(doc_dir, KeywordEncoder())
    _downgrade_to_pre_window_format(doc_dir / EMBEDDINGS)

    rows = rebuild_index(doc_dir, TITLE, len(chunks), KeywordEncoder(), dense.DEFAULT_MODEL)

    assert rows == len(chunks)
    quiet = KeywordEncoder()
    result = load_index(doc_dir, TITLE, len(chunks), quiet, dense.DEFAULT_MODEL)
    assert not quiet.seen, "the rebuilt cache was not read back; it was embedded again"
    assert set(result.chunks) == {c.id for c in chunks}


def test_rebuild_index_ranks_the_same_as_the_original(tmp_path):
    """A repair that quietly changed the answers would be worse than the failure it fixes."""
    doc_dir = tmp_path / "doc-5"
    chunks = _seed_document(doc_dir, KeywordEncoder())
    query = chunks[1].text[:20]
    before = [h.id for h in load_index(
        doc_dir, TITLE, len(chunks), KeywordEncoder(), dense.DEFAULT_MODEL).dense.search(query, k=2)]

    _downgrade_to_pre_window_format(doc_dir / EMBEDDINGS)
    rebuild_index(doc_dir, TITLE, len(chunks), KeywordEncoder(), dense.DEFAULT_MODEL)

    after = [h.id for h in load_index(
        doc_dir, TITLE, len(chunks), KeywordEncoder(), dense.DEFAULT_MODEL).dense.search(query, k=2)]
    assert after == before


def test_rebuild_index_refuses_when_the_chunks_are_the_damaged_part(tmp_path):
    """chunks.jsonl is the document; embeddings.npz is derived from it. Lose the first and
    there is nothing to derive from — re-uploading the file is the only repair, and saying so
    is more use than a rebuilt index over the wrong text."""
    doc_dir = tmp_path / "doc-6"
    chunks = _seed_document(doc_dir, KeywordEncoder())
    (doc_dir / CHUNKS).write_text("{not json at all\n", encoding="utf-8")

    with pytest.raises(IndexDamaged):
        rebuild_index(doc_dir, TITLE, len(chunks), KeywordEncoder(), dense.DEFAULT_MODEL)


def test_rebuild_index_refuses_a_chunk_count_that_does_not_match_the_metadata(tmp_path):
    doc_dir = tmp_path / "doc-7"
    chunks = _seed_document(doc_dir, KeywordEncoder())

    with pytest.raises(IndexDamaged, match="not 99"):
        rebuild_index(doc_dir, TITLE, 99, KeywordEncoder(), dense.DEFAULT_MODEL)
