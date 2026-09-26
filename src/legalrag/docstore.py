"""One stored document on disk, and where an upload is staged (ADR-023).

    <docs>/<doc_id>/source.pdf | source.txt   the upload's bytes
                    meta.json                  DocMeta, replaced atomically
                    chunks.jsonl               one chunking.Chunk per line
                    embeddings.npz             dense.DenseIndex's own cache
    <tmp>/<uuid4 hex>/                         an upload being staged

Files only. Which document is live, what may be replaced and which error a
caller sees are `library`'s decisions: a damaged directory comes back from
here as a description, never as an exception raised on purpose.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .atomic import write_json_atomic
from .chunking import Chunk

META = "meta.json"
CHUNKS = "chunks.jsonl"
EMBEDDINGS = "embeddings.npz"


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    kind: str            # "statute" | "generic"
    suffix: str          # ".pdf" | ".txt"
    size_bytes: int
    sha256: str
    pages: int
    chunks: int
    chars: int           # non-whitespace characters extracted: the count held to MIN_TEXT_CHARS
    created_at: str      # UTC ISO-8601, seconds
    deleted: bool = False
    deleted_at: str | None = None
    # A statute-like PDF whose Arabic-only pass hit its deadline and so was chunked generically
    # instead (the page limit cannot fire here: both passes share MAX_PDF_PAGES, and the first
    # pass already raised PdfTooLarge before this one ever ran, had the page count been over it):
    # same bytes, a slower run, a different `kind`. Internal only — never serialized by the HTTP
    # API. False for every document saved before this field existed, which is the only reading a
    # missing key can have: this never happened.
    fell_back_to_pages: bool = False


def missing(directory: Path, names: Iterable[str]) -> list[str]:
    """Those of `names` that are not a file in `directory`, in order."""
    return [name for name in names if not (directory / name).is_file()]


def read_meta(doc_dir: Path) -> tuple[DocMeta | None, str]:
    """A stored document's metadata — or None, and what keeps the directory
    from being served: meta.json, chunks.jsonl or embeddings.npz missing, a
    meta.json that does not parse, or one naming another document."""
    absent = missing(doc_dir, (META, CHUNKS, EMBEDDINGS))
    if absent:
        return None, "missing " + ", ".join(absent)
    try:
        meta = DocMeta(**json.loads((doc_dir / META).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as e:
        return None, f"unreadable {META}: {e}"
    if meta.doc_id != doc_dir.name:
        return None, f"{META} names document {meta.doc_id!r}"
    return meta, ""


def write_meta(path: Path, meta: DocMeta) -> None:
    write_json_atomic(path, asdict(meta))


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    """The chunks `write_chunks` wrote; OSError, ValueError or TypeError for a
    file it did not write."""
    # "\n" only: JSON escapes it inside strings, but not U+2028, which splitlines() splits on.
    lines = path.read_text(encoding="utf-8").split("\n")
    return [Chunk(**json.loads(line)) for line in lines if line.strip()]


def clear_stale(tmp: Path, max_age_seconds: float) -> None:
    """Remove each entry of `tmp` unmodified for `max_age_seconds`. A staging
    directory's mtime moves whenever its upload adds a file, so a younger one
    may belong to an upload still running, and it stays."""
    cutoff = time.time() - max_age_seconds
    for entry in tmp.iterdir():
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError:
            continue  # gone meanwhile, or held open: a later open tries again
