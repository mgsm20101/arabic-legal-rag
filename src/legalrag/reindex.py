"""Recompute stored documents' embeddings after the index format changes.

    python tasks.py reindex [--dir data/app] [--all] [--dry-run]

Why this exists. `docindex.load_index` refuses to embed a passage: a search
that quietly paid for a full re-encode would surprise its caller by minutes,
and a cache that will not load is the signal that its vectors should stop being
trusted. The cost is that a change to the cache format strands every document
stored before it — ADR-026 added the window row map, ADR-027 recorded which
spelling the vectors are in, and an index written before either is refused on
load. The library then reports the document as damaged, which reads to a user
as "my file is broken" when nothing about their file changed.

Re-uploading the same bytes does repair it, and that is the only repair the app
itself offers. This is the one that does not ask a user to go and find a PDF
from six months ago: chunks.jsonl IS the document as the library parsed it, and
embeddings.npz is derived from it, so deriving it again is always correct while
the chunks are intact.

By default only documents whose index will not load are touched. `--all`
recomputes every one, for the case where the vectors load but were produced by
something that has since been found wrong (a truncating encoder, say) — the
format is intact there, so nothing else would notice.

The app caches indexes in memory, so restart it after a run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .docindex import (
    EncoderUnavailable,
    IndexDamaged,
    IndexUnreadable,
    LazyEncoder,
    load_index,
    rebuild_index,
)
from .docstore import read_meta
from .library import DOC_ID

DEFAULT_DIR = "data/app"

OK = "ok"
STALE = "stale"
BROKEN = "broken"
UNREADABLE = "unreadable"


def inspect(doc_dir: Path, title: str, chunk_count: int, encoder, model_name: str) -> tuple[str, str]:
    """Whether this document's index loads, and what to say about it.

    `IndexDamaged` covers both "the cache is in an older format" and "chunks.jsonl is not what
    the library wrote". Only the second is unrecoverable here, and it is not distinguished by
    the exception: the rebuild itself re-reads the chunks and raises again if they are the
    problem. So a stale-looking document is reported as `STALE` and the rebuild has the last word.
    """
    try:
        load_index(doc_dir, title, chunk_count, encoder, model_name)
    except IndexDamaged as e:
        return STALE, str(e)
    except IndexUnreadable as e:
        return UNREADABLE, str(e)
    return OK, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reindex", description=__doc__)
    parser.add_argument("--dir", default=DEFAULT_DIR, help=f"the library root (default: {DEFAULT_DIR})")
    parser.add_argument("--all", action="store_true", help="recompute every document, not only the stale ones")
    parser.add_argument("--dry-run", action="store_true", help="report what would be recomputed, change nothing")
    args = parser.parse_args(argv)

    docs_dir = Path(args.dir) / "docs"
    if not docs_dir.is_dir():
        print(f"no library at {docs_dir}")
        return 2

    # Named here rather than imported from webapp: this runs without the app, and the app's own
    # default is the same string. A mismatch would rebuild vectors the app then rejects.
    from .dense import DEFAULT_MODEL

    encoder = LazyEncoder(DEFAULT_MODEL)

    stored = [d for d in sorted(docs_dir.iterdir()) if d.is_dir() and DOC_ID.fullmatch(d.name)]
    if not stored:
        print(f"no documents under {docs_dir}")
        return 0

    print(f"library : {docs_dir}  ({len(stored)} documents)")
    print(f"model   : {DEFAULT_MODEL}")
    print()

    rebuilt = failed = skipped = 0
    for doc_dir in stored:
        meta, problem = read_meta(doc_dir)
        if meta is None:
            print(f"  {doc_dir.name}  BROKEN    meta.json: {problem}")
            failed += 1
            continue

        state, detail = inspect(doc_dir, meta.title, meta.chunks, encoder, DEFAULT_MODEL)
        if state == UNREADABLE:
            # Says nothing about what is stored — a file held open, a permission. Rebuilding
            # would overwrite an index that may be perfectly good.
            print(f"  {doc_dir.name}  SKIPPED   {meta.title}: {detail}")
            failed += 1
            continue
        if state == OK and not args.all:
            print(f"  {doc_dir.name}  ok        {meta.title}")
            skipped += 1
            continue

        why = "stale index" if state == STALE else "--all"
        if args.dry_run:
            print(f"  {doc_dir.name}  WOULD     {meta.title}  ({why})")
            rebuilt += 1
            continue

        try:
            rows = rebuild_index(doc_dir, meta.title, meta.chunks, encoder, DEFAULT_MODEL)
        except IndexDamaged as e:
            # The chunks themselves, then — nothing here can recover that.
            print(f"  {doc_dir.name}  BROKEN    {meta.title}: {e}")
            print("              re-upload the original file to replace it")
            failed += 1
        except IndexUnreadable as e:
            print(f"  {doc_dir.name}  SKIPPED   {meta.title}: {e}")
            failed += 1
        except EncoderUnavailable as e:
            print(f"\nthe embedding model could not be loaded: {e}")
            return 3
        else:
            print(f"  {doc_dir.name}  REBUILT   {meta.title}  ({rows} vectors, {why})")
            rebuilt += 1

    print()
    verb = "would rebuild" if args.dry_run else "rebuilt"
    print(f"{verb} {rebuilt} · already current {skipped} · could not {failed}")
    if rebuilt and not args.dry_run:
        print("restart the app: it holds its indexes in memory.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
