"""Corpus ingestion: official PDF -> article-level chunks.

Chunking is by ARTICLE, not by a fixed token window. An Egyptian statute is
already segmented into self-contained numbered units, and every ground-truth
answer in ``evals/`` is keyed to an article number — a 512-token window would
make that key meaningless.

Input : data/raw/*.pdf | data/raw/*.txt   (one law, downloaded by hand)
Output: data/processed/articles.jsonl

The source is an official PDF, not a redistributed dataset (DECISIONS.md
ADR-006, reinstated by ADR-012). ``--law`` is mandatory: the law name is
stamped onto every article so the ADR-010 corpus/question binding guard has
something to check. Without it that guard silently passes.

Run: ``python tasks.py ingest --law "<اسم القانون>"``
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .normalize import evaluation_normalize, normalize_digits, search_normalize

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/articles.jsonl")
SOURCE_SUFFIXES = {".pdf", ".txt"}
# Bookkeeping files that live in data/raw/ but are not corpus text.
NOT_CORPUS = {"source.txt"}

# مادة (1) / المادة 1 / مادة رقم ( ١ ) — and NOTHING else on the line.
#
# The "own line" requirement is what separates a header from a citation. The
# 151/2020 PDF contains "المادة (21) من هذا القانون ..." at the start of a
# wrapped line; without the anchor it opened a second article 21 and cut the
# real article 21 in half. Measured on that PDF: 56 header-only lines (7
# issuance + 49 law, exactly the published structure) against 1 citation.
# Arabic justification stretches letters with tatweel (U+0640), so the gazette
# sets the same word as مادة, مـادة and مــادة on different lines. The
# republication the corpus was first built from does not use it, which is why
# this only surfaced when a gazette OCR was ingested and returned 48 articles
# where the law has 49. A missed header does not error: it welds two articles
# into one and the count is quietly short.
_T = "ـ"
_MADA = f"م{_T}*ا{_T}*د{_T}*ة"

NUMERIC_ARTICLE = re.compile(
    r"(?:^|\n)[ \t]*(?:ال)?" + _MADA + r"[ \t]*(?:رقم)?[ \t]*"
    r"[\(\[]?[ \t]*([0-9٠-٩۰-۹]{1,3})[ \t]*[\)\]]?[ \t]*[:\-–]?[ \t]*(?=\n|$)",
)

# المادة الأولى .. المادة العاشرة  (issuance articles carry ordinal names)
ORDINALS = {
    "الأولى": 1, "الاولى": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4,
    "الخامسة": 5, "السادسة": 6, "السابعة": 7, "الثامنة": 8, "التاسعة": 9,
    "العاشرة": 10,
}
# The gazette prints issuance headers inside brackets — «(المادة الأولى)» —
# where the republication printed them bare. Unmatched, all seven issuance
# articles vanish and the annexed law's own numbering is taken for theirs,
# which is exactly what the first gazette ingest reported: 36 "issuance"
# articles and 12 "law" ones.
ORDINAL_ARTICLE = re.compile(
    r"(?:^|\n)[ \t]*[\(\[]?[ \t]*(?:ال)?" + _MADA + r"[ \t]+("
    + "|".join(ORDINALS)
    + r")[ \t]*[\)\]]?[ \t]*[:\-–]?[ \t]*(?=\n|$)",
)

ARABIC_CHAR = re.compile(r"[؀-ۿ]")
NON_SPACE = re.compile(r"\S")

# Below this, the document almost certainly carried no text layer at all.
MIN_EXTRACTED_CHARS = 200
# An Arabic statute extracted correctly is overwhelmingly Arabic script.
MIN_ARABIC_RATIO = 0.30


@dataclass
class Article:
    id: str          # "law-7" | "issuance-3"
    book: str        # "law" | "issuance"
    number: int
    text: str        # verbatim
    search_text: str # aggressively normalized, for indexing
    char_len: int
    source_file: str
    law_name: str = ""


def sources() -> list[Path]:
    """The .pdf/.txt files directly under data/raw/.

    Deliberately NOT recursive: a leftover ``data/raw/egypt-legal-corpus/``
    from the Hugging Face era (ADR-007, reverted by ADR-012) must not be
    picked up as a source of raw text. ``SOURCE.txt`` is the provenance card,
    not corpus text — ingesting it would inject provenance prose into the
    articles and it is a .txt, so it has to be excluded by name.
    """
    return sorted(
        p
        for p in RAW_DIR.glob("*")
        if p.suffix.lower() in SOURCE_SUFFIXES and p.name.lower() not in NOT_CORPUS
    )


def _read_raw(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber  # type: ignore  # noqa: F401
        except ImportError:
            sys.exit(
                "pdfplumber is required to read PDFs.\n"
                "  pip install pdfplumber\n"
                "or convert the PDF to UTF-8 text and drop it in data/raw/."
            )
        from .pdf_text import extract_text

        # NOT pdfplumber's own extract_text: it returns glyphs in painting
        # order, which for Arabic is visual order — every word reversed and
        # every article number scrambled (ADR-013).
        return extract_text(path)
    sys.exit(f"Unsupported source file: {path.name} (use .pdf or .txt)")


def _headers(text: str) -> list[tuple[int, int, str, int]]:
    """All article headers in document order: (start, end, book, number).

    Both header styles are collected in ONE pass and sorted by position. Two
    independent passes would let an issuance article run past the first
    numbered article and swallow it — the bug that this ordering prevents.
    """
    found: list[tuple[int, int, str, int]] = []
    for m in ORDINAL_ARTICLE.finditer(text):
        found.append((m.start(), m.end(), "issuance", ORDINALS[m.group(1)]))
    for m in NUMERIC_ARTICLE.finditer(text):
        found.append((m.start(), m.end(), "law", int(normalize_digits(m.group(1)))))
    found.sort(key=lambda h: h[0])

    # Drop a header that starts inside the span of an earlier one (overlapping
    # regex matches), keeping the earliest.
    deduped: list[tuple[int, int, str, int]] = []
    for h in found:
        if deduped and h[0] < deduped[-1][1]:
            continue
        deduped.append(h)
    return _split_issuance(deduped)


def _split_issuance(
    headers: list[tuple[int, int, str, int]],
) -> list[tuple[int, int, str, int]]:
    """Re-label the leading run of numbered articles as the issuance law.

    An Egyptian statute of this shape is two documents in one file: a short
    issuance law, then the law it annexes, each numbered from 1. Some sources
    write the issuance articles in words (المادة الأولى) and the ORDINAL
    pattern already catches those. The 151/2020 PDF numbers them — المادة (1)
    — so both books arrive as "law" and every number 1..7 duplicates.

    The signal is the restart: the first header whose number is not greater
    than its predecessor. Everything before it belongs to the issuance law.
    A document that never restarts is left exactly as it is.
    """
    numeric = [i for i, h in enumerate(headers) if h[2] == "law"]
    if len(numeric) < 2:
        return headers
    restart = next(
        (
            i
            for prev, i in zip(numeric, numeric[1:])
            if headers[i][3] <= headers[prev][3]
        ),
        None,
    )
    if restart is None:
        return headers
    return [
        (start, end, "issuance" if i < restart else book, number)
        for i, (start, end, book, number) in enumerate(headers)
    ]


def article_headers(text: str) -> list[tuple[int, int, str, int]]:
    """Every article header `parse` splits `text` on, in document order:
    ``(start, body_start, book, number)``.

    The same scan `parse` runs, exposed for a caller that needs a header's
    POSITION — the upload chunker maps it to a page — so what counts as a
    header is never re-implemented outside this module.
    """
    return _headers(text)


def extraction_problems(text: str, name: str) -> list[str]:
    """Catch the three ways Arabic PDF extraction fails *plausibly*.

    All three used to surface as "ingested 0 articles" with no explanation,
    which reads like a splitter bug and sends the fix to the wrong place.
    """
    problems: list[str] = []
    body = text.strip()

    if len(body) < MIN_EXTRACTED_CHARS:
        problems.append(
            f"{name}: no text layer ({len(body)} chars extracted). Probably a scanned "
            f"image — run OCR (ocrmypdf --language ara) and re-ingest."
        )
        return problems  # the other two checks are meaningless on nothing

    arabic = len(ARABIC_CHAR.findall(body))
    total = len(NON_SPACE.findall(body))
    ratio = arabic / total if total else 0.0
    if ratio < MIN_ARABIC_RATIO:
        problems.append(
            f"{name}: only {ratio:.0%} of the extracted characters are Arabic. The "
            f"embedded font is not mapping to Unicode — re-export the PDF, or OCR it."
        )

    if not _headers(body):
        problems.append(
            f"{name}: no article header (مادة ...) found in {len(body):,} chars. Either "
            f"the extraction scrambled the RTL run order, or this document numbers its "
            f"articles in a style the splitter does not know yet."
        )
    return problems


def parse(text: str, source_file: str, law_name: str = "") -> list[Article]:
    headers = _headers(text)
    articles: list[Article] = []

    for i, (_, body_start, book, number) in enumerate(headers):
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        articles.append(
            Article(
                id=f"{book}-{number}",
                book=book,
                number=number,
                text=evaluation_normalize(body),
                search_text=search_normalize(body),
                char_len=len(body),
                source_file=source_file,
                law_name=law_name,
            )
        )
    return articles


# Shorter than this is a scrap, not an article: a header matched where there was
# none leaves one between it and the next real header. It was 40, a number no
# statute had been measured against -- law 151's shortest article is 77 -- and
# law 174/2025 article 286, «لا يجوز رد الشهود لأي سبب من الأسباب.», is a whole
# article in 37. A complete legal sentence does not fit in 20 characters; a
# stray «(الفقرة الأولى)» (15) still does.
MIN_ARTICLE_CHARS = 20


def validate(articles: list[Article]) -> list[str]:
    """Structural checks. Returns a list of human-readable problems."""
    problems: list[str] = []
    for book in ("issuance", "law"):
        nums = sorted(a.number for a in articles if a.book == book)
        if not nums:
            continue
        dupes = {n for n in nums if nums.count(n) > 1}
        if dupes:
            problems.append(f"[{book}] duplicate article numbers: {sorted(dupes)}")
        expected = set(range(min(nums), max(nums) + 1))
        missing = sorted(expected - set(nums))
        if missing:
            problems.append(f"[{book}] missing article numbers: {missing}")
        if min(nums) != 1:
            problems.append(f"[{book}] numbering starts at {min(nums)}, expected 1")

    tiny = [a.id for a in articles if a.char_len < MIN_ARTICLE_CHARS]
    if tiny:
        problems.append(f"suspiciously short articles (possible bad split): {tiny}")
    return problems


def _write_articles_atomic(path: Path, articles: list[Article]) -> None:
    """Write `articles` as JSONL beside `path`, then ``os.replace`` over it.

    Same pattern as ``docstore.write_json_atomic``: a reader of `path` sees the
    old corpus or the new one, never a half-written one, and a failure here —
    disk full while writing the temp file, or the final replace itself — raises
    with `path` completely untouched, because `path` is never opened directly.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        content = "".join(json.dumps(asdict(a), ensure_ascii=False) + "\n" for a in articles)
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


NO_SOURCES = """Nothing to ingest — no .pdf or .txt in data/raw/.

The corpus text comes from an official PDF downloaded by hand (ADR-006/012):
public pages return summaries and paraphrase, not verbatim text. Download it,
drop it in data/raw/, and fill in the source card in data/raw/README.md.

Then:  python tasks.py ingest --law "<اسم القانون>\""""

NEEDS_LAW = """ingest needs --law.

The law name is stamped onto every article; the ADR-010 binding guard uses it
to refuse scoring when the question set was written for a different law. Leave
it out and that guard passes silently on an empty name.

  python tasks.py ingest --law "قانون حماية البيانات الشخصية\""""


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    law_name = ""
    if "--law" in argv:
        i = argv.index("--law")
        if i + 1 >= len(argv):
            sys.exit('--law needs a value, e.g. --law "حماية البيانات"')
        law_name = argv[i + 1].strip()

    srcs = sources()
    if not srcs:
        print(NO_SOURCES)
        return 1
    if not law_name:
        print(NEEDS_LAW)
        return 1

    all_articles: list[Article] = []
    extraction: list[str] = []
    for src in srcs:
        text = _read_raw(src)
        print(f"{src.name}: {len(text):,} chars extracted")
        problems = extraction_problems(text, src.name)
        if problems:
            extraction.extend(problems)
            continue
        all_articles.extend(parse(text, src.name, law_name=law_name))

    if extraction:
        print("\nEXTRACTION PROBLEMS (nothing written — the text itself is damaged):")
        for p in extraction:
            print(f"  - {p}")
        return 2

    problems = validate(all_articles)
    if problems:
        print("\nSTRUCTURAL PROBLEMS (fix before trusting any retrieval number):")
        for p in problems:
            print(f"  - {p}")
        return 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_articles_atomic(OUT_PATH, all_articles)

    print(f"\nlaw: {law_name}")
    print(f"ingested {len(all_articles)} articles -> {OUT_PATH}")
    for book in ("issuance", "law"):
        n = sum(1 for a in all_articles if a.book == book)
        print(f"  {book:9s}: {n}")
    print("\nstructure OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
