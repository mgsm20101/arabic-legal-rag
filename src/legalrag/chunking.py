"""Chunk an uploaded document — the P1 demo layer (ADR-023).

Two kinds of document reach the app, and each is cut where a reader can
check a citation against it:

* **statute** — ingest's own splitter (`ingest.parse`) finds at least
  `MIN_STATUTE_ARTICLES` articles and `ingest.validate` finds no problem:
  one chunk per article, so the article number stays the key `cite.gate`
  checks a claim against, exactly as on the benchmark corpus.
* **generic** — everything else, including a document whose articles fail
  validation: chunks built from one page's lines, never crossing a page,
  so every source shown can name the one page to open.

"Statute" is decided by the splitter the benchmark already trusts, not by
a second heuristic here: one definition of an article in the project. A
document that only half-matches it (a missing or duplicated number) falls
back to pages instead of carrying article labels nobody can trust.

Nothing here reads a file or calls a model — pages in, chunks out — so the
library above it owns every I/O and failure decision.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from . import ingest
from .normalize import evaluation_normalize

MAX_CHUNK_CHARS = 1000
MIN_CHUNK_CHARS = 200

# Fewer valid articles than this is not a statute worth article labels:
# two «مادة N» lines are a document that quotes a law, not the law.
MIN_STATUTE_ARTICLES = 3

# A line ending in one of these ends a sentence: the marks `cite.SENTENCE_END`
# splits sentences on, less the newline. Closing quotes and brackets after
# the mark («... العمل.») are looked through.
_SENTENCE_END = (".", "!", "؟", "؛")
_AFTER_SENTENCE_END = ")]»\"'"

_CLOSING = r".,،؛:؟!)\]»"
_OPENING = r"(\[«"
# Only a run of marks that ENDS its token (whitespace or the end of the text
# follows the whole run) loses the space before it, and only a run that
# STARTS its token loses the space after it (see `tidy`).
_SPACE_BEFORE_CLOSING = re.compile(rf"\s+(?=[{_CLOSING}]+(?:\s|$))")
_SPACE_AFTER_OPENING = re.compile(rf"(?<!\S)([{_OPENING}]+)\s+")
_WRAPPED_LATIN_HYPHEN = re.compile(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])")
_VISIBLE = re.compile(r"\S")


@dataclass(frozen=True)
class Chunk:
    id: str              # f"{doc_id}:{number}", unique across documents
    doc_id: str
    number: int          # 1-based position in the document (DenseIndex/Hit need an int)
    article: int | None  # the article number of a "law" article; None for everything else
    label: str           # "مادة 7" | "مادة 2 (إصدار)" | "ص 3"
    page: int            # 1-based page the chunk starts on
    text: str            # tidy(evaluation_normalize(...)): what the model reads and the UI shows


def chunk_document(doc_id: str, pages: list[str], title: str = "") -> tuple[str, list[Chunk]]:
    """Returns ("statute" | "generic", chunks).

    `pages` is one string per page, in order — `pdf_text.extract_pages`, or
    a text file split on form feeds; a blank page produces nothing. `title`
    is stamped onto the parsed articles as their law name. For both kinds,
    `number` runs 1..N over the whole document.
    """
    articles = statute_chunks(doc_id, pages, title)
    if articles is not None:
        return "statute", articles
    return "generic", page_chunks(doc_id, pages)


def may_be_statute(pages: list[str]) -> bool:
    """Whether `pages` carry any article header: reason enough for a statute check, which for a
    PDF means a second, Arabic-only extraction. One is enough to look, because an extraction that
    welds a translation into the Arabic lines hides headers (Law 151/2020 shows 4 of its 56);
    being a statute still takes MIN_STATUTE_ARTICLES valid articles (`statute_chunks`)."""
    return bool(ingest.article_headers("\n".join(_universal_newlines(pages))))


def tidy(text: str) -> str:
    """Punctuation spacing only, never a letter: no space before . , ، ؛ : ؟ ! ) ] », none after ( [ «,
    and a Latin word hyphen-wrapped across a line ("Wi- Fi") rejoined ("Wi-Fi").

    A run of marks loses the space before it only when the whole run ENDS
    its token, and the space after it only when it STARTS one. A run glued
    to the neighbouring token («يناير ..2026») closes or opens nothing, and
    taking that space away would weld two words into one. Whitespace is all
    this ever removes.
    """
    # «يناير .2026» keeps its space: that period comes from pdf_text's bidi ordering, not the source.
    text = _SPACE_BEFORE_CLOSING.sub("", text)
    text = _SPACE_AFTER_OPENING.sub(r"\1", text)
    return _WRAPPED_LATIN_HYPHEN.sub("-", text)


# ------------------------------------------------------------------ statute


def statute_chunks(doc_id: str, pages: list[str], title: str = "") -> list[Chunk] | None:
    """One chunk per article, or None when `pages` are not a statute: fewer than
    MIN_STATUTE_ARTICLES articles, or any problem `ingest.validate` finds."""
    pages = _universal_newlines(pages)
    text = "\n".join(pages)
    articles = ingest.parse(text, source_file=doc_id, law_name=title)
    if len(articles) < MIN_STATUTE_ARTICLES or ingest.validate(articles):
        return None

    chunks: list[Chunk] = []
    header_pages = _header_pages(text, pages)
    for number, (article, page) in enumerate(zip(articles, header_pages, strict=True), start=1):
        law = article.book == "law"
        chunks.append(Chunk(
            id=f"{doc_id}:{number}",
            doc_id=doc_id,
            number=number,
            article=article.number if law else None,
            label=f"مادة {article.number}" if law else f"مادة {article.number} (إصدار)",
            page=page,
            text=tidy(article.text),
        ))
    return chunks


def _universal_newlines(pages: list[str]) -> list[str]:
    """As a text-mode read would give them: ingest's header patterns end a header line at
    "\\n", so a Windows-saved statute ("مادة 1\\r\\n") would never be recognised as one."""
    return [p.replace("\r\n", "\n").replace("\r", "\n") for p in pages]


def _header_pages(text: str, pages: list[str]) -> list[int]:
    """The 1-based page of every header `ingest.parse` made an article of,
    in document order.

    A header's match can start on the newline BEFORE it (the patterns open
    with ``(?:^|\\n)``), and when the header opens a page, that newline is
    the one joining two pages — so a header's page is read at its first
    visible character, not at the match start. A header with an empty body
    is skipped because `ingest.parse` skips it, which keeps this list
    one-to-one with the articles.
    """
    page_starts: list[int] = []
    offset = 0
    for page in pages:
        page_starts.append(offset)
        offset += len(page) + 1  # the "\n" the pages were joined with

    headers = ingest.article_headers(text)
    result: list[int] = []
    for i, (start, body_start, _book, _number) in enumerate(headers):
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        if not text[body_start:body_end].strip():
            continue
        visible = _VISIBLE.search(text, start)
        result.append(bisect.bisect_right(page_starts, visible.start()))
    return result


# ------------------------------------------------------------------ generic


def page_chunks(doc_id: str, pages: list[str]) -> list[Chunk]:
    """Chunks built from each page's own lines, never crossing a page — whatever headers they hold."""
    chunks: list[Chunk] = []
    for page_number, page in enumerate(pages, start=1):
        for piece in _page_pieces(page):
            text = tidy(evaluation_normalize(piece))
            if not text:
                continue  # a line of nothing but tatweel or marks normalizes away
            number = len(chunks) + 1
            chunks.append(Chunk(
                id=f"{doc_id}:{number}",
                doc_id=doc_id,
                number=number,
                article=None,
                label=f"ص {page_number}",
                page=page_number,
                text=text,
            ))
    return chunks


def _page_pieces(page: str) -> list[str]:
    """One page's non-empty lines, packed into pieces of at most MAX_CHUNK_CHARS;
    a last piece under MIN_CHUNK_CHARS joins the one before it, which puts that
    piece at most MIN_CHUNK_CHARS past the cap. Lines are measured normalized,
    as the chunk text will be: NFKC can turn one character into a phrase."""
    units: list[str] = []
    for line in page.splitlines():
        line = evaluation_normalize(line)
        if line:
            units.extend(_split_long_line(line) if len(line) > MAX_CHUNK_CHARS else [line])

    pieces = _pack(units)
    if len(pieces) > 1 and len(pieces[-1]) < MIN_CHUNK_CHARS:
        tail = pieces.pop()
        pieces[-1] = f"{pieces[-1]}\n{tail}"
    return pieces


def _split_long_line(line: str) -> list[str]:
    """A line over the cap, split on whitespace. A word is hard-split only
    when that one word is over the cap by itself."""
    parts: list[str] = []
    current = ""
    for word in line.split():
        while len(word) > MAX_CHUNK_CHARS:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:MAX_CHUNK_CHARS])
            word = word[MAX_CHUNK_CHARS:]
        if current and len(current) + 1 + len(word) <= MAX_CHUNK_CHARS:
            current = f"{current} {word}"
        else:
            if current:
                parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def _pack(units: list[str]) -> list[str]:
    """`units` (each within the cap) packed greedily, joined by "\\n", into
    pieces of at most MAX_CHUNK_CHARS. A full piece closes where
    `_keep_count` says; the lines it leaves out open the next piece."""
    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        while current and _joined_len(current) + 1 + len(unit) > MAX_CHUNK_CHARS:
            keep = _keep_count(current)
            pieces.append("\n".join(current[:keep]))
            current = current[keep:]
        current.append(unit)
    if current:
        pieces.append("\n".join(current))
    return pieces


def _keep_count(lines: list[str]) -> int:
    """How many of a full piece's lines close it: up to and including its
    last line that ends a sentence, when that leaves at least
    MIN_CHUNK_CHARS; otherwise all of them, the plain greedy cut.

    A page's lines break wherever the page edge falls, not where sentences
    end, so the plain cut regularly lands mid-sentence — and a phrase a
    question needs, split across two chunks, is whole in neither. Carrying
    the unfinished sentence into the next piece keeps it together; the
    MIN_CHUNK_CHARS floor keeps the cut from leaving a sliver behind. A
    piece never grows past the cap this way: what stays is a prefix of a
    piece that was already within it, and `_pack` re-checks what moves on.
    """
    length = _joined_len(lines)
    for count in range(len(lines), 0, -1):
        if length < MIN_CHUNK_CHARS:
            break
        if lines[count - 1].rstrip(_AFTER_SENTENCE_END).endswith(_SENTENCE_END):
            return count
        length -= len(lines[count - 1]) + 1
    return len(lines)


def _joined_len(lines: list[str]) -> int:
    return sum(len(line) for line in lines) + len(lines) - 1
