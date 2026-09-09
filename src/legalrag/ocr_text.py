"""Turn a page-keyed OCR result into raw text `ingest` can read.

Three things stand between an OCR JSON and a corpus file, and none of them is
visible until the corpus is already wrong:

**Markup.** surya returns lines carrying HTML (`<b>مادة</b> (١)`). Left in, the
article-header regex stops matching, and `ingest` reports "0 articles" while
pointing at the splitter.

**Page furniture.** Every gazette page repeats a running header — «الجريدة
الرسمية - العدد ٢٨ مكرر (ه) فى ١٥ يولية سنة ٢٠٢٠» — and a bare page number.
Ingested, they attach to whichever article the page break falls inside, so 29
articles quietly acquire a sentence that is not part of them, and BM25 sees the
same header 29 times. Nothing errors; the corpus is just wrong in a way that is
tedious to notice later.

**Page order.** JSON object key order is not numeric order. `p10` sorts before
`p2` as a string, and the law arrives shuffled.

The header is matched by *shape*, not by a hardcoded string: a short line whose
tokens are dominated by the fixed furniture words. A gazette from another issue
carries different numbers, and a rule keyed to this issue's numbers would fail
silently on the next one.

Run: ``python tasks.py ocr-to-raw <ocr.json> <out.txt>``
"""

from __future__ import annotations

import html as html_mod
import json
import re
import sys
from pathlib import Path

TAG = re.compile(r"<[^>]+>")
DIGITS_ONLY = re.compile(r"^[\s٠-٩0-9.,،ـ\-]+$")
PAGE_KEY = re.compile(r"(\d+)")

# Words that make up the running header of an Egyptian gazette issue. The
# issue number, date and page differ per issue and per page, so they are not
# part of the rule -- only the invariant words are.
FURNITURE_WORDS = {"الجريدة", "الرسمية", "العدد", "مكرر", "يولية", "سنة", "فى", "في"}

# A line is furniture when it is short AND mostly furniture words. Both halves
# matter: "سنة" alone appears inside real articles, and a long line that merely
# contains "العدد" is ordinary legal prose.
FURNITURE_MAX_TOKENS = 14
FURNITURE_MIN_RATIO = 0.5


def strip_markup(text: str) -> str:
    return html_mod.unescape(TAG.sub("", text))


def is_page_furniture(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if DIGITS_ONLY.match(stripped):  # a bare page number on its own line
        return True
    tokens = [t for t in re.split(r"\s+", stripped) if t]
    if len(tokens) > FURNITURE_MAX_TOKENS:
        return False
    words = [t for t in tokens if not DIGITS_ONLY.match(t)]
    if not words:
        return True
    hits = sum(1 for t in words if t.strip("()،.:-") in FURNITURE_WORDS)
    return hits / len(words) >= FURNITURE_MIN_RATIO


def page_order(key: str) -> int:
    m = PAGE_KEY.search(key)
    return int(m.group(1)) if m else 0


def to_raw_text(pages: dict[str, str]) -> tuple[str, int]:
    """Return (raw text in page order, number of furniture lines dropped)."""
    out: list[str] = []
    dropped = 0
    for key in sorted(pages, key=page_order):
        raw = pages[key]
        text = raw if isinstance(raw, str) else "\n".join(raw)
        for line in strip_markup(text).splitlines():
            if is_page_furniture(line):
                dropped += 1
                continue
            out.append(line.rstrip())
    return "\n".join(out).strip() + "\n", dropped


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    if len(argv) < 2:
        print("usage: python tasks.py ocr-to-raw <ocr.json> <out.txt>")
        return 2

    src, dst = Path(argv[0]), Path(argv[1])
    if not src.exists():
        print(f"no such file: {src}")
        return 2

    pages = json.loads(src.read_text(encoding="utf-8"))
    text, dropped = to_raw_text(pages)
    dst.write_text(text, encoding="utf-8")

    headers = len(re.findall(r"^\s*مادة\s*\(", text, flags=re.M))
    print(f"pages     : {len(pages)}")
    print(f"dropped   : {dropped} page-furniture lines (running header / page number)")
    print(f"wrote     : {dst}  ({len(text)} chars)")
    print(f"looks like: {headers} article headers")
    print(
        "\nBefore ingesting: move any other .pdf/.txt out of data/raw/ first.\n"
        "`ingest` reads every source file in that directory and concatenates\n"
        "them, so leaving the republication PDF in place would ingest both\n"
        "texts and duplicate every article."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
