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

That is not a hypothetical warning: the rule did it to itself. `يولية` sat in
the word set alone, which is this issue's month and nothing more invariant than
its number, and the next issue was November. The same header scored 0.875 here
and 0.667 there — still above the bar, but one OCR error from falling under it,
which is what Run 12 caught a model doing. The lesson the fix encodes is that a
category belongs in the rule whole (all twelve months) or not at all.

Run: ``python tasks.py ocr-to-raw <ocr.json> <out.txt>``
"""

from __future__ import annotations

import html as html_mod
import json
import re
import sys
from pathlib import Path

TAG = re.compile(r"<[^>]+>")
# Markdown an OCR engine emitted as LAYOUT, at the start of a line. surya
# returns HTML and deepseek-ocr returns markdown, and one corpus may hold pages
# from either, so both leave here.
#
# Not cosmetic. «### مادة (٦٦)» is not matched by `ingest`'s article-header
# regex, so the article is not split out and is not in the corpus at all --
# sixteen articles out of fifty vanished this way on law 174/2025, with no
# error anywhere. A missing article in a sequence of 800 is invisible.
#
# Deliberately narrow, to two constructs measured on that document (14 heading
# lines and 8 bullet lines out of 285; no bold, italics, tables, code fences or
# blockquotes anywhere). Both require whitespace after the marker, so `#٦٦` and
# a dash used as punctuation mid-sentence are left exactly as they are: this
# module removes an engine's rendering, never edits the text of record.
MD_PREFIX = re.compile(r"^[ \t]*(?:#{1,6}|[-*+])[ \t]+", re.MULTILINE)
# A line that is bold from end to end: «**مادة (37)**». The full 137-page run
# found what those first pages did not -- p014 wrote all four of its headers
# this way, and articles 37-40 were absent. Whole lines only: the inner text
# may not itself contain `**`, so «**أ** و **ب**» is not read as one bold run
# and bold inside a sentence stays exactly as the engine wrote it.
MD_BOLD_LINE = re.compile(r"^[ \t]*\*\*(?!\s)((?:(?!\*\*).)*?\S)\*\*[ \t]*$", re.MULTILINE)
# All three digit blocks Arabic text uses. Extended Arabic-Indic (U+06F0-06F9,
# ۰۱۲) is not optional: surya emits it for roughly one digit in six here, so
# a page number written `۲` slips past a pattern that knows only U+0660.
DIGITS_ONLY = re.compile(r"^[\s٠-٩۰-۹0-9.,،ـ\-]+$")

# An article header whose number the engine wrote BEFORE the word: «(143) مادة»
# where the page prints «مادة (١٤٣)». The page is not ambiguous -- this is one
# right-to-left line written back out left-to-right -- but `ingest`'s header
# pattern requires the word first, so an unreordered header is not a header at
# all: the article is never split out and is simply absent from the corpus.
# 27 articles on 7 pages of law 174/2025 were lost this way, among them the
# unbroken run 314-320. Same silent shape as the markdown headings in ADR-032:
# nothing raises, the count is just short.
#
# Anchored to a line that is ONLY this. A number cited mid-sentence never owns
# a line, and that is the same test `ingest` uses to tell a header from a
# reference. Three digits at most, for the same reason `ingest` caps it there:
# «(2025) مادة» is a year that happened to land alone on a line.
_TATWEEL = "ـ"
_MADA = f"م{_TATWEEL}*ا{_TATWEEL}*د{_TATWEEL}*ة"
REVERSED_HEADER = re.compile(
    r"^[ \t]*[\(\[][ \t]*([0-9٠-٩۰-۹]{1,3})[ \t]*[\)\]][ \t]*"
    r"((?:ال)?" + _MADA + r")[ \t]*$",
    re.MULTILINE,
)
PAGE_KEY = re.compile(r"(\d+)")

# Words that make up the running header of an Egyptian gazette issue. The
# issue number, date and page differ per issue and per page, so they are not
# part of the rule -- only the invariant words are.
#
# `يولية` used to sit in this set on its own, which contradicted the sentence
# above it: a month is part of the date, as issue-specific as the number. It
# was the month of the one gazette this rule was written against, and it
# inflated that corpus's scores while leaving every other issue short. The same
# header measured 0.875 in July and 0.667 in November. The fix is not to drop
# months but to know all of them -- twelve names is closed, invariant
# vocabulary, unlike an issue number. Both Egyptian spellings of the two months
# that have them are included.
MONTH_WORDS = {
    "يناير", "فبراير", "مارس", "أبريل", "إبريل", "مايو", "يونية", "يونيو",
    "يولية", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
}
FURNITURE_WORDS = {
    "الجريدة", "الرسمية", "العدد", "مكرر", "سنة", "فى", "في",
} | MONTH_WORDS

# A line is furniture when it is short AND mostly furniture words. Both halves
# matter: "سنة" alone appears inside real articles, and a long line that merely
# contains "العدد" is ordinary legal prose.
FURNITURE_MAX_TOKENS = 14
# A STRICT majority: "mostly" means more than half, and the `>=` this used to be
# deleted «( الموافق ١٣ يولية سنة ٢٠٢٠م ) .» -- the law's own issuance date, two
# of whose four words are in the set -- along with two-word sentence tails like
# «في شأنها .». Both are content, and both vanished without an error.
FURNITURE_MIN_RATIO = 0.5
# Punctuation to peel off a token before asking what it is. A token that has
# one character or none left after this carries no evidence either way, so it
# is not counted at all. The gazette's own series marker «(د)» and the en-dash
# between «الرسمية» and «العدد» both land here, and both used to be counted
# AGAINST the line being the very header they appear in.
FURNITURE_PUNCT = "()،.:-–—"


def strip_markup(text: str) -> str:
    """Remove an engine's own rendering: HTML tags, then markdown line prefixes.

    Markdown second, because an engine can emit both on one line
    (`### <b>مادة</b> (٦٦)`) and the tag has to go before the prefix is at the
    start of the line.
    """
    return MD_BOLD_LINE.sub(r"\1", MD_PREFIX.sub("", html_mod.unescape(TAG.sub("", text))))


def normalise_headers(text: str) -> str:
    """Put an article header's number back after the word, where it is printed.

    Reordering only. The digits and the word are the engine's own; nothing here
    invents, drops or renumbers anything.
    """
    return REVERSED_HEADER.sub(r"\g<2> (\g<1>)", text)


def is_page_furniture(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if DIGITS_ONLY.match(stripped):  # a bare page number on its own line
        return True
    tokens = [t for t in re.split(r"\s+", stripped) if t]
    if len(tokens) > FURNITURE_MAX_TOKENS:
        return False
    words = [t.strip(FURNITURE_PUNCT) for t in tokens if not DIGITS_ONLY.match(t)]
    evidence = [w for w in words if len(w) > 1]
    # Nothing but digits, punctuation and stray single characters: a bare page
    # number, or one of the fragments surya emits between columns (`.N`).
    if not evidence:
        return True
    hits = sum(1 for w in evidence if w in FURNITURE_WORDS)
    return hits / len(evidence) > FURNITURE_MIN_RATIO


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
        for line in normalise_headers(strip_markup(text)).splitlines():
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
