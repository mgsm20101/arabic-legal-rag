"""Logical-order text extraction from a visually-ordered Arabic PDF.

``pdfplumber.Page.extract_text()`` returns glyphs in the order they are painted,
which for Arabic is *visual* order — right-to-left text comes out reversed.
``البيانات`` arrives as ``تانايبلا``. Nothing downstream notices: the characters
are valid Arabic, the article splitter simply matches nothing, and the run ends
as "ingested 0 articles" — which reads like a splitter bug.

Three details separate a correct reconstruction from a plausible-looking one:

1. **Reverse glyphs, never the string.** A lam-alef ligature (``لا``, ``لأ``,
   ``لإ``) is ONE glyph whose ``ToUnicode`` maps to TWO characters, already in
   logical order. ``"".join(reversed(text))`` splits it into ``ال`` and quietly
   corrupts every word containing it. Reversing the glyph list keeps it intact.
2. **Order by glyph CENTRE, not by ``x0``.** Arabic fonts give ``ر`` a wide left
   side bearing, so its ``x0`` can fall left of the ``ا`` that visually precedes
   it. Sorting on ``x0`` swaps the pair: ``إجراءات`` came out ``إجارءات``,
   ``الآراء`` came out ``الآارء``. Sorting on ``(x0 + x1) / 2`` fixes all of them.
3. **Split the columns before anything else.** These PDFs set the Arabic
   statute beside an English translation. Dropping Latin *letters* leaves the
   translation's digits and brackets in the Arabic line, so ``مادة (1)`` arrives
   as ``مادة (1)  )1(`` — two article numbers where the law has one.

Measured on the 151/2020 dual-language PDFs: (2) alone accounts for every
mis-spelled word found, and (3) for every duplicated article number.

A browser-rendered PDF (``evals/app/policy_ar.pdf``, ADR-023) broke four
further assumptions the scanned/printed statute PDFs never exercised:

4. **Contextual presentation forms are still Arabic.** Edge encodes 572
   of page 1's 965 glyphs as U+FB50-FDFF/U+FE70-FEFF; the old
   U+0600-06FF-only range missed all of them. See ``ARABIC_LETTER``.
5. **NFKC alone does not undo a font's Persian substitution — and ArialMT
   sometimes skips NFKC's path entirely.** Farsi-yeh and medial/final heh
   presentation forms NFKC-decompose to Persian code points; ArialMT's
   INITIAL-position heh instead arrives as raw base-block U+06BE, with no
   presentation form at all. See ``_ALWAYS_FOLD_RAW``.
6. **NFKC injects a stray leading space for some presentation forms.** 6
   shadda ligatures and 8 isolated-haraka forms decompose with a leading
   U+0020 ahead of the mark, splitting a mid-word glyph in two. See
   ``_NFKC_LEADING_SPACE``.
7. **A single foreign word is not a second column.** ``arabic_column``
   used to split on any Latin glyph at all, dropping half the Arabic text
   on a page with just one English word. See ``arabic_column``.
8. **Whether to mirror brackets, and whether to mirror «», is a property of
   the WHOLE DOCUMENT, decided once — never of one line, and never assumed
   from the file's origin.** A non-browser renderer (151/2020's statute
   PDF) draws the mirror glyph for every bracket in RTL text; a
   browser-rendered one (``policy_ar.pdf``) does not, and mirroring it a
   second time is what turned "مادة (1)" into "مادة )1(". «» were never
   mirrored at all, so a browser-rendered PDF whose «» genuinely ARE
   mirror-drawn stayed reversed. A per-line rule cannot fix this safely
   either: a legal parenthetical wraps across lines, so a line may
   legitimately begin with ")" and end with "(" on its own. See
   ``mirror_pages``.
"""

from __future__ import annotations

import re
import unicodedata
from itertools import islice
from pathlib import Path
from time import monotonic

# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits live INSIDE the
# Arabic Unicode block but are not letters: bidi treats them as a weak LTR run.
# Classifying them as Arabic reverses them — ‎٧٢ ساعة‎ becomes ‎٢٧ ساعة‎, and
# مادة (٢٥) becomes مادة (٥٢). Silent, and fatal to a corpus keyed on numbers.
DIGIT = re.compile(r"[0-9٠-٩۰-۹]")

# Swaps `()[]{}<>`. A non-browser renderer (151/2020's statute PDF) paints
# the MIRROR of each bracket in an RTL line — "مادة (1)" is drawn ")1("
# left to right — so reading the glyphs back in logical order has to
# mirror them again. A browser-rendered PDF (`policy_ar.pdf`) does not: it
# already stores the logical glyph, and mirroring it a second time is a
# bug, not a fix. Applied conditionally, once per document, by
# `mirror_pages` — `logical_line` itself no longer touches this table.
# (The statute's own bug only became visible once the English column was
# removed — its un-mirrored "(1)" used to land on the same line and made
# the Arabic column's brackets look correct by accident.)
MIRRORED = {ord(a): b for a, b in
            [("(", ")"), (")", "("), ("[", "]"), ("]", "["),
             ("{", "}"), ("}", "{"), ("<", ">"), (">", "<")]}

# Swaps «»· never in `MIRRORED`, so a browser-rendered PDF whose «» arrive
# mirror-drawn (`policy_ar.pdf` does; the statute PDF does not use «» at
# all) had no way to be corrected. Same conditional, per-document
# application as `MIRRORED`, via `mirror_pages`.
MIRRORED_QUOTES = {ord("«"): "»", ord("»"): "«"}

# Evidence for `mirror_pages`'s decision: a bracket or a «» pair that reads
# in the wrong order means the glyph codes run the opposite way from what
# a straight left-to-right reading implies. Bounded to a short run
# (quotes) or a short number (brackets) so an unrelated pair elsewhere on
# the page is never miscounted as the same broken one; `[^«»\n]` also
# keeps a quote match inside a single line, the way a wrapped parenthetical
# — legitimate, and NOT evidence either way — cannot be for brackets.
_BRACKET_LOGICAL = re.compile(r"\(\s*[0-9٠-٩]+\s*\)")
_BRACKET_MIRRORED = re.compile(r"\)\s*[0-9٠-٩]+\s*\(")
_QUOTE_LOGICAL = re.compile(r"«[^«»\n]{1,200}»")
_QUOTE_MIRRORED = re.compile(r"»[^«»\n]{1,200}«")

# Arabic base blocks + presentation forms A/B — Edge encodes 572 of page
# 1's 965 glyphs on evals/app/policy_ar.pdf this way. \u escapes, not
# literal characters: several render as invisible glyphs in an editor.
ARABIC_LETTER = re.compile(
    "[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFC]"
)
LATIN = re.compile(r"[A-Za-z]")

# A glyph in this range is a CONTEXTUAL FORM, eligible for folding
# (`_fold_presentation_form`) — U+FEFD/FEFE are unassigned and U+FEFF is
# the BOM, so the second range stops at U+FEFC, the last real one.
_PRESENTATION_FORM = re.compile("[\uFB50-\uFDFF\uFE70-\uFEFC]")

# NFKC decomposes these 14 presentation forms — 6 shadda ligatures and 8
# isolated-haraka forms — with a LEADING U+0020 SPACE ahead of the actual
# mark(s), splitting a mid-word glyph in two. Confirmed directly with
# `unicodedata`, not guessed.
_NFKC_LEADING_SPACE = frozenset(
    chr(cp) for cp in (
        0xFC5E, 0xFC5F, 0xFC60, 0xFC61, 0xFC62, 0xFC63,
        0xFE70, 0xFE72, 0xFE74, 0xFE76, 0xFE78, 0xFE7A, 0xFE7C, 0xFE7E,
    )
)

# Persian/Urdu look-alikes -> Arabic equivalents (`_ALWAYS_FOLD_RAW` below
# says what ArialMT actually emits); \u escapes for the look-alike pairs.
_PERSIAN_TO_ARABIC = str.maketrans({
    "\u06CC": "\u064A",  # farsi yeh       -> arabic yeh
    "\u06BE": "\u0647",  # heh doachashmee -> arabic heh
    "\u06C1": "\u0647",  # heh goal        -> arabic heh
    "\u06A9": "\u0643",  # keheh           -> arabic kaf
})

# ArialMT emits every INITIAL-position heh-doachashmee as raw U+06BE, never
# via a presentation form — 11 occurrences, pages 1/2/5/6 (confirmed with a
# direct pdfplumber scan); medial/final heh arrives as U+FBAD/U+FBAB
# instead. U+06C1 (heh goal), U+06A9 (keheh) and U+06CC (Farsi yeh) are NOT
# here: none was ever observed raw on this fixture, and a base-form
# occurrence of any of them is legitimate Persian text, not a bug to fix.
_ALWAYS_FOLD_RAW = re.compile("\u06BE")

# Fallback when a page has too few lines to measure its own spacing. Measured
# on the 151/2020 PDF: within-line wobble stays under 0.5pt, line spacing is
# 17.5pt or more, so anything in between separates lines correctly.
DEFAULT_LINE_TOL = 3.0


def _is_arabic(text: str) -> bool:
    """True for Arabic letters and Arabic punctuation, False for Arabic digits.

    Punctuation stays on the RTL side deliberately: a neutral character takes
    the direction of the text around it, so ‎«،»‎ must travel with the words.
    """
    return bool(ARABIC_LETTER.search(text)) and not bool(DIGIT.search(text))


def _is_latin(text: str) -> bool:
    return bool(LATIN.search(text))


def _fold_presentation_form(text: str) -> str:
    """Fold one glyph: NFKC-expand a presentation form, strip the stray
    leading space NFKC injects for some of them, then fold this font's
    Persian look-alikes to Arabic. Applied PER GLYPH, never the assembled
    line, so a lam-alef ligature still expands in place and a neighbouring
    base-form glyph (or a genuine space glyph next to this one) is never
    touched by either step.

    Folds a base-form glyph only for U+06BE (`_ALWAYS_FOLD_RAW`) — a
    base-form U+06CC is legitimate Persian text and must survive as-is.
    """
    is_presentation_form = bool(_PRESENTATION_FORM.search(text))
    if is_presentation_form:
        injects_leading_space = text in _NFKC_LEADING_SPACE
        text = unicodedata.normalize("NFKC", text)
        if injects_leading_space and text.startswith(" "):
            text = text[1:]
    if is_presentation_form or _ALWAYS_FOLD_RAW.search(text):
        text = text.translate(_PERSIAN_TO_ARABIC)
    return text


def _centre(c: dict) -> float:
    return (c["x0"] + c["x1"]) / 2


# 20%: evals/app/policy_ar.pdf page 2 has 21 Latin letters against several
# hundred Arabic ones (nowhere near this ratio); a real second column
# (151/2020) is comparable in length to the Arabic column itself. Full rule
# in `arabic_column`.
COLUMN_LATIN_RATIO = 0.20

# 5%: splitting at the latin/arabic median midpoint puts every glyph on its
# own side for a genuine two-column page (0% bleed, 151/2020) — this caps
# how much bleed still counts as two real columns. Full rule in
# `arabic_column`.
COLUMN_BLEED_RATIO = 0.05


def arabic_column(chars: list[dict]) -> list[dict]:
    """Keep only the glyphs in the Arabic column of a genuinely two-column page.

    The circulating PDFs of 151/2020 set the Arabic statute beside an English
    translation. Dropping Latin *letters* is not enough: the translation's
    digits and brackets survive and land in the Arabic line, so ``مادة (1)``
    arrives as ``مادة (1)  )1(`` and the splitter sees two article numbers.

    But splitting on ANY Latin glyph is too eager: evals/app/policy_ar.pdf
    page 2 has three Latin words (``VPN``, ``Wi-Fi``, ``Microsoft Teams``)
    inline in otherwise single-column Arabic prose, and the old rule (no
    ratio check at all) cut the page in two and silently dropped about half
    its Arabic text. A page only counts as two columns when Latin content is
    a substantial fraction of the Arabic content (``COLUMN_LATIN_RATIO``)
    AND the two scripts actually separate on x, with little enough bleed
    across the median-midpoint boundary (``COLUMN_BLEED_RATIO``) — otherwise
    the page is returned untouched, since a wrong boundary would silently
    delete statute (or policy) text either way.
    """
    latin = [_centre(c) for c in chars if _is_latin(c["text"])]
    arabic = [_centre(c) for c in chars if _is_arabic(c["text"])]
    if not latin or not arabic:
        return chars
    if len(latin) < COLUMN_LATIN_RATIO * len(arabic):
        return chars

    latin_sorted = sorted(latin)
    arabic_sorted = sorted(arabic)
    latin_mid = latin_sorted[len(latin_sorted) // 2]
    arabic_mid = arabic_sorted[len(arabic_sorted) // 2]
    boundary = (latin_mid + arabic_mid) / 2
    arabic_on_right = arabic_mid > latin_mid

    def is_arabic_side(x: float) -> bool:
        return x >= boundary if arabic_on_right else x <= boundary

    latin_bleed = sum(1 for x in latin if is_arabic_side(x)) / len(latin)
    arabic_bleed = sum(1 for x in arabic if not is_arabic_side(x)) / len(arabic)
    if latin_bleed > COLUMN_BLEED_RATIO or arabic_bleed > COLUMN_BLEED_RATIO:
        return chars

    return [c for c in chars if is_arabic_side(_centre(c))]


def _cluster(chars: list[dict], tol: float) -> list[list[dict]]:
    ordered = sorted(chars, key=lambda c: c["bottom"])
    lines: list[list[dict]] = [[ordered[0]]]
    for c in ordered[1:]:
        if c["bottom"] - lines[-1][-1]["bottom"] > tol:
            lines.append([c])
        else:
            lines[-1].append(c)
    return lines


def group_lines(chars: list[dict], tol: float | None = None) -> list[list[dict]]:
    """Cluster glyphs into visual lines, each ordered left -> right by centre.

    Clusters on the GAP between baselines, not on ``round(bottom / tol)``.
    Bucketing by division splits a line whenever its glyphs straddle a bucket
    edge: on page 4 of the 151/2020 PDF, baselines 623.5/623.7 landed in one
    bucket and 624.0 in the next, so two letters of ``القانون`` were hoisted
    onto their own line and the word was left as ``القانو``. The corpus keeps
    the damaged spelling and no check downstream can see it.

    ``tol`` is the largest vertical gap still counted as one line. Left unset,
    it is derived from the document: 45% of the median line spacing, which
    adapts to the font size instead of assuming one.
    """
    if not chars:
        return []
    if tol is None:
        rows = _cluster(chars, 1.0)
        tops = [min(c["bottom"] for c in r) for r in rows]
        gaps = sorted(b - a for a, b in zip(tops, tops[1:]))
        median_gap = gaps[len(gaps) // 2] if gaps else 0.0
        tol = max(0.45 * median_gap, 1.0) if median_gap else DEFAULT_LINE_TOL
    return [sorted(line, key=_centre) for line in _cluster(chars, tol)]


def drop_marks(line: list[dict], ratio: float = 0.35) -> list[dict]:
    """Remove glyphs painted above the line's baseline: they are diacritics.

    Not cosmetic. This font maps the tanween glyph inconsistently — ``أولاً``
    comes back with a correct ``ً`` but ``فعلاً`` comes back with a second
    ``ا``, because that glyph's ToUnicode entry is wrong. The spurious letter
    then survives ``strip_tashkeel`` (it is not a diacritic character) and the
    corpus keeps ``فعلاا``.

    Geometry separates them where the character code cannot: on this PDF marks
    sit 6.5pt above a 12pt baseline (0.54 of font size) while the largest
    honest deviation, a space glyph, is 1.7pt (0.17). The cut is at 0.35.
    """
    if len(line) < 3:
        return line
    bottoms = sorted(c["bottom"] for c in line)
    median_bottom = bottoms[len(bottoms) // 2]
    sizes = sorted(c.get("size", 0.0) for c in line)
    median_size = sizes[len(sizes) // 2]
    if not median_size:
        return line
    return [c for c in line if median_bottom - c["bottom"] <= ratio * median_size]


def logical_line(chars: list[dict], keep_latin: bool = False) -> str:
    """One visual line (left -> right glyphs) -> logical-order text.

    ``chars`` are pdfplumber char dicts; only ``text``, ``x0`` and ``x1`` are
    read, so callers can synthesise them in tests.

    Returns brackets and «» exactly as the glyph codes read — never
    mirrored here. Whether a document's glyph codes need mirroring at all
    is a whole-document question (``mirror_pages``), not a per-line one: a
    legal parenthetical wraps across lines, so a line can legitimately
    begin with ")" and end with "(" on its own, and deciding per line
    would corrupt exactly that line.
    """
    if not keep_latin:
        chars = [c for c in chars if not _is_latin(c["text"])]
    if not chars or not any(_is_arabic(c["text"]) for c in chars):
        return ""

    # Split into maximal runs of Arabic / non-Arabic glyphs.
    runs: list[tuple[bool, list[dict]]] = []
    for c in chars:
        arabic = _is_arabic(c["text"])
        if runs and runs[-1][0] == arabic:
            runs[-1][1].append(c)
        else:
            runs.append((arabic, [c]))

    # RTL base direction: the rightmost run comes first. Digits keep their
    # left-to-right order inside an RTL line (bidi treats them as a weak LTR
    # run) — "72 ساعة" must not become "27 ساعة".
    parts = []
    for arabic, run in reversed(runs):
        glyphs = reversed(run) if arabic else run
        text = "".join(_fold_presentation_form(c["text"]) for c in glyphs)
        parts.append(text)

    # A space glyph sitting on the boundary between two directions is emitted
    # on the far side of its run, so "مادة" and "(1)" arrive welded together.
    # Re-separate them: neighbouring runs of different direction always need
    # one space between them.
    out = parts[0]
    for part in parts[1:]:
        if out and part and not out[-1].isspace() and not part[0].isspace():
            out += " "
        out += part
    return re.sub(r"[ \t]{2,}", " ", out).strip()


class ExtractionLimitExceeded(Exception):
    """`extract_pages` stopped at a limit its caller set: the PDF is too big to take, not damaged."""


class TooManyPages(ExtractionLimitExceeded):
    def __init__(self, max_pages: int):
        super().__init__(f"the PDF has more than {max_pages} pages")
        self.max_pages = max_pages


class DeadlineExceeded(ExtractionLimitExceeded):
    def __init__(self, pages_done: int, pages: int):
        super().__init__(f"extraction ran out of time after {pages_done} of {pages} pages")
        self.pages_done = pages_done
        self.pages = pages


def mirror_pages(pages: list[str]) -> tuple[list[str], dict]:
    """Decide, once for the whole document, whether brackets and «» need
    mirroring, then apply that one decision to every page.

    ``pages`` are raw, unmirrored page texts — the shape ``logical_line``
    now always returns (see its docstring: it stops mirroring itself,
    because that decision cannot be made safely one line at a time).

    Counts ``_BRACKET_LOGICAL``/``_BRACKET_MIRRORED`` and
    ``_QUOTE_LOGICAL``/``_QUOTE_MIRRORED`` across every page joined
    together. ``{}``/``<>`` follow the bracket decision, exactly as
    ``MIRRORED`` always bound them to it. Mirror only on a clear majority
    in that direction; a tie or no evidence at all keeps each family's own
    long-standing default — brackets mirror (unconditional ``MIRRORED``
    was always right for a statute-style PDF), quotes stay as read (they
    were never mirrored before this function existed).

    Returns the mirrored pages and the decision itself — the four counts
    and the two booleans — so a caller (a test, or ``extract_pages``'s
    ``report``) can see the evidence, not just trust the result.
    """
    text = "\n".join(pages)
    bracket_logical = len(_BRACKET_LOGICAL.findall(text))
    bracket_mirrored = len(_BRACKET_MIRRORED.findall(text))
    quote_logical = len(_QUOTE_LOGICAL.findall(text))
    quote_mirrored = len(_QUOTE_MIRRORED.findall(text))

    decision = {
        "bracket_logical": bracket_logical,
        "bracket_mirrored": bracket_mirrored,
        "mirror_brackets": bracket_mirrored >= bracket_logical,
        "quote_logical": quote_logical,
        "quote_mirrored": quote_mirrored,
        "mirror_quotes": quote_mirrored > quote_logical,
    }

    table: dict[int, str] = {}
    if decision["mirror_brackets"]:
        table.update(MIRRORED)
    if decision["mirror_quotes"]:
        table.update(MIRRORED_QUOTES)
    mirrored_pages = [p.translate(table) if table else p for p in pages]
    return mirrored_pages, decision


def extract_pages(
    path: Path | str,
    keep_latin: bool = False,
    line_tol: float | None = None,
    *,
    max_pages: int | None = None,
    deadline: float | None = None,
    report: dict | None = None,
) -> list[str]:
    """One logical-order string per PDF page — its lines joined by ``\\n``,
    or ``""`` for a page with no text.

    Page boundaries matter to the upload pipeline (ADR-023): a document like
    ``evals/app/policy_ar.pdf`` has no «مادة» headers to chunk on, so it is
    chunked by page instead, and that needs each page's text kept apart
    rather than flattened into one string the way ``extract_text`` returns
    it. A page always gets an entry (even an empty one), so a caller can
    zip page index <-> text 1:1 without the list silently shrinking on a
    blank page.

    ``keep_latin=False`` drops Latin-script glyphs. The circulating PDFs of
    151/2020 are dual-language (Arabic statute beside an English translation);
    the translation is not the corpus and would otherwise be indexed as if it
    were part of the law.

    Two limits for a PDF nobody vetted, both off by default so every other
    caller extracts exactly as before. ``max_pages`` raises TooManyPages
    before pdfplumber even opens the file. ``deadline``, a
    ``time.monotonic()`` value, is checked between pages: once it has passed,
    the page in progress completes and DeadlineExceeded is raised instead of
    starting the next.

    Every page is read once, in raw (unmirrored) form, before ``mirror_pages``
    decides — once, for the document as a whole — whether brackets and «»
    need mirroring; see its docstring. Passing a dict as ``report`` fills it
    with that decision (the four counts and the two booleans), for a caller
    that wants to log what was decided; ``report=None`` (the default) costs
    nothing extra and changes no other caller's behaviour.
    """
    import pdfplumber  # imported lazily: only the PDF path needs it

    if max_pages is not None and _page_count(path, stop=max_pages + 1) > max_pages:
        raise TooManyPages(max_pages)
    raw_pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            if deadline is not None and raw_pages and monotonic() > deadline:
                raise DeadlineExceeded(len(raw_pages), len(pdf.pages))
            try:
                chars = page.chars if keep_latin else arabic_column(page.chars)
                lines: list[str] = []
                for line in group_lines(chars, line_tol):
                    text = logical_line(drop_marks(line), keep_latin=keep_latin).strip()
                    if text:
                        lines.append(text)
                raw_pages.append("\n".join(lines))
            finally:
                # pdfplumber keeps a page's parsed content in memory until
                # closed — 293 MB at 150 pages, 1.03 GB at 600 (measured),
                # on an app machine with ~3 GB free. `finally` closes it
                # even when this page's own extraction raised.
                page.close()
    pages, decision = mirror_pages(raw_pages)
    if report is not None:
        report.update(decision)
    return pages


def _page_count(path: Path | str, stop: int) -> int:
    """How many pages the PDF's page tree holds, counted no further than `stop`.

    Counted with pdfminer, before pdfplumber opens the file: closing a
    pdfplumber PDF builds a page object for every page, read or not, so a
    refusal made through it still paid for all of them. Measured on 100,000
    blank pages (10 MiB): 35 s that way, about 3 s this way. What pdfminer's
    lazy walk still pays in full is the parse of the cross-reference table and
    page-tree nodes, which grows with the file's size, not the pages counted.
    """
    from pdfminer.pdfdocument import PDFDocument  # the parser pdfplumber is built on
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser

    with open(path, "rb") as fh:
        document = PDFDocument(PDFParser(fh))
        return sum(1 for _ in islice(PDFPage.create_pages(document), stop))


def extract_text(
    path: Path | str,
    keep_latin: bool = False,
    line_tol: float | None = None,
) -> str:
    """Full document text in logical order, one visual line per output line.

    Exactly ``extract_pages`` joined back into one string, with blank pages
    dropped — kept as its own function because most callers (``ingest``, the
    statute corpus) want the whole document and never need a page boundary.
    """
    pages = extract_pages(path, keep_latin=keep_latin, line_tol=line_tol)
    return "\n".join(p for p in pages if p)
