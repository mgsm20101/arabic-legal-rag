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
   presentation form at all. See ``_fold_presentation_form``.
6. **NFKC injects a stray leading space for some presentation forms.** 6
   shadda ligatures and 8 isolated-haraka forms decompose with a leading
   U+0020 ahead of the mark, splitting a mid-word glyph in two. See
   ``_NFKC_LEADING_SPACE``.
7. **A single foreign word is not a second column.** ``arabic_column``
   used to split on any Latin glyph at all, dropping half the Arabic text
   on a page with just one English word. See ``arabic_column``.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits live INSIDE the
# Arabic Unicode block but are not letters: bidi treats them as a weak LTR run.
# Classifying them as Arabic reverses them — ‎٧٢ ساعة‎ becomes ‎٢٧ ساعة‎, and
# مادة (٢٥) becomes مادة (٥٢). Silent, and fatal to a corpus keyed on numbers.
DIGIT = re.compile(r"[0-9٠-٩۰-۹]")

# In an RTL line the renderer paints the MIRROR of each bracket: "مادة (1)" is
# drawn ")1(" left to right. Reading the glyphs back in logical order therefore
# has to mirror them again. (This only became visible once the English column
# was removed — its un-mirrored "(1)" was landing in the same line and made the
# Arabic column's brackets look correct by accident.)
MIRRORED = {ord(a): b for a, b in
            [("(", ")"), (")", "("), ("[", "]"), ("]", "["),
             ("{", "}"), ("}", "{"), ("<", ">"), (">", "<")]}

# Arabic base blocks + presentation forms A/B (module docstring, point 4).
# \u escapes, not literal characters: several render as invisible glyphs.
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
# mark(s), splitting a mid-word glyph in two (module docstring, point 6).
# Confirmed directly with `unicodedata`, not guessed.
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


# A single foreign word inside Arabic prose is not a second column: Latin
# content below this fraction of the page's Arabic glyph count is a term
# like "VPN" or "Wi-Fi", not a translation running the length of the page.
# evals/app/policy_ar.pdf page 2 has 21 Latin letters ("VPN", "Wi-Fi",
# "Microsoft Teams") against several hundred Arabic ones — nowhere near this
# ratio — while the 151/2020 dual-language pages, a real second column, are
# comparable in length to the Arabic column itself.
COLUMN_LATIN_RATIO = 0.20

# Even past that ratio, a genuine two-column layout separates almost
# perfectly on x: on the 151/2020 PDFs, splitting at the latin/arabic median
# midpoint puts every glyph on its own side. Text where the two scripts are
# genuinely interleaved (not columned) would instead spread glyphs of EITHER
# script across both sides of that boundary; this caps how much of that
# "bleed" is still consistent with two real columns rather than one column
# with foreign words scattered through it. 5% survives the 151/2020 fixture
# (0% bleed there) with room to spare for a boundary that is only ever an
# approximation (the median midpoint, not a fitted separator).
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
        parts.append(text if arabic else text.translate(MIRRORED))

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


def extract_pages(
    path: Path | str,
    keep_latin: bool = False,
    line_tol: float | None = None,
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
    """
    import pdfplumber  # imported lazily: only the PDF path needs it

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                chars = page.chars if keep_latin else arabic_column(page.chars)
                lines: list[str] = []
                for line in group_lines(chars, line_tol):
                    text = logical_line(drop_marks(line), keep_latin=keep_latin).strip()
                    if text:
                        lines.append(text)
                pages.append("\n".join(lines))
            finally:
                # pdfplumber keeps a page's parsed content in memory until
                # closed — 293 MB at 150 pages, 1.03 GB at 600 (measured),
                # on an app machine with ~3 GB free. `finally` closes it
                # even when this page's own extraction raised.
                page.close()
    return pages


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
