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
"""

from __future__ import annotations

import re
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
ARABIC_LETTER = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")

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


def _centre(c: dict) -> float:
    return (c["x0"] + c["x1"]) / 2


def arabic_column(chars: list[dict]) -> list[dict]:
    """Keep only the glyphs in the Arabic column of a dual-language page.

    The circulating PDFs of 151/2020 set the Arabic statute beside an English
    translation. Dropping Latin *letters* is not enough: the translation's
    digits and brackets survive and land in the Arabic line, so ``مادة (1)``
    arrives as ``مادة (1)  )1(`` and the splitter sees two article numbers.

    The two columns separate cleanly on x, so the split is a boundary halfway
    between the Latin and Arabic medians. Pages with no Latin at all (or with
    the two scripts interleaved rather than columned) are returned untouched —
    a wrong boundary would silently delete statute text.
    """
    latin = [_centre(c) for c in chars if _is_latin(c["text"])]
    arabic = [_centre(c) for c in chars if _is_arabic(c["text"])]
    if not latin or not arabic:
        return chars

    latin.sort()
    arabic.sort()
    latin_mid = latin[len(latin) // 2]
    arabic_mid = arabic[len(arabic) // 2]
    boundary = (latin_mid + arabic_mid) / 2
    if arabic_mid > latin_mid:
        return [c for c in chars if _centre(c) >= boundary]
    return [c for c in chars if _centre(c) <= boundary]


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
        text = "".join(c["text"] for c in glyphs)
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


def extract_text(
    path: Path | str,
    keep_latin: bool = False,
    line_tol: float | None = None,
) -> str:
    """Full document text in logical order, one visual line per output line.

    ``keep_latin=False`` drops Latin-script glyphs. The circulating PDFs of
    151/2020 are dual-language (Arabic statute beside an English translation);
    the translation is not the corpus and would otherwise be indexed as if it
    were part of the law.
    """
    import pdfplumber  # imported lazily: only the PDF path needs it

    lines: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            chars = page.chars if keep_latin else arabic_column(page.chars)
            for line in group_lines(chars, line_tol):
                text = logical_line(drop_marks(line), keep_latin=keep_latin).strip()
                if text:
                    lines.append(text)
    return "\n".join(lines)
