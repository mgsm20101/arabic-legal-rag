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

A browser-rendered PDF (``evals/app/policy_ar.pdf``, ADR-023) broke two
further assumptions the scanned/printed statute PDFs never exercised:

4. **Contextual presentation forms are still Arabic.** Edge encodes 572 of
   page 1's 965 Arabic glyphs as U+FB50-FDFF/U+FE70-FEFF presentation forms
   instead of plain U+0600-06FF letters. The old range treated every one of
   them as non-Arabic — never reversed, never emitted — and the extractor
   matched zero headings and zero keywords although the page count was
   right.
5. **NFKC alone does not undo a font's Persian substitution.** This font
   maps yeh/heh/kaf presentation forms so that even after
   ``unicodedata.normalize("NFKC", ...)`` they land on Persian/Urdu code
   points (``ی`` U+06CC, ``ھ`` U+06BE, ``ہ`` U+06C1, ``ک`` U+06A9) instead of
   their Arabic look-alikes (``ي``, ``ه``, ``ك``) — so ``أيام`` never matched
   ``أیام``. Folding those four letters, on top of NFKC, and ONLY for
   glyphs that were presentation forms in the first place (a base-form
   glyph must survive untouched), fixes it — except point 6.
6. **Heh sometimes skips the presentation form entirely.** This same font
   also emits U+06BE directly, as a RAW base-block glyph, for what should be
   plain heh in "القاھرة" and "المقاھي" (page 2) — confirmed with a direct
   pdfplumber scan, not merely after NFKC. Heh-doachashmee/heh-goal/keheh
   have no legitimate use in an Arabic corpus, so any occurrence of those
   three is folded whether or not it came from a presentation form. Farsi
   yeh keeps the presentation-form-only rule (point 5): it is common as a
   stylistic substitute even in ordinary fonts, so a raw occurrence is not
   assumed to be this bug. See ``_fold_presentation_form``.
7. **A single foreign word is not a second column.** ``arabic_column``
   split a page on ANY Latin glyph, with no check that the page actually
   had two columns. A single Latin word inline in Arabic prose (page 2:
   ``VPN``, ``Wi-Fi``, ``Microsoft Teams``) lost about half its Arabic text
   to a boundary that was never there. See ``arabic_column`` for the
   proportional/bleed rule that replaces it.
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

# U+0600-06FF (Arabic) and U+0750-077F/U+08A0-08FF (its Supplement and
# Extended-A blocks) are base letters. U+FB50-FDFF/U+FE70-FEFF (Presentation
# Forms A/B) are the SAME letters shaped for a specific joining position —
# Edge encodes 572 of 965 Arabic glyphs on evals/app/policy_ar.pdf's page 1
# this way. Excluding them is what made the first extraction of that PDF
# match zero headings: they were never recognised as Arabic, so never
# reversed, never folded, never emitted as anything sensible.
ARABIC_LETTER = re.compile(
    "[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)
LATIN = re.compile(r"[A-Za-z]")

# A glyph in this range is a CONTEXTUAL FORM, not a base letter — only such
# a glyph may be folded (`_fold_presentation_form` below). A base-form
# glyph must never be folded, even if it happens to already be one of the
# Persian look-alikes in `_PERSIAN_TO_ARABIC` (`test_a_base_letter_glyph_is_never_folded`).
_PRESENTATION_FORM = re.compile("[ﭐ-﷿ﹰ-﻿]")

# This font's yeh/heh/kaf presentation forms NFKC-decompose to Persian/Urdu
# code points, not their Arabic look-alikes — confirmed with `unicodedata`,
# not guessed: U+FBFE (FARSI YEH INITIAL FORM) -> NFKC -> U+06CC, U+FBAA
# (HEH DOACHASHMEE ISOLATED) -> U+06BE, U+FBA6 (HEH GOAL ISOLATED) -> U+06C1,
# U+FB8E (KEHEH ISOLATED) -> U+06A9. Left alone, ``أيام`` (U+064A yeh) never
# matches this font's ``أیام`` (U+06CC yeh) — see evals/app/README.md.
_PERSIAN_TO_ARABIC = str.maketrans({"ی": "ي", "ھ": "ه", "ہ": "ه", "ک": "ك"})

# Confirmed on evals/app/policy_ar.pdf (direct pdfplumber inspection, not
# just after NFKC): this font also emits heh-doachashmee/heh-goal as a RAW
# base-block glyph outright — U+06BE with no presentation-form code point
# anywhere near it — in "القاھرة" and "المقاھي" (page 2), never routed
# through FB50-FDFF/FE70-FEFF at all. Heh-doachashmee/heh-goal/keheh have no
# legitimate use in this project's Arabic corpus (they exist for Kurdish,
# Urdu and similar languages), so any occurrence — presentation form or raw
# — is folded. Farsi yeh (U+06CC) is excluded from this: unlike the other
# three, it is a common stylistic substitute for Arabic yeh even in ordinary
# (non-buggy) Arabic fonts, so treating every raw occurrence as this font's
# mistake would be a much wider, unverified assumption — it is folded only
# where the presentation-form signal actually says so
# (`test_a_base_letter_glyph_is_never_folded` pins the raw case unfolded).
# Confirmed empirically that this is safe for the existing statute corpus:
# data/raw/*.pdf contain zero raw or presentation-form occurrences of any of
# the four letters at all.
_ALWAYS_FOLD_RAW = re.compile("[ھہک]")

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
    """NFKC-expand a presentation-form glyph, then fold this font's Persian
    look-alikes to their Arabic equivalent — a no-op for a base-letter glyph.

    Applied PER GLYPH, before glyphs are joined, for the same reason
    reversal is per-glyph (module docstring, point 1): a lam-alef
    presentation ligature (``U+FEFB``, ``U+FEF5``, ...) is ONE glyph whose
    NFKC decomposition is TWO characters (``لا``, ``لآ``, ...), already in
    logical order — folding the assembled line instead of the individual
    glyph would risk normalising text a caller never classified as a
    presentation form in the first place, and could not tell a base-form
    ``ی`` (must survive untouched) from one that arrived via folding.

    Checking for a presentation-form code point BEFORE normalising, rather
    than just always normalising, is what keeps a base-letter glyph
    byte-identical: NFKC is a no-op for plain Arabic letters here, but the
    Persian fold below is not, and it must never touch a base-form ``ی``.

    Heh-doachashmee/heh-goal/keheh are folded whenever they appear at all,
    presentation form or raw (``_ALWAYS_FOLD_RAW`` above) — confirmed on
    evals/app/policy_ar.pdf, this font emits U+06BE directly for "المقاھي"
    and "القاھرة", with no presentation-form code point involved anywhere.
    Farsi yeh is not: it is a common stylistic substitute for Arabic yeh
    even outside this bug, so only the presentation-form signal folds it.
    """
    is_presentation_form = bool(_PRESENTATION_FORM.search(text))
    if is_presentation_form:
        text = unicodedata.normalize("NFKC", text)
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
            chars = page.chars if keep_latin else arabic_column(page.chars)
            lines: list[str] = []
            for line in group_lines(chars, line_tol):
                text = logical_line(drop_marks(line), keep_latin=keep_latin).strip()
                if text:
                    lines.append(text)
            pages.append("\n".join(lines))
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
