"""Digit-accuracy gate for an OCR engine (ADR-019).

ADR-019 rejected easyocr and wrote down the condition every later engine has to
meet: *measured on digit accuracy on a known page before it is run over the
whole document*. That condition was a sentence in a document, which is exactly
the shape ADR-013's "0 remaining broken words" had before it turned out to be
false. So it lives here instead.

**Why digits and not characters.** An OCR engine is normally judged on overall
character accuracy, and by that measure a legal gazette scores well: it is
clean digital print and the prose comes out mostly right. But the prose is not
what breaks. A wrong letter inside a sentence is visible to anyone reading it;
a wrong *number* is not, because ``المادة (١٢)`` and ``المادة (١٢١)`` are
equally well-formed. In this corpus article numbers are the primary key: they
segment the text, they are the ground truth of every eval question, and every
cross-reference resolves through them. One corrupted digit silently rewires all
three, and nothing anywhere errors.

**Why a multiset and not an edit distance.** An engine that reads ``٢٠٢٠`` as
``٢ ٠`` has lost the number even though it kept three of four characters, and
an engine that reads ``٣٦،`` as ``٣٦١`` has invented an article that does not
exist. Comparing whole digit *tokens* scores both as misses, which is the
honest reading. ``missing`` and ``spurious`` are reported separately because
they mean different things: a missing number fails loudly at segmentation, an
invented one fails silently forever.

Ground truth is in ``evals/ocr/digit_ground_truth.json``, read off the page
images by eye. It is deliberately not derived from any engine's output — an
engine cannot be its own ruler.

``--gt`` points the same scorer at a different document's ground truth —
``evals/ocr/law174_digit_ground_truth.json`` is one (Run 11). The threshold and
the multiset logic do not move with it: what a passing score *licenses* is a
question for the ADR that reads the number, not for the ruler.

``--content`` scores the body of the page and not its furniture. The running
header of a gazette is printed identically on every page, so on a three-page
ground truth it can be 41% of the measurement — the same four tokens, counted
three times, next to the article numbers that actually decide an answer. An
engine that reads the header well scores points it has not earned, and one that
misreads it is punished for text ``ocr_text`` strips before ingestion anyway.
The flag takes the furniture off both sides, by two different routes because
the two sides know different things. On the ground truth it subtracts the
tokens a page lists as ``furniture``, by name. On the engine's output it runs
``ocr_text``'s own header stripper, by shape — an engine that misread the issue
number produced a token no ground truth lists, and only the shape rule finds
it. That is also the honest question: production drops those lines, so the
digits left are the digits that reach the corpus.

Threshold and multiset logic are unchanged; what the flag changes is which
tokens are on the ruler, and that has to be declared before the run (Run 12).

Run: ``python tasks.py ocr-gate <ocr.json> [engine] [--gt <path>] [--content]``
where the JSON maps a page key (``p07`` or ``07`` or ``7``) to that page's text.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from .ocr_text import is_page_furniture, strip_markup

GROUND_TRUTH = Path("evals/ocr/digit_ground_truth.json")

# A digit token is a maximal run of digits in ANY of the three blocks Arabic
# text uses, folded together: an engine that writes ٤٧ or ۴۷ or 47 has read the
# same number, and the number is what is being measured.
#
# The third block is not optional trivia. Arabic-Indic (U+0660-0669, ٠١٢) and
# EXTENDED Arabic-Indic (U+06F0-06F9, ۰۱۲) are separate code points that render
# almost identically, and surya emits both — sometimes inside one number
# (`مادة ( ۲٤ )` is U+06F2 followed by U+0664). A pattern covering only the
# first block scored 16% of this document's digits as absent and split mixed
# numbers in half, which understated the best engine by a wide margin. The rest
# of the codebase already knew this: normalize.normalize_digits, ingest's header
# regex and pdf_text all cover both blocks. This module did not.
DIGIT_RUN = re.compile(r"[٠-٩۰-۹0-9]+")
TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# This is a screening gate, not a certificate. Passing means "worth running
# over the whole document, then verifying against it" — 31 digits on two
# pages cannot establish that 29 pages are correct. Failing, on the other
# hand, is decisive: an engine that cannot read two hand-checked pages will
# not read twenty-nine. The bar is set at the point past which the corpus
# cannot be segmented at all, because article headers *are* digits.
PASS_THRESHOLD = 0.99


def digit_tokens(text: str) -> list[str]:
    return DIGIT_RUN.findall(text)


def fold(token: str) -> str:
    return token.translate(TO_ASCII)


def drop_furniture(tokens: list[str], furniture: list[str]) -> list[str]:
    """Remove ONE occurrence of each furniture token, comparing folded.

    One occurrence, not all of them: ``١٢`` is both this gazette's issue date
    and a plausible article number, and deleting every ``١٢`` on the page would
    delete the article too. Applied to the ground truth and to the engine's
    output alike, so an engine that fails to read the header is not rewarded
    for it — the header is simply not on the ruler in either direction.
    """
    budget = Counter(fold(t) for t in furniture)
    kept = []
    for tok in tokens:
        folded = fold(tok)
        if budget[folded]:
            budget[folded] -= 1
            continue
        kept.append(tok)
    return kept


def body_text(text: str) -> str:
    """The page as `ocr_text` hands it to `ingest`: furniture lines removed.

    Not a second implementation of the same idea as `drop_furniture`. That one
    works on the ground truth, where the furniture tokens are known by name.
    This one works on the engine's output, where they are not: an engine that
    misread the issue number wrote a token nobody listed, and subtracting by
    name cannot find it. `is_page_furniture` matches the running header by its
    SHAPE -- a short line dominated by the gazette's fixed words -- so it drops
    that line whatever number the engine put in it.

    Using production's own stripper is the point. The question the body score
    asks is "of the digits that reach the corpus, how many are right", and the
    digits that reach the corpus are exactly the ones this function keeps. It
    cuts both ways and is meant to: a header the engine merged into a body line
    is not short enough to match, so its digits stay in and count against the
    engine -- which is what they would do in the index.
    """
    return "\n".join(
        line for line in strip_markup(text).splitlines()
        if not is_page_furniture(line)
    )


def score_page(expected: list[str], got: list[str]) -> dict:
    """Multiset comparison of digit tokens."""
    want = Counter(fold(t) for t in expected)
    have = Counter(fold(t) for t in got)
    hit = want & have
    return {
        "expected": sum(want.values()),
        "found": sum(have.values()),
        "correct": sum(hit.values()),
        "missing": sorted((want - have).elements()),
        "spurious": sorted((have - want).elements()),
        "recall": sum(hit.values()) / sum(want.values()) if want else 0.0,
    }


def _page_text(ocr: dict, page: str) -> str | None:
    """Accept `p07`, `07` or `7` as the key for the same page."""
    bare = page.lstrip("p").lstrip("0") or "0"
    for key in (page, bare, bare.zfill(2)):
        if key in ocr:
            raw = ocr[key]
            return raw if isinstance(raw, str) else "\n".join(raw)
    return None


def _no_matching_page(gt: dict, ocr: dict) -> bool:
    """True when the OCR output has text for none of the ground-truth pages.

    That is the wrong-file case (wrong engine output, empty file, ...), and
    it is a different failure from a real run where some pages are simply
    missing -- `run` now scores each of those as a full miss on its own.
    """
    return not any(_page_text(ocr, page) is not None for page in gt)


def load_ground_truth(path: Path = GROUND_TRUTH) -> dict:
    gt = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in gt.items() if not k.startswith("_")}


def run(gt: dict, ocr: dict, *, content_only: bool = False) -> dict:
    pages = {}
    for page in sorted(gt):
        text = _page_text(ocr, page)
        # A page with no matching OCR text at all is not skipped: it is
        # scored as if the engine had read nothing on it, so a page missing
        # from the OCR output counts as a full miss instead of silently
        # dropping out of `expected`/`correct` and every critical check.
        want = gt[page]["tokens"]
        if content_only:
            # Expected side by name, observed side by shape: see `body_text`.
            # A ground truth with no `furniture` key is every token content,
            # which is what the two older files already meant.
            want = drop_furniture(want, gt[page].get("furniture", []))
            text = body_text(text) if text is not None else None
        got = digit_tokens(text) if text is not None else []
        r = score_page(want, got)
        # Checked against the same list the score is computed from: a critical
        # token is a content token by construction, so in content mode the
        # header copy of it must not be what satisfies the check.
        r["critical"] = {
            tok: fold(tok) in {fold(t) for t in got}
            for tok in gt[page].get("critical", {})
        }
        pages[page] = r
    expected = sum(r["expected"] for r in pages.values())
    correct = sum(r["correct"] for r in pages.values())
    return {
        "pages": pages,
        "expected": expected,
        "correct": correct,
        "recall": correct / expected if expected else 0.0,
        "critical_failures": [
            f"{page}:{tok}"
            for page, r in pages.items()
            for tok, ok in r["critical"].items()
            if not ok
        ],
        # Zero tolerance, same as critical_failures: an invented digit is a
        # silent failure by design (see module docstring), not a precision
        # threshold to tune.
        "spurious_failures": [
            f"{page}:{tok}"
            for page, r in pages.items()
            for tok in r["spurious"]
        ],
    }


def _take_gt_flag(argv: list[str]) -> tuple[list[str], str | None]:
    """Split `--gt <path>` out of argv, leaving the positional arguments alone."""
    if "--gt" not in argv:
        return argv, None
    i = argv.index("--gt")
    if i + 1 >= len(argv):
        raise ValueError("--gt needs a path")
    return argv[:i] + argv[i + 2:], argv[i + 1]


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    # A bare switch, so it can sit anywhere among the positional arguments and
    # cannot swallow the one after it the way `--gt` has to.
    content_only = "--content" in argv
    argv = [a for a in argv if a != "--content"]
    try:
        argv, gt_arg = _take_gt_flag(list(argv))
    except ValueError as e:
        print(f"{e}")
        return 2
    if not argv:
        print(__doc__.strip().splitlines()[-2])
        print("usage: python tasks.py ocr-gate <ocr.json> [engine] "
              "[--gt <path>] [--content]")
        return 2

    ocr_path = Path(argv[0])
    if not ocr_path.exists():
        print(f"no such file: {ocr_path}")
        return 2
    name = argv[1] if len(argv) > 1 else ocr_path.stem

    if gt_arg is not None and not Path(gt_arg).exists():
        print(f"no such ground truth: {gt_arg}")
        return 2
    # Called with no argument when no --gt was given: that is the signature the
    # tests replace, and the default has to keep going through it.
    gt = load_ground_truth(Path(gt_arg)) if gt_arg is not None else load_ground_truth()
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))

    if _no_matching_page(gt, ocr):
        print(f"{ocr_path} has no page matching the ground truth "
              f"({', '.join(sorted(gt))}).")
        return 2

    result = run(gt, ocr, content_only=content_only)

    scope = "body digits only" if content_only else "every digit on the page"
    print(f"\n  OCR digit gate · {name}")
    print(f"  scoring: {scope}")
    print("  " + "-" * 64)
    for page, r in sorted(result["pages"].items()):
        print(f"  {page}: {r['correct']:2d}/{r['expected']:2d} "
              f"({r['recall']:6.1%})   read {r['found']} tokens")
        if r["missing"]:
            print(f"        missing  : {' '.join(r['missing'])}")
        if r["spurious"]:
            print(f"        invented : {' '.join(r['spurious'])}")
        for tok, ok in r["critical"].items():
            # Folded for printing, as `missing` and `spurious` above already are.
            # Not cosmetic: a console on a legacy Arabic code page (cp1256, the
            # default here) cannot encode ٠-٩ at all, and printing them raw
            # crashed the report half-written — after the first page's numbers
            # had scrolled past, which reads like the gate itself failing. The
            # comparison has always been on folded tokens anyway.
            print(f"        critical {fold(tok)}: {'ok' if ok else 'FAIL'}")
    print("  " + "-" * 64)
    print(f"  digit recall: {result['correct']}/{result['expected']} "
          f"= {result['recall']:.1%}   (gate: {PASS_THRESHOLD:.0%})")

    # `fold` over the joined line, for the same encoding reason as above: these
    # entries are `page:token` and only the token half carries Arabic digits.
    if result["critical_failures"]:
        print(f"  critical failures: {fold(', '.join(result['critical_failures']))}")
    if result["spurious_failures"]:
        print(f"  spurious digits  : {fold(', '.join(result['spurious_failures']))}")

    passed = (
        result["recall"] >= PASS_THRESHOLD
        and not result["critical_failures"]
        and not result["spurious_failures"]
    )
    print(f"\n  {'PASS' if passed else 'FAIL'} — "
          + ("passed the digit check (ADR-019); not a certification of the "
             "text of record."
             if passed else
             "failed the digit check (ADR-019)."))
    if not passed:
        print("  A wrong article number is not a typo here: it is the primary "
              "key.\n  It segments the corpus, it is the ground truth of every "
              "eval question,\n  and every cross-reference resolves through it.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
