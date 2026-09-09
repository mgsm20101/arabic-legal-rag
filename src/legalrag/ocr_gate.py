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

Run: ``python tasks.py ocr-gate <ocr-output.json> [engine name]``
where the JSON maps a page key (``p07`` or ``07`` or ``7``) to that page's text.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

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


def load_ground_truth(path: Path = GROUND_TRUTH) -> dict:
    gt = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in gt.items() if not k.startswith("_")}


def run(gt: dict, ocr: dict) -> dict:
    pages = {}
    for page in sorted(gt):
        text = _page_text(ocr, page)
        if text is None:
            continue
        r = score_page(gt[page]["tokens"], digit_tokens(text))
        r["critical"] = {
            tok: fold(tok) in {fold(t) for t in digit_tokens(text)}
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
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    if not argv:
        print(__doc__.strip().splitlines()[-2])
        print("usage: python tasks.py ocr-gate <ocr-output.json> [engine name]")
        return 2

    ocr_path = Path(argv[0])
    if not ocr_path.exists():
        print(f"no such file: {ocr_path}")
        return 2
    name = argv[1] if len(argv) > 1 else ocr_path.stem

    gt = load_ground_truth()
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    result = run(gt, ocr)

    if not result["pages"]:
        print(f"{ocr_path} has no page matching the ground truth "
              f"({', '.join(sorted(gt))}).")
        return 2

    print(f"\n  OCR digit gate · {name}")
    print("  " + "-" * 64)
    for page, r in sorted(result["pages"].items()):
        print(f"  {page}: {r['correct']:2d}/{r['expected']:2d} "
              f"({r['recall']:6.1%})   read {r['found']} tokens")
        if r["missing"]:
            print(f"        missing  : {' '.join(r['missing'])}")
        if r["spurious"]:
            print(f"        invented : {' '.join(r['spurious'])}")
        for tok, ok in r["critical"].items():
            print(f"        critical {tok}: {'ok' if ok else 'FAIL'}")
    print("  " + "-" * 64)
    print(f"  digit recall: {result['correct']}/{result['expected']} "
          f"= {result['recall']:.1%}   (gate: {PASS_THRESHOLD:.0%})")

    if result["critical_failures"]:
        print(f"  critical failures: {', '.join(result['critical_failures'])}")

    passed = result["recall"] >= PASS_THRESHOLD and not result["critical_failures"]
    print(f"\n  {'PASS' if passed else 'FAIL'} — "
          + ("eligible as the text of record."
             if passed else
             "not eligible as the text of record (ADR-019)."))
    if not passed:
        print("  A wrong article number is not a typo here: it is the primary "
              "key.\n  It segments the corpus, it is the ground truth of every "
              "eval question,\n  and every cross-reference resolves through it.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
