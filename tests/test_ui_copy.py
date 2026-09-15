"""Copy checks on the chat page: what ui/app.html and its scripts say, in Arabic.

No browser. Each test pins something the page says that an edit could get wrong
with no visible symptom until the case comes up:

- every abstain reason the API can return has its Arabic sentence;
- every count the page shows agrees with its noun, and the limits the page
  states are the ones the server enforces;
- the page claims only what the citation gate checks.

Every scan first runs on known-bad samples, so a pattern that has quietly
stopped matching cannot pass for a clean page. The test that runs js/copy.js
needs Node on PATH, and skips without it.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui_page import all_js, copy_scripts_for_node, needs_node, read, run_node  # noqa: E402

# --- every abstain reason has its sentence ----------------------------------

# The API contract's abstain reasons, and the sentence the page shows for each.
ABSTAIN_SENTENCES = {
    "no_sources": "لا توجد مستندات للبحث فيها. ارفع مستنداً أولاً.",
    "relevance_no": "لا أستطيع الإجابة من المستندات المتاحة.",
    "model_abstained": "لا أستطيع الإجابة من المستندات المتاحة.",
    "all_dropped": "حُذفت كل الجمل لأنها بلا مصدر متحقَّق منه.",
    "relevance_failure": "تعذّر توليد إجابة صالحة. أعد المحاولة.",
    "schema_failure": "تعذّر توليد إجابة صالحة. أعد المحاولة.",
    "no_claims": "تعذّر توليد إجابة صالحة. أعد المحاولة.",
}

ARABIC_LETTER = re.compile(r"[ء-ي]")


def _nfc(text: str) -> str:
    """Harakat order is not significant; compare the canonical form."""
    return unicodedata.normalize("NFC", text)


def test_every_abstain_reason_in_the_contract_has_an_arabic_sentence():
    source = all_js()
    table = re.search(r"const ABSTAIN_SENTENCES = Object\.freeze\(\{(.*?)\}\);", source, re.S)
    assert table, "no script has an ABSTAIN_SENTENCES table"
    entries = re.findall(r"^\s*([a-z_]+)\s*:\s*\"([^\"]*)\",?\s*$", table.group(1), re.M)
    assert {reason: _nfc(text) for reason, text in entries} == {
        reason: _nfc(text) for reason, text in ABSTAIN_SENTENCES.items()
    }

    fallback = re.search(r"const ABSTAIN_FALLBACK = \"([^\"]+)\";", source)
    assert fallback and ARABIC_LETTER.search(fallback.group(1)), "no Arabic sentence for an unknown reason"


# --- every count agrees with its noun -----------------------------------------

# How each counted noun reads, in the order Intl.PluralRules("ar") names the
# categories: «one» and «two» stand in for the number itself; «few» follows
# 3-10; «many» follows 11-99, where the noun is a singular tamyiz, so a
# masculine noun takes tanween; «other» follows 100-102 and the like. Zero
# takes the «few» form.
COUNT_FORMS = {
    "page": ("صفحة واحدة", "صفحتان", "صفحات", "صفحة", "صفحة"),
    "article": ("مادة واحدة", "مادتان", "مواد", "مادة", "مادة"),
    "chunk": ("مقطع واحد", "مقطعان", "مقاطع", "مقطعاً", "مقطع"),
    "sentence": ("جملة واحدة", "جملتان", "جمل", "جملة", "جملة"),
    "second": ("ثانية واحدة", "ثانيتان", "ثوانٍ", "ثانية", "ثانية"),
    "document": ("مستند واحد", "مستندان", "مستندات", "مستنداً", "مستند"),
    "letter": ("حرف واحد", "حرفان", "أحرف", "حرفاً", "حرف"),
}

HARAKAT = re.compile(r"[ً-ْ]")
_COUNTED_WORDS = sorted(
    {HARAKAT.sub("", form) for forms in COUNT_FORMS.values() for form in forms[2:]}, key=len, reverse=True
)
NUMBER_BEFORE_NOUN = re.compile(
    r"(?:\$\{[^}]*\}|[0-9٠-٩]+)\s*(?:" + "|".join(map(re.escape, _COUNTED_WORDS)) + ")"
)
COUNT_TABLE = re.compile(r"export const COUNT_FORMS = Object\.freeze\(\{.*?\}\);", re.S)
FORMAT_COUNT_CALL = re.compile(r"(?<!function )\bformatCount\(")

COUNT_SAMPLES = {
    "a number interpolated before a noun": "const pages = (n) => `${n} صفحة`;",
    "a count written out": 'const kind = "قانون · 49 مادة";',
    "a dual written by hand": 'const note = "حُذفت جملتان";',
    "an unknown noun": 'const size = formatCount(n, "pages");',
    "a call the scan cannot read": 'const wait = formatCount(Math.floor(ms / 1000), "second");',
}

GOOD_COUNT_SAMPLE = (
    "export const COUNT_FORMS = Object.freeze({\n"
    '  page: countForms("صفحة واحدة", "صفحتان", "صفحات", "صفحة", "صفحة"),\n'
    "});\n"
    "export function formatCount(n, noun) { return `${n} ${noun}`; }\n"
    'const kind = (n) => `قانون · ${formatCount(n, "article")}`;\n'
)


def _count_problems(js: str, html: str = "") -> list[str]:
    """Counts shown without the plural formatter, or through it with a noun it does not know."""
    outside = COUNT_TABLE.sub("", js)
    text = outside + "\n" + html
    problems = [f"a number set against a noun: {m.group(0)!r}" for m in NUMBER_BEFORE_NOUN.finditer(text)]
    problems += [
        f"a count form outside COUNT_FORMS: {form!r}"
        for forms in COUNT_FORMS.values()
        for form in forms[:2]
        if _nfc(form) in _nfc(text)
    ]
    calls = re.findall(r"(?<!function )\bformatCount\(([^()]*)\)", outside)
    if len(calls) != len(FORMAT_COUNT_CALL.findall(outside)):
        problems.append("a formatCount call whose arguments the scan cannot read")
    for arguments in calls:
        noun = re.fullmatch(r'[^,]+,\s*"(\w+)"', arguments.strip())
        if noun is None or noun.group(1) not in COUNT_FORMS:
            problems.append(f"formatCount({arguments}) names no noun in COUNT_FORMS")
    return problems


def test_every_count_on_the_page_goes_through_the_plural_formatter():
    for label, sample in COUNT_SAMPLES.items():
        assert _count_problems(sample), f"the scan misses {label}"
    assert _count_problems(GOOD_COUNT_SAMPLE) == []

    assert _count_problems(all_js(), read("app.html")) == []


# --- the scripts run: js/copy.js under Node ------------------------------------

COUNT_NUMBERS = (0, 1, 2, 3, 10, 11, 99, 100, 101, 102, 103, 111)


def _arabic_plural_category(n: int) -> str:
    """CLDR's Arabic plural rules for a whole number, which Intl.PluralRules("ar") follows."""
    if n in (0, 1, 2):
        return ("zero", "one", "two")[n]
    if 3 <= n % 100 <= 10:
        return "few"
    if 11 <= n % 100 <= 99:
        return "many"
    return "other"


def _expected_count(n: int, noun: str) -> str:
    one, two, few, many, other = COUNT_FORMS[noun]
    category = _arabic_plural_category(n)
    if category in ("one", "two"):
        return one if category == "one" else two
    return f"{n} {({'zero': few, 'few': few, 'many': many, 'other': other})[category]}"


@needs_node
def test_the_copy_module_counts_and_states_the_limits_the_server_enforces(tmp_path):
    from legalrag.library import MAX_UPLOAD_BYTES
    from legalrag.webapp import MAX_DOC_IDS

    copy_url = (copy_scripts_for_node(tmp_path) / "js" / "copy.js").as_uri()
    script = (
        f"const copy = await import({json.dumps(copy_url)});\n"
        f"const numbers = {json.dumps(COUNT_NUMBERS)};\n"
        "const nouns = Object.keys(copy.COUNT_FORMS);\n"
        "console.log(JSON.stringify({\n"
        "  counts: Object.fromEntries(nouns.map((noun) => [noun, numbers.map((n) => copy.formatCount(n, noun))])),\n"
        "  maxDocIds: copy.MAX_DOC_IDS,\n"
        "  maxUploadBytes: copy.MAX_UPLOAD_BYTES,\n"
        "  tooLarge: copy.TEXT.tooLarge,\n"
        "  tooMany: copy.TEXT.scopeTooMany,\n"
        "  tooShort: copy.TEXT.questionTooShort,\n"
        "  retry: [1, 2, 12].map((n) => copy.TEXT.retryAfter(n)),\n"
        '  notes: [[3, "all_dropped"], [2, null], [3, null], [0, null]].map(([n, why]) => copy.droppedNote(n, why)),\n'
        '  abstain: ["all_dropped", "constructor"].map((why) => copy.abstainSentence(why)),\n'
        "}));\n"
    )
    result = run_node("--input-type=module", stdin=script)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    nfc = lambda value: _nfc(value) if isinstance(value, str) else [nfc(v) for v in value]  # noqa: E731

    assert {noun: nfc(forms) for noun, forms in out["counts"].items()} == {
        noun: [_nfc(_expected_count(n, noun)) for n in COUNT_NUMBERS] for noun in COUNT_FORMS
    }

    assert (out["maxDocIds"], out["maxUploadBytes"]) == (MAX_DOC_IDS, MAX_UPLOAD_BYTES)
    megabytes = MAX_UPLOAD_BYTES // (1024 * 1024)
    # a file just over the limit is told the limit, never its own size rounded to read as the limit
    assert nfc(out["tooLarge"]) == _nfc(f"الحد الأقصى لحجم الملف {megabytes} ميجابايت.")
    assert nfc(out["tooMany"]) == _nfc(
        f"يمكن اختيار {_expected_count(MAX_DOC_IDS, 'document')} كحد أقصى، أو اختيار الكل."
    )
    assert nfc(out["tooShort"]) == _nfc(f"الحد الأدنى لطول السؤال: {_expected_count(3, 'letter')}.")
    assert nfc(out["retry"]) == [
        _nfc(f"مدة الانتظار قبل إعادة المحاولة: {_expected_count(n, 'second')}.") for n in (1, 2, 12)
    ]

    # all_dropped already says every sentence went; the count note would say it twice
    assert nfc(out["notes"]) == [
        "",
        _nfc("حُذفت جملتان لأنهما بلا مصدر متحقَّق منه"),
        _nfc("حُذفت 3 جمل لأنها بلا مصدر متحقَّق منه"),
        "",
    ]
    assert nfc(out["abstain"][0]) == _nfc(ABSTAIN_SENTENCES["all_dropped"])
    # a reason named like an inherited property ("constructor") gets the fallback, not what the prototype holds
    assert ARABIC_LETTER.search(out["abstain"][1]) and out["abstain"][1] not in ABSTAIN_SENTENCES.values()


# --- the page claims only what the citation gate checks ------------------------

HONEST_COPY = (
    "محلي بالكامل · البحث في المستندات المرفوعة فقط · يتحقّق الكود من أن كل جملة معروضة تستشهد بمقطع من مستنداتك",
    "تأتي الإجابة جملاً، ومع كل جملة المقطع الذي تستشهد به. اضغط على المصدر لتقرأ نصه وتتحقّق بنفسك.",
    "نص المقطع كما استُخرج من المستند. يتحقّق الكود من الاستشهاد، لا من المعنى.",
    "للعرض فقط — ليست استشارة قانونية. الإجابات مولّدة آلياً وقد تكون ناقصة أو غير دقيقة؛ راجع المصدر قبل الاعتماد عليها.",
)

# The gate checks that a sentence cites a retrieved chunk, not that the chunk supports it.
OVERCLAIMS = ("مصدر تحقّق منه الكود", "مصدرها في المستند")


def test_the_page_claims_only_what_the_citation_gate_checks():
    page = _nfc(re.sub(r"\s+", " ", read("app.html")))
    for sentence in HONEST_COPY:
        assert _nfc(sentence) in page, sentence
    for claim in OVERCLAIMS:
        assert _nfc(claim) not in page, claim
