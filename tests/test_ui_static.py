"""Static checks on the chat page: ui/app.html, ui/app.css, and ui/app.js with its modules.

No browser. Each test reads the files as text and pins one property that the
page's safety or its API contract rests on, and that an edit could break with
no visible symptom:

- text from the server never reaches an HTML parser;
- nothing on the page needs an exception to its Content-Security-Policy, and
  nothing is loaded from another origin;
- every request goes through one helper that sends the app header;
- every script the page imports is one the server serves;
- every abstain reason the API can return has its Arabic sentence;
- every count the page shows agrees with its noun.

Every scan first runs on known-bad samples, so a pattern that has quietly
stopped matching cannot pass for a clean page.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"

TAG = re.compile(r"<[a-zA-Z][^>]*>")

# The scripts the server serves, and nothing else: the entry and its five modules.
ALLOWLISTED_JS = ("app.js", "js/copy.js", "js/api.js", "js/dom.js", "js/documents.js", "js/chat.js")


def _read(name: str) -> str:
    return (UI / name).read_text(encoding="utf-8")


def _all_js() -> str:
    """Every allowlisted script as one text, so that no scan can miss a module."""
    return "\n;\n".join(_read(name) for name in ALLOWLISTED_JS)


def _attr(tag: str, name: str) -> str | None:
    """The value of attribute `name` in one start tag, or None."""
    match = re.search(
        rf"\s{re.escape(name)}\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))", tag, re.I
    )
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


# --- server text never becomes markup or code ------------------------------

HTML_SINKS = {
    "innerHTML": r"\binnerHTML\b",
    "outerHTML": r"\bouterHTML\b",
    "insertAdjacentHTML": r"\binsertAdjacentHTML\b",
    "setHTMLUnsafe": r"\bsetHTMLUnsafe\b",
    "createContextualFragment": r"\bcreateContextualFragment\b",
    "parseFromString": r"\bparseFromString\b",
    "srcdoc": r"\bsrcdoc\b",
    "document.write": r"\bdocument\s*\.\s*write",
    "eval": r"\beval\s*\(",
    "the Function constructor": r"\bFunction\s*\(",
    "a timer given a string": r"\bset(?:Timeout|Interval)\s*\(\s*[\"'`]",
}

SINK_SAMPLES = [
    "card.innerHTML = reply.message_ar;",
    "card.outerHTML = row;",
    "list.insertAdjacentHTML('beforeend', row);",
    "card.setHTMLUnsafe(text);",
    "range.createContextualFragment(text);",
    "new DOMParser().parseFromString(text, 'text/html');",
    "frame.srcdoc = text;",
    "document.write(title);",
    "window.eval(body);",
    "const run = new Function('return ' + body);",
    "setTimeout('refresh()', 500);",
    'setInterval( "poll()", 500);',
    "setTimeout(`tick()`, 500);",
]

SAFE_JS_SAMPLE = "card.textContent = reply.message_ar; setTimeout(() => hide(toast), 500);"


def _html_sinks(source: str) -> list[str]:
    return [name for name, pattern in HTML_SINKS.items() if re.search(pattern, source)]


def test_the_chat_page_never_assigns_html_from_strings():
    for sample in SINK_SAMPLES:
        assert _html_sinks(sample), f"the scan misses {sample!r}"
    assert _html_sinks(SAFE_JS_SAMPLE) == []

    source = _all_js()
    assert _html_sinks(source) == []
    assert "textContent" in source, "server text is expected to render as text"


# --- nothing the CSP would refuse -------------------------------------------

INLINE_SAMPLES = [
    "<script>start()</script>",
    "<style>body{color:red}</style>",
    '<p style="color:red">نص</p>',
    '<button type="button" onclick="ask()">اسأل</button>',
    "<body onload = 'init()'>",
    '<a href="javascript:ask()">اسأل</a>',
]

CLEAN_HTML_SAMPLE = (
    '<button type="button" class="btn">اسأل</button><script src="app.js" defer></script>'
)


def _inline_code(html: str) -> list[str]:
    """What in `html` a policy without 'unsafe-inline' refuses to run or apply."""
    found = [
        "a <script> with a body"
        for body in re.findall(r"<script\b[^>]*>(.*?)</script\s*>", html, re.I | re.S)
        if body.strip()
    ]
    if re.search(r"<style\b", html, re.I):
        found.append("a <style> block")
    for tag in TAG.findall(html):
        if re.search(r"\sstyle\s*=", tag, re.I):
            found.append(f"a style attribute: {tag}")
        if re.search(r"\son[a-z]+\s*=", tag, re.I):
            found.append(f"an event-handler attribute: {tag}")
        if re.search(r"=\s*[\"']?\s*javascript:", tag, re.I):
            found.append(f"a javascript: URL: {tag}")
    return found


def test_the_chat_page_has_no_inline_script_or_style():
    for sample in INLINE_SAMPLES:
        assert _inline_code(sample), f"the scan misses {sample!r}"
    assert _inline_code(CLEAN_HTML_SAMPLE) == []

    assert _inline_code(_read("app.html")) == []


# --- RTL Arabic, and only its own two assets --------------------------------

URL_ATTRIBUTES = ("src", "href", "xlink:href", "srcset", "action", "formaction", "poster", "data")

EXTERNAL_URL = re.compile(r"\bhttps?:", re.I)
CSS_LOAD = re.compile(r"@import|url\(\s*[\"']?\s*(?!data:)", re.I)
JS_PROTOCOL_RELATIVE = re.compile(r"[\"'`]//")


def _urls(html: str) -> list[str]:
    """Every URL the markup points at, except same-document #fragments."""
    urls = []
    for tag in TAG.findall(html):
        for name in URL_ATTRIBUTES:
            value = _attr(tag, name)
            if value is not None and not value.startswith("#"):
                urls.append(value)
    return sorted(urls)


def test_the_chat_page_is_rtl_arabic_and_loads_only_its_own_assets():
    assert EXTERNAL_URL.search('<link rel="stylesheet" href="https://fonts.example/a.css">')
    assert EXTERNAL_URL.search("fetch('http://127.0.0.1:11434/api/chat')")
    assert CSS_LOAD.search('@import "theme.css";') and CSS_LOAD.search("background: url(/bg.png)")
    assert not CSS_LOAD.search("background: url(data:image/png;base64,AAAA)")
    assert JS_PROTOCOL_RELATIVE.search('fetch("//cdn.example/x.js")')
    assert _urls('<img src="//cdn.example/x.png"><a href="#top">') == ["//cdn.example/x.png"]

    html = _read("app.html")
    root = re.search(r"<html\b[^>]*>", html, re.I)
    assert root, "no <html> tag"
    assert _attr(root.group(0), "lang") == "ar"
    assert _attr(root.group(0), "dir") == "rtl"

    tags = TAG.findall(html)
    stylesheets = [
        _attr(tag, "href")
        for tag in tags
        if re.match(r"<link\b", tag, re.I) and (_attr(tag, "rel") or "").lower() == "stylesheet"
    ]
    scripts = [_attr(tag, "src") for tag in tags if re.match(r"<script\b", tag, re.I)]
    assert stylesheets == ["app.css"]
    assert scripts == ["app.js"]
    assert _urls(html) == ["app.css", "app.js"]

    for name in ("app.html", "app.css", *ALLOWLISTED_JS):
        assert not EXTERNAL_URL.search(_read(name)), f"{name} names an http(s) URL"
    assert not CSS_LOAD.search(_read("app.css")), "app.css loads a resource"
    assert not JS_PROTOCOL_RELATIVE.search(_all_js()), "a script names a protocol-relative URL"


# --- one request helper, and it sends the app header ------------------------

FETCH = re.compile(r"\bfetch\s*\(")
APP_HEADER = re.compile(r"[\"']X-LegalRAG[\"']\s*:\s*[\"']1[\"']")
OTHER_REQUEST_APIS = re.compile(
    r"\bXMLHttpRequest\b|\bsendBeacon\b|\bEventSource\b|\bWebSocket\b|\bimport\s*\("
)

_WITH_HEADER = '{ headers: { "X-LegalRAG": "1" } }'
REQUEST_SAMPLES = {
    "a second fetch call": (
        f'async function api(p) {{ return fetch(p, {_WITH_HEADER}); }}\n'
        'api("/api/health");\nfetch("/api/documents");'
    ),
    "a helper without the header": 'async function api(p) { return fetch(p); }\napi("/api/health");',
    "a computed URL": (
        f'async function api(p) {{ return fetch(p, {_WITH_HEADER}); }}\napi(base + "/api/health");'
    ),
    "a fetch outside any named helper": (
        f'const get = (p) => fetch(p, {_WITH_HEADER});\nget("/api/health");'
    ),
    "a request made without fetch": (
        f'async function api(p) {{ return fetch(p, {_WITH_HEADER}); }}\n'
        'api("/api/health");\nnew XMLHttpRequest();'
    ),
}

GOOD_REQUEST_SAMPLE = (
    'async function api(path, { method = "GET" } = {}) {\n'
    '  const headers = { "X-LegalRAG": "1" };\n'
    "  return fetch(path, { method, headers });\n"
    "}\n"
    'api("/api/health");\n'
    'api(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" });\n'
)


def _closing(source: str, opening: int) -> int:
    """Index of the bracket that closes the one at `opening`.

    Skips string literals and comments; app.js has no regular-expression
    literal inside its request helper, which is all this is used on.
    """
    closer = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    i = opening
    while i < len(source):
        char = source[i]
        if char in "'\"`":
            i += 1
            while i < len(source) and source[i] != char:
                i += 2 if source[i] == "\\" else 1
        elif source.startswith("//", i):
            i = source.find("\n", i)
            if i == -1:
                break
        elif source.startswith("/*", i):
            i = source.index("*/", i) + 1
        elif char in closer:
            stack.append(closer[char])
        elif char in ")]}":
            if not stack or stack.pop() != char:
                break
            if not stack:
                return i
        i += 1
    raise AssertionError(f"unbalanced bracket at offset {opening}")


def _enclosing_function(source: str, index: int) -> tuple[str, str] | None:
    """Name and body of the innermost named function whose body holds `index`."""
    declarations = re.finditer(r"\bfunction\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(", source[:index])
    for declaration in reversed(list(declarations)):
        params_end = _closing(source, declaration.end() - 1)
        body_start = source.find("{", params_end)
        body_end = _closing(source, body_start)
        if body_start < index < body_end:
            return declaration.group(1), source[body_start : body_end + 1]
    return None


def _request_problems(source: str) -> list[str]:
    calls = [match.start() for match in FETCH.finditer(source)]
    if len(calls) != 1:
        return [f"{len(calls)} fetch calls; every request should share one"]
    problems = []
    if OTHER_REQUEST_APIS.search(source):
        problems.append("a request made without fetch")
    helper = _enclosing_function(source, calls[0])
    if helper is None:
        return problems + ["fetch is not inside a named helper function"]
    name, body = helper
    if not APP_HEADER.search(body):
        problems.append(f"{name}() does not send X-LegalRAG: 1")
    uses = [
        match
        for match in re.finditer(rf"(?<![\w$.]){re.escape(name)}\s*\(", source)
        if not source[: match.start()].rstrip().endswith("function")
    ]
    if not uses:
        problems.append(f"{name}() is never called")
    for use in uses:
        if not re.match(r"[\"'`]/api/", source[use.end() :].lstrip()):
            problems.append(f"{name}() is called with something other than a literal /api/ path")
    return problems


def test_every_request_goes_through_one_helper_that_sends_the_app_header():
    for label, sample in REQUEST_SAMPLES.items():
        assert _request_problems(sample), f"the scan misses {label}"
    assert _request_problems(GOOD_REQUEST_SAMPLE) == []

    assert _request_problems(_all_js()) == []


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
    source = _all_js()
    table = re.search(r"const ABSTAIN_SENTENCES = Object\.freeze\(\{(.*?)\}\);", source, re.S)
    assert table, "no script has an ABSTAIN_SENTENCES table"
    entries = re.findall(r"^\s*([a-z_]+)\s*:\s*\"([^\"]*)\",?\s*$", table.group(1), re.M)
    assert {reason: _nfc(text) for reason, text in entries} == {
        reason: _nfc(text) for reason, text in ABSTAIN_SENTENCES.items()
    }

    fallback = re.search(r"const ABSTAIN_FALLBACK = \"([^\"]+)\";", source)
    assert fallback and ARABIC_LETTER.search(fallback.group(1)), "no Arabic sentence for an unknown reason"
    # a reason named like an inherited property ("constructor") must not index the prototype
    assert "Object.hasOwn(ABSTAIN_SENTENCES, " in source


# --- nothing about the documents or the conversation outlives the tab -------

BROWSER_STORAGE = re.compile(
    r"\blocalStorage\b|\bsessionStorage\b|\bindexedDB\b|\bdocument\s*\.\s*cookie\b|\bcaches\s*\."
)

STORAGE_SAMPLES = [
    "localStorage.setItem('thread', json);",
    "window.sessionStorage.last = question;",
    "indexedDB.open('legalrag');",
    "document.cookie = 'scope=' + ids;",
    "caches.open('answers');",
]


def test_the_chat_page_keeps_nothing_in_browser_storage():
    """The documents are private; the thread lives in memory and dies with the tab."""
    for sample in STORAGE_SAMPLES:
        assert BROWSER_STORAGE.search(sample), f"the scan misses {sample!r}"

    assert not BROWSER_STORAGE.search(_all_js())


# --- every script the page imports is one the server serves ------------------

IMPORT_SPECIFIER = re.compile(
    r"^\s*(?:import|export)\b[^;'\"`]*?\bfrom\s*(['\"])(?P<spec>[^'\"]+)\1"
    r"|^\s*import\s*(['\"])(?P<bare>[^'\"]+)\3",
    re.M,
)

IMPORT_SAMPLES = {
    "a bare specifier": {"app.js": 'import { html } from "lit";'},
    "a URL": {"app.js": 'import confetti from "https://cdn.example/confetti.js";'},
    "a module the server does not serve": {"app.js": 'import { x } from "./js/extra.js";'},
    "a path that climbs out of ui/": {"app.js": 'import "../src/legalrag/secret.js";'},
    "a multi-line import of a missing module": {"app.js": 'import {\n  a,\n  b,\n} from "./js/gone.js";'},
    "a re-export of a missing module": {"app.js": 'export { a } from "./js/gone.js";'},
    "a module nothing imports": {
        "app.js": 'import { a } from "./js/a.js";',
        "js/a.js": "export const a = 1;",
        "js/b.js": "export const b = 2;",
    },
}

GOOD_IMPORT_SAMPLE = {
    "app.js": 'import { api } from "./js/api.js";\nimport {\n  TEXT,\n} from "./js/copy.js";',
    "js/api.js": 'import { TEXT } from "./copy.js";\nexport async function api() {}',
    "js/copy.js": 'export const TEXT = Object.freeze({ from: "a key, not an import" });',
}


def _import_problems(files: dict[str, str]) -> list[str]:
    """Imports that are not relative or resolve outside `files`, and files app.js never reaches."""
    problems: list[str] = []
    imports: dict[str, set[str]] = {name: set() for name in files}
    for name, source in files.items():
        for match in IMPORT_SPECIFIER.finditer(source):
            spec = match.group("spec") or match.group("bare")
            if not spec.startswith(("./", "../")):
                problems.append(f"{name} imports {spec!r}, which is not a relative path")
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), spec))
            if target in files:
                imports[name].add(target)
            else:
                problems.append(f"{name} imports {spec!r}, which resolves to {target!r}, not an allowlisted file")
    reached, pending = {"app.js"}, ["app.js"]
    while pending:
        for target in imports.get(pending.pop(), set()) - reached:
            reached.add(target)
            pending.append(target)
    return problems + [f"{name} is never imported from app.js" for name in files if name not in reached]


def test_every_relative_import_resolves_to_an_allowlisted_file():
    for label, files in IMPORT_SAMPLES.items():
        assert _import_problems(files), f"the scan misses {label}"
    assert _import_problems(GOOD_IMPORT_SAMPLE) == []

    on_disk = sorted(path.relative_to(UI).as_posix() for path in UI.rglob("*.js"))
    assert on_disk == sorted(ALLOWLISTED_JS), "a script under ui/ that the server would not serve"
    assert _import_problems({name: _read(name) for name in ALLOWLISTED_JS}) == []

    scripts = [tag for tag in TAG.findall(_read("app.html")) if re.match(r"<script\b", tag, re.I)]
    assert [(_attr(tag, "type"), _attr(tag, "src")) for tag in scripts] == [("module", "app.js")]


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

    assert _count_problems(_all_js(), _read("app.html")) == []


def test_the_count_forms_follow_arabic_number_agreement():
    copy = _read("js/copy.js")
    assert 'new Intl.PluralRules("ar")' in copy
    assert "function countForms(one, two, few, many, other)" in copy
    table = COUNT_TABLE.search(copy)
    assert table, "js/copy.js has no COUNT_FORMS table"
    rows = re.findall(r'^\s*(\w+): countForms\(((?:"[^"]+"(?:, )?){5})\),$', table.group(0), re.M)
    parsed = {noun: tuple(_nfc(form) for form in re.findall(r'"([^"]+)"', forms)) for noun, forms in rows}
    assert parsed == {noun: tuple(_nfc(form) for form in forms) for noun, forms in COUNT_FORMS.items()}


# --- the limits the server enforces ---------------------------------------------


def test_the_limits_are_stated_as_the_server_enforces_them():
    copy = _read("js/copy.js")
    assert "export const MAX_DOC_IDS = 20;" in copy
    assert "export const MAX_UPLOAD_MB = 20;" in copy
    # a file just over the limit is told the limit, never its own size rounded to read as the limit
    assert "tooLarge: `الحد الأقصى لحجم الملف ${MAX_UPLOAD_MB} ميجابايت.`," in copy
    assert "formatSize(file.size)" not in _all_js()
    assert 'يمكن اختيار ${formatCount(MAX_DOC_IDS, "document")} كحد أقصى، أو اختيار الكل.' in copy
