"""Structure checks on the chat page: ui/app.html, ui/app.css, and ui/app.js with its modules.

No browser and no build step, so nothing else notices when an edit breaks how
the files fit together, until the page is opened and fails. Each test pins one
link:

- every script the page imports is one the server serves, and app.js reaches
  every script the server serves;
- every name a script imports is exported by its module;
- every script parses as an ES module;
- every id, template, icon and class a script reaches for is in the markup;
- the wide layout starts at the same width in the script and the stylesheet.

Every scan first runs on known-bad samples, so a pattern that has quietly
stopped matching cannot pass for a clean page. The parse check needs Node on
PATH, and skips without it.
"""

from __future__ import annotations

import posixpath
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui_page import (  # noqa: E402
    ALLOWLISTED_JS,
    TAG,
    UI,
    all_js,
    attr,
    copy_scripts_for_node,
    needs_node,
    read,
    run_node,
)

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
    assert _import_problems({name: read(name) for name in ALLOWLISTED_JS}) == []

    scripts = [tag for tag in TAG.findall(read("app.html")) if re.match(r"<script\b", tag, re.I)]
    assert [(attr(tag, "type"), attr(tag, "src")) for tag in scripts] == [("module", "app.js")]


# --- every name a script imports is exported by its module --------------------

NAMED_IMPORT = re.compile(r"^\s*import\s*\{([^}]*)\}\s*from\s*(['\"])([^'\"]+)\2", re.M)
EXPORTED_NAME = re.compile(r"^export\s+(?:async\s+)?(?:function\*?|class|const|let)\s+([A-Za-z_$][\w$]*)", re.M)

EXPORT_SAMPLES = {
    "a name the module never exported": {
        "app.js": 'import { refresh } from "./js/api.js";',
        "js/api.js": "export async function api() {}",
    },
    "a renamed export": {
        "app.js": 'import {\n  api,\n  isObject,\n} from "./js/api.js";',
        "js/api.js": "export async function api() {}\nexport const isPlainObject = () => true;",
    },
}

GOOD_EXPORT_SAMPLE = {
    "app.js": 'import { api, ApiError as Failure } from "./js/api.js";',
    "js/api.js": "export class ApiError extends Error {}\nexport async function api() {}",
}


def _missing_exports(files: dict[str, str]) -> list[str]:
    """Named imports that their module does not export: the page would die at link time."""
    exports = {name: set(EXPORTED_NAME.findall(source)) for name, source in files.items()}
    problems = []
    for name, source in files.items():
        for names, _quote, spec in NAMED_IMPORT.findall(source):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), spec))
            for imported in (part.split(" as ")[0].strip() for part in names.split(",")):
                if imported and target in exports and imported not in exports[target]:
                    problems.append(f"{name} imports {imported!r}, which {target} does not export")
    return problems


def test_every_imported_name_is_exported_by_its_module():
    for label, files in EXPORT_SAMPLES.items():
        assert _missing_exports(files), f"the scan misses {label}"
    assert _missing_exports(GOOD_EXPORT_SAMPLE) == []

    assert _missing_exports({name: read(name) for name in ALLOWLISTED_JS}) == []


# --- every script parses as an ES module --------------------------------------


@needs_node
def test_every_script_parses_as_an_es_module(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    (broken / "bad.js").write_text('import { a } from "./a.js";\nexport const = a;\n', encoding="utf-8")
    assert run_node("--check", str(broken / "bad.js")).returncode != 0, "node --check misses a syntax error"

    scripts = copy_scripts_for_node(tmp_path / "ui")
    for name in ALLOWLISTED_JS:
        result = run_node("--check", str(scripts / name))
        assert result.returncode == 0, f"{name}: {result.stderr}"


# --- what the scripts reach for is in the markup -------------------------------

WIRING = {
    "id": re.compile(r"\bbyId\(\s*\"([^\"]+)\"\s*\)"),
    "template": re.compile(r"\bfromTemplate\(\s*\"([^\"]+)\"\s*\)"),
    "icon": re.compile(r"\bicon(?:\(\s*|:\s*)\"([^\"]+)\""),
    "class": re.compile(r"\bquerySelector(?:All)?\(\s*\"\.([\w-]+)\"\s*\)"),
}

WIRING_HTML_SAMPLE = (
    '<p id="health"></p><svg><symbol id="i-check"></symbol></svg>'
    '<template id="tpl-doc"><li class="doc"><input class="doc-check"></li></template>'
)

WIRING_SAMPLES = {
    "a renamed id": 'const chip = byId("health-chip");',
    "a missing template": 'const row = fromTemplate("tpl-exchange");',
    "a missing icon": 'node.append(icon("spark"));',
    "a status icon with no symbol": 'Object.freeze({ text: "x", icon: "abstain" });',
    "a class no template has": 'row.querySelector(".doc-delete");',
}

GOOD_WIRING_SAMPLE = 'byId("health"); fromTemplate("tpl-doc"); icon("check"); row.querySelector(".doc-check");'


def _markup_names(html: str) -> dict[str, set[str]]:
    tags = TAG.findall(html)
    ids_of = lambda element: {attr(t, "id") for t in tags if re.match(rf"<{element}\b", t, re.I)} - {None}  # noqa: E731
    return {
        "id": {attr(tag, "id") for tag in tags} - {None},
        "template": ids_of("template"),
        "icon": {symbol[2:] for symbol in ids_of("symbol") if symbol.startswith("i-")},
        "class": {name for tag in tags for name in (attr(tag, "class") or "").split()},
    }


def _wiring_problems(js: str, html: str) -> list[str]:
    markup = _markup_names(html)
    return [
        f"a script reaches for {kind} {name!r}, which app.html does not have"
        for kind, pattern in WIRING.items()
        for name in pattern.findall(js)
        if name not in markup[kind]
    ]


def test_every_id_template_icon_and_class_the_scripts_use_is_in_the_markup():
    for label, sample in WIRING_SAMPLES.items():
        assert _wiring_problems(sample, WIRING_HTML_SAMPLE), f"the scan misses {label}"
    assert _wiring_problems(GOOD_WIRING_SAMPLE, WIRING_HTML_SAMPLE) == []

    assert _wiring_problems(all_js(), read("app.html")) == []


# --- the wide layout starts at the same width in the script and the stylesheet ---


def test_the_wide_layout_breakpoint_is_the_same_in_the_script_and_the_stylesheet():
    match = re.search(r'WIDE_LAYOUT = "\(min-width: (\d+)px\)"', read("js/copy.js"))
    assert match, "js/copy.js has no WIDE_LAYOUT media query"
    width = int(match.group(1))
    css = re.sub(r"\s+", "", read("app.css"))
    assert f"@media(min-width:{width}px)" in css, "the source column starts at another width in app.css"
    assert f"@media(max-width:{width - 1}.98px)" in css, "the source drawer ends at another width in app.css"
