"""Behaviour checks on the chat page's scripts, run under Node instead of a browser.

A test imports one module from a copy of ui/, stubs the few DOM names the module
meets, and drives it with plain objects in place of events. The tests need Node
on PATH, and skip without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui_page import copy_scripts_for_node, needs_node, run_node  # noqa: E402

# --- a stray file drop never opens in the tab --------------------------------------


@needs_node
def test_only_a_stray_file_drop_is_cancelled_and_an_idle_picker_still_takes_one(tmp_path):
    """documents.js's drop guard, run under Node with the two DOM names it meets stubbed.

    A file dropped outside the picker would open in the tab and wipe the memory-only
    conversation. A text drag, into the question box say, carries no file and is left
    alone. The picker takes a file only while its input is enabled: mid-upload the
    input is disabled, Chromium refuses the drop there, and the file would open.
    """
    documents_url = (copy_scripts_for_node(tmp_path) / "js" / "documents.js").as_uri()
    script = (
        "globalThis.window = { matchMedia: () => ({ matches: false }) };\n"
        "globalThis.Node = class {};\n"
        f"const {{ guardStrayDrop }} = await import({json.dumps(documents_url)});\n"
        "const element = () => Object.create(Node.prototype);\n"
        "const picker = element(), input = element(), textarea = element(), body = element();\n"
        "const inPicker = new Set([picker, input]);\n"
        "const ui = (inputDisabled) => ({\n"
        "  filePicker: { contains: (node) => inPicker.has(node) }, fileInput: { disabled: inputDisabled } });\n"
        "const drag = (type, target, dataTransfer, inputDisabled = false) => {\n"
        "  const event = { type, target, dataTransfer, defaultPrevented: false,\n"
        "    preventDefault() { this.defaultPrevented = true; } };\n"
        "  guardStrayDrop({ ui: ui(inputDisabled) }, event);\n"
        "  return [event.defaultPrevented, dataTransfer === null ? null : dataTransfer.dropEffect];\n"
        "};\n"
        "const text = () => ({ types: ['text/plain'], dropEffect: 'copy' });\n"
        "const file = () => ({ types: ['Files'], dropEffect: 'copy' });\n"
        "console.log(JSON.stringify({\n"
        "  textOverQuestion: drag('dragover', textarea, text()),\n"
        "  textIntoQuestion: drag('drop', textarea, text()),\n"
        "  fileOverPage: drag('dragover', body, file()),\n"
        "  fileOnPage: drag('drop', body, file()),\n"
        "  fileOverIdlePicker: drag('dragover', picker, file()),\n"
        "  fileOnIdlePicker: drag('drop', input, file()),\n"
        "  fileOverPickerMidUpload: drag('dragover', input, file(), true),\n"
        "  fileOnPickerMidUpload: drag('drop', input, file(), true),\n"
        "  noTransfer: drag('drop', body, null),\n"
        "}));\n"
    )
    result = run_node("--input-type=module", stdin=script)
    assert result.returncode == 0, result.stderr
    # [cancelled, dropEffect]: a cancelled dragover also shows the no-drop cursor
    assert json.loads(result.stdout) == {
        "textOverQuestion": [False, "copy"],
        "textIntoQuestion": [False, "copy"],
        "fileOverPage": [True, "none"],
        "fileOnPage": [True, "copy"],
        "fileOverIdlePicker": [False, "copy"],
        "fileOnIdlePicker": [False, "copy"],
        "fileOverPickerMidUpload": [True, "none"],
        "fileOnPickerMidUpload": [True, "copy"],
        "noTransfer": [False, None],
    }
