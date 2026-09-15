"""Behaviour checks on the chat page's scripts, run under Node instead of a browser.

A test imports functions from one module in a copy of ui/js/, stubs the few DOM
names the module meets, and drives them with plain objects in place of elements
and events. The tests need Node on PATH, and skip without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui_page import TAG, attr, copy_scripts_for_node, needs_node, read, run_node  # noqa: E402

# All the modules read of the window: dom.js asks matchMedia about reduced motion when it
# loads, and chat.js reads the viewport height, here a 500×844 phone's.
PRELUDE = "globalThis.window = { matchMedia: () => ({ matches: false }), innerHeight: 844 };\n"

# A stand-in element with the properties and attribute calls the page's functions use.
STUB = """
const stub = (props = {}) => {
  const attributes = new Map();
  return {
    hidden: false, disabled: false, textContent: "", value: "", dataset: {},
    classList: { toggle() {} },
    setAttribute(name, value) { attributes.set(name, String(value)); },
    getAttribute(name) { return attributes.has(name) ? attributes.get(name) : null; },
    removeAttribute(name) { attributes.delete(name); },
    ...props,
  };
};
"""

# The chat panel's elements and state, with a thread of `exchanges` and a log of what the page did.
CHAT_CONTEXT = STUB + """
const chatContext = ({ asking, exchanges }) => {
  const log = [];
  const ui = {
    question: stub({ focus: () => log.push("focus the question") }),
    questionError: stub({ hidden: true }),
    scopeNote: stub(), counter: stub(), askButton: stub(),
    newChat: stub(), newChatNote: stub({ hidden: true }),
    thread: stub({
      childElementCount: exchanges,
      replaceChildren() { this.childElementCount = 0; log.push("empty the thread"); },
    }),
    threadIntro: stub({ hidden: exchanges > 0 }),
    sourcePanel: stub({ hidden: true }),
  };
  const state = { asking, documents: [{ id: "a1" }], excluded: new Set(), docsStatus: "ready" };
  return { ui, state, log, announce: (message) => log.push(`announce: ${message}`) };
};
"""


def _run_module(tmp_path: Path, module: str, names: str, body: str):
    """Imports `names` from ui/js/`module`, runs `body`, and hands back the JSON it logs."""
    url = (copy_scripts_for_node(tmp_path) / "js" / module).as_uri()
    script = PRELUDE + f"const {{ {names} }} = await import({json.dumps(url)});\n" + body
    result = run_node("--input-type=module", stdin=script)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# --- a stray file drop never opens in the tab --------------------------------------


@needs_node
def test_only_a_stray_file_drop_is_cancelled_and_an_idle_picker_still_takes_one(tmp_path):
    """documents.js's drop guard, run under Node with the two DOM names it meets stubbed.

    A file dropped outside the picker would open in the tab and wipe the memory-only
    conversation. A text drag, into the question box say, carries no file and is left
    alone. The picker takes a file only while its input is enabled: mid-upload the
    input is disabled, Chromium refuses the drop there, and the file would open.
    """
    body = (
        "globalThis.Node = class {};\n"
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
    # [cancelled, dropEffect]: a cancelled dragover also shows the no-drop cursor
    assert _run_module(tmp_path, "documents.js", "guardStrayDrop", body) == {
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


# --- a button that is unavailable for a while keeps keyboard focus -------------------


@needs_node
def test_mid_upload_the_upload_button_keeps_focus_and_only_says_it_is_unavailable(tmp_path):
    """syncFilePicker. A button disabled under the keyboard's focus drops it to the page;
    mid-upload the button stays enabled, says aria-disabled, and upload() ignores the press."""
    body = STUB + """
const button = (uploading, files) => {
  const ui = { fileInput: stub({ files }), fileName: stub(), filePicker: stub(), uploadButton: stub() };
  syncFilePicker({ ui, state: { uploading } });
  return [ui.uploadButton.disabled, ui.uploadButton.getAttribute("aria-disabled")];
};
const pdf = { name: "law.pdf", size: 2048 }, image = { name: "scan.png", size: 2048 };
console.log(JSON.stringify({
  noFile: button(false, []),
  goodFile: button(false, [pdf]),
  badFile: button(false, [image]),
  midUpload: button(true, [pdf]),
}));
"""
    # [disabled, aria-disabled]
    assert _run_module(tmp_path, "documents.js", "syncFilePicker", body) == {
        "noFile": [True, "false"],
        "goodFile": [False, "false"],
        "badFile": [True, "false"],
        "midUpload": [False, "true"],
    }


# --- a new chat waits for the answer in flight ---------------------------------------


@needs_node
def test_new_chat_is_refused_while_an_answer_is_on_its_way(tmp_path):
    """clearThread. The server keeps generating an answer nobody would wait for, so while one is
    on its way a press on «محادثة جديدة» leaves the thread, the focus and the live region alone."""
    body = CHAT_CONTEXT + """
const clear = (asking) => {
  const ctx = chatContext({ asking, exchanges: 1 });
  clearThread(ctx);
  return { log: ctx.log, introShown: !ctx.ui.threadIntro.hidden };
};
console.log(JSON.stringify({ asking: clear(true), idle: clear(false) }));
"""
    out = _run_module(tmp_path, "chat.js", "clearThread", body)
    assert out["asking"] == {"log": [], "introShown": False}
    assert out["idle"] == {
        "log": ["empty the thread", "focus the question", "announce: بدأت محادثة جديدة."],
        "introShown": True,
    }


@needs_node
def test_while_an_answer_is_on_its_way_new_chat_keeps_focus_and_says_why_it_waits(tmp_path):
    """updateAskState. A title on a disabled button reaches neither keyboard, touch nor screen-reader
    users. So «محادثة جديدة» stays enabled and says aria-disabled, and a visible note, which the button
    names with aria-describedby, says why; clearThread ignores the press."""
    body = CHAT_CONTEXT + """
const newChat = (asking, exchanges) => {
  const ctx = chatContext({ asking, exchanges });
  updateAskState(ctx);
  const { newChat: button, newChatNote: note } = ctx.ui;
  return {
    disabled: button.disabled,
    ariaDisabled: button.getAttribute("aria-disabled"),
    note: note.hidden ? null : note.textContent,
  };
};
console.log(JSON.stringify({
  answerOnItsWay: newChat(true, 1),
  threadIdle: newChat(false, 1),
  threadEmpty: newChat(false, 0),
}));
"""
    assert _run_module(tmp_path, "chat.js", "updateAskState", body) == {
        "answerOnItsWay": {"disabled": False, "ariaDisabled": "true", "note": "انتظر انتهاء الإجابة الجارية"},
        "threadIdle": {"disabled": False, "ariaDisabled": "false", "note": None},
        "threadEmpty": {"disabled": True, "ariaDisabled": "false", "note": None},
    }
    button = next(tag for tag in TAG.findall(read("app.html")) if attr(tag, "id") == "new-chat")
    assert attr(button, "aria-describedby") == "new-chat-note"


# --- an answer follows only a reader who is watching for it --------------------------


@needs_node
def test_an_answer_follows_the_reader_only_while_they_can_see_its_loading_card(tmp_path):
    """showAnswer, with plain rectangles in place of layout. On a 500×844 phone the page scrolls and
    the composer sticks over the bottom of the viewport: a loading card behind it is not being
    watched, and neither is one the reader scrolled away from."""
    body = STUB + """
const at = (top, bottom) => () => ({ top, bottom });
const follow = ({ card, frame = [0, 844], composer }) => {
  const scrolled = [];
  const exchange = {
    root: { scrollIntoView: ({ block }) => scrolled.push(block) },
    answer: stub({ getBoundingClientRect: at(...card), replaceChildren() {} }),
  };
  const ui = {
    threadScroll: { getBoundingClientRect: at(...frame) },
    askForm: { getBoundingClientRect: at(...composer) },
    chatPanel: { inert: false },
  };
  showAnswer({ ui, announce: () => {} }, exchange, "answered", []);
  return scrolled;
};
console.log(JSON.stringify({
  watchingTheCard: follow({ card: [420, 580], composer: [614, 844] }),
  cardPeeksAboveTheComposer: follow({ card: [560, 720], composer: [614, 844] }),
  cardBehindTheComposer: follow({ card: [640, 800], composer: [614, 844] }),
  readerScrolledUp: follow({ card: [900, 1060], composer: [614, 844] }),
  readerScrolledPast: follow({ card: [-260, -100], composer: [614, 844] }),
  desktopWatching: follow({ card: [500, 600], frame: [120, 640], composer: [640, 820] }),
  desktopCardBelowThePanel: follow({ card: [660, 760], frame: [120, 640], composer: [640, 820] }),
}));
"""
    # the block each answer was scrolled to: "start" follows it, none leaves the reader where they are
    assert _run_module(tmp_path, "chat.js", "showAnswer", body) == {
        "watchingTheCard": ["start"],
        "cardPeeksAboveTheComposer": ["start"],
        "cardBehindTheComposer": [],
        "readerScrolledUp": [],
        "readerScrolledPast": [],
        "desktopWatching": ["start"],
        "desktopCardBelowThePanel": [],
    }
