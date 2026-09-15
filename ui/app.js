/*
 * Chat page of the arabic-legal-rag demo layer (ADR-023): upload an Arabic PDF
 * or TXT document, ask about it, and read every displayed sentence beside the
 * chunk it cites.
 *
 * The entry module: the element map, the in-memory state, and the event
 * wiring. The work lives in ./js/, and the server serves exactly these six
 * scripts. Rules the modules keep, pinned by tests/test_ui_security.py:
 * - server text reaches the page only as textContent or attribute values;
 * - every request goes through one helper, which sends the X-LegalRAG header;
 * - nothing is written to browser storage: the documents are private, and the
 *   conversation lives only as long as the tab.
 */
import { WIDE_LAYOUT } from "./js/copy.js";
import { byId, createAnnouncer } from "./js/dom.js";
import {
  guardStrayDrop,
  onFileChosen,
  refreshDocuments,
  refreshHealth,
  renderDocuments,
  retryDocuments,
  syncFilePicker,
  upload,
} from "./js/documents.js";
import {
  applySourceMode,
  ask,
  clearThread,
  closeSource,
  onDocumentKeydown,
  onQuestionKeydown,
  updateAskState,
} from "./js/chat.js";

const ui = Object.freeze({
  skipLink: byId("skip-link"),
  header: byId("app-header"),
  footer: byId("app-footer"),
  layout: byId("layout"),
  health: byId("health"),
  healthText: byId("health-text"),
  docsPanel: byId("docs-panel"),
  docsCount: byId("docs-count"),
  uploadForm: byId("upload-form"),
  filePicker: byId("file-picker"),
  fileInput: byId("file-input"),
  fileName: byId("file-name"),
  fileError: byId("file-error"),
  uploadButton: byId("upload-btn"),
  docsCaption: byId("docs-caption"),
  docList: byId("doc-list"),
  docsEmpty: byId("docs-empty"),
  docsError: byId("docs-error"),
  docsErrorText: byId("docs-error-text"),
  docsRetry: byId("docs-retry"),
  chatPanel: byId("chat-panel"),
  newChat: byId("new-chat"),
  newChatNote: byId("new-chat-note"),
  threadScroll: byId("thread-scroll"),
  threadIntro: byId("thread-intro"),
  thread: byId("thread"),
  askForm: byId("ask-form"),
  question: byId("question"),
  questionError: byId("question-error"),
  scopeNote: byId("scope-note"),
  counter: byId("question-count"),
  askButton: byId("ask-btn"),
  sourcePanel: byId("source-panel"),
  sourceTitle: byId("source-title"),
  sourceLabel: byId("source-label"),
  sourcePageRow: byId("source-page-row"),
  sourcePage: byId("source-page"),
  sourceText: byId("source-text"),
  sourceClose: byId("source-close"),
  scrim: byId("scrim"),
  toast: byId("toast"),
});

// In memory only: nothing about the documents or the conversation outlives the tab.
const state = {
  documents: [], // frozen documents, newest first
  excluded: new Set(), // doc_ids unticked out of the scope; a new document starts ticked
  deleting: new Set(), // doc_ids whose delete request is in flight
  documentsRequest: 0, // numbers each documents refresh, so a late reply cannot overwrite a newer one
  docsStatus: "loading", // "loading" | "ready" | "error"
  uploading: false,
  pendingRow: null,
  stopPendingClock: null,
  asking: false, // a question is in flight
  sourceOpener: null, // the citation chip whose source is open
};

const ctx = Object.freeze({
  ui,
  state,
  announce: createAnnouncer(ui.toast),
  wideLayout: window.matchMedia(WIDE_LAYOUT),
  changed: () => updateAskState(ctx),
});

function init() {
  ui.uploadForm.addEventListener("submit", (event) => upload(ctx, event));
  ui.fileInput.addEventListener("change", () => onFileChosen(ctx));
  ui.docsRetry.addEventListener("click", () => retryDocuments(ctx));
  ui.askForm.addEventListener("submit", (event) => ask(ctx, event));
  ui.question.addEventListener("input", () => updateAskState(ctx));
  ui.question.addEventListener("keydown", (event) => onQuestionKeydown(ctx, event));
  ui.newChat.addEventListener("click", () => clearThread(ctx));
  ui.sourceClose.addEventListener("click", () => closeSource(ctx));
  ui.scrim.addEventListener("click", () => closeSource(ctx));
  document.addEventListener("keydown", (event) => onDocumentKeydown(ctx, event));
  document.addEventListener("dragover", (event) => guardStrayDrop(ctx, event));
  document.addEventListener("drop", (event) => guardStrayDrop(ctx, event));
  ctx.wideLayout.addEventListener("change", () => applySourceMode(ctx));

  syncFilePicker(ctx);
  renderDocuments(ctx);
  refreshHealth(ctx);
  refreshDocuments(ctx);
}

init();
