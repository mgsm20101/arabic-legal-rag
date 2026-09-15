/*
 * The documents panel: the health chip, the documents list and the question
 * scope, upload, and delete. Every function takes the page context that app.js
 * builds: { ui, state, announce, wideLayout, changed }.
 */
import { ApiError, api, isObject, messageOf, textOf, wholeNumber } from "./api.js";
import {
  MAX_DOC_IDS,
  MAX_UPLOAD_BYTES,
  SKELETON_ROWS,
  TEXT,
  UPLOAD_EXTENSION,
  formatDate,
  formatSize,
} from "./copy.js";
import { el, fromTemplate, setBusy, setText, startClock } from "./dom.js";

// --- health ------------------------------------------------------------------

export async function refreshHealth({ ui }) {
  try {
    renderHealth(ui, await api("/api/health"));
  } catch (error) {
    if (!(error instanceof ApiError)) console.error(error);
    ui.health.dataset.state = "unknown";
    ui.healthText.replaceChildren(TEXT.healthUnknown);
  }
}

function renderHealth(ui, health) {
  const generator = isObject(health.generator) ? health.generator : {};
  if (generator.reachable !== true) {
    ui.health.dataset.state = "down";
    ui.healthText.replaceChildren(TEXT.modelDown);
    return;
  }
  const model = textOf(generator.model, "—").replace(/^ollama:/, "");
  const parts = [TEXT.modelPrefix, el("bdi", "", model)];
  const share = generator.gpu_share;
  if (typeof share === "number" && Number.isFinite(share)) {
    const percent = Math.round(Math.min(Math.max(share, 0), 1) * 100);
    parts.push(" · ", el("bdi", "", `GPU ${percent}%`));
  }
  ui.health.dataset.state = "ok";
  ui.healthText.replaceChildren(...parts);
}

// --- documents ---------------------------------------------------------------

function toDocument(raw) {
  if (!isObject(raw) || typeof raw.doc_id !== "string" || raw.doc_id === "") return null;
  return Object.freeze({
    id: raw.doc_id,
    title: textOf(raw.title, TEXT.untitled),
    statute: raw.kind === "statute",
    chunks: wholeNumber(raw.chunks) ?? 0,
    pages: wholeNumber(raw.pages),
    sizeBytes: wholeNumber(raw.size_bytes),
    createdAt: typeof raw.created_at === "string" ? raw.created_at : "",
  });
}

export async function refreshDocuments(ctx) {
  const { state } = ctx;
  state.documentsRequest += 1;
  const request = state.documentsRequest;
  try {
    const data = await api("/api/documents");
    if (request !== state.documentsRequest) return; // a newer refresh went out; its reply wins
    const documents = (Array.isArray(data.documents) ? data.documents : [])
      .map((raw) => toDocument(raw))
      .filter((doc) => doc !== null)
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    const ids = new Set(documents.map((doc) => doc.id));
    state.documents = documents;
    state.excluded = new Set([...state.excluded].filter((id) => ids.has(id)));
    state.docsStatus = "ready";
  } catch (error) {
    const message = messageOf(error);
    if (request !== state.documentsRequest) return;
    if (state.docsStatus === "ready") {
      ctx.announce(message, { tone: "error" });
    } else {
      state.docsStatus = "error";
      ctx.ui.docsErrorText.textContent = `${TEXT.docsLoadError} ${message}`;
    }
  }
  renderDocuments(ctx);
}

export function retryDocuments(ctx) {
  ctx.state.docsStatus = "loading";
  renderDocuments(ctx);
  refreshDocuments(ctx);
}

export function renderDocuments(ctx) {
  const { ui, state } = ctx;
  const { documents, docsStatus, pendingRow } = state;
  const ready = docsStatus === "ready";
  const pending = pendingRow === null ? [] : [pendingRow];
  const skeletons =
    docsStatus === "loading" ? Array.from({ length: SKELETON_ROWS }, () => fromTemplate("tpl-doc-skeleton")) : [];
  const rows = ready ? documents.map((doc) => documentRow(ctx, doc)) : [];
  ui.docList.replaceChildren(...pending, ...skeletons, ...rows);
  ui.docList.setAttribute("aria-busy", String(docsStatus === "loading"));

  ui.docsError.hidden = docsStatus !== "error";
  ui.docsEmpty.hidden = !ready || documents.length > 0 || pendingRow !== null;
  ui.docsCaption.hidden = !ready || documents.length === 0;
  ui.docsCount.hidden = !ready || documents.length === 0;
  ui.docsCount.textContent = String(documents.length);
  ctx.changed();
}

let rowSerial = 0;

function documentRow(ctx, doc) {
  rowSerial += 1;
  const row = fromTemplate("tpl-doc");
  const check = row.querySelector(".doc-check");
  const title = row.querySelector(".doc-title");
  const meta = row.querySelector(".doc-meta");
  const date = row.querySelector(".doc-date");
  const remove = row.querySelector(".doc-delete");

  check.id = `doc-check-${rowSerial}`;
  meta.id = `doc-meta-${rowSerial}`;
  check.checked = !ctx.state.excluded.has(doc.id);
  check.setAttribute("aria-describedby", meta.id);
  check.addEventListener("change", () => setIncluded(ctx, doc.id, check.checked));
  title.htmlFor = check.id;
  title.querySelector(".doc-title-text").textContent = doc.title;

  setText(row.querySelector(".doc-kind"), doc.statute ? TEXT.statute(doc.chunks) : TEXT.generic(doc.chunks));
  setText(row.querySelector(".doc-pages"), doc.pages ? TEXT.pages(doc.pages) : "");
  setText(row.querySelector(".doc-size"), doc.sizeBytes === null ? "" : formatSize(doc.sizeBytes));
  setText(date, formatDate(doc.createdAt));
  if (!date.hidden) date.dateTime = doc.createdAt;

  // a list refresh must not hand an enabled button back to a row whose delete is in flight
  const deleting = ctx.state.deleting.has(doc.id);
  remove.disabled = deleting;
  setBusy(remove, deleting);
  remove.setAttribute("aria-label", TEXT.deleteLabel(doc.title));
  remove.addEventListener("click", () => deleteDocument(ctx, doc, remove));
  return row;
}

// --- scope -------------------------------------------------------------------

function setIncluded(ctx, id, included) {
  const excluded = new Set(ctx.state.excluded);
  if (included) excluded.delete(id);
  else excluded.add(id);
  ctx.state.excluded = excluded;
  ctx.changed();
}

function selectedIds(state) {
  return state.documents.filter((doc) => !state.excluded.has(doc.id)).map((doc) => doc.id);
}

/** The scope note under the question, and whether a question may be asked in that scope. */
export function scopeStatus(state) {
  const total = state.documents.length;
  const selected = selectedIds(state).length;
  if (state.docsStatus === "loading") return { note: TEXT.scopeLoading, tone: "muted", askable: false };
  if (state.docsStatus === "error") return { note: TEXT.docsLoadError, tone: "warn", askable: false };
  if (total === 0) return { note: TEXT.scopeNoDocuments, tone: "muted", askable: false };
  if (selected === 0) return { note: TEXT.scopeNone, tone: "warn", askable: false };
  if (selected === total) return { note: TEXT.scopeAll(total), tone: "muted", askable: true };
  if (selected > MAX_DOC_IDS) return { note: TEXT.scopeTooMany, tone: "warn", askable: false };
  return { note: TEXT.scopeSome(selected, total), tone: "muted", askable: true };
}

/** null asks about every document, as the API contract defines; otherwise the ticked ids. */
export function scopeForRequest(state) {
  const selected = selectedIds(state);
  return selected.length === state.documents.length ? null : selected;
}

// --- upload ------------------------------------------------------------------

function selectedFile(ui) {
  const { files } = ui.fileInput;
  return files !== null && files.length > 0 ? files[0] : null;
}

function fileProblem(file) {
  if (file === null) return TEXT.noFile;
  if (!UPLOAD_EXTENSION.test(file.name)) return TEXT.badExtension;
  if (file.size === 0) return TEXT.emptyFile;
  if (file.size > MAX_UPLOAD_BYTES) return TEXT.tooLarge;
  return null;
}

function showFileError(ui, message) {
  ui.fileError.textContent = message ?? "";
  ui.fileError.hidden = message === null;
  ui.filePicker.classList.toggle("is-invalid", message !== null);
  if (message === null) ui.fileInput.removeAttribute("aria-invalid");
  else ui.fileInput.setAttribute("aria-invalid", "true");
}

/** Mirrors the chosen file in the picker and the button; leaves a shown error alone. */
export function syncFilePicker({ ui, state }) {
  const file = selectedFile(ui);
  ui.fileName.textContent = file === null ? TEXT.pickFile : file.name;
  ui.filePicker.classList.toggle("is-chosen", file !== null);
  // Mid-upload the button stays focusable and only says it is unavailable; upload() ignores it.
  ui.uploadButton.setAttribute("aria-disabled", String(state.uploading));
  ui.uploadButton.disabled = !state.uploading && (file === null || fileProblem(file) !== null);
  return file;
}

export function onFileChosen(ctx) {
  const file = syncFilePicker(ctx);
  showFileError(ctx.ui, file === null ? null : fileProblem(file));
}

export async function upload(ctx, event) {
  event.preventDefault();
  const { ui, state } = ctx;
  if (state.uploading) return;
  const file = selectedFile(ui);
  const problem = fileProblem(file);
  if (problem !== null) {
    showFileError(ui, problem);
    ui.fileInput.focus();
    return;
  }

  state.uploading = true;
  ui.fileInput.disabled = true;
  syncFilePicker(ctx);
  setBusy(ui.uploadButton, true);
  showFileError(ui, null);
  showPendingRow(ctx, file.name);
  ctx.announce(TEXT.uploading, { visible: false });

  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const created = toDocument(await api("/api/documents", { method: "POST", form }));
    ui.uploadForm.reset();
    await refreshDocuments(ctx);
    ctx.announce(TEXT.uploaded(created === null ? file.name : created.title), { tone: "success" });
  } catch (error) {
    const message = messageOf(error);
    showFileError(ui, message);
    ctx.announce(message, { tone: "error", visible: false });
  } finally {
    clearPendingRow(state);
    state.uploading = false;
    ui.fileInput.disabled = false;
    setBusy(ui.uploadButton, false);
    const buttonHadFocus = document.activeElement === ui.uploadButton;
    syncFilePicker(ctx);
    renderDocuments(ctx);
    // A successful upload empties the picker, which disables the button. Browsers drop focus from a
    // disabled element only at the next render, so move it to the picker now rather than lose it.
    if ((buttonHadFocus && ui.uploadButton.disabled) || document.activeElement === document.body) {
      ui.fileInput.focus();
    }
  }
  refreshHealth(ctx);
}

function showPendingRow({ ui, state }, fileName) {
  const row = fromTemplate("tpl-doc-pending");
  row.querySelector(".pending-name").textContent = fileName;
  state.stopPendingClock = startClock(row.querySelector(".elapsed"));
  state.pendingRow = row;
  ui.docsEmpty.hidden = true;
  ui.docList.prepend(row);
}

function clearPendingRow(state) {
  state.stopPendingClock?.();
  state.pendingRow?.remove();
  state.pendingRow = null;
  state.stopPendingClock = null;
}

/**
 * A file dropped anywhere but an idle picker would open in the tab and wipe the conversation.
 * Only file drags are touched: text dragged into the question box is the browser's to handle.
 */
export function guardStrayDrop({ ui }, event) {
  if (!event.dataTransfer?.types.includes("Files")) return;
  // mid-upload the input is disabled, Chromium refuses a drop on it, and the file would open instead
  if (!ui.fileInput.disabled && ui.filePicker.contains(event.target)) return;
  event.preventDefault();
  if (event.type === "dragover") event.dataTransfer.dropEffect = "none";
}

// --- delete ------------------------------------------------------------------

function withoutId(ids, id) {
  return new Set([...ids].filter((item) => item !== id));
}

async function deleteDocument(ctx, doc, button) {
  const { state } = ctx;
  if (state.deleting.has(doc.id) || !window.confirm(TEXT.confirmDelete(doc.title))) return;
  const position = state.documents.findIndex((item) => item.id === doc.id);
  state.deleting = new Set([...state.deleting, doc.id]);
  button.disabled = true;
  setBusy(button, true);
  try {
    await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
  } catch (error) {
    state.deleting = withoutId(state.deleting, doc.id);
    if (button.isConnected) {
      button.disabled = false;
      setBusy(button, false);
    } else {
      renderDocuments(ctx);
    }
    ctx.announce(messageOf(error), { tone: "error" });
    return;
  }
  ctx.announce(TEXT.deleted(doc.title), { tone: "success" });
  // The row and its busy state go now: a refresh that fails below must not leave them behind.
  state.documents = state.documents.filter((item) => item.id !== doc.id);
  state.deleting = withoutId(state.deleting, doc.id);
  renderDocuments(ctx);
  await refreshDocuments(ctx);
  focusDocumentAt(ctx.ui, position);
  refreshHealth(ctx);
}

/** Once a row is gone, keyboard focus moves to its neighbour, or to the picker. */
function focusDocumentAt(ui, position) {
  const checks = ui.docList.querySelectorAll(".doc-check");
  const target = position >= 0 ? checks[Math.min(position, checks.length - 1)] : undefined;
  (target ?? ui.fileInput).focus();
}
