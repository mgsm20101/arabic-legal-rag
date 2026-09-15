/*
 * The chat panel: asking, the answer card with its citation chips, and the
 * source panel. Every function takes the page context that app.js builds:
 * { ui, state, announce, wideLayout, changed }.
 */
import { ApiError, api, isObject, messageOf, textOf, wholeNumber } from "./api.js";
import {
  COUNTER_WARNING_LENGTH,
  MAX_QUESTION_LENGTH,
  MIN_QUESTION_LENGTH,
  STATUS_TAGS,
  TEXT,
  abstainSentence,
  droppedNote,
  formatDuration,
} from "./copy.js";
import { el, fromTemplate, icon, reveal, setBusy, startClock } from "./dom.js";
import { refreshHealth, scopeForRequest, scopeStatus } from "./documents.js";

// Failures that can mean the model server went away, so the health chip is checked again.
const GENERATOR_TROUBLE = new Set([0, 503]);

// --- asking ------------------------------------------------------------------

export function updateAskState({ ui, state }) {
  const typed = ui.question.value.length;
  const length = ui.question.value.trim().length;
  const scope = scopeStatus(state);

  ui.scopeNote.textContent = scope.note;
  ui.scopeNote.dataset.tone = scope.tone;
  ui.counter.textContent = `${typed}/${MAX_QUESTION_LENGTH}`;
  ui.counter.dataset.tone = typed >= COUNTER_WARNING_LENGTH ? "warn" : "muted";
  if (length >= MIN_QUESTION_LENGTH) hideQuestionError(ui);

  ui.askButton.disabled = state.asking || !scope.askable || length < MIN_QUESTION_LENGTH;
  // The server keeps generating an answer nobody waits for, so a new chat waits for it instead.
  ui.newChat.disabled = state.asking || ui.thread.childElementCount === 0;
  if (state.asking) ui.newChat.title = TEXT.newChatBusy;
  else ui.newChat.removeAttribute("title");
}

function showQuestionError(ui) {
  ui.questionError.textContent = TEXT.questionTooShort;
  ui.questionError.hidden = false;
  ui.question.setAttribute("aria-invalid", "true");
}

function hideQuestionError(ui) {
  ui.questionError.textContent = "";
  ui.questionError.hidden = true;
  ui.question.removeAttribute("aria-invalid");
}

export async function ask(ctx, event) {
  event?.preventDefault();
  const { ui, state } = ctx;
  const question = ui.question.value.trim();
  if (question.length > 0 && question.length < MIN_QUESTION_LENGTH) {
    showQuestionError(ui);
    return;
  }
  if (ui.askButton.disabled || state.asking || !scopeStatus(state).askable) return;

  const docIds = scopeForRequest(state);
  const refocus = document.activeElement === ui.askButton;
  const exchange = addExchange(ui, question);
  state.asking = true;
  ui.question.value = "";
  setBusy(ui.askButton, true);
  updateAskState(ctx);
  if (refocus) ui.question.focus();

  try {
    const data = await api("/api/chat", { method: "POST", json: { question, doc_ids: docIds } });
    renderAnswer(ctx, exchange, data);
  } catch (error) {
    renderError(ctx, exchange, error);
    if (ui.question.value === "") ui.question.value = question;
    if (error instanceof ApiError && GENERATOR_TROUBLE.has(error.status)) refreshHealth(ctx);
  } finally {
    exchange.stopClock();
    state.asking = false;
    setBusy(ui.askButton, false);
    updateAskState(ctx);
  }
}

function addExchange(ui, question) {
  const root = fromTemplate("tpl-exchange");
  root.querySelector(".question-text").textContent = question;
  const answer = root.querySelector(".answer");
  const stopClock = startClock(answer.querySelector(".elapsed"));
  ui.threadIntro.hidden = true;
  ui.thread.append(root);
  reveal(root, "nearest");
  return Object.freeze({ root, answer, stopClock });
}

export function clearThread(ctx) {
  const { ui, state } = ctx;
  if (state.asking) return; // the button is disabled meanwhile; see updateAskState
  closeSource(ctx, { restoreFocus: false });
  ui.thread.replaceChildren();
  ui.threadIntro.hidden = false;
  hideQuestionError(ui);
  updateAskState(ctx);
  ui.question.focus();
  ctx.announce(TEXT.newChat);
}

// --- answers -----------------------------------------------------------------

function toSource(raw) {
  return Object.freeze({
    label: textOf(raw.label, TEXT.sourceLabel(raw.n)),
    docTitle: textOf(raw.doc_title, TEXT.untitled),
    page: Number.isInteger(raw.page) && raw.page > 0 ? raw.page : null,
    text: typeof raw.text === "string" ? raw.text : "",
  });
}

function toAnswer(data) {
  const sources = new Map(
    (Array.isArray(data.sources) ? data.sources : [])
      .filter((raw) => isObject(raw) && Number.isInteger(raw.n))
      .map((raw) => [raw.n, toSource(raw)]),
  );
  const claims = (Array.isArray(data.claims) ? data.claims : [])
    .filter((raw) => isObject(raw) && typeof raw.text === "string" && raw.text.trim() !== "")
    .map((raw) =>
      Object.freeze({
        text: raw.text.trim(),
        sources: [...new Set(Array.isArray(raw.sources) ? raw.sources : [])]
          .filter((n) => sources.has(n))
          .map((n) => sources.get(n)),
      }),
    );
  return Object.freeze({
    status: data.status,
    abstainReason: data.abstain_reason,
    claims,
    dropped: droppedCount(data.dropped),
    timings: isObject(data.timings_ms) ? data.timings_ms : null,
  });
}

function droppedCount(dropped) {
  if (!isObject(dropped)) return 0;
  return Object.values(dropped).reduce((sum, value) => sum + (wholeNumber(value) ?? 0), 0);
}

function renderAnswer(ctx, exchange, data) {
  const answer = toAnswer(data);
  const abstained = answer.status === "abstained";
  // A sentence reaches the page only with a source that resolves in this same reply.
  const shown = abstained ? [] : answer.claims.filter((claim) => claim.sources.length > 0);
  const unsourced = abstained ? 0 : answer.claims.length - shown.length;
  const status = shown.length === 0 ? "abstained" : answer.status === "partial" ? "partial" : "answered";

  const parts = [
    statusTag(status),
    shown.length > 0 ? claimList(ctx, shown) : el("p", "abstain-text", abstainSentence(answer.abstainReason)),
  ];
  const note = droppedNote(answer.dropped + unsourced, answer.abstainReason);
  if (note !== "") parts.push(el("p", "answer-note", note));
  const timings = timingsText(answer.timings);
  if (timings !== "") parts.push(el("p", "answer-timings", timings));
  showAnswer(ctx, exchange, status, parts);
}

function renderError(ctx, exchange, error) {
  showAnswer(ctx, exchange, "error", [statusTag("error"), el("p", "error-text", messageOf(error))]);
}

/** Whether any part of `node` shows inside `frame` and inside the viewport. */
function isOnScreen(node, frame) {
  const box = node.getBoundingClientRect();
  const bounds = frame.getBoundingClientRect();
  return box.bottom > Math.max(bounds.top, 0) && box.top < Math.min(bounds.bottom, window.innerHeight);
}

function showAnswer(ctx, exchange, status, parts) {
  const { ui } = ctx;
  // Follow the answer only if the reader is still watching its loading card, never pull them back down.
  const watched = isOnScreen(exchange.answer, ui.threadScroll);
  exchange.answer.dataset.status = status;
  exchange.answer.replaceChildren(...parts);
  exchange.answer.setAttribute("aria-busy", "false");
  if (watched) reveal(exchange.root, "start");
  // Behind the source drawer the thread is inert, and its live region is silent with it.
  if (ui.chatPanel.inert) ctx.announce(TEXT.answerArrived(STATUS_TAGS[status].text));
}

function statusTag(status) {
  const { text, icon: iconName } = STATUS_TAGS[status];
  const node = el("p", "status-tag");
  node.append(icon(iconName), el("span", "", text));
  return node;
}

function claimList(ctx, claims) {
  const list = el("ol", "claims");
  for (const claim of claims) {
    const cites = el("span", "cites");
    cites.append(...claim.sources.map((source) => citeChip(ctx, source)));
    const item = el("li", "claim");
    item.append(el("span", "claim-text", claim.text), " ", cites);
    list.append(item);
  }
  return list;
}

function citeChip(ctx, source) {
  const chip = fromTemplate("tpl-cite");
  chip.querySelector(".cite-label").textContent = source.label;
  chip.title = source.docTitle;
  chip.setAttribute("aria-label", TEXT.citeName(source.label));
  chip.addEventListener("click", () => openSource(ctx, source, chip));
  return chip;
}

function timingsText(timings) {
  if (timings === null) return "";
  const ms = (key) => (Number.isFinite(timings[key]) && timings[key] >= 0 ? timings[key] : null);
  const retrieval = ms("retrieval");
  const relevance = ms("relevance");
  const claims = ms("claims");
  const parts = [];
  if (retrieval !== null) parts.push(TEXT.retrieval(formatDuration(retrieval)));
  if (relevance !== null || claims !== null) {
    parts.push(TEXT.generation(formatDuration((relevance ?? 0) + (claims ?? 0))));
  }
  return parts.join(" · ");
}

// --- the source panel --------------------------------------------------------

function openSource(ctx, source, chip) {
  const { ui, state } = ctx;
  ui.sourceTitle.replaceChildren(el("bdi", "", source.docTitle));
  ui.sourceLabel.textContent = source.label;
  ui.sourcePage.textContent = source.page === null ? "" : String(source.page);
  ui.sourcePageRow.hidden = source.page === null;
  ui.sourceText.textContent = source.text;

  state.sourceOpener?.setAttribute("aria-expanded", "false");
  state.sourceOpener = chip;
  chip.setAttribute("aria-expanded", "true");

  ui.sourcePanel.hidden = false;
  ui.sourcePanel.scrollTop = 0;
  ui.layout.classList.add("has-source");
  applySourceMode(ctx);
  ui.sourceTitle.focus();
}

export function closeSource(ctx, { restoreFocus = true } = {}) {
  const { ui, state } = ctx;
  if (ui.sourcePanel.hidden) return;
  ui.sourcePanel.hidden = true;
  ui.layout.classList.remove("has-source");
  applySourceMode(ctx);

  const opener = state.sourceOpener;
  state.sourceOpener = null;
  if (opener === null) return;
  opener.setAttribute("aria-expanded", "false");
  if (!restoreFocus) return;
  if (opener.isConnected) opener.focus();
  else ui.question.focus();
}

/** Below the wide breakpoint the panel is a drawer, and everything behind it goes inert. */
export function applySourceMode({ ui, wideLayout }) {
  const overlay = !ui.sourcePanel.hidden && !wideLayout.matches;
  ui.scrim.hidden = !overlay;
  document.body.classList.toggle("has-overlay", overlay);
  for (const region of [ui.skipLink, ui.header, ui.docsPanel, ui.chatPanel, ui.footer]) {
    region.inert = overlay;
  }
}

// --- keyboard ----------------------------------------------------------------

export function onQuestionKeydown(ctx, event) {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  ask(ctx);
}

/** Escape closes the source panel from inside it or from the page itself, never while typing elsewhere. */
export function onDocumentKeydown(ctx, event) {
  const { ui } = ctx;
  if (event.key !== "Escape" || ui.sourcePanel.hidden) return;
  const focus = document.activeElement;
  const onPage = focus === null || focus === document.body || focus === document.documentElement;
  if (!onPage && !ui.sourcePanel.contains(focus)) return;
  event.preventDefault();
  closeSource(ctx);
}
