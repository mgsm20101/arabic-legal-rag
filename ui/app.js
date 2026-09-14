/*
 * Chat page of the arabic-legal-rag demo layer (ADR-023): upload an Arabic PDF
 * or TXT document, ask about it, and read every displayed sentence beside the
 * source the pipeline verified for it.
 *
 * Rules this file keeps; tests/test_ui_static.py pins them:
 * - server text reaches the page only as textContent or attribute values;
 * - every request goes through one helper, which sends the X-LegalRAG header;
 * - nothing is written to browser storage: the documents are private, and the
 *   conversation lives only as long as the tab.
 *
 * One file by design, past the usual size ceiling: the server serves exactly
 * /app.css and /app.js with no build step, so splitting the script into modules
 * would change that contract. The sections below run top-down.
 */
(() => {
  "use strict";

  // --- limits ----------------------------------------------------------------

  const MAX_UPLOAD_BYTES = 20 * 1024 * 1024; // the server enforces it too
  const UPLOAD_EXTENSION = /\.(pdf|txt)$/i;
  const MIN_QUESTION_LENGTH = 3;
  const MAX_QUESTION_LENGTH = 500;
  const COUNTER_WARNING_LENGTH = 450;
  const BYTES_PER_KB = 1024;
  const BYTES_PER_MB = 1024 * 1024;
  const KB_SHOWN_BELOW = 1000;
  const SKELETON_ROWS = 3;
  const TOAST_MS = 6000;
  const ANNOUNCE_DELAY_MS = 80;
  const TICK_MS = 1000;
  const WIDE_LAYOUT = "(min-width: 1280px)"; // where the source panel gets its own column; see app.css

  // --- copy ------------------------------------------------------------------

  // Every abstain_reason in the API contract, and the sentence the reader sees.
  const ABSTAIN_SENTENCES = Object.freeze({
    no_sources: "لا توجد مستندات للبحث فيها. ارفع مستنداً أولاً.",
    relevance_no: "لا أستطيع الإجابة من المستندات المتاحة.",
    model_abstained: "لا أستطيع الإجابة من المستندات المتاحة.",
    all_dropped: "حُذفت كل الجمل لأنها بلا مصدر متحقَّق منه.",
    relevance_failure: "تعذّر توليد إجابة صالحة. أعد المحاولة.",
    schema_failure: "تعذّر توليد إجابة صالحة. أعد المحاولة.",
    no_claims: "تعذّر توليد إجابة صالحة. أعد المحاولة.",
  });
  const ABSTAIN_FALLBACK = "لا توجد إجابة يمكن عرضها لهذا السؤال.";

  const STATUS_TAGS = Object.freeze({
    answered: Object.freeze({ text: "إجابة", icon: "check" }),
    partial: Object.freeze({ text: "إجابة جزئية", icon: "partial" }),
    abstained: Object.freeze({ text: "امتناع", icon: "abstain" }),
    error: Object.freeze({ text: "خطأ", icon: "alert" }),
  });

  const TEXT = Object.freeze({
    networkError: "تعذّر الاتصال بالخادم. تأكد أنه يعمل ثم أعد المحاولة.",
    genericError: "حدث خطأ غير متوقع. أعد المحاولة.",
    retryAfter: (seconds) => `انتظر ${seconds} ثانية ثم أعد المحاولة.`,
    retryLater: "انتظر قليلاً ثم أعد المحاولة.",
    modelPrefix: "النموذج: ",
    modelDown: "خادم النموذج غير متاح",
    healthUnknown: "تعذّر فحص حالة النموذج",
    pickFile: "اختر ملفاً أو أفلته هنا",
    noFile: "اختر ملفاً أولاً.",
    badExtension: "نوع الملف غير مدعوم. اختر ملف PDF أو TXT.",
    emptyFile: "الملف فارغ.",
    tooLarge: (size) => `حجم الملف ${size}، والحد الأقصى 20 م.ب.`,
    uploading: "جارٍ تجهيز المستند…",
    uploaded: (title) => `رُفع «${title}» وصار ضمن نطاق السؤال.`,
    confirmDelete: (title) => `حذف «${title}»؟ لن يُستخدم في الإجابات بعد ذلك.`,
    deleteLabel: (title) => `حذف «${title}»`,
    deleted: (title) => `حُذف «${title}».`,
    docsLoadError: "تعذّر تحميل المستندات.",
    untitled: "مستند بلا عنوان",
    statute: (n) => `قانون · ${n} مادة`,
    generic: (n) => `مستند · ${n} مقطع`,
    pages: (n) => `${n} صفحة`,
    kilobytes: (n) => `${n} ك.ب`,
    megabytes: (n) => `${n} م.ب`,
    seconds: (n) => `${n} ث`,
    scopeLoading: "جارٍ تحميل المستندات…",
    scopeNoDocuments: "ارفع مستنداً أولاً لتسأل عنه.",
    scopeNone: "اختر مستنداً واحداً على الأقل.",
    scopeAll: (total) => `البحث في كل المستندات (${total})`,
    scopeSome: (selected, total) => `البحث في ${selected} من ${total} مستندات`,
    questionTooShort: `اكتب ${MIN_QUESTION_LENGTH} أحرف على الأقل.`,
    newChat: "بدأت محادثة جديدة.",
    dropped: (n) => `حُذفت ${n} جملة لأنها بلا مصدر متحقَّق منه`,
    retrieval: (duration) => `الاسترجاع ${duration}`,
    generation: (duration) => `التوليد ${duration}`,
    citeName: (label, title) => `المصدر ${label}، ${title}`,
    sourceLabel: (n) => `مصدر ${n}`,
  });

  // --- values from the server are checked, never trusted ---------------------

  const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  const countOf = (value) => (Number.isInteger(value) && value >= 0 ? value : null);
  const textOf = (value, fallback) =>
    typeof value === "string" && value.trim() !== "" ? value.trim() : fallback;

  const dateFormat = new Intl.DateTimeFormat("ar-EG-u-nu-latn", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  function formatSize(bytes) {
    const kilobytes = bytes / BYTES_PER_KB;
    if (kilobytes < KB_SHOWN_BELOW) return TEXT.kilobytes(Math.max(1, Math.round(kilobytes)));
    return TEXT.megabytes((bytes / BYTES_PER_MB).toFixed(1));
  }

  function formatDate(iso) {
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? "" : dateFormat.format(date);
  }

  function formatDuration(ms) {
    return TEXT.seconds((ms / 1000).toFixed(1));
  }

  // --- DOM -------------------------------------------------------------------

  function byId(id) {
    const node = document.getElementById(id);
    if (node === null) throw new Error(`app.html has no #${id}`);
    return node;
  }

  function el(tag, className = "", text = null) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== null) node.textContent = text;
    return node;
  }

  function fromTemplate(id) {
    return byId(id).content.firstElementChild.cloneNode(true);
  }

  function icon(name) {
    const svg = fromTemplate("tpl-icon");
    svg.querySelector("use").setAttribute("href", `#i-${name}`);
    return svg;
  }

  function setText(node, text) {
    node.textContent = text;
    node.hidden = text === "";
  }

  function setBusy(button, busy) {
    button.classList.toggle("is-busy", busy);
    const label = button.querySelector(".btn-label");
    if (label !== null && label.dataset.idle && label.dataset.busy) {
      label.textContent = busy ? label.dataset.busy : label.dataset.idle;
    }
  }

  /** Counts whole seconds into `node` and hands back a stop callback. */
  function startClock(node) {
    const started = Date.now();
    const timer = setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      node.textContent = TEXT.seconds(seconds);
      node.hidden = seconds < 1;
    }, TICK_MS);
    return () => clearInterval(timer);
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const wideLayout = window.matchMedia(WIDE_LAYOUT);

  function reveal(node, block) {
    node.scrollIntoView({ block, behavior: reducedMotion.matches ? "auto" : "smooth" });
  }

  const ui = Object.freeze({
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

  // --- state: in memory only -------------------------------------------------

  const state = {
    documents: [], // frozen documents, newest first
    excluded: new Set(), // doc_ids unticked out of the scope; a new document starts ticked
    docsStatus: "loading", // "loading" | "ready" | "error"
    uploading: false,
    pendingRow: null,
    stopPendingClock: null,
    chat: null, // AbortController of the question in flight
    sourceOpener: null, // the citation chip whose source is open
  };

  // --- the API ---------------------------------------------------------------

  class ApiError extends Error {
    constructor(status, code, messageAr) {
      super(`${status} ${code}`);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
      this.messageAr = messageAr;
    }
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  function retryAfterText(header) {
    const seconds = Number.parseInt(header ?? "", 10);
    return Number.isInteger(seconds) && seconds > 0 ? TEXT.retryAfter(seconds) : TEXT.retryLater;
  }

  /**
   * The one way this page talks to the server: same-origin paths, the app
   * header on every request, and every failure turned into an ApiError that
   * carries an Arabic sentence to show.
   */
  async function api(path, { method = "GET", json, form, signal } = {}) {
    const headers = { "X-LegalRAG": "1", Accept: "application/json" };
    let body;
    if (json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(json);
    } else if (form !== undefined) {
      body = form;
    }

    let response;
    try {
      response = await fetch(path, {
        method,
        headers,
        body,
        signal,
        cache: "no-store",
        credentials: "same-origin",
      });
    } catch (error) {
      if (signal?.aborted) throw error;
      throw new ApiError(0, "network", TEXT.networkError);
    }

    const data = await readJson(response);
    if (!response.ok) {
      const payload = isObject(data) ? data : {};
      const message = textOf(payload.message_ar, TEXT.genericError);
      const code = typeof payload.error === "string" ? payload.error : `http_${response.status}`;
      const wait = response.status === 429 ? ` ${retryAfterText(response.headers.get("Retry-After"))}` : "";
      throw new ApiError(response.status, code, message + wait);
    }
    if (!isObject(data)) throw new ApiError(response.status, "bad_response", TEXT.genericError);
    return data;
  }

  function messageOf(error) {
    if (error instanceof ApiError) return error.messageAr;
    console.error(error);
    return TEXT.genericError;
  }

  // --- feedback --------------------------------------------------------------

  let toastTimer = 0;

  /** Says `message` through the status region; `visible` also shows it as a toast. */
  function announce(message, { tone = "info", visible = true } = {}) {
    clearTimeout(toastTimer);
    ui.toast.classList.remove("is-visible");
    ui.toast.textContent = "";
    // emptied first, so a repeated message is still a change that gets announced
    toastTimer = setTimeout(() => {
      ui.toast.dataset.tone = tone;
      ui.toast.textContent = message;
      ui.toast.classList.toggle("is-visible", visible);
      toastTimer = setTimeout(() => ui.toast.classList.remove("is-visible"), TOAST_MS);
    }, ANNOUNCE_DELAY_MS);
  }

  // --- health ----------------------------------------------------------------

  async function refreshHealth() {
    try {
      renderHealth(await api("/api/health"));
    } catch (error) {
      if (!(error instanceof ApiError)) console.error(error);
      ui.health.dataset.state = "unknown";
      ui.healthText.replaceChildren(TEXT.healthUnknown);
    }
  }

  function renderHealth(health) {
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

  // --- documents -------------------------------------------------------------

  function toDocument(raw) {
    if (!isObject(raw) || typeof raw.doc_id !== "string" || raw.doc_id === "") return null;
    return Object.freeze({
      id: raw.doc_id,
      title: textOf(raw.title, TEXT.untitled),
      statute: raw.kind === "statute",
      chunks: countOf(raw.chunks) ?? 0,
      pages: countOf(raw.pages),
      sizeBytes: countOf(raw.size_bytes),
      createdAt: typeof raw.created_at === "string" ? raw.created_at : "",
    });
  }

  async function refreshDocuments() {
    try {
      const data = await api("/api/documents");
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
      if (state.docsStatus === "ready") {
        announce(message, { tone: "error" });
      } else {
        state.docsStatus = "error";
        ui.docsErrorText.textContent = `${TEXT.docsLoadError} ${message}`;
      }
    }
    renderDocuments();
  }

  function retryDocuments() {
    state.docsStatus = "loading";
    renderDocuments();
    refreshDocuments();
  }

  function renderDocuments() {
    const { documents, docsStatus, pendingRow } = state;
    const ready = docsStatus === "ready";
    const pending = pendingRow === null ? [] : [pendingRow];
    const skeletons =
      docsStatus === "loading"
        ? Array.from({ length: SKELETON_ROWS }, () => fromTemplate("tpl-doc-skeleton"))
        : [];
    const rows = ready ? documents.map((doc) => documentRow(doc)) : [];
    ui.docList.replaceChildren(...pending, ...skeletons, ...rows);
    ui.docList.setAttribute("aria-busy", String(docsStatus === "loading"));

    ui.docsError.hidden = docsStatus !== "error";
    ui.docsEmpty.hidden = !ready || documents.length > 0 || pendingRow !== null;
    ui.docsCaption.hidden = !ready || documents.length === 0;
    ui.docsCount.hidden = !ready || documents.length === 0;
    ui.docsCount.textContent = String(documents.length);
    updateAskState();
  }

  let rowSerial = 0;

  function documentRow(doc) {
    rowSerial += 1;
    const row = fromTemplate("tpl-doc");
    const check = row.querySelector(".doc-check");
    const title = row.querySelector(".doc-title");
    const meta = row.querySelector(".doc-meta");
    const date = row.querySelector(".doc-date");
    const remove = row.querySelector(".doc-delete");

    check.id = `doc-check-${rowSerial}`;
    meta.id = `doc-meta-${rowSerial}`;
    check.checked = !state.excluded.has(doc.id);
    check.setAttribute("aria-describedby", meta.id);
    check.addEventListener("change", () => setIncluded(doc.id, check.checked));
    title.htmlFor = check.id;
    title.querySelector(".doc-title-text").textContent = doc.title;

    setText(row.querySelector(".doc-kind"), doc.statute ? TEXT.statute(doc.chunks) : TEXT.generic(doc.chunks));
    setText(row.querySelector(".doc-pages"), doc.pages ? TEXT.pages(doc.pages) : "");
    setText(row.querySelector(".doc-size"), doc.sizeBytes === null ? "" : formatSize(doc.sizeBytes));
    setText(date, formatDate(doc.createdAt));
    if (!date.hidden) date.dateTime = doc.createdAt;

    remove.setAttribute("aria-label", TEXT.deleteLabel(doc.title));
    remove.addEventListener("click", () => deleteDocument(doc, remove));
    return row;
  }

  function setIncluded(id, included) {
    const excluded = new Set(state.excluded);
    if (included) excluded.delete(id);
    else excluded.add(id);
    state.excluded = excluded;
    updateAskState();
  }

  function selectedIds() {
    return state.documents.filter((doc) => !state.excluded.has(doc.id)).map((doc) => doc.id);
  }

  /** null asks about every document, as the API contract defines it. */
  function scopeForRequest() {
    const selected = selectedIds();
    return selected.length === state.documents.length ? null : selected;
  }

  // --- upload ----------------------------------------------------------------

  function selectedFile() {
    const { files } = ui.fileInput;
    return files !== null && files.length > 0 ? files[0] : null;
  }

  function fileProblem(file) {
    if (file === null) return TEXT.noFile;
    if (!UPLOAD_EXTENSION.test(file.name)) return TEXT.badExtension;
    if (file.size === 0) return TEXT.emptyFile;
    if (file.size > MAX_UPLOAD_BYTES) return TEXT.tooLarge(formatSize(file.size));
    return null;
  }

  function showFileError(message) {
    ui.fileError.textContent = message ?? "";
    ui.fileError.hidden = message === null;
    ui.filePicker.classList.toggle("is-invalid", message !== null);
    if (message === null) ui.fileInput.removeAttribute("aria-invalid");
    else ui.fileInput.setAttribute("aria-invalid", "true");
  }

  /** Mirrors the chosen file in the picker and the button; leaves a shown error alone. */
  function syncFilePicker() {
    const file = selectedFile();
    ui.fileName.textContent = file === null ? TEXT.pickFile : file.name;
    ui.filePicker.classList.toggle("is-chosen", file !== null);
    ui.uploadButton.disabled = state.uploading || file === null || fileProblem(file) !== null;
    return file;
  }

  function onFileChosen() {
    const file = syncFilePicker();
    showFileError(file === null ? null : fileProblem(file));
  }

  async function upload(event) {
    event.preventDefault();
    if (state.uploading) return;
    const file = selectedFile();
    const problem = fileProblem(file);
    if (problem !== null) {
      showFileError(problem);
      ui.fileInput.focus();
      return;
    }

    state.uploading = true;
    ui.fileInput.disabled = true;
    ui.uploadButton.disabled = true;
    setBusy(ui.uploadButton, true);
    showFileError(null);
    showPendingRow(file.name);
    announce(TEXT.uploading, { visible: false });

    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const created = toDocument(await api("/api/documents", { method: "POST", form }));
      ui.uploadForm.reset();
      await refreshDocuments();
      announce(TEXT.uploaded(created === null ? file.name : created.title), { tone: "success" });
    } catch (error) {
      const message = messageOf(error);
      showFileError(message);
      announce(message, { tone: "error", visible: false });
    } finally {
      clearPendingRow();
      state.uploading = false;
      ui.fileInput.disabled = false;
      setBusy(ui.uploadButton, false);
      syncFilePicker();
      renderDocuments();
      if (document.activeElement === null || document.activeElement === document.body) {
        ui.fileInput.focus();
      }
    }
    refreshHealth();
  }

  function showPendingRow(fileName) {
    const row = fromTemplate("tpl-doc-pending");
    row.querySelector(".pending-name").textContent = fileName;
    state.stopPendingClock = startClock(row.querySelector(".elapsed"));
    state.pendingRow = row;
    ui.docsEmpty.hidden = true;
    ui.docList.prepend(row);
  }

  function clearPendingRow() {
    state.stopPendingClock?.();
    state.pendingRow?.remove();
    state.pendingRow = null;
    state.stopPendingClock = null;
  }

  // --- delete ----------------------------------------------------------------

  async function deleteDocument(doc, button) {
    if (!window.confirm(TEXT.confirmDelete(doc.title))) return;
    const position = state.documents.findIndex((item) => item.id === doc.id);
    button.disabled = true;
    setBusy(button, true);
    try {
      await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
    } catch (error) {
      button.disabled = false;
      setBusy(button, false);
      announce(messageOf(error), { tone: "error" });
      return;
    }
    announce(TEXT.deleted(doc.title), { tone: "success" });
    await refreshDocuments();
    focusDocumentAt(position);
    refreshHealth();
  }

  /** Once a row is gone, keyboard focus moves to its neighbour, or to the picker. */
  function focusDocumentAt(position) {
    const checks = ui.docList.querySelectorAll(".doc-check");
    const target = position >= 0 ? checks[Math.min(position, checks.length - 1)] : undefined;
    (target ?? ui.fileInput).focus();
  }

  // --- asking ----------------------------------------------------------------

  function scopeNote(total, selected) {
    if (state.docsStatus === "loading") return [TEXT.scopeLoading, "muted"];
    if (state.docsStatus === "error") return [TEXT.docsLoadError, "warn"];
    if (total === 0) return [TEXT.scopeNoDocuments, "muted"];
    if (selected === 0) return [TEXT.scopeNone, "warn"];
    return [selected === total ? TEXT.scopeAll(total) : TEXT.scopeSome(selected, total), "muted"];
  }

  function updateAskState() {
    const total = state.documents.length;
    const selected = selectedIds().length;
    const typed = ui.question.value.length;
    const length = ui.question.value.trim().length;

    const [note, tone] = scopeNote(total, selected);
    ui.scopeNote.textContent = note;
    ui.scopeNote.dataset.tone = tone;
    ui.counter.textContent = `${typed}/${MAX_QUESTION_LENGTH}`;
    ui.counter.dataset.tone = typed >= COUNTER_WARNING_LENGTH ? "warn" : "muted";
    if (length >= MIN_QUESTION_LENGTH) hideQuestionError();

    ui.askButton.disabled =
      state.chat !== null ||
      state.docsStatus !== "ready" ||
      selected === 0 ||
      length < MIN_QUESTION_LENGTH;
    ui.newChat.disabled = ui.thread.childElementCount === 0;
  }

  function showQuestionError() {
    ui.questionError.textContent = TEXT.questionTooShort;
    ui.questionError.hidden = false;
    ui.question.setAttribute("aria-invalid", "true");
  }

  function hideQuestionError() {
    ui.questionError.textContent = "";
    ui.questionError.hidden = true;
    ui.question.removeAttribute("aria-invalid");
  }

  async function ask(event) {
    event?.preventDefault();
    const question = ui.question.value.trim();
    if (question.length > 0 && question.length < MIN_QUESTION_LENGTH) {
      showQuestionError();
      return;
    }
    if (ui.askButton.disabled) return;

    const controller = new AbortController();
    const docIds = scopeForRequest();
    const refocus = document.activeElement === ui.askButton;
    const exchange = addExchange(question);
    state.chat = controller;
    ui.question.value = "";
    setBusy(ui.askButton, true);
    updateAskState();
    if (refocus) ui.question.focus();

    try {
      const data = await api("/api/chat", {
        method: "POST",
        json: { question, doc_ids: docIds },
        signal: controller.signal,
      });
      if (!controller.signal.aborted) renderAnswer(exchange, data);
    } catch (error) {
      if (controller.signal.aborted) return;
      renderError(exchange, error);
      if (ui.question.value === "") ui.question.value = question;
    } finally {
      exchange.stopClock();
      if (state.chat === controller) {
        state.chat = null;
        setBusy(ui.askButton, false);
        updateAskState();
      }
    }
  }

  function addExchange(question) {
    const root = fromTemplate("tpl-exchange");
    root.querySelector(".question-text").textContent = question;
    const answer = root.querySelector(".answer");
    const stopClock = startClock(answer.querySelector(".elapsed"));
    ui.threadIntro.hidden = true;
    ui.thread.append(root);
    reveal(root, "nearest");
    return Object.freeze({ root, answer, stopClock });
  }

  function clearThread() {
    if (state.chat !== null) {
      state.chat.abort();
      state.chat = null;
      setBusy(ui.askButton, false);
    }
    closeSource({ restoreFocus: false });
    ui.thread.replaceChildren();
    ui.threadIntro.hidden = false;
    hideQuestionError();
    updateAskState();
    ui.question.focus();
    announce(TEXT.newChat);
  }

  // --- answers ---------------------------------------------------------------

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
    return Object.values(dropped).reduce((sum, value) => sum + (countOf(value) ?? 0), 0);
  }

  function renderAnswer(exchange, data) {
    const answer = toAnswer(data);
    const abstained = answer.status === "abstained";
    // A sentence reaches the page only with a source that resolves in this same reply.
    const shown = abstained ? [] : answer.claims.filter((claim) => claim.sources.length > 0);
    const unsourced = abstained ? 0 : answer.claims.length - shown.length;
    const status = shown.length === 0 ? "abstained" : answer.status === "partial" ? "partial" : "answered";

    const parts = [
      statusTag(status),
      shown.length > 0 ? claimList(shown) : el("p", "abstain-text", abstainSentence(answer.abstainReason)),
    ];
    const dropped = answer.dropped + unsourced;
    if (dropped > 0) parts.push(el("p", "answer-note", TEXT.dropped(dropped)));
    const timings = timingsText(answer.timings);
    if (timings !== "") parts.push(el("p", "answer-timings", timings));
    showAnswer(exchange, status, parts);
  }

  function renderError(exchange, error) {
    showAnswer(exchange, "error", [statusTag("error"), el("p", "error-text", messageOf(error))]);
  }

  function showAnswer(exchange, status, parts) {
    exchange.answer.dataset.status = status;
    exchange.answer.replaceChildren(...parts);
    exchange.answer.setAttribute("aria-busy", "false");
    reveal(exchange.root, "start");
  }

  function statusTag(status) {
    const { text, icon: iconName } = STATUS_TAGS[status];
    const node = el("p", "status-tag");
    node.append(icon(iconName), el("span", "", text));
    return node;
  }

  function claimList(claims) {
    const list = el("ol", "claims");
    for (const claim of claims) {
      const cites = el("span", "cites");
      cites.append(...claim.sources.map((source) => citeChip(source)));
      const item = el("li", "claim");
      item.append(el("span", "claim-text", claim.text), " ", cites);
      list.append(item);
    }
    return list;
  }

  function citeChip(source) {
    const chip = fromTemplate("tpl-cite");
    chip.querySelector(".cite-label").textContent = source.label;
    chip.title = source.docTitle;
    chip.setAttribute("aria-label", TEXT.citeName(source.label, source.docTitle));
    chip.addEventListener("click", () => openSource(source, chip));
    return chip;
  }

  function abstainSentence(reason) {
    return typeof reason === "string" && Object.hasOwn(ABSTAIN_SENTENCES, reason)
      ? ABSTAIN_SENTENCES[reason]
      : ABSTAIN_FALLBACK;
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

  // --- the source panel ------------------------------------------------------

  function openSource(source, chip) {
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
    applySourceMode();
    ui.sourceTitle.focus();
  }

  function closeSource({ restoreFocus = true } = {}) {
    if (ui.sourcePanel.hidden) return;
    ui.sourcePanel.hidden = true;
    ui.layout.classList.remove("has-source");
    applySourceMode();

    const opener = state.sourceOpener;
    state.sourceOpener = null;
    if (opener === null) return;
    opener.setAttribute("aria-expanded", "false");
    if (!restoreFocus) return;
    if (opener.isConnected) opener.focus();
    else ui.question.focus();
  }

  /** Below the wide breakpoint the panel is a drawer, and the page behind it goes inert. */
  function applySourceMode() {
    const overlay = !ui.sourcePanel.hidden && !wideLayout.matches;
    ui.scrim.hidden = !overlay;
    document.body.classList.toggle("has-overlay", overlay);
    for (const region of [ui.header, ui.docsPanel, ui.chatPanel, ui.footer]) {
      region.inert = overlay;
    }
  }

  // --- events ----------------------------------------------------------------

  function onQuestionKeydown(event) {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    ask();
  }

  function onDocumentKeydown(event) {
    if (event.key !== "Escape" || ui.sourcePanel.hidden) return;
    event.preventDefault();
    closeSource();
  }

  function init() {
    ui.uploadForm.addEventListener("submit", upload);
    ui.fileInput.addEventListener("change", onFileChosen);
    ui.docsRetry.addEventListener("click", retryDocuments);
    ui.askForm.addEventListener("submit", ask);
    ui.question.addEventListener("input", () => updateAskState());
    ui.question.addEventListener("keydown", onQuestionKeydown);
    ui.newChat.addEventListener("click", clearThread);
    ui.sourceClose.addEventListener("click", () => closeSource());
    ui.scrim.addEventListener("click", () => closeSource());
    document.addEventListener("keydown", onDocumentKeydown);
    wideLayout.addEventListener("change", applySourceMode);

    syncFilePicker();
    renderDocuments();
    refreshHealth();
    refreshDocuments();
  }

  init();
})();
