/*
 * Constants, the page's Arabic sentences, and the formatters that set numbers,
 * sizes, dates and durations into them.
 */

// --- limits ------------------------------------------------------------------

export const MAX_UPLOAD_MB = 20;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024; // the server enforces it too
export const UPLOAD_EXTENSION = /\.(pdf|txt)$/i;
export const MIN_QUESTION_LENGTH = 3;
export const MAX_QUESTION_LENGTH = 500;
export const COUNTER_WARNING_LENGTH = 450;
export const MAX_DOC_IDS = 20; // the most doc_ids the server accepts in one question
export const SKELETON_ROWS = 3;
export const TOAST_MS = 6000;
export const ANNOUNCE_DELAY_MS = 80;
export const TICK_MS = 1000;
export const WIDE_LAYOUT = "(min-width: 1280px)"; // where the source panel gets its own column; see app.css

const BYTES_PER_KB = 1024;
const BYTES_PER_MB = 1024 * 1024;
const KB_SHOWN_BELOW = 1000;

// --- counts ------------------------------------------------------------------

const pluralRules = new Intl.PluralRules("ar");

function countForms(one, two, few, many, other) {
  return Object.freeze({ one, two, few, many, other });
}

// How each counted noun reads after its number, by plural category. The one and
// two forms stand in for the number itself; zero takes the few form. The grammar
// behind each column is written out in tests/test_ui_copy.py.
export const COUNT_FORMS = Object.freeze({
  page: countForms("صفحة واحدة", "صفحتان", "صفحات", "صفحة", "صفحة"),
  article: countForms("مادة واحدة", "مادتان", "مواد", "مادة", "مادة"),
  chunk: countForms("مقطع واحد", "مقطعان", "مقاطع", "مقطعاً", "مقطع"),
  sentence: countForms("جملة واحدة", "جملتان", "جمل", "جملة", "جملة"),
  second: countForms("ثانية واحدة", "ثانيتان", "ثوانٍ", "ثانية", "ثانية"),
  document: countForms("مستند واحد", "مستندان", "مستندات", "مستنداً", "مستند"),
  letter: countForms("حرف واحد", "حرفان", "أحرف", "حرفاً", "حرف"),
});

/** A whole-number count with its noun in the form Arabic number agreement needs. */
export function formatCount(n, noun) {
  const forms = COUNT_FORMS[noun];
  const category = pluralRules.select(n);
  if (category === "one" || category === "two") return forms[category];
  return `${n} ${category === "zero" ? forms.few : forms[category]}`;
}

// --- sentences ---------------------------------------------------------------

// Every abstain_reason in the API contract, and the sentence the reader sees.
export const ABSTAIN_SENTENCES = Object.freeze({
  no_sources: "لا توجد مستندات للبحث فيها. ارفع مستنداً أولاً.",
  relevance_no: "لا أستطيع الإجابة من المستندات المتاحة.",
  model_abstained: "لا أستطيع الإجابة من المستندات المتاحة.",
  all_dropped: "حُذفت كل الجمل لأنها بلا مصدر متحقَّق منه.",
  relevance_failure: "تعذّر توليد إجابة صالحة. أعد المحاولة.",
  schema_failure: "تعذّر توليد إجابة صالحة. أعد المحاولة.",
  no_claims: "تعذّر توليد إجابة صالحة. أعد المحاولة.",
});
export const ABSTAIN_FALLBACK = "لا توجد إجابة يمكن عرضها لهذا السؤال.";

export const STATUS_TAGS = Object.freeze({
  answered: Object.freeze({ text: "إجابة", icon: "check" }),
  partial: Object.freeze({ text: "إجابة جزئية", icon: "partial" }),
  abstained: Object.freeze({ text: "امتناع", icon: "abstain" }),
  error: Object.freeze({ text: "خطأ", icon: "alert" }),
});

export const TEXT = Object.freeze({
  networkError: "تعذّر الاتصال بالخادم. تأكد أنه يعمل ثم أعد المحاولة.",
  genericError: "حدث خطأ غير متوقع. أعد المحاولة.",
  retryAfter: (seconds) => `مدة الانتظار قبل إعادة المحاولة: ${formatCount(seconds, "second")}.`,
  retryLater: "انتظر قليلاً ثم أعد المحاولة.",
  modelPrefix: "النموذج: ",
  modelDown: "خادم النموذج غير متاح",
  healthUnknown: "تعذّر فحص حالة النموذج",
  pickFile: "اختر ملفاً أو أفلته هنا",
  noFile: "اختر ملفاً أولاً.",
  badExtension: "نوع الملف غير مدعوم. اختر ملف PDF أو TXT.",
  emptyFile: "الملف فارغ.",
  tooLarge: `الحد الأقصى لحجم الملف ${MAX_UPLOAD_MB} ميجابايت.`,
  uploading: "جارٍ تجهيز المستند…",
  uploaded: (title) => `رُفع «${title}» وصار ضمن نطاق السؤال.`,
  confirmDelete: (title) => `حذف «${title}»؟ لن يُستخدم في الإجابات بعد ذلك.`,
  deleteLabel: (title) => `حذف «${title}»`,
  deleted: (title) => `حُذف «${title}».`,
  docsLoadError: "تعذّر تحميل المستندات.",
  untitled: "مستند بلا عنوان",
  statute: (chunks) => `قانون · ${formatCount(chunks, "article")}`,
  generic: (chunks) => `مستند · ${formatCount(chunks, "chunk")}`,
  pages: (pages) => formatCount(pages, "page"),
  elapsed: (seconds) => formatCount(seconds, "second"),
  scopeLoading: "جارٍ تحميل المستندات…",
  scopeNoDocuments: "ارفع مستنداً أولاً لتسأل عنه.",
  scopeNone: "اختر مستنداً واحداً على الأقل.",
  scopeTooMany: `يمكن اختيار ${formatCount(MAX_DOC_IDS, "document")} كحد أقصى، أو اختيار الكل.`,
  scopeAll: (total) => `نطاق البحث: كل المستندات (${total})`,
  scopeSome: (selected, total) => `نطاق البحث: ${formatCount(selected, "document")} من أصل ${total}`,
  questionTooShort: `الحد الأدنى لطول السؤال: ${formatCount(MIN_QUESTION_LENGTH, "letter")}.`,
  newChat: "بدأت محادثة جديدة.",
  newChatBusy: "انتظر انتهاء الإجابة الجارية",
  answerArrived: (status) => `وصل الرد على سؤالك: ${status}.`,
  // a dual subject takes the dual pronoun; a non-human plural takes the feminine singular
  dropped: (n) => `حُذفت ${formatCount(n, "sentence")} ${n === 2 ? "لأنهما" : "لأنها"} بلا مصدر متحقَّق منه`,
  retrieval: (duration) => `الاسترجاع ${duration}`,
  generation: (duration) => `التوليد ${duration}`,
  citeName: (label) => `المصدر ${label}`,
  sourceLabel: (n) => `مصدر ${n}`,
});

/** The note under an answer about sentences the gate dropped; all_dropped already says so itself. */
export function droppedNote(count, abstainReason) {
  return count > 0 && abstainReason !== "all_dropped" ? TEXT.dropped(count) : "";
}

// --- formatters --------------------------------------------------------------

const dateFormat = new Intl.DateTimeFormat("ar-EG-u-nu-latn", {
  day: "numeric",
  month: "long",
  year: "numeric",
});

export function formatSize(bytes) {
  const kilobytes = bytes / BYTES_PER_KB;
  if (kilobytes < KB_SHOWN_BELOW) return `${Math.max(1, Math.round(kilobytes))} ك.ب`;
  return `${(bytes / BYTES_PER_MB).toFixed(1)} م.ب`;
}

export function formatDate(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : dateFormat.format(date);
}

/** A measured duration, to one decimal, with the unit abbreviated as the spec shows it. */
export function formatDuration(ms) {
  return `${(ms / 1000).toFixed(1)} ث`;
}

export function retryAfterText(header) {
  const seconds = Number.parseInt(header ?? "", 10);
  return Number.isInteger(seconds) && seconds > 0 ? TEXT.retryAfter(seconds) : TEXT.retryLater;
}

export function abstainSentence(reason) {
  return typeof reason === "string" && Object.hasOwn(ABSTAIN_SENTENCES, reason)
    ? ABSTAIN_SENTENCES[reason]
    : ABSTAIN_FALLBACK;
}
