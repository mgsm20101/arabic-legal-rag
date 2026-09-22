# Internal review — an audit trail, not a roadmap

This folder is a **point-in-time critical review** of the repository, carried out
against commit `ed2107e` on **2026-09-17**. It is kept in the repo because the
findings and the probes that produced them are part of the record, not because
the task list describes the project's current state.

**Read it as history.** `IMPLEMENTATION-TASKS-AR.md` opens by saying every task is
proposed and unimplemented. That was true on 2026-09-17. It is no longer true:
**21 commits have landed since that baseline**, and four of the review's own P1
tasks are among them.

| Review task | Landed in |
|---|---|
| **T01** — atomic ingestion | `36377af` *fix(ingest): validate before writing, and write the corpus atomically* |
| **T02** — OCR gate | `aef010f` *fix(ocr_gate): score missing pages as full misses and fail on spurious digits* · `32dd43e` *feat(ocr): …score the gate* |
| **T03** — benchmark server hardening | `1d48b99` *fix(server): reject foreign Host headers and bound k in the dev benchmark server* |
| **T04** — isolate file extraction | `580d756` *feat(isolate): add run_isolated, a killable subprocess with a wall-clock timeout* · `00bf4c6` *fix(pdf): isolate PDF extraction in a subprocess with enforceable limits (F04)* |

The remaining tasks are still open. Nothing here has been edited to look better
in hindsight — only absolute filesystem paths were rewritten as repository-relative
links so they resolve on GitHub.

## Files

| File | What it is |
|---|---|
| `PROJECT-REVIEW-AR.md` | The critical review itself (Arabic), with a code reference per finding |
| `IMPLEMENTATION-TASKS-AR.md` | 22 proposed tasks with acceptance criteria (Arabic) — **a 2026-09-17 snapshot** |
| `reproduce_findings.py` | Read-only probes that reproduce the findings. Uses temporary directories; changes nothing. Run from the repo root: `python docs/review/reproduce_findings.py` |
| `reproductions.json` · `VERIFICATION.json` | What each probe checked and what it returned |
| `dependency-audit-summary.json` | `pip-audit` over `constraints.txt` — 61 packages, 2 unique advisories. Scope is stated in the file: package-version matches, **not** demonstrated exploitability |
| `pytest-ci-environment.txt` | The environment the suite ran in |

---

## مقروءاً بالعربية

هذا المجلد **مراجعة نقدية في لحظة بعينها** — على commit `ed2107e` بتاريخ 2026-09-17 — ومحفوظ
لأن النتائج والفحوص التي أنتجتها جزء من السجل، لا لأن قائمة التاسكات تصف الحالة الراهنة.

`IMPLEMENTATION-TASKS-AR.md` يبدأ بجملة «كل التاسكات مقترحة وغير منفذة». كان ذلك صحيحاً في
تاريخه. **لم يعد صحيحاً**: نزل بعده 21 commit، ومنها أربع من تاسكات P1 نفسها (الجدول أعلاه).
باقي التاسكات ما زالت مفتوحة.

لم يُعدَّل شيء هنا ليبدو أفضل بأثر رجعي — التعديل الوحيد كان تحويل المسارات المطلقة إلى روابط
نسبية داخل المستودع حتى تعمل على GitHub.
