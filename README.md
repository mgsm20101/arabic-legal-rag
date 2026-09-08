# arabic-legal-rag

**Measured retrieval over Egyptian statutory text — Arabic, article-level, eval-first.**

Not a legal assistant. This repository answers one engineering question: *on a
real Arabic legal corpus, how much does each component of a RAG pipeline
actually buy — and at what latency and cost?*

The deliverable is a defensible table of numbers, not a demo.

---

## Status

| Milestone | Scope | State |
|---|---|---|
| **M1** | Article-level corpus · 60-question eval set · BM25 vs dense vs hybrid vs +reranker | 🚧 in progress — corpus ingested (56 articles), eval set 20/60, BM25 measured: Recall@5 0.333 |
| **M2** | Citation-forced generation · abstention · measured tool use · workflow-vs-agent ablation | ⬜ not started |

## Quick start

```bash
python tasks.py setup                    # install dependencies
# download the official law PDF into data/raw/ by hand — see data/raw/README.md
python tasks.py ingest --law "<name>"    # PDF -> article-level chunks + validation
python tasks.py verify-refs              # check every ground-truth article ref
python tasks.py eval                     # BM25 scoreboard — fast, no model needed
python tasks.py ablate                   # 4 configs compared: BM25/dense/hybrid/+rerank
python tasks.py serve                    # local test page at http://127.0.0.1:8000
python tasks.py test                     # run the test suite
```

`tasks.py` is the canonical runner and needs nothing but Python. `make <target>`
is an equivalent alias on Linux/CI.

`eval` and `ablate` are separate on purpose: `eval` must stay runnable on a
fresh clone with no model and no torch (it takes ~0.2s), while `ablate` loads
~1 GB of local models. Neither calls a paid API; retrieval runs entirely on CPU.

`python tasks.py eval` runs from a fresh clone with no corpus present — it
reports the question set and an empty result column rather than failing. That is
deliberate: the scoreboard exists before the system does.

## The test page

`python tasks.py serve` opens a local page whose primary job is **running the
eval set against the current retriever and showing which questions fail**, not
demoing answers. Ad-hoc querying and corpus browsing are secondary tabs. It is
a local server, not a hosted page: the corpus is not redistributable, and the
page reads it off disk. Standard library only — nothing to install.

## Why this corpus

Egyptian statutory text, one law at a time, taken from the **official PDF**
(الجريدة الرسمية / منشورات قانونية) downloaded by hand and re-split here into
articles. Chosen for engineering reasons, not subject matter:

1. **The text is pre-segmented into numbered articles**, so every ground-truth
   answer has an objective key (an article number). A corpus without stable
   references cannot be evaluated, only demoed.
2. **It is a public government instrument** — no publication constraint.
3. **The PDF is the text of record.** Public pages paraphrase and summarise,
   and a redistributed dataset is "extracted and processed" by someone else —
   both corrupt the corpus and the ground truth at the same time, invisibly.
   ADR-006, reinstated by ADR-012 after the Hugging Face detour.

## Design decisions worth reading

* **Article-level chunking, not fixed token windows.** A 512-token window
  splits an article in half and destroys the verification key. See
  [`DECISIONS.md`](DECISIONS.md) ADR-002.
* **Two Arabic normalizers, deliberately different.** An aggressive normalizer
  helps retrieval (`الأسماء` ≈ `الاسماء`) and *inflates scores* if reused at
  evaluation time. Search and scoring use separate functions. ADR-004.
* **Ground-truth references are machine-checked.** Every question starts as
  `ref_status: "unverified"`; `make verify-refs` confirms the cited article
  exists and contains the expected keywords. A wrong article number does not
  look like a bug — it looks like poor retrieval, and gets "fixed" in the wrong
  place. ADR-005.
* **Arabic PDFs do not extract, they mislead.** `pdfplumber.extract_text()`
  returns Arabic in *visual* order — every word reversed — and the failure looks
  like a broken splitter, not a broken reader. Reconstructing logical order also
  has to keep lam-alef ligatures whole, order glyphs by centre rather than `x0`,
  and keep Arabic-Indic digits left-to-right (otherwise `مادة (٢٥)` silently
  becomes `مادة (٥٢)`). ADR-013.
* **Locked test split.** 20 of the 60 questions are never used for
  configuration choices — one final run, reported with its date. PRD §5.
* **A public generated QA set is deliberately NOT the eval set.**
  [`fr3on/eg-legal-qa`](https://huggingface.co/datasets/fr3on/eg-legal-qa) was
  generated from the article text, so its questions share wording with the
  passages that answer them — lexical retrieval scores high for an artificial
  reason, and the set contains no unanswerable questions. It is kept as a
  secondary generalization set: the *gap* between it and the hand-written set is
  the interesting number. ADR-008.

## Evaluation design

| Category | n (target) | What it probes |
|---|---|---|
| `direct` | 20 | single-article factual retrieval |
| `multi_article` | 15 | answers that require combining two articles |
| `out_of_corpus` | 15 | **abstention** — the honest answer is "not in the corpus" |
| `colloquial` | 10 | Egyptian-dialect phrasing against formal legal Arabic |

Metrics: Recall@5, MRR, groundedness, abstention accuracy, latency, cost —
each reported with the split it was measured on.

## Limitations

* Small evaluation set (60 questions). Confidence intervals are wide; the
  numbers are a signal, not proof, and are reported as such.
* Single corpus, single jurisdiction.
* **Not suitable for legal use.** Outputs are not legal advice and are not
  validated by a lawyer.

## Repository layout

```
tasks.py           task runner — the entry point on every platform
src/legalrag/
  normalize.py     search vs evaluation normalization (ADR-004)
  pdf_text.py      logical-order Arabic out of a visually-ordered PDF (ADR-013)
  ingest.py        PDF -> article chunks + extraction & structural validation
  retrieve.py      BM25 baseline + Recall@k / MRR
  evaluate.py      the scoreboard
  verify_refs.py   ground-truth reference checker (ADR-005)
  server.py        local test page server (stdlib only)
ui/index.html      the test page
evals/retrieval/   the question set — written before any retrieval code
PRD.md             problem, scope, acceptance criteria
DECISIONS.md       ADRs with rejected alternatives and revisit conditions
EVAL.md            results (created at first measured run)
```

## License

Code: MIT. The statutory text in `data/` is an Egyptian government
instrument and is not redistributed in this repository.
