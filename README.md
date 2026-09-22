# arabic-legal-rag

**Measured retrieval and citation-forced generation over Egyptian statutory text — Arabic, article-level, eval-first.**

Not a legal assistant. This repository answers one engineering question: *on a
real Arabic legal corpus, how much does each component of a RAG pipeline
actually buy — and at what latency?*

The deliverable is a defensible table of numbers, not a demo. Every number
below has a row in **[`EVIDENCE.md`](EVIDENCE.md)** naming the command that
produced it, the commit it ran at, the environment, the eval set and the raw
result file. A number with no row there is not quotable.

> **Citing this from somewhere else?** Use the
> [`portfolio-v1`](https://github.com/mgsm20101/arabic-legal-rag/tree/portfolio-v1)
> tag, not `main`. The tag pins the numbers to the code that produced them;
> `main` will move.

---

## Problem

Arabic legal text breaks the defaults of a RAG pipeline in three specific ways,
and each one is silent.

1. **Extraction lies rather than fails.** This corpus's PDF stores Arabic in
   *visual* order: `pdfplumber.extract_text()` returns every word reversed, and
   the result looks like a broken chunker rather than a broken reader. Arabic-Indic
   digits sit inside the Arabic Unicode block, so a naive reversal turns
   `مادة (٢٥)` into `مادة (٥٢)` — a real article, wrong number, no error. ADR-013.
2. **Normalization that helps retrieval inflates the score.** Folding hamza
   forms (`الأسماء` ≈ `الاسماء`) genuinely improves search. Reusing the same
   function at evaluation time marks answers correct that the system did not
   find. ADR-004 keeps them as two separate functions on purpose.
3. **Ground truth is the thing most likely to be wrong.** A mistyped article
   number does not look like bad data — it looks like bad retrieval, and gets
   "fixed" in the retriever. ADR-005 machine-checks every reference before any
   question is scored.

So the eval set and its guards were built before the retriever, and the
scoreboard exists before the system does.

## Architecture

```
data/raw/<law>.pdf
     │  pdf_text.py      visual → logical order, ligatures, digit runs (ADR-013)
     ▼
  ingest.py              article-level split + extraction & structural validation
     │                   refuses the whole run rather than write a partial corpus
     ▼
data/processed/articles.jsonl     56 articles, each with a stable number
     │
     ├──► retrieve.py    BM25 baseline                       (rank-bm25)
     ├──► dense.py       multilingual-e5-base, cached        (sentence-transformers)
     ├──► fusion.py      reciprocal rank fusion, depth 20
     └──► rerank.py      cross-encoder over the fused candidates
     │
     ▼
  generate.py            citation-forced answer, local Ollama, greedy
     │
     ▼
  cite.py                audits every citation against the corpus AND against
                         the exact articles the retriever handed over.
                         No model in it — the check holds whatever wrote the text.
```

Chunks are **articles, not token windows**. A 512-token window splits an
article in half and destroys the verification key that makes any of this
scoreable. ADR-002.

## Run

```bash
python tasks.py setup                    # install dependencies
# download the law PDF by hand into data/raw/ — see data/raw/README.md
python tasks.py ingest --law "<name>"    # PDF -> article-level chunks + validation
python tasks.py verify-refs              # check every ground-truth article ref
python tasks.py eval                     # BM25 scoreboard — fast, no model needed
python tasks.py ablate                   # 4 configs: BM25/dense/hybrid/+rerank
python tasks.py answer-eval --model ollama:gemma3:4b   # end-to-end answers
python tasks.py serve                    # local test page at http://127.0.0.1:8000
python tasks.py app                      # document Q&A app (needs Ollama)
python tasks.py reindex                  # repair stored documents after an upgrade
python tasks.py test                     # the test suite
```

`tasks.py` is the canonical runner and needs nothing but Python. `make <target>`
is an equivalent alias on Linux/CI.

`eval` and `ablate` are separate on purpose: `eval` must stay runnable on a
fresh clone with no model and no torch (~0.2 s), while `ablate` loads ~1 GB of
local models. Neither calls a paid API; retrieval runs entirely on CPU.

Models are cached under `HF_HUB_CACHE` (default `~/.cache/huggingface/hub`).
Point it at a disk with room before the first run — e5-base alone is ~1.1 GB.
The demo layer's other settings are in `.env.example`.

`python tasks.py eval` runs from a fresh clone with no corpus present: it
reports the question set and an empty result column rather than failing.

### Running the app

`python tasks.py app` serves the document Q&A layer (ADR-023) on
http://127.0.0.1:8000. It answers only about documents you upload — it never
reads the ingested statute corpus, which exists for the eval harness and has
not passed its fidelity check (see **Limitations**).

```bash
ollama serve                             # in its own terminal
ollama pull gemma3:4b                    # once; ADR-024 pins this model
curl -s http://127.0.0.1:11434/api/tags  # confirm the server sees it
```

Then `curl -s -H 'X-LegalRAG: 1' http://127.0.0.1:8000/api/health` should report
`"reachable": true`. If `gpu_share` comes back low (under ~0.3), every answer
runs 3–4× slower; unload the model and let the next request reload it.

### After upgrading

The embedding cache records the format its vectors were written in, and an index
written by an older version is refused rather than trusted. That is deliberate —
ADR-026 found the old vectors were silently truncated — but it means **documents
uploaded before an upgrade stop opening**, and the app reports them as damaged,
which reads as "my file is broken" when nothing about the file changed.

```bash
python tasks.py reindex --dry-run        # which stored documents are stale
python tasks.py reindex                  # recompute them from their chunks
```

### The test page

`python tasks.py serve` opens a local page whose primary job is **running the
eval set against the current retriever and showing which questions fail**, not
demoing answers. Ad-hoc querying and corpus browsing are secondary tabs. It is a
local server, not a hosted page: the corpus is not redistributable, and the page
reads it off disk. Standard library only — nothing to install.

## Eval

**The corpus.** Egyptian statutory text, one law at a time, from a PDF
downloaded by hand and re-split here into articles. Chosen for engineering
reasons, not subject matter: the text is pre-segmented into numbered articles,
so every ground-truth answer has an objective key. A corpus without stable
references cannot be evaluated, only demoed.

**The question set — what exists today.** 20 questions, hand-written, every
article reference machine-verified against the ingested corpus. All of them are
the **dev** split; there is no held-out test split yet, so nothing here has been
validated out of sample.

| Category | Written | What it probes |
|---|---:|---|
| `direct` | 4 | single-article factual retrieval |
| `multi_article` | 3 | answers requiring two articles combined |
| `colloquial` | 8 | Egyptian-dialect phrasing against formal legal Arabic |
| `out_of_corpus` | 5 | **abstention** — the honest answer is "not in the corpus" |

The 60-question set with a locked test split is the design in [`PRD.md`](PRD.md).
It is a plan, not a state.

**Two guards worth knowing about, because both have fired.** Scoring is refused
when the ingested corpus is a different law — 151/2020 questions run against
قانون المرافعات would read as bad retrieval, not as a mismatched eval set
(ADR-010). And `ingest` refuses the entire run when any source file fails
extraction, rather than writing a partial corpus.

**Lexical overlap is capped *and* floored** (ADR-009). Too much shared wording
and BM25 wins for free; too little in a `direct` question and the lexical
baseline is crushed unfairly — that belongs in `colloquial`, where paraphrase
robustness is tested on purpose.

**A public generated QA set is deliberately NOT the eval set.**
[`fr3on/eg-legal-qa`](https://huggingface.co/datasets/fr3on/eg-legal-qa) was
generated from the article text, so its questions share wording with the
passages that answer them, and it contains no unanswerable questions. Kept as a
secondary generalization set: the *gap* between it and the hand-written set is
the interesting number. ADR-008.

## Results

Full protocol, environment and raw files: **[`EVIDENCE.md`](EVIDENCE.md)**.

> Want to see it run first? [`docs/demo/`](docs/demo/) has one screenshot of the
> local test page answering a colloquial Arabic question. It is labelled
> **Demo — not measurement evidence**, and it is kept out of this section on
> purpose: a screenshot has no command, no commit and no raw result behind it.

### Retrieval quality — 15 scored questions, k=5, dev split

| Configuration | Recall@5 | 95% CI | MRR |
|---|---:|---|---:|
| BM25 | 0.333 | 0.133 – 0.567 | 0.289 |
| **dense** (e5-base) | **0.767** | 0.567 – 0.933 | **0.588** |
| hybrid (RRF) | 0.600 | 0.400 – 0.800 | 0.530 |
| hybrid + rerank | 0.667 | 0.433 – 0.867 | 0.567 |

Candidate ceiling (Recall@20 of the hybrid stage): 0.933.

### Retrieval latency — CPU, warm, interleaved, n=45 per configuration

| Configuration | Median | p95 |
|---|---:|---:|
| BM25 | 1.1 ms | 3.1 ms |
| dense | 100.1 ms | 326.6 ms |
| hybrid (RRF) | 98.7 ms | 253.5 ms |
| hybrid + rerank | 6 369.0 ms | 11 547.3 ms |

### Generation — gemma3:4b, greedy, over dense top-5

Scored by `cite.audit`, a program: it resolves every citation against the corpus
and against the exact articles the retriever returned. No judge model.

| | |
|---|---:|
| Fully grounded | 9/15 = 60.0% |
| Cited an article not in the law | **0** |
| Cited a real article it was not given | 1 |
| Made a claim with no citation | 5 |
| Cited the expected article | 10/15 = 66.7% |
| Used the requested `[مادة N]` form | 15/15 = 100% |
| Abstained on `out_of_corpus` | 4/5 = 80.0% |
| Falsely abstained on answerable | 0/15 = 0.0% |

Against thresholds pre-registered in [`EVAL.md`](EVAL.md) before the run:
**B1 (grounding) FAIL · B2 (abstention) PASS**.

### The reranker did not help — the interesting part of this repo

**Observed.**

* Dense alone scores highest (0.767). Adding RRF fusion *lowered* it to 0.600;
  adding the cross-encoder on top brought it back to 0.667 — still under dense.
* The reranker did not run out of candidates: the hybrid stage's Recall@20 is
  0.933, so the right article was usually in the pool and was ranked out of the
  top 5.
* Per-category Recall@5 shows the two stages failing in opposite places:

  | Configuration | `direct` | `multi_article` | `colloquial` |
  |---|---:|---:|---:|
  | BM25 | 0.750 | 0.333 | **0.125** |
  | dense | 0.750 | 0.667 | **0.812** |
  | hybrid (RRF) | 0.750 | 0.500 | 0.562 |
  | hybrid + rerank | **1.000** | **0.833** | **0.438** |

* The reranker costs **6 369 ms** at the median against dense's **100 ms** —
  about 64× — for a quality difference this eval set cannot resolve.

**Hypotheses** — consistent with the numbers, *not established by them*.

* Equal-weight RRF may dilute the stronger dense signal by fusing it with a
  weaker retriever on equal terms.
* The reranker is trained on mMARCO, machine-translated. A training
  distribution that does not cover Arabic paraphrase would explain help on
  lexically-anchored questions and harm on reworded ones — which is the shape
  of the table above. It would not be the only explanation.

**Next experiments.** Tune RRF weights on a validation split — not on dev,
which is what produced these numbers. Try a different multilingual or Arabic
reranker. Extend the eval set to 60 questions and re-run before claiming any
ordering among dense, hybrid and rerank.

**The decision this supports:** ship dense. The cost is certain, the benefit is
unmeasured, and a configuration chosen on 15 overlapping questions is a
configuration chosen on noise.

## Limitations

Every limitation here carries *why* and *what would settle it*.

* **15 scored retrieval questions; 15 answerable + 5 abstention for
  generation.** The intervals overlap, so only `dense > BM25` survives them.
  One question moves generation grounding by 6.7 points and abstention by 20.
  *Why:* hand-written questions with machine-verified references, bought at the
  cost of volume. *Next:* 60 questions under the same protocol.
* **The corpus failed its fidelity check.** The ingested PDF is a third-party
  republication and 9 of 11 sampled articles differ from the gazette — whole
  different words, reproduced independently by two OCR engines, so the
  difference is in the source and not in this repository's extraction. **Every
  number here is retrieval and generation performance *on this corpus*, and
  none of them is a statement about Egyptian law.** All four configurations saw
  exactly the same corpus, so the comparison between them stands. *Why:* the
  authoritative text is a 29-page scan with no text layer. *Next:* the measured
  OCR gate in `evals/ocr/` reproduces the published 56-article structure at 0.988
  mean similarity; adopting it is an open decision, declined deliberately
  (ADR-021) because correcting misread numbers by hand buys a stronger claim
  rather than a better measurement.
* **Citations are checked mechanically, not semantically.** The audit confirms
  a citation is real and was retrieved, not that it supports the sentence it is
  attached to. ADR-022. *Next:* that needs a judge model with its own
  validation, or human adjudication.
* **One corpus, one jurisdiction, one machine.** No GPU retrieval numbers, no
  production-scale numbers, no held-out split.
* **Not suitable for legal use.** Outputs are not legal advice and are not
  validated by a lawyer.

## Reproduction steps

```bash
git clone https://github.com/mgsm20101/arabic-legal-rag && cd arabic-legal-rag
python tasks.py setup
```

**Download exactly one PDF into `data/raw/`.** The source, URL and sha256 are in
[`data/raw/SOURCE.txt`](data/raw/SOURCE.txt); the text is an Egyptian government
instrument and is not redistributed here.

> ⚠️ **One file, not two.** `ingest` reads every `.pdf` directly under
> `data/raw/` and refuses the whole run if any of them fails extraction. The
> gazette scan has no text layer, so keeping it beside the corpus breaks the
> first ingest. `sources()` is non-recursive — put reference material in
> `data/raw/reference/` and it stays on disk without being ingested.

```bash
python tasks.py ingest --law "قانون حماية البيانات الشخصية"
python tasks.py verify-refs          # every ground-truth reference re-checked
python tasks.py test                 # 890 tests, no model needed (2 skip on a
                                     # fresh clone — they read the ignored corpus)

python evals/environment.py          # regenerate the environment record
python evals/ablation_result.py      # -> evals/registry/ablation_<sha>.json
python evals/retrieval_latency.py --reps 3 --warmup 3

ollama serve && ollama pull gemma3:4b
python tasks.py answer-eval --model ollama:gemma3:4b
python evals/answer_eval_result.py   # -> evals/registry/answer_eval_<sha>.json
```

Each harness refuses to describe a dirty worktree: it runs `git status
--porcelain` and stamps `worktree_clean` into its own output, so a result
produced from an edited tree says so.

Retrieval results should reproduce exactly — the pipeline is deterministic.
Generation reproduced verdict-for-verdict across three commits here (greedy,
temperature 0), but that is an observation about this setup, not a guarantee
about yours. Latency will not reproduce: it is hardware, and `env-001` in
[`EVIDENCE.md`](EVIDENCE.md) says which.

## Repository layout

```
tasks.py             task runner — the entry point on every platform
src/legalrag/
  normalize.py       search vs evaluation normalization (ADR-004)
  pdf_text.py        logical-order Arabic out of a visually-ordered PDF (ADR-013)
  ingest.py          PDF -> article chunks + extraction & structural validation
  retrieve.py        BM25 baseline + Recall@k / MRR
  dense.py           e5-base index, cached and fingerprinted
  fusion.py          reciprocal rank fusion
  rerank.py          cross-encoder over fused candidates
  generate.py        citation-forced generation (local Ollama / transformers)
  cite.py            the citation audit — no model in it
  evaluate.py        the scoreboard
  server.py          local test page server (stdlib only)
evals/
  retrieval/         the question set — written before any retrieval code
  environment.py     -> environment.json, the environment_ref every row uses
  ablation_result.py retrieval quality, with bootstrap intervals
  retrieval_latency.py  latency, interleaved and warmed
  answer_eval_result.py generation quality, promoted out of git-ignored runs/
  registry/          tracked raw results — the only files a number may cite
EVIDENCE.md          the reproducibility registry
EVAL.md              every measured run, in order, with its pre-registrations
DECISIONS.md         ADRs with rejected alternatives and revisit conditions
LICENSES.md          models, corpus, dependencies, dependency audit
PRD.md               problem, scope, acceptance criteria
```

## Attribution

This project uses, without redistributing, work released by others:

| | | |
|---|---|---|
| `intfloat/multilingual-e5-base` | embeddings | MIT |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | reranking | Apache-2.0 |
| `gemma3:4b` via Ollama | generation | Gemma Terms of Use — use restrictions and the Prohibited Use Policy apply to whoever runs it |
| `sentence-transformers`, `rank-bm25`, `pdfplumber`, `fastapi`, `httpx`, `numpy` | the pipeline | permissive (Apache-2.0 / MIT / BSD) |

Model weights are downloaded on your machine, not shipped from here. Exact
revisions, digests, the clause-by-clause Gemma reading and the dated dependency
audit are in **[`LICENSES.md`](LICENSES.md)**.

## License

Code: MIT — see [`LICENSE`](LICENSE). The statutory text in `data/` is an
Egyptian government instrument and is **not** redistributed in this repository.
