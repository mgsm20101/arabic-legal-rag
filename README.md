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
> [`portfolio-v6`](https://github.com/mgsm20101/arabic-legal-rag/tree/portfolio-v6)
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

## Structure

Two products share one core. The **app** answers questions about documents you
upload. The **statute bench** measures retrieval and generation on one ingested
law and produces every number under Results. They never read each other's data.

### Entry points

Everything runs through `python tasks.py <command>` from the repo root.
`python tasks.py --help` lists them.

| Command | What it is | Reads | Writes |
|---|---|---|---|
| `app` | **The app**: upload a document, ask it questions (needs Ollama) | `ui/app.html`, `ui/app.js`, `ui/js/`, `data/app/` | `data/app/docs/<doc_id>/` |
| `reindex` | Recompute stored documents' embeddings after an upgrade | `data/app/docs/` | `data/app/docs/*/embeddings.npz` |
| `app-eval` | Score the app's upload pipeline on the synthetic `app-dev` set | `evals/app/` | `runs/app_eval-*.json` |
| `ingest --law "…"` | Statute PDF to article-level chunks, validated | `data/raw/*.pdf` | `data/processed/articles.jsonl` |
| `verify-refs [--write]` | Check every ground-truth article reference | `evals/retrieval/questions.jsonl`, corpus | stdout (`--write`: the questions file) |
| `eval` | BM25 scoreboard, no model | questions, corpus | stdout |
| `ablate` | BM25 / dense / hybrid / hybrid + rerank compared | questions, corpus, local models | stdout, `data/processed/embeddings.npz` |
| `adversarial` | Named failure-mode probes (not a score) | `evals/adversarial/` | stdout |
| `answer-eval` | End-to-end answers: citations and abstention | questions, corpus, Ollama | `runs/answer_eval*.json` |
| `all --law "…"` | `ingest`, then `verify-refs`, then `eval` | as above | as above |
| `broken-words` | Find words split by a stray space in the corpus | corpus | stdout |
| `ocr-gate <json>` | Score an OCR engine's digit accuracy | OCR JSON, `evals/ocr/` | stdout |
| `ocr-to-raw <in> <out>` | OCR pages to raw text that `ingest` can read | OCR JSON | `<out>` |
| `serve` | **Legacy** benchmark test page (BM25, stdlib server) | `ui/index.html`, questions, corpus | nothing |
| `test` | The test suite | `tests/` | nothing |
| `setup` | Install pinned dependencies | `requirements-dev.txt`, `constraints.txt` | the environment |

Registry results come from the scripts in `evals/` (below), not from `tasks.py`.

### Request flow

**`POST /api/chat`** (question in, JSON answer out):

1. `web_guard.py` `Guard`: host, `X-LegalRAG` header, rate limits, before any route.
2. `webapp.py` `_add_chat_route.chat`: validates the body (`ChatRequest`).
3. `pipeline.py` `Pipeline.ask`: rejects a too-short or too-long question.
4. `library.py` `Library.search`: `dense.py` `encode_query` once, then each live document's
   `DenseIndex.search_with_vector` (loaded by `docindex.py` `load_index`); top 5 overall.
5. `pipeline.py` `Pipeline._generate`: one generation at a time (`TooBusy` otherwise), calling
   `claims.py` `ClaimsGenerator.answer`: a relevance check, then JSON claims, both through
   `ollama.py` `OllamaChat`.
6. `cite.py` `gate`: drops every claim whose citation is not among the retrieved sources or
   not grounded in them.
7. `claims.py` `final_abstain_reason`, then `pipeline.py` `ChatResult.to_dict` is the response.
   Anticipated errors become HTTP codes in `webapp.py` `_add_error_handlers`.

**`POST /api/documents`** (upload):

1. `webapp.py` `_add_document_routes.upload` calls `library.py` `Library.add`: size, suffix and
   PDF checks first; identical bytes return the existing document.
2. PDF: `docextract.py` `extract_pdf` runs `pdf_text.py` `extract_pages` in a killable
   subprocess (`isolate.py` `run_isolated`), then `chunking.py` cuts statute articles if the
   text validates as a statute, page chunks otherwise. Text files go to `chunk_document`.
3. `docstore.py` writes `chunks.jsonl` and `meta.json`, `dense.py` `DenseIndex.save` writes
   `embeddings.npz`, all in a staging directory moved into `data/app/docs/<doc_id>/` at the end.

### Code map

`src/legalrag/`, 36 modules. The app uses dense retrieval only; BM25, fusion and
rerank exist only in the bench.

**App**

| Module | Role |
|---|---|
| `webapp.py` | FastAPI app, routes, error-to-HTTP mapping, `main()` for `tasks.py app` |
| `web_guard.py` | ASGI middleware: host and header checks, rate limits, security headers |
| `pipeline.py` | `Pipeline.ask`: one question from retrieval to a gated answer |
| `library.py` | The on-disk document library: add, search, soft delete |
| `docextract.py` | A stored PDF's pages and chunks, extracted in a subprocess |
| `isolate.py` | Run a callable in its own process with a hard timeout |
| `docstore.py` | One stored document's files, atomic JSON writes |
| `docindex.py` | A document's dense index and the shared lazily loaded encoder |
| `docerrors.py` | Every error the library raises, each with a stable code |
| `reindex.py` | `tasks.py reindex`: rebuild stale embeddings |
| `app_eval.py` | `tasks.py app-eval`: the upload pipeline on the `app-dev` split |

**Shared core**

| Module | Role |
|---|---|
| `normalize.py` | Search normalization vs evaluation normalization, kept apart |
| `pdf_text.py` | Logical-order Arabic from a visually ordered PDF |
| `chunking.py` | Article chunks for statutes, page chunks for everything else |
| `dense.py` | multilingual-e5-base index, cached and fingerprinted |
| `cite.py` | Citation checks with no model in them: `audit` (bench) and `gate` (app) |
| `claims.py` | The claims contract: relevance step, JSON claims, abstain reason |
| `ollama.py` | Local Ollama client and health probe |
| `generate.py` | Model-spec parsing and loading, used by both products. Its `Generator` class (text and JSON contracts, `transformers` path) is legacy, kept to reproduce E3 |

**Statute bench**

| Module | Role |
|---|---|
| `ingest.py` | `tasks.py ingest`: PDF to validated article chunks |
| `verify_refs.py` | `tasks.py verify-refs`: every ground-truth reference re-checked |
| `overlap.py` | Lexical-overlap guard for the question set |
| `evaluate.py` | `tasks.py eval`: question loading, split handling, BM25 scoreboard |
| `retrieve.py` | BM25 baseline, Recall@k, MRR |
| `fusion.py` | Reciprocal rank fusion (ablation arm only) |
| `rerank.py` | Cross-encoder reranking (ablation arm only) |
| `ablate.py` | `tasks.py ablate`: the four retrieval configurations |
| `adversarial.py` | `tasks.py adversarial`: named failure-mode probes |
| `answer_eval.py` | `tasks.py answer-eval`: end-to-end answers over dense top-5. The `text` and `json` contracts are legacy, kept to reproduce E3; `gated` is what the app runs |
| `answer_report.py` | Console reports for `answer-eval` |
| `envcheck.py` | Refuses to stamp an environment record the interpreter does not match |
| `provenance.py` | The commit and worktree state stamped on every result file |
| `server.py` | `tasks.py serve` (legacy, benchmark test page) |

**Tools**

| Module | Role |
|---|---|
| `broken_words.py` | `tasks.py broken-words`: split-word detector for the corpus |
| `ocr_gate.py` | `tasks.py ocr-gate`: digit-accuracy gate for an OCR engine |
| `ocr_text.py` | `tasks.py ocr-to-raw`: OCR output to raw text for `ingest` |
| `__init__.py` | Package version |

### Other folders

| Path | What is in it |
|---|---|
| `ui/` | `app.html`, `app.css`, `app.js`, `js/`: the app's page. `index.html`: the legacy `serve` page |
| `tests/` | The test suite; no model or network needed |
| `evals/retrieval/`, `evals/adversarial/`, `evals/app/`, `evals/ocr/` | Question sets and ground truth, each with a `meta.json` |
| `evals/*.py` | Registry harnesses: each stamps commit and environment and writes `evals/registry/*.json`. `ablation_result.py`, `retrieval_latency.py`, `answer_eval_result.py`, `reweighting_check.py`, `grounding_breakdown.py`, `generation_repeat.py`; `generation_variance.py` and `placement_probe.py` are targeted experiments behind E3c–E3e; `environment.py` writes `evals/environment.json` |
| `evals/app/coldstart_check.py` | One-off reproduction against a running container; writes `coldstart_result.json` beside it |
| `evals/registry/` | Tracked raw results: the only files a number may cite |
| `scripts/make_constraints.py` | Regenerates `constraints.txt` |
| `docs/demo/` | One screenshot, not evidence |
| `docs/internal/` | Working notes: `PRD.md`, `PROJECT-STATE.md`, and a point-in-time review (`README.md` there) |
| `data/raw/`, `data/processed/`, `data/app/` | Statute PDF, ingested corpus, uploaded documents. Git-ignored except `data/raw/README.md` and `data/raw/SOURCE.txt` |

`EVIDENCE.md` binds each number to its command, commit and raw file; `EVAL.md`
records every run in order; `DECISIONS.md` holds the ADRs; `LICENSES.md` covers
models, corpus and dependencies.

### Read the code in this order

1. `tasks.py`: every entry point in one table.
2. `webapp.py` `create_app`, then `pipeline.py` `Pipeline.ask`: the app's whole round trip.
3. `library.py` `Library.add` and `Library.search`, then `dense.py` `DenseIndex`.
4. `claims.py` `ClaimsGenerator.answer`, then `cite.py` `gate`.
5. For the bench: `ingest.py`, `evaluate.py`, `ablate.py`, `answer_eval.py`, then `EVIDENCE.md`.

## Architecture

**App** (`python tasks.py app`, dense retrieval only)

```
upload ──► webapp ──► library.add
                        │  docextract: pdf_text in a killable subprocess
                        │  chunking:   article chunks if it validates as a statute, page chunks otherwise
                        ▼
              data/app/docs/<doc_id>/  source · chunks.jsonl · meta.json · embeddings.npz

question ─► webapp ──► pipeline.ask
                        │  library.search   dense (multilingual-e5-base), top 5 across documents
                        ▼
                      claims              relevance check, then JSON claims (local Ollama, greedy)
                        ▼
                      cite.gate           drops every claim citing a source it was not handed
                        ▼
                      JSON: claims · sources · abstain_reason · timings
```

**Statute bench** (`ingest`, `verify-refs`, `eval`, `ablate`, `answer-eval`)

```
data/raw/<law>.pdf
     │  pdf_text.py      visual → logical order, ligatures, digit runs (ADR-013)
     ▼
  ingest.py              article-level split + extraction & structural validation
     │                   refuses the whole run rather than write a partial corpus
     ▼
data/processed/articles.jsonl     56 articles, each with a stable number
     │
     ├──► evaluate.py    BM25 scoreboard (retrieve.py, rank-bm25)
     ├──► ablate.py      BM25 · dense · hybrid (fusion.py, RRF depth 20) · hybrid + rerank.py
     │
     ▼
  answer_eval.py         dense top-5 → claims.py / generate.py → local Ollama, greedy
     │
     ▼
  cite.py                audits every citation against the corpus AND against
     │                   the exact articles the retriever handed over.
     │                   No model in it — the check holds whatever wrote the text.
     ▼
runs/*.json ──► evals/*.py ──► evals/registry/*.json ──► EVIDENCE.md
```

**Shared core:** `normalize`, `pdf_text`, `chunking`, `dense`, `cite`, and the
generation clients `claims`, `ollama`, `generate`.

In the bench, chunks are **articles, not token windows**. A 512-token window
splits an article in half and destroys the verification key that makes any of
this scoreable. ADR-002.

## Run

```bash
python tasks.py setup                    # install dependencies
# download the law PDF by hand into data/raw/ — see data/raw/README.md
python tasks.py ingest --law "<name>"    # PDF -> article-level chunks + validation
python tasks.py verify-refs              # check every ground-truth article ref
python tasks.py eval                     # BM25 scoreboard — fast, no model needed
python tasks.py ablate                   # 4 configs: BM25/dense/hybrid/+rerank
python tasks.py answer-eval --model ollama:gemma3:4b   # end-to-end answers
python tasks.py app                      # document Q&A app (needs Ollama)
python tasks.py reindex                  # repair stored documents after an upgrade
python tasks.py test                     # the test suite
```

`tasks.py` is the only runner and needs nothing but Python. The full command
list is under **Structure**.

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

### The legacy test page

Kept for the benchmark, not part of the app. `python tasks.py serve` opens a local page whose primary job is **running the
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

**The question set.** 60 questions, hand-written, every article reference
machine-verified against the ingested corpus.

| Category | Total | dev | test | What it probes |
|---|---:|---:|---:|---|
| `direct` | 20 | 12 | 8 | single-article factual retrieval |
| `multi_article` | 15 | 9 | 6 | answers requiring two articles combined |
| `colloquial` | 10 | 9 | 1 | Egyptian-dialect phrasing against formal legal Arabic |
| `out_of_corpus` | 15 | 10 | 5 | **abstention** — the honest answer is "not in the corpus" |

**20 of them are held out and have never been run.** `load_questions` returns
`dev` unless a caller names the split, so forgetting to exclude the test rows
is not a way to include them — a split kept by convention is not held out. They
exist to answer one question later: does a configuration chosen on `dev`
survive questions that never informed it? The original 20 cannot serve that
role, because they have already informed every choice in this repository.

Questions 21–66 were written **with the corpus text visible**, which is weaker
provenance than the first 20 (paraphrased from secondary sources). The
containment guard is what makes that tolerable, and it proved the point
immediately: 22 of the first 40 drafts failed it — 19 above band, where a
question quotes its own answer and hands BM25 a match it did not earn, and 3
below band, where a question has no lexical anchor at all and crushes the
lexical baseline just as unfairly. All were rewritten and re-measured before
anything was committed. `evals/retrieval/meta.json` records this rather than
hiding it.
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

### Retrieval quality — 30 scored questions, k=5, dev split

| Configuration | Recall@5 | 95% CI | MRR | 95% CI |
|---|---:|---|---:|---|
| BM25 | 0.467 | 0.300 – 0.633 | 0.403 | 0.250 – 0.561 |
| dense (e5-base) | 0.750 | 0.617 – 0.883 | 0.562 | 0.423 – 0.699 |
| hybrid (RRF) | 0.750 | 0.600 – 0.883 | 0.591 | 0.449 – 0.726 |
| hybrid + rerank | 0.767 | 0.617 – 0.900 | 0.668 | 0.512 – 0.811 |

Candidate ceiling (Recall@20 of the hybrid stage): 0.967.

**Every interval overlaps every other.** No ordering between these four is
supported by this eval set — not on Recall@5 and not on MRR. Read the section
below before quoting any row.

### Retrieval latency — CPU, warm, interleaved, n=90 per configuration

| Configuration | Median | p95 |
|---|---:|---:|
| BM25 | 1.3 ms | 3.3 ms |
| dense | 145.1 ms | 331.7 ms |
| hybrid (RRF) | 136.3 ms | 311.6 ms |
| hybrid + rerank | 7 994.2 ms | 13 137.8 ms |

The reranker costs about **55× dense** at the median. Earlier runs on the same
machine gave 61× and 64×: the multiple drifts by a few between runs on an
unisolated desktop, the order of magnitude does not. Read it as roughly 55–65×.

### Generation — gemma3:4b, greedy, over dense top-5

Scored by `cite.audit`, a program: it resolves every citation against the corpus
and against the exact articles the retriever returned. No judge model.

Same code, same model digest, greedy decoding, 30 answerable + 10
out-of-corpus questions — at the two GPU placements measured end to end:

| | 34 of 35 layers on GPU | 2 of 35 layers on GPU |
|---|---:|---:|
| Fully grounded (of answered) | 12/30 = 40.0% | 11/29 = 37.9% |
| Cited an article not in the law | **0** | **0** |
| Cited a real article it was not given | 3 | 2 |
| Made a claim with no citation | 15 | 16 |
| Cited the expected article | 14/30 = 46.7% | 16/29 = 55.2% |
| Abstained on `out_of_corpus` | 7/10 = 70.0% | 8/10 = 80.0% |
| Falsely abstained on answerable | 0/30 = 0.0% | 1/30 = 3.3% |
| B1 grounding (pre-registered) | FAIL | FAIL |
| B2 abstention ≥ 80% (pre-registered) | **FAIL** | **PASS** |

**Zero fabricated citations in every run — 21 runs, six distinct sets of
answers, five GPU placements.** That is the claim that holds at any placement. The failure is
attribution, not invention: the model omits the reference; it does not invent law.

**The abstention verdict depends on where the model runs.** Ten runs, each after
a full Ollama restart, produced identical text: Ollama put 34 of 35 layers on
the GPU every time. The one run that differed had recorded a 9% GPU share —
another program held the card. Pinning 2 layers, the placement that measures
closest to 9%, reproduced that run's text on 37 of 40 questions (against 14/40
at 34 layers) and every one of its verdicts, B2 PASS included. So the output is
deterministic *per placement*, and Ollama picks the placement from the VRAM
other programs leave free.

Measured at five placements, the pre-registered abstention verdict is FAIL at
0 layers, PASS at 2, 8 and 17, and FAIL at 34 — it changes twice, so it is
stated only with its layer count. Every placement was deterministic across
restarts. Read abstention as *70–80% at 0–7% false abstention, depending on
placement* — never as a pass. The layer count is now recorded with every run.
[`EVIDENCE.md`](EVIDENCE.md) E3c–E3e, pre-registered as Runs 13, 13b and 13c in
[`EVAL.md`](EVAL.md).

### An eval set overturned this repo's own headline — the interesting part

This section used to be titled **"The reranker did not help."** It was wrong,
and it is worth keeping the correction visible rather than quietly editing it.

At 15 scored questions, dense led at 0.767 and hybrid+rerank came fourth at
0.667. That ordering inverted at 30. But the retrievers did not change:

| | `direct` | `multi_article` | `colloquial` |
|---|---:|---:|---:|
| hybrid + rerank @ n=15 | 1.000 | 0.833 | 0.438 |
| hybrid + rerank @ n=30 | 1.000 | 0.833 | **0.389** |

What changed is that `colloquial` — the one category the reranker is bad at —
fell from **53% of the scored set to 30%**. The old set was badly unbalanced,
and the aggregate was reporting that imbalance.

**The check.** Take the n=30 per-category rates and re-weight them by the n=15
category mix. If composition explains the reversal, the old ordering should
come back:

| Configuration | measured @ n=15 | measured @ n=30 | n=30 rates, n=15 mix |
|---|---:|---:|---:|
| dense | 0.767 | 0.750 | **0.778** |
| hybrid (RRF) | 0.600 | 0.750 | 0.704 |
| hybrid + rerank | 0.667 | **0.767** | **0.641** |
| BM25 | 0.333 | 0.467 | 0.348 |

It does at both ends — dense first, BM25 last — and **not in the middle**:
re-weighted hybrid (0.704) lands above hybrid+rerank (0.641), where the
15-question run had them the other way round. So the mix explains dense losing
first place, not every position. `python evals/reweighting_check.py` prints the
match position by position:
[`reweighting_check_4fe53a93.json`](evals/registry/reweighting_check_4fe53a93.json).

**What the numbers now support.** Per-category, the two stages still fail in
opposite places, and those gaps are large enough to be worth something:

| Configuration | `direct` | `multi_article` | `colloquial` |
|---|---:|---:|---:|
| BM25 | 0.667 | 0.556 | **0.111** |
| dense | 0.750 | 0.667 | **0.833** |
| hybrid (RRF) | 0.833 | 0.778 | 0.611 |
| hybrid + rerank | **1.000** | **0.833** | 0.389 |

**What they do not support: any aggregate ordering at all.** Every interval
overlaps every other, on both metrics. Even `dense > BM25`, the single
comparison that survived at 15 questions, no longer does — BM25's point estimate
rose faster (0.333 → 0.467) than the intervals shrank.

**Hypotheses** — consistent with the numbers, *not established by them*.

* The reranker is trained on mMARCO, machine-translated. A training
  distribution that does not cover Arabic paraphrase would explain help on
  lexically-anchored questions and harm on reworded ones — which is the shape
  of the per-category table, in both runs.
* Equal-weight RRF may dilute the stronger dense signal by fusing it with a
  weaker retriever on equal terms.

**Next experiments.** A paired comparison — bootstrap over per-question
differences rather than four marginal intervals — which has the power these
lack, since every configuration answers the same questions. Then the 20
held-out `test` questions, still unrun. Then RRF weights tuned on a validation
split, not on dev.

**The decision this supports:** ship dense, but for a different reason than
before. It is no longer "the reranker is worse" — that claim is dead. It is
that the aggregate difference is unresolvable while the cost is certain at
**roughly 55–65×**, and that the reranker's advantage is concentrated in `direct`
questions while its deficit is concentrated in `colloquial` ones, which is the
register real users actually write in. Routing to the reranker by question
shape is the obvious follow-up and has not been tried.

## Limitations

Every limitation here carries *why* and *what would settle it*.

* **30 scored retrieval questions; 30 answerable + 10 abstention for
  generation.** Every interval overlaps every other, so **no** configuration
  ordering survives them — including `dense > BM25`, which did survive at 15.
  Abstention still moves 10 points per question. *Why:* hand-written questions
  with machine-verified references, bought at the cost of volume. *Next:* a
  paired comparison, which has the power four marginal intervals lack; then the
  20 held-out questions.
* **Generation depends on GPU placement.** Identical across ten restarts at a
  fixed placement; different text and a different abstention verdict at
  another (E3d). Five of 36 placements are measured; the abstention verdict
  changes twice across them and is unmeasured between them (E3e). The remaining
  3/40 difference from the old run is unexplained (its Ollama version was not
  recorded).
* **Generation latency is not measured.** Runs recorded 17.9–38.7 s per answer
  for the same model at the same quantization, depending on GPU share on a
  shared 4 GB card. None is a benchmark and none is quoted as one. Retrieval latency (E2) is the row
  with a real timing protocol behind it. *Next:* an idle machine, if a
  generation number is ever needed.
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
  production-scale numbers, and the 20-question held-out split is still unrun.
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
python tasks.py test                 # 956 tests, no model needed (2 skip on a
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
