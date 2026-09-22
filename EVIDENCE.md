# EVIDENCE — reproducibility registry

Every number this project publishes — in its README, in a CV, on a portfolio
page — has a row here. A row names the command that produced it, the commit of
the code that ran, the environment, the data, and the raw result file.

**A number without a row is not quotable.** That is the whole rule.

## How to read a row

| Field | What it means |
|---|---|
| `source_commit_sha` | The commit of the **code that produced the number** — deliberately *not* the commit that stored the result. Measure first, record second; the two are never the same commit. |
| `environment_ref` | A key into [`evals/environment.json`](evals/environment.json): machine, interpreter, dependency pin hashes, model revisions, Ollama digests. Stated once so it cannot drift between rows. |
| Raw result | A tracked file under `evals/registry/`. `runs/` is git-ignored, so nothing there can back a published number. |
| Limitations | What the number does **not** support. Filled in on every row, with *why* and *the next experiment*. |

### The protocol every measurement follows

```
1. commit the code, the config and the eval set
2. git status --porcelain  →  empty, or stop
3. source_commit_sha = HEAD
4. run the measurement at exactly that sha
5. write the raw result with that sha inside it
6. commit the result separately
```

Each harness checks step 2 itself and stamps `worktree_clean` into its output,
so a result produced from an edited tree says so in the file. It has fired in
anger: a generation run was discarded and repeated because documentation edits
landed while it was going. The re-run reproduced it exactly, which is the
outcome that makes the discipline cheap.

---

## env-001 — the environment behind every row below

| | |
|---|---|
| CPU | Intel, 12 logical cores · 15.9 GB RAM |
| GPU | **NVIDIA GeForce GTX 1050 Ti, 4 GB** — present, but invisible to torch (`+cpu` build). Retrieval and reranking run on CPU; Ollama reaches the card directly and offloads part of a generator into it. |
| Runtime | Python 3.14.6 · Windows 11 (10.0.26200) |
| Key packages | torch 2.14.0+cpu · sentence-transformers 6.0.1 · transformers 5.16.1 · numpy 2.5.1 |
| Embedding model | `intfloat/multilingual-e5-base` @ `d128750597153bb5` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` @ `1427fd652930e4ba` |
| Generator | `gemma3:4b`, Ollama digest `a2af6cc3eb7f`, Q4_K_M, 4.3B |

Regenerate with `python evals/environment.py` →
[`evals/environment.json`](evals/environment.json).

> The GPU line was wrong until 2026-09-23. The file asserted "CPU-only … no
> CUDA device present", reasoning from `torch.cuda.is_available()` — which is a
> statement about the installed wheel, not about the machine. `nvidia-smi`
> disagreed, and the generation harness had been printing `GPU share: 55%` on
> every run the whole time. Recorded here because a registry that hides its own
> corrections is worth less than one that shows them.

---

## E1 · Retrieval quality — four-configuration ablation

| | |
|---|---|
| **Command** | `python evals/ablation_result.py` |
| **source_commit_sha** | `bd04e8e8` · worktree clean |
| **environment_ref** | `env-001` |
| **Dataset** | 56 articles (Egypt Law 151/2020) · 20 questions written, **15 scored**, 5 abstention · dev split · k=5 |
| **Raw result** | [`evals/registry/ablation_bd04e8e8.json`](evals/registry/ablation_bd04e8e8.json) |
| **Date** | 2026-09-22 |

| Configuration | Recall@5 | 95% CI | MRR |
|---|---:|---|---:|
| BM25 | 0.333 | 0.133 – 0.567 | 0.289 |
| **dense** (e5-base) | **0.767** | 0.567 – 0.933 | **0.588** |
| hybrid (RRF) | 0.600 | 0.400 – 0.800 | 0.530 |
| hybrid + rerank | 0.667 | 0.433 – 0.867 | 0.567 |

Candidate ceiling — Recall@20 of the hybrid stage: **0.933**. The reranker can
reorder those candidates; it cannot exceed them.

Intervals are a percentile bootstrap over per-question recall (10 000 resamples,
seed `20260923`). Not Wilson: recall is fractional on `multi_article` questions,
so the Bernoulli assumption does not hold.

> **Limitations.** 15 scored questions. **Exactly one comparison survives the
> intervals: dense clears BM25.** Dense, hybrid and hybrid+rerank overlap almost
> completely, so this eval set cannot order them and no row here should be read
> as if it does. A 56-article corpus also puts top-5 at roughly 9% of the
> corpus, which makes the task easier than it would be at scale. *Why this
> limit:* the questions are hand-written with machine-verified article
> references, and that was bought at the cost of volume. *Next experiment:*
> extend to 60 questions under the same verification protocol, then re-run
> before making any between-configuration claim.

---

## E2 · Retrieval latency

| | |
|---|---|
| **Command** | `python evals/retrieval_latency.py --reps 3 --warmup 3` |
| **source_commit_sha** | `d5cee7ff` · worktree clean |
| **environment_ref** | `env-001` |
| **Dataset** | Same corpus and 15 scored questions · k=5 |
| **Raw result** | [`evals/registry/latency_d5cee7ff.json`](evals/registry/latency_d5cee7ff.json) |
| **Date** | 2026-09-22 |

| Configuration | Median | p95 | n |
|---|---:|---:|---:|
| BM25 | 1.1 ms | 3.1 ms | 45 |
| dense | 100.1 ms | 326.6 ms | 45 |
| hybrid (RRF) | 98.7 ms | 253.5 ms | 45 |
| **hybrid + rerank** | **6 369.0 ms** | 11 547.3 ms | 45 |

**Protocol.** CPU, single process, warm. Model load and 3 warm-up queries per
configuration excluded. 3 repetitions × 15 questions = 45 samples per
configuration. Percentile method: nearest-rank. Configurations are
**interleaved** — one question through all four before the next — so ambient
machine load is spread across them instead of landing on whichever happened to
be running.

> Interleaving is not decoration. The first version of this harness timed each
> configuration to completion and reported hybrid at **85.4 ms against dense's
> 197.8 ms** — impossible, because hybrid calls dense and then does more. The
> ordering, not the code, produced that. Both runs are kept:
> [`latency_run_aborted.log`](evals/registry/latency_run_aborted.log).

> **Limitations.** p95 over 45 samples is coarse; read it next to `n`, never
> alone. Interleaving spreads drift, it does not remove it. Dense and hybrid
> differ by 1.4 ms at the median, **below what this sample resolves** — the
> fusion step is effectively free and the cost is the dense encode. *Why this
> limit:* one machine, one process, no isolation from the desktop it runs on.
> *Next experiment:* pin CPU affinity and repeat on an idle machine if a tighter
> number is ever needed. Nothing published here depends on one.

---

## E3 · Generation quality — grounding, abstention, citation

| | |
|---|---|
| **Command** | `python tasks.py answer-eval --model ollama:gemma3:4b` then `python evals/answer_eval_result.py` |
| **source_commit_sha** | `46797bef` · worktree clean |
| **environment_ref** | `env-001` |
| **Dataset** | 20 questions — 15 answerable, 5 `out_of_corpus` · dev split · corpus fingerprint `498155cd96f12f6b` |
| **Raw judgments** | [`evals/registry/answer_rows_46797bef.json`](evals/registry/answer_rows_46797bef.json) — every answer, citation and verdict |
| **Summary** | [`evals/registry/answer_eval_46797bef.json`](evals/registry/answer_eval_46797bef.json) |
| **Date** | 2026-09-23 |

**Generation protocol.** `gemma3:4b` through a local Ollama (digest
`a2af6cc3eb7f`, Q4_K_M), greedy — temperature 0.0, seed 0 — capped at 128 new
tokens, free-text contract, over the dense retriever's top-5. The Arabic system
and user prompts are fixed in `src/legalrag/generate.py`; nothing is templated
per question beyond the retrieved articles.

**Scoring is a program, not a judge.** `legalrag.cite.audit` resolves every
citation against the corpus and against the exact article set the retriever
handed over. There is no LLM in the loop, so there is no judge model, judge
prompt or judge temperature to report, and the verdicts replay from the saved
rows with `answer-eval --report-only`.

| Metric | Result |
|---|---:|
| Answered, of 15 answerable | 15/15 |
| Fully grounded | **9/15 = 60.0%** |
| Cited an article not in the law (fabricated) | **0** |
| Cited a real article it was not given | 1 |
| Made a claim with no citation | 5 |
| Cited the article the eval set names | 10/15 = 66.7% |
| Used the requested `[مادة N]` form | 15/15 = 100% |
| Abstained on `out_of_corpus` | 4/5 = 80.0% |
| Falsely abstained on answerable | **0/15 = 0.0%** |
| Median seconds per answer | 17.9 s |

**Verdicts against thresholds pre-registered in `EVAL.md` before the run:**
B1 (grounding) **FAIL** · B2 (abstention) **PASS**.

**Observed.** Zero fabrications and 100% format compliance, against five
answers carrying no citation at all. The failure is attribution, not invention:
the model is not making up law, it is omitting the reference. One further
citation resolved to article 10, which the retriever never returned.

**Read B2 only beside the line under it.** 80.0% is 4 of 5 — exactly the
pre-registered floor, one question away from failing. It sits next to a 0%
false-abstention rate on purpose: a model that abstained on everything would
score 100% on the first line and be useless.

**Hypotheses** — consistent with these numbers, *not established by them*: the
free-text contract leaves citation optional in practice, so a schema-constrained
claims contract should move the uncited count rather than a larger model would.

> **Limitations.** 15 answerable and 5 abstention questions; one question is
> 6.7 points, and B2 moves 20 points per question. The audit checks that a
> citation is *mechanically* valid — real article, actually retrieved — not that
> it *supports* the sentence it is attached to (ADR-022). *Why this limit:*
> semantic support needs either a judge model, which would need its own
> validation, or human adjudication. *Next experiment:* run the `gated` claims
> contract on the same set and compare the uncited count, which is the one
> number the hypothesis above predicts will move.

---

## E4 · Corpus fidelity — a check this project failed, and published anyway

| | |
|---|---|
| **Method** | Manual comparison, 11 sampled articles against the gazette, reproduced independently by two OCR engines |
| **Recorded in** | [`EVAL.md`](EVAL.md) § *Corpus fidelity* · [`data/raw/SOURCE.txt`](data/raw/SOURCE.txt) · ADR-018 |
| **Date** | 2026-09-08 |
| **Result** | **FAILED — 9 of 11 sampled articles differ from the text of record** |

The ingested PDF is a third-party republication. The gazette itself is on disk
but has no text layer — 29 embedded images, 0 extractable characters — so it
cannot currently be ingested.

The differences are whole different words, not extraction noise: the gazette
reads «اثنتين وسبعين ساعة» where the republication reads «اثنين وسبعين ساعة».
Two OCR engines of different architectures reproduced the split with zero
crossover, so it is in the source, not in this repository's reconstruction.

**What this does to every number above:** they measure *retrieval and
generation performance on this corpus*. They are **not** statements about
Egyptian law, and no row here may be quoted as one. All four retrieval
configurations saw exactly the same corpus, so the comparison between them
stands.

> *Why this limit:* the authoritative source is a scan. *Next experiment:* a
> measured OCR gate — scored against digit and page-role ground truth read by
> eye, rather than judged by eye — is already built (`evals/ocr/`) and has been
> run over the whole gazette: it reproduces the published 56-article structure
> at mean similarity 0.988. Adopting it as the corpus is the open decision, and
> it was declined deliberately: correcting three misread article numbers from
> their position in the sequence is a human hand on the text of record, and at
> 0.988 the retrieval numbers would barely move. That buys a stronger *claim*,
> not a better *measurement*. ADR-019/020/021.

---

## Secondary signal

| | |
|---|---|
| **Command** | `python tasks.py test` |
| **source_commit_sha** | `6484c7d7` |
| **Result** | **890 passed**, 1 warning, 90.9 s |

A test count is not a quality metric and is not presented as one. It is here so
that the number quoted elsewhere is the *passing* count, taken from a run,
rather than the collected count taken from `--collect-only`.

---

## Not measured — and therefore not claimable

| Claim someone might expect | Status |
|---|---|
| Whether a citation *supports* the sentence it is attached to | **Not measured.** The audit is mechanical (ADR-022). |
| Cost per 1 000 queries against a hosted API | Not measured. Everything here is local; there are no API calls to price. Any cost figure would be an assumed rate multiplied by a latency, which is the latency restated. |
| Held-out test-split results | **There is no test split yet.** The eval set is 20 questions, all dev (`evals/retrieval/meta.json`). A 60-question set with a locked test split is the design in `PRD.md`; it is not built, and nothing here has been validated out of sample. |
| Anything about laws other than 151/2020 | Out of scope — one corpus, one jurisdiction. |
| GPU or production-scale retrieval latency | Not measured. Retrieval ran on CPU on one desktop. |
| `qwen2.5-coder:3b` as a generator, or tool-calling behaviour | Not measured in any run recorded here. |
