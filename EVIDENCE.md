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
| **source_commit_sha** | `618693ef` · worktree clean |
| **environment_ref** | `env-001` |
| **Dataset** | 56 articles (Egypt Law 151/2020) · 60 questions written, **30 scored**, 10 abstention · dev split · k=5 |
| **Raw result** | [`evals/registry/ablation_618693ef.json`](evals/registry/ablation_618693ef.json) |
| **Date** | 2026-09-23 |

| Configuration | Recall@5 | 95% CI | MRR | 95% CI |
|---|---:|---|---:|---|
| BM25 | 0.467 | 0.300 – 0.633 | 0.403 | 0.250 – 0.561 |
| dense (e5-base) | 0.750 | 0.617 – 0.883 | 0.562 | 0.423 – 0.699 |
| hybrid (RRF) | 0.750 | 0.600 – 0.883 | 0.591 | 0.449 – 0.726 |
| hybrid + rerank | 0.767 | 0.617 – 0.900 | 0.668 | 0.512 – 0.811 |

Candidate ceiling — Recall@20 of the hybrid stage: **0.967**. The reranker can
reorder those candidates; it cannot exceed them.

Intervals are a percentile bootstrap over per-question values (10 000 resamples,
seed `20260923`). Not Wilson: recall is fractional on `multi_article` questions,
so the Bernoulli assumption does not hold. MRR carries an interval too, because
once the recall intervals overlap MRR is the metric a decision would rest on.

**No ordering between these four is supported.** Every Recall@5 interval
overlaps every other, and so does every MRR interval — including BM25 against
dense, which was the one comparison that survived at 15 questions. Doubling the
question count did not narrow the gaps: BM25's point estimate rose faster
(0.333 → 0.467) than the intervals shrank.

#### E1b · Why the earlier run looked different — `reweighting_check_618693ef.json`

The 15-question run put **dense on top at 0.767** and hybrid+rerank fourth at
0.667, and this repository published a section titled *"The reranker did not
help"* on the strength of it. At 30 questions that ordering inverts. The
retrievers did not change:

| | direct | multi_article | colloquial |
|---|---:|---:|---:|
| hybrid + rerank @ n=15 | 1.000 | 0.833 | 0.438 |
| hybrid + rerank @ n=30 | 1.000 | 0.833 | 0.389 |

What changed is that `colloquial` — the one category the reranker is bad at —
fell from **53% of the scored set to 30%**. Taking the n=30 per-category rates
and re-weighting them by the n=15 category mix reproduces the old ordering
(dense 0.778, hybrid+rerank 0.641, BM25 0.348). The old aggregate was a
statement about the question mix, not about retrieval.

The per-category gaps are large and are where this table is informative:
BM25 scores 0.111 on `colloquial` against dense's 0.833; hybrid+rerank scores
1.000 on `direct` against BM25's 0.667.

> **Limitations.** 30 scored questions, and no aggregate comparison survives its
> interval. A 56-article corpus also puts top-5 at roughly 9% of the corpus,
> which makes the task easier than it would be at scale. These are **unpaired
> marginal intervals**, which is conservative — every configuration answers the
> same questions. *Why this limit:* the questions are hand-written with
> machine-verified article references, bought at the cost of volume. *Next
> experiment:* a paired comparison (bootstrap over per-question differences),
> which has the power these intervals lack; and the 20 held-out `test`
> questions, still unrun, to check whether any choice made on `dev` survives
> questions that never informed it.

> The 15-question run is kept at
> [`ablation_bd04e8e8.json`](evals/registry/ablation_bd04e8e8.json). It is
> superseded, not deleted — a registry that removes its own overturned rows
> cannot be used to check whether anyone was ever wrong.

---

## E2 · Retrieval latency

| | |
|---|---|
| **Command** | `python evals/retrieval_latency.py --reps 3 --warmup 3` |
| **source_commit_sha** | `170c906a` · worktree clean |
| **environment_ref** | `env-001` |
| **Dataset** | Same corpus and 30 scored questions · k=5 |
| **Raw result** | [`evals/registry/latency_318ba702.json`](evals/registry/latency_318ba702.json) |
| **Date** | 2026-09-23 |

| Configuration | Median | p95 | n |
|---|---:|---:|---:|
| BM25 | 1.5 ms | 3.9 ms | 90 |
| dense | 125.5 ms | 469.8 ms | 90 |
| hybrid (RRF) | 124.8 ms | 346.3 ms | 90 |
| **hybrid + rerank** | **7 710.0 ms** | 14 161.2 ms | 90 |

The reranker costs about **61× dense** at the median. The 15-question run put
that at 64×, so unlike the quality side, the cost side did not move when the
eval set doubled.

**Protocol.** CPU, single process, warm. Model load and 3 warm-up queries per
configuration excluded. 3 repetitions × 30 questions = 90 samples per
configuration. Percentile method: nearest-rank. Configurations are
**interleaved** — one question through all four before the next — so ambient
machine load is spread across them instead of landing on whichever happened to
be running.

> Interleaving is not decoration. The first version of this harness timed each
> configuration to completion and reported hybrid at **85.4 ms against dense's
> 197.8 ms** — impossible, because hybrid calls dense and then does more. The
> ordering, not the code, produced that. Both runs are kept:
> [`latency_run_aborted.log`](evals/registry/latency_run_aborted.log).

> **Limitations.** p95 over 90 samples is coarse; read it next to `n`, never
> alone. Interleaving spreads drift, it does not remove it. Hybrid comes out
> **0.7 ms faster than dense**, which is impossible — hybrid calls dense and
> then does more — so that gap is noise and the two are indistinguishable here:
> the fusion step is effectively free and the cost is the dense encode. *Why
> this limit:* one machine, one process, no isolation from the desktop it runs
> on. *Next experiment:* pin CPU affinity and repeat on an idle machine if a
> tighter number is ever needed. Nothing published here depends on one.

---

## E3 · Generation quality — grounding, abstention, citation

| | |
|---|---|
| **Command** | `python tasks.py answer-eval --model ollama:gemma3:4b` then `python evals/answer_eval_result.py` |
| **source_commit_sha** | `83384b2b` · worktree clean |
| **environment_ref** | `env-001` |
| **Dataset** | 40 dev questions — 30 answerable, 10 `out_of_corpus` · corpus fingerprint `498155cd96f12f6b` |
| **Raw judgments** | [`evals/registry/answer_rows_83384b2b.json`](evals/registry/answer_rows_83384b2b.json) — every answer, citation and verdict |
| **Summary** | [`evals/registry/answer_eval_83384b2b.json`](evals/registry/answer_eval_83384b2b.json) |
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

| Metric | 15 answerable (prev) | **30 answerable (now)** |
|---|---:|---:|
| Answered | 15/15 | 29/30 |
| Fully grounded | 9/15 = 60.0% | **11/29 = 37.9%** |
| Cited an article not in the law (fabricated) | **0** | **0** |
| Cited a real article it was not given | 1 | 2 |
| Made a claim with no citation | 5 | 16 |
| Cited the article the eval set names | 10/15 = 66.7% | 16/29 = 55.2% |
| Used the requested `[مادة N]` form | 15/15 = 100% | 25/29 = 86.2% |
| Abstained on `out_of_corpus` | 4/5 = 80.0% | 8/10 = 80.0% |
| Falsely abstained on answerable | 0/15 = 0.0% | 1/30 = 3.3% |

**Verdicts against thresholds pre-registered in `EVAL.md` before the run:**
B1 (grounding) **FAIL** · B2 (abstention) **PASS** — the same verdicts as the
earlier run.

**Zero fabricated citations, at twice the questions.** That is the claim this
project rests on and it did not move. The failure is attribution, not
invention: the model is not making up law, it is omitting the reference.

#### E3b · Where the 22-point grounding drop came from — `grounding_breakdown_83384b2b.json`

"The model got worse" is the reading that would be wrong.

| | n | grounded | uncited | cut by 128-token cap |
|---|---:|---:|---:|---:|
| `direct` | 12 | 7 | 4 | 0 |
| **`multi_article`** | 9 | **0** | **8** | 2 |
| `colloquial` | 9 | 5 | 4 | 3 |

The generator did not change: the 15 batch-1 questions scored **8/15** here
against **9/15** before — the same behaviour, within one answer. The whole fall
is in the questions added, and within those it is one category. The earlier run
had 3 `multi_article` questions; this one has 9, and the model grounded **none**
of them.

The 128-token cap contributes but does not explain it. All 5 answers it cut
were uncited — a truncated Arabic answer loses its trailing `[مادة N]` — but 11
of the 16 uncited answers were never truncated, and only 2 of the 9
`multi_article` failures were. Raising the cap recovers at most 5 and would not
touch the `multi_article` result.

> This is the **second** aggregate in this run that moved because the question
> mix changed rather than the system — E1b is the same shape from the retrieval
> side. Neither was visible until the numbers were split by category, which is
> the argument for recording per-category rates even when nobody asks for them.

**Read B2 only beside the line under it.** 80.0% is 8 of 10 — exactly the
pre-registered floor again. It sits next to the false-abstention rate on
purpose: a model that abstained on everything would score 100% on the first
line and be useless. That rate is no longer zero (1/30), which is the cost of
the abstention behaviour becoming visible at a larger sample.

**Hypotheses** — consistent with these numbers, *not established by them*: the
free-text contract leaves citation optional in practice, and `multi_article`
questions are where that bites hardest because the answer has to attribute
several claims rather than one.

> **Limitations.** 30 answerable and 10 abstention questions; B2 moves 10 points
> per question. The audit checks that a citation is *mechanically* valid — real
> article, actually retrieved — not that it *supports* the sentence it is
> attached to (ADR-022). *Why this limit:* semantic support needs either a judge
> model, which would need its own validation, or human adjudication. *Next
> experiments:* raise `max_new_tokens` to separate the cap's contribution from
> the model's; then run the `gated` claims contract on `multi_article`
> specifically — an answer that must name its articles before writing prose is
> the intervention aimed at exactly this failure.

> **Timing is not comparable between the two runs and neither is a benchmark.**
> GPU share was **9%** here against **55%** before — another process held the
> 4 GB card — so mean 38.7 s/answer against the earlier 17.9 s median says
> nothing about the model. Quantization (`Q4_K_M`) and digest (`a2af6cc3eb7f`)
> are identical, so the quality numbers above are unaffected. Generation latency
> on a shared desktop is not measured here; E2 is the row with a real timing
> protocol behind it.

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
| **source_commit_sha** | `83384b2b` |
| **Result** | **911 passed** locally · **909 passed, 2 skipped** from a clone of this repository |

A test count is not a quality metric and is not presented as one. It is here so
that the number quoted elsewhere is the *passing* count, taken from a run,
rather than the collected count taken from `--collect-only`.

The two numbers differ for a reason worth stating: the statute corpus is
git-ignored, so the two tests that read it skip on any machine that has not run
`ingest`. CI reports the same 909/2. A reader who clones this and sees 909
should see it explained here rather than wonder which number was inflated.

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
