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
landed while it was going. The re-run matched it — inside the same Ollama server
session, and it does across restarts too — as long as Ollama puts the same
number of layers on the GPU (E3d). A different placement produces different
text; the layer count is now recorded with every run.

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

## Corrections to this registry — 2026-09-23

Three things were wrong with the rows below as they were first published, and
all of them were found by re-running rather than by reading.

1. **Their commits could not be opened.** E1, E2 and E3 named
   `618693ef`, `318ba702` and `83384b2b`. Those were real commits when the
   measurements ran, and were then rewritten before they were pushed, so none
   of them is on `origin/main`. E2's row named `170c906a`, which exists
   nowhere. A row whose commit cannot be checked out is a claim that only looks
   documented. Every row now names a commit on `origin/main`; the old result
   files stay in `evals/registry/`, read for their history only.
2. **"env-001" was a string, not an observation.** The harnesses wrote it into
   every result unconditionally. A re-run started from a shell whose `python`
   was a different interpreter — Python 3.12, CUDA torch that could see the GPU
   — was stamped env-001 like any other. Retrieval quality came out identical;
   latency came out 5–11× faster. None of it was pushed. `legalrag.envcheck`
   now compares the running interpreter with `evals/environment.json`, stops
   before any model loads on a mismatch, and writes what actually ran into each
   result as `runtime_observed`.
3. **"The re-run reproduced it exactly" hid a condition.** The same model
   digest under greedy decoding produced different text on 26 of 40 questions
   between two runs (E3c). Ten further runs across restarts (E3d) showed it is
   not the restart: it is how many layers Ollama placed on the GPU, which it
   decides from the VRAM other programs leave free. Output is identical at a
   fixed placement and different across placements.

Every row below was re-measured on 2026-09-23 at a commit on `origin/main`,
with the environment checked rather than asserted.

---

## E1 · Retrieval quality — four-configuration ablation

| | |
|---|---|
| **Command** | `python evals/ablation_result.py` |
| **source_commit_sha** | `4fe53a93` · worktree clean |
| **environment_ref** | `env-001` — checked by the harness, recorded in the file as `runtime_observed` |
| **Dataset** | 56 articles (Egypt Law 151/2020) · 60 questions written — 40 `dev`, 20 `test` held out and unrun · **30 scored**, 10 abstention · k=5 |
| **Raw result** | [`evals/registry/ablation_4fe53a93.json`](evals/registry/ablation_4fe53a93.json) |
| **Date** | 2026-09-23 |

| Configuration | Recall@5 | 95% CI | MRR | 95% CI |
|---|---:|---|---:|---|
| BM25 | 0.467 | 0.300 – 0.633 | 0.403 | 0.250 – 0.561 |
| dense (e5-base) | 0.750 | 0.617 – 0.883 | 0.562 | 0.423 – 0.699 |
| hybrid (RRF) | 0.750 | 0.600 – 0.883 | 0.591 | 0.449 – 0.726 |
| hybrid + rerank | 0.767 | 0.617 – 0.900 | 0.668 | 0.512 – 0.811 |

Candidate ceiling — Recall@20 of the hybrid stage: **0.967**. The reranker can
reorder those candidates; it cannot exceed them.

This is the third time these numbers have been produced — at `618693ef`, on the
wrong interpreter, and here — and they have been identical every time.
Retrieval is deterministic; that is what lets a registry row stand for it.

Intervals are a percentile bootstrap over per-question values (10 000 resamples,
seed `20260923`). Not Wilson: recall is fractional on `multi_article` questions,
so the Bernoulli assumption does not hold. MRR carries an interval too, because
once the recall intervals overlap MRR is the metric a decision would rest on.

**No ordering between these four is supported.** Every Recall@5 interval
overlaps every other, and so does every MRR interval — including BM25 against
dense, which was the one comparison that survived at 15 questions. Doubling the
question count did not narrow the gaps: BM25's point estimate rose faster
(0.333 → 0.467) than the intervals shrank.

Per category the gaps are large, and this is where the table is informative:

| | direct (12) | multi_article (9) | colloquial (9) |
|---|---:|---:|---:|
| BM25 | 0.667 | 0.556 | 0.111 |
| dense | 0.750 | 0.667 | **0.833** |
| hybrid (RRF) | 0.833 | 0.778 | 0.611 |
| hybrid + rerank | **1.000** | 0.833 | 0.389 |

#### E1b · Why the 15-question run looked different — `reweighting_check_4fe53a93.json`

| | |
|---|---|
| **Command** | `python evals/reweighting_check.py` — reads two ablation files, no model |
| **source_commit_sha** | `4fe53a93` · worktree clean |
| **Raw result** | [`evals/registry/reweighting_check_4fe53a93.json`](evals/registry/reweighting_check_4fe53a93.json) |

The 15-question run put **dense on top at 0.767** and hybrid+rerank third at
0.667, and this repository published a section titled *"The reranker did not
help"* on the strength of it. At 30 questions that ordering inverts. The
retrievers did not change — hybrid+rerank scores 1.000 / 0.833 on `direct` /
`multi_article` in both runs, and 0.438 → 0.389 on `colloquial`.

What changed is that `colloquial` — the one category the reranker is bad at —
fell from **53% of the scored set to 30%**. The script takes the n=30
per-category rates, re-weights them by the n=15 category mix (both mixes
computed from `questions.jsonl`, not typed), and reports every rank position:

| Position | 15-question run | n=30 rates at the n=15 mix | |
|---|---|---|---|
| 1 | dense 0.767 | dense 0.778 | match |
| 2 | hybrid + rerank 0.667 | hybrid (RRF) 0.704 | **differs** |
| 3 | hybrid (RRF) 0.600 | hybrid + rerank 0.641 | **differs** |
| 4 | BM25 0.333 | BM25 0.348 | match |

So the mix explains the headline reversal — dense losing first place — and
both ends, but not the middle, and at these interval widths the middle two were
never separable anyway.

> **Correction.** This row previously said the re-weighting "reproduces the old
> ordering". It restores both ends and not the middle. The first version of the
> check was computed by hand with no command; the script replaces it and prints
> the position-by-position match so the sentence cannot be written again
> without the file contradicting it.

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

> Superseded, kept: the 15-question run
> [`ablation_bd04e8e8.json`](evals/registry/ablation_bd04e8e8.json); the first
> 30-question run [`ablation_618693ef.json`](evals/registry/ablation_618693ef.json)
> (unreachable SHA, and a `reading` field written before the harness's prose was
> corrected); [`reweighting_check_618693ef.json`](evals/registry/reweighting_check_618693ef.json)
> (hand-computed). A registry that removes its overturned rows cannot be used
> to check whether anyone was ever wrong.

---

## E2 · Retrieval latency

| | |
|---|---|
| **Command** | `python evals/retrieval_latency.py --reps 3 --warmup 3` — Ollama stopped |
| **source_commit_sha** | `4fe53a93` · worktree clean |
| **environment_ref** | `env-001` — checked; `torch_sees_gpu: false` in `runtime_observed` |
| **Dataset** | Same corpus and 30 scored questions · k=5 |
| **Raw result** | [`evals/registry/latency_4fe53a93.json`](evals/registry/latency_4fe53a93.json) |
| **Date** | 2026-09-23 |

| Configuration | Median | p95 | n |
|---|---:|---:|---:|
| BM25 | 1.3 ms | 3.3 ms | 90 |
| dense | 145.1 ms | 331.7 ms | 90 |
| hybrid (RRF) | 136.3 ms | 311.6 ms | 90 |
| **hybrid + rerank** | **7 994.2 ms** | 13 137.8 ms | 90 |

The reranker costs about **55× dense** at the median. Earlier runs on the same
machine and interpreter put it at 64× (15 questions) and 61× (30 questions,
unreachable SHA, [`latency_318ba702.json`](evals/registry/latency_318ba702.json)).
The ratio moves by a few multiples between runs of an unisolated desktop; the
order of magnitude does not. Quote it as "roughly 55–65×", not as one number.

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
> **8.8 ms faster than dense**, which is impossible — hybrid calls dense and
> then does more — so that gap is noise and the two are indistinguishable here:
> the fusion step is effectively free and the cost is the dense encode. *Why
> this limit:* one machine, one process, no isolation from the desktop it runs
> on. *Next experiment:* pin CPU affinity and repeat on an idle machine if a
> tighter number is ever needed. Nothing published here depends on one.

---

## E3 · Generation quality — grounding, abstention, citation

| | |
|---|---|
| **Command** | `python tasks.py answer-eval --model ollama:gemma3:4b --overwrite` then `python evals/answer_eval_result.py` |
| **source_commit_sha** | `4fe53a93` · worktree clean |
| **environment_ref** | `env-001` — the run's own metadata records its interpreter; the promoter refuses one that differs |
| **Dataset** | 40 dev questions — 30 answerable, 10 `out_of_corpus` · corpus fingerprint `498155cd96f12f6b` |
| **Raw judgments** | [`evals/registry/answer_rows_4fe53a93.json`](evals/registry/answer_rows_4fe53a93.json) — every answer, citation and verdict |
| **Summary** | [`evals/registry/answer_eval_4fe53a93.json`](evals/registry/answer_eval_4fe53a93.json) |
| **Date** | 2026-09-23 |

**Generation protocol.** `gemma3:4b` through a local Ollama 0.20.3 (digest
`a2af6cc3eb7f`, Q4_K_M, GPU share 55% on the GTX 1050 Ti), greedy — temperature
0.0, seed 0 — capped at 128 new tokens, free-text contract, over the dense
retriever's top-5. The Arabic system and user prompts are fixed in
`src/legalrag/generate.py`; nothing is templated per question beyond the
retrieved articles.

**Scoring is a program, not a judge.** `legalrag.cite.audit` resolves every
citation against the corpus and against the exact article set the retriever
handed over. There is no LLM in the loop, so there is no judge model, judge
prompt or judge temperature to report, and the verdicts replay from the saved
rows with `answer-eval --report-only`.

| Metric | this run (`4fe53a93`) | earlier run (`83384b2b`) |
|---|---:|---:|
| Answered | 30/30 | 29/30 |
| Fully grounded (of answered) | **12/30 = 40.0%** | 11/29 = 37.9% |
| Cited an article not in the law (fabricated) | **0** | **0** |
| Cited a real article it was not given | 3 | 2 |
| Made a claim with no citation | 15 | 16 |
| Cited the article the eval set names | 14/30 = 46.7% | 16/29 = 55.2% |
| Used the requested `[مادة N]` form | 30/30 = 100% | 25/29 = 86.2% |
| Abstained on `out_of_corpus` | **7/10 = 70.0%** | 8/10 = 80.0% |
| Falsely abstained on answerable | 0/30 = 0.0% | 1/30 = 3.3% |
| **B1** grounding (pre-registered) | **FAIL** | FAIL |
| **B2** abstention ≥ 80% (pre-registered) | **FAIL** | PASS |

**The one claim that holds in every run: zero fabricated citations** — 0 in
all 21 generation runs recorded here (six distinct sets of answers, at five
GPU placements), and 0 in every earlier run. The failure is
attribution, not invention: the model omits the reference far more than it
misplaces one, and it never cites an article the law does not have.

**B2 depends on GPU placement, irregularly.** It sits one question from the
pre-registered floor: FAIL at 0 and at 34 of 35 layers on the GPU — the
placement Ollama chose in every run since — and PASS at 2, 8 and 17 (E3d–E3e). Read it beside the
false-abstention line, always — a model that abstained on everything would
score 100% on the first and be useless — and read the pair as "abstains on
70–80% of out-of-corpus questions, at 0–7% false abstention, depending on
placement", not as a pass or a fail.

#### E3b · Grounding by category — `grounding_breakdown_5c76fa46.json`

| | |
|---|---|
| **Command** | `python evals/grounding_breakdown.py --rows evals/registry/answer_rows_4fe53a93.json` — reads saved rows only, no model |
| **source_commit_sha** | `5c76fa46` · worktree clean |
| **Raw result** | [`evals/registry/grounding_breakdown_5c76fa46.json`](evals/registry/grounding_breakdown_5c76fa46.json) |

The denominator is **answered** questions. A declined answerable question
carries `grounded: true` in its row because it made no claim; it is a false
abstention, reported in its own column, never counted as grounded.

| | answered | grounded — this run | grounded — `83384b2b` | uncited | cut by 128-token cap |
|---|---:|---:|---:|---:|---:|
| `direct` | 12 | 4 | 7 | 6 | 0 |
| `multi_article` | 9 | 3 | **0** | 5 | 2 |
| `colloquial` | 9 | 5 | 4 (of 8) | 4 | 1 |
| **total** | **30** | **12** | 11 (of 29) | 15 | 3 |

The totals barely moved between runs; the categories under them did. The
earlier run grounded **no** `multi_article` question and this repository
attributed the grounding drop to that category. This run grounds 3 of 9, and
loses the difference on `direct` instead. With nine questions a category, a
category-level finding from one generation run is not a finding. What both runs
agree on is the shape of the failure: uncited answers, not fabricated ones.

The token cap contributes and does not explain it: 3 answers were cut here, and
12 of the 15 uncited answers were never truncated.

> **Correction.** The first breakdown, `grounding_breakdown_83384b2b.json`, was
> computed by hand and counted the one false abstention (Q014) as grounded,
> which is how its two tables summed to 12 against a headline of 11/29. The
> script replaced it, with a test holding that run to 11/29. Its
> `multi_article 0/9` stood as computed — and is now shown by E3c/E3d to be one
> placement's result, not a property of the model.

#### E3c · Does the generation run repeat? — `generation_repeat_5250b8b0.json`

| | |
|---|---|
| **Command** | `python evals/generation_repeat.py --rows …83384b2b.json --rows …4fe53a93.json --rows runs/answer_eval-ollama-gemma3-4b.json` |
| **source_commit_sha** | `5250b8b0` · worktree clean |
| **Raw result** | [`evals/registry/generation_repeat_5250b8b0.json`](evals/registry/generation_repeat_5250b8b0.json) — embeds the third run's rows, since `runs/` is not tracked |

Three runs of the same 40 questions, same model digest, greedy decoding:

| Pair | same retrieval | same generated text | verdict flips |
|---|---:|---:|---|
| `83384b2b` vs `4fe53a93` — **across an Ollama restart** | 40/40 | **14/40** | grounded on 9 questions; abstained on Q014, Q016 |
| `4fe53a93` vs a repeat — **same server session** | 40/40 | **40/40** | none |

Retrieval is identical in every run. Generation is identical **within** one
Ollama server session and differs on 26 of 40 questions **across** one. The two
sessions also ran at different speeds (prompt evaluation 10.5 s against 2.7 s
on the same first question), which pointed at a different placement of work
between CPU and GPU. E3d tested it.

This qualified a sentence that stood at the top of this file: that a
generation run discarded and repeated "reproduced it exactly". It did — and E3d
shows the missing condition was placement, not the server session.

#### E3d · Where the difference came from — Run 13 and Run 13b

Both pre-registered in [`EVAL.md`](EVAL.md) — hypothesis, the rule for every
chosen value, and how each outcome would be read — and pushed before any run
(`d5379c39`, `7455aba8`).

**Run 13 — ten runs, each after a full Ollama restart.**

| | |
|---|---|
| **Command** | `python evals/generation_variance.py --runs 5 --label auto` · then `--label pinned --num-gpu 34` |
| **source_commit_sha** | `d5379c39` · worktree clean · env-001 checked |
| **Raw results** | [`generation_variance_d5379c39_auto.json`](evals/registry/generation_variance_d5379c39_auto.json) · [`generation_variance_d5379c39_pinned.json`](evals/registry/generation_variance_d5379c39_pinned.json) |

| Arm | runs | distinct outputs | layers on GPU (server log) | VRAM held by other processes | B2 met |
|---|---:|---:|---|---|---:|
| auto — Ollama chooses | 5 | **1** | 34/35 in every run | 134–216 MiB | 0/5 |
| pinned — `num_gpu 34` | 5 | **1** | 34/35 in every run | 215–238 MiB | 0/5 |

All ten produced the same text as each other and as `4fe53a93`. A restart does
not change the output. Ollama chose the same split every time, so the pinned
arm had nothing to remove — by the table written beforehand, this arm could not
confirm or refute placement as the cause.

**Run 13b — the placement of the one run that differed.**

| | |
|---|---|
| **Command** | `python evals/placement_probe.py --layers 0 2 4 6 8 10 12 --target-share 0.09` · `python evals/generation_variance.py --runs 2 --label low2 --num-gpu 2` · `python evals/generation_repeat.py …` |
| **source_commit_sha** | `7455aba8` · worktree clean · env-001 checked |
| **Raw results** | [`placement_probe_7455aba8.json`](evals/registry/placement_probe_7455aba8.json) · [`generation_variance_7455aba8_low2.json`](evals/registry/generation_variance_7455aba8_low2.json) · [`generation_repeat_7455aba8.json`](evals/registry/generation_repeat_7455aba8.json) |

`83384b2b` recorded a GPU share of **9%** — another process held most of the
card — and generated at 2.6 tokens/s against 8.3 since. It recorded no layer
count. The probe measured the share at each pin (0.000, **0.078**, 0.109, 0.140,
0.169, 0.198, 0.227 for 0–12 layers), and the pre-registered rule picked
`num_gpu 2`. Two runs there, each after a restart:

| | same text as `83384b2b` | grounded | abstained OOC | false abstention | B2 |
|---|---:|---:|---:|---:|---|
| 34 layers (all ten Run 13 runs) | 14/40 | 12/30 | 7/10 | 0/30 | FAIL |
| **2 layers (both Run 13b runs, identical)** | **37/40** | **11/29** | **8/10** | **1/30** | **PASS** |
| `83384b2b` itself | — | 11/29 | 8/10 | 1/30 | PASS |

**What this establishes.** Generation here is deterministic *per placement*
and changes *between* placements. At two layers on the GPU every headline
number and every per-question verdict of the old run comes back; the three
answers that still differ diverge late (from character 52, 93 and 183) and
change no verdict. By the table written beforehand this is the **partial**
row, not a confirmation: placement moves agreement from 14/40 to 37/40, but the
text is not reproduced exactly. The share at two layers (0.078) is not the
recorded 0.09, and the old run's Ollama version was not recorded — either could
account for three answers.

**What it means for every generation number in this file.** The placement is
Ollama's choice, made at load time from the VRAM free on a card that other
desktop programs also use. So the pre-registered abstention verdict is not
noise and not a property of the model alone: it is **PASS with 2 layers on the
GPU and FAIL with 34**. A generation result on this stack is not reproducible
unless the layer count is recorded — and from `bc520d3` on it is, in every
run's metadata and in the promoted result. Quote E3 as *at 34/35 layers*.

> **Limitations.** Two placements measured end to end here; E3e adds three.

#### E3e · Five placements — Run 13c

Pre-registered in [`EVAL.md`](EVAL.md) and pushed before any run (`34690455`).

| | |
|---|---|
| **Command** | `python evals/generation_variance.py --runs 2 --num-gpu N` for N = 0, 8, 17 · then `python evals/generation_repeat.py` over all five placements |
| **source_commit_sha** | `34690455` · worktree clean · env-001 checked |
| **Raw results** | [`…_low0.json`](evals/registry/generation_variance_34690455_low0.json) · [`…_low8.json`](evals/registry/generation_variance_34690455_low8.json) · [`…_mid17.json`](evals/registry/generation_variance_34690455_mid17.json) · [`generation_repeat_34690455.json`](evals/registry/generation_repeat_34690455.json) |

| Layers on GPU | GPU share | runs, all identical | grounded | abstained OOC | false abstention | fabricated | B2 |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 0.000 | 2 ✔ | 12/28 | 7/10 | 2/30 | 0 | **FAIL** |
| 2 | 0.078 | 2 ✔ (13b) | 11/29 | 8/10 | 1/30 | 0 | **PASS** |
| 8 | 0.169 | 2 ✔ | 11/29 | 8/10 | 1/30 | 0 | **PASS** |
| 17 | 0.301 | 2 ✔ | 11/30 | 8/10 | 0/30 | 0 | **PASS** |
| 34 | 0.551 | 10 ✔ (13) | 12/30 | 7/10 | 0/30 | 0 | **FAIL** |

**Read by the table written beforehand:**

1. **Deterministic at every placement measured** — two or more runs, each after
   a restart, identical at all five. That includes the first 0-layer run, which
   the machine slept through for six and a half hours mid-question; its
   repeat, uninterrupted, matched it on all 40 answers.
2. **B2 changes twice** — FAIL at 0, PASS at 2, 8 and 17, FAIL at 34. The
   pre-registered reading for more than one change is **irregular**: the
   verdict is stated only with the exact layer count, never as "below N" or
   "above N". Descriptively the three PASS placements are contiguous, but five
   points cannot say whether that holds between them, and every one of these
   verdicts is one out-of-corpus question from the floor.

**Text agreement falls with distance in placement.** Neighbours share the most
answers — 2 and 8 layers agree on 31 of 40, 8 and 17 on 27 — and every
placement agrees with 34 layers on only 13 or 14. Retrieval is identical
throughout (40/40 in every pair).

**What this leaves standing.** Grounding is 11–12 answers at every placement;
abstention is 7–8 of 10; false abstention 0–2 of 30; and **zero fabricated
citations in all 21 runs** — six distinct sets of answers. Those ranges, not
any single run, are the generation result of this repository.

> **Limitations.** Five of 36 possible placements. The verdict between them is
> unmeasured, and B2's two changes are each one question. *Why this limit:* a
> 0-layer run takes about 90 minutes on this CPU. *Next experiment:* none on
> placement — the finding is that the verdict is placement-bound, and more
> points would refine that without changing it. The next generation experiment
> is the `gated` contract, measured at a recorded placement.


**Hypotheses** — consistent with these numbers, *not established by them*: the
free-text contract leaves citation optional in practice, so whether a given
answer carries its reference is close to a coin toss that decoding noise can
flip. *Next experiment:* the `gated` claims contract, where an answer must name
its articles before writing prose — aimed at exactly this failure, and one that
should make grounding far less sensitive to placement if it works.

> **Timing is not a benchmark.** Median 20.7 s per answer here, on a desktop,
> with GPU share left to Ollama. E2 is the row with a real timing protocol.

> Superseded, kept: `answer_eval_83384b2b.json` and `answer_rows_83384b2b.json`
> (unreachable SHA, interpreter not recorded; still the second sample in E3c),
> and `grounding_breakdown_83384b2b.json` (hand-computed).

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
| **source_commit_sha** | `993583df` |
| **environment_ref** | `env-001` — both runs on the recorded interpreter |
| **Result** | **956 passed** locally · **954 passed, 2 skipped** from a fresh clone of this repository |

A test count is not a quality metric and is not presented as one. It is here so
that the number quoted elsewhere is the *passing* count, taken from a run,
rather than the collected count taken from `--collect-only`.

The two numbers differ for a reason worth stating: the statute corpus is
git-ignored, so the two tests that read it skip on any machine that has not run
`ingest`. CI reports the same split. A reader who clones this and sees 954
should see it explained here rather than wonder which number was inflated.

---

## Not measured — and therefore not claimable

| Claim someone might expect | Status |
|---|---|
| B2 between the measured placements | **Five of 36 placements measured** (E3e). The verdict between them is not. |
| Whether a citation *supports* the sentence it is attached to | **Not measured.** The audit is mechanical (ADR-022). |
| Cost per 1 000 queries against a hosted API | Not measured. Everything here is local; there are no API calls to price. Any cost figure would be an assumed rate multiplied by a latency, which is the latency restated. |
| Held-out test-split results | **Not run.** The 60-question set has a locked `test` split of 20 questions that no measurement here has touched. Nothing in this file has been validated out of sample. |
| Anything about laws other than 151/2020 | Out of scope — one corpus, one jurisdiction. |
| GPU or production-scale retrieval latency | Not measured. Retrieval ran on CPU on one desktop. |
| `qwen2.5-coder:3b` as a generator, or tool-calling behaviour | Not measured in any run recorded here. |
