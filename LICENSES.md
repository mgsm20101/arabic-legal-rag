# Licenses, provenance and redistribution

Every third-party thing this repository depends on, what governs it, and what
that means here. Verified by reading the source named in each row on the date
given — not inferred from a package name.

**The short version:** the code is MIT. No model weights and no corpus text are
redistributed from this repository — both are fetched by the reader on first
run. The one licence that is not a standard open-source licence is Gemma, and
the clause that would bind a redistributor does not trigger here because nothing
is redistributed.

---

## 1. This repository

| | |
|---|---|
| Licence | **MIT** — see [`LICENSE`](LICENSE) |
| Copyright | © 2026 Mohamed |
| Redistribution | Permitted with the notice retained |
| Verified | 2026-09-23 |

---

## 2. Models

Weights are **not** stored in this repository. `models/` is git-ignored;
sentence-transformers downloads from the Hugging Face Hub on first run and
Ollama pulls its own. The revision column is the snapshot this project was
measured against, recorded in [`evals/environment.json`](evals/environment.json) —
a tag can be repointed, a revision cannot.

| Model | Role | Licence | Revision measured | Redistributed here | Attribution required | Verified |
|---|---|---|---|---|---|---|
| [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) | Dense retrieval | **MIT** (`license: mit` in the model card) | `d128750597153bb5987e10b1c3493a34e5a4502a` | No | No | 2026-09-23 |
| [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | Reranking | **Apache-2.0** (model-card metadata) | `1427fd652930e4ba29e8149678df786c240d8825` | No | Standard Apache-2.0 notice on redistribution | 2026-09-23 |
| [`gemma3:4b`](https://ollama.com/library/gemma3) | Generation (default) | **[Gemma Terms of Use](https://ai.google.dev/gemma/terms)** — *not* a standard open-source licence | Ollama digest `a2af6cc3eb7f…`, Q4_K_M | No | See below | 2026-09-23 |

### 2.1 Gemma — the one that needs reading, not skimming

`gemma3:4b` is the default generator (ADR-024). It is served by a local Ollama
instance; this repository ships no Gemma weights and produces no model
derivative.

| Clause | What it says | Does it bind this repository? |
|---|---|---|
| **§3.1 Distribution** | Anyone who *reproduces or distributes* Gemma or a Model Derivative must pass these terms on | **No.** Nothing is redistributed — Ollama pulls the weights on the reader's machine |
| **§3.2 Use restrictions** | Use must comply with the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) and applicable law | **Yes** — on whoever runs it, including you |
| **Commercial use** | Not prohibited by the terms | Permitted |
| **§4.2 Trademarks** | Grants no right to Google's marks, and nothing may suggest endorsement | **Yes** — this project is not affiliated with or endorsed by Google |

Swapping the generator removes §3.2 from your obligations entirely: the provider
is a setting, not a dependency. `LEGALRAG_MODEL` accepts any `ollama:<model>` or
`hf:<repo>` spec — see [`.env.example`](.env.example).

---

## 3. Corpus and datasets

**No legal text is stored in this repository.** `data/raw/*` is git-ignored except
the provenance card. The reader downloads the source themselves, following
[`data/raw/README.md`](data/raw/README.md); the full card with checksums is in
[`data/raw/SOURCE.txt`](data/raw/SOURCE.txt).

| Item | What it is | Redistributed here | Notes |
|---|---|---|---|
| Egypt Law 151/2020 (Personal Data Protection) | The ingested corpus | **No** — reader downloads it | Obtained from `privacylaws.com`, a third-party republication. `sha256 406b649d…` |
| Official Gazette 28 *mukarrar* (h), 2020-07-15 | The text of record | **No** | Via `manshurat.org`. No text layer — 29 scanned images. `sha256 deb05826…` |
| Evaluation question set | Written by hand for this project | Yes — `evals/` | Covered by this repository's MIT licence |

> **Fidelity, stated because it bears on what the numbers mean:** the ingested
> file is *not* byte-identical to the gazette. A check on 2026-09-08 sampled 11
> articles and **9 differed** (ADR-018). Retrieval scores here describe
> performance *on this corpus* — they are not statements about the law. The full
> record is in [`EVAL.md`](EVAL.md) under *Corpus fidelity*.

Egyptian legislation is official state text. This repository takes no position
on its copyright status and sidesteps the question by not republishing it.

---

## 4. Direct dependencies

Pinned in [`constraints.txt`](constraints.txt); the sha256 of each pin file is
recorded in `evals/environment.json`. Licences below are read from the installed
distribution metadata.

| Package | Version | Licence | Role |
|---|---|---|---|
| `sentence-transformers` | 6.0.1 | Apache-2.0 | Loads the embedding and reranker models |
| `rank-bm25` | 0.2.2 | Apache-2.0 | Lexical baseline |
| `numpy` | 2.5.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Embedding matrix |
| `pdfplumber` | 0.11.10 | MIT | Reads the source PDF |
| `fastapi` | 0.141.1 | MIT | Local app and dev server |
| `uvicorn` | 0.52.4 | BSD-3-Clause | ASGI server |
| `python-multipart` | 0.0.32 | Apache-2.0 | Upload parsing |
| `httpx` | 0.28.1 | BSD-3-Clause | Ollama client |
| `pytest` | 9.1.1 | MIT | Test runner (dev only — not in the image) |

All permissive. No copyleft and no non-commercial terms among the direct
dependencies, so none of them constrains how this repository may be used or
redistributed. Versions are the ones recorded in `evals/environment.json`;
licences read from installed distribution metadata on 2026-09-23.

Transitive dependencies are not enumerated by hand — see the generated report in
§5 instead.

---

## 5. Security audit

[`docs/review/dependency-audit-summary.json`](docs/review/dependency-audit-summary.json)
— `pip-audit -r constraints.txt --no-deps`, run from an isolated venv so the
environment being audited is not the environment doing the auditing.

| | |
|---|---|
| Date | **2026-09-23** |
| Packages audited | 61 |
| Unique advisories | **4** (6 raw records — two sources report the same two twice) |
| Scope, as stated in the file | Package-version matches — **not** demonstrated application exploitability |

| Package | Version | Advisory | Fixed in | First seen |
|---|---|---|---|---|
| `anyio` | 4.12.1 | `CVE-2026-63374` / `GHSA-82r6-8w77-94w6` | 4.14.2 | 2026-09-23 |
| `anyio` | 4.12.1 | `CVE-2026-64847` / `GHSA-5p39-cfhj-2xmp` | 4.14.2 | 2026-09-23 |
| `cryptography` | 49.0.0 | `PYSEC-2026-3552` / `CVE-2026-69247` | 50.0.0 | 2026-09-17 |
| `setuptools` | 82.0.1 | `PYSEC-2026-3447` / `CVE-2026-59890` | 83.0.0 | 2026-09-17 |

All four are transitive, all four are open, and none sits in the retrieval
path. They are recorded rather than quietly dropped: an audit whose findings
disappear is not an audit.

**Why they are not patched in this commit.** Upgrading a pin changes
`constraints.txt`, and `constraints.txt`'s hash *is* part of `environment_ref`
`env-001` — the environment every measured row in
[`EVIDENCE.md`](EVIDENCE.md) points at. Bumping the pins and re-measuring is
one action, not two, and doing the first half alone would leave the registry
describing an environment that no longer exists. The upgrade is queued behind
the next measurement pass, not forgotten.

**The count went up between audits, and that is the useful part.** The
2026-09-17 run found two advisories; this one finds four. Nothing in the
repository changed — the advisory database did. A dated snapshot is the only
honest form this section can take.

---

## 6. If you fork this

1. Keep the MIT notice.
2. Download the corpus yourself — `data/raw/README.md` says how, and why the
   source matters more than it looks.
3. Read the Gemma Prohibited Use Policy, or point `LEGALRAG_MODEL` elsewhere.
4. Re-run `pip-audit`; §5 is a snapshot, not a standing guarantee.
