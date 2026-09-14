# App eval set — `app-dev`

The evaluation for the upload-and-chat demo layer (ADR-023). It is written and
committed **before** any upload-pipeline or app code, for the same reason the
retrieval questions were written before the retriever: questions written after
the system tend to be the questions the system can answer.

## The document

`policy_ar` is a **synthetic** Arabic remote-work policy for a fictional company.
It is authored for this repository, so it can be committed and redistributed —
unlike the statute corpus, which never enters git (ADR-023). It is deliberately
**not** a statute: no «مادة» headers, so the upload pipeline must chunk it by page
instead of by article.

| File | Role |
|---|---|
| `policy_ar.html` | the source — 12 sections on 6 A4 pages (explicit page breaks) |
| `policy_ar.pdf` | rendered **once** from the HTML and committed; nothing regenerates it |
| `policy_ar.txt` | the same text as plain UTF-8, derived from the HTML |

Rendering the PDF (any Chromium; the output is committed, so this is provenance,
not a build step):

```bash
chromium --headless --no-pdf-header-footer --print-to-pdf=policy_ar.pdf policy_ar.html
```

## What the document is built to test

- **Numbers in three scripts:** `24 ساعة` (Latin digits), `٤٠٠ جنيه` (Arabic-Indic
  digits), `عشرين يوم عمل` (words).
- **Latin words inside Arabic lines:** `VPN`, `Wi-Fi`, `Microsoft Teams`. A single
  Latin word on an Arabic page used to make `pdf_text.arabic_column` cut the page
  into columns and silently drop Arabic text.
- **Distractors:** the same durations appear in different sections, e.g.
  «خمسة أيام عمل» and «ثلاثة أيام عمل», or «قبل أسبوعين» and «مدته أسبوعان». A
  retriever that matches the number without the context fails.
- **A hard abstention (`O01`):** section six mentions the annual-leave balance but
  never gives a number of days. The model may answer from general knowledge.

## What rendering it revealed (found, not designed)

The first extraction of the browser-rendered PDF matched **zero** headings and
**zero** keywords, although the page count was right. Two properties of this PDF
had never appeared in the statute PDF:

- **Presentation forms.** On page 1, Edge encodes 572 of 965 Arabic glyphs as
  contextual presentation forms (U+FB50–U+FEFF). `pdf_text` only treated
  U+0600–U+06FF as Arabic, so these glyphs were never reversed as Arabic.
- **Persian code points.** The font maps yeh and heh to `ی` (U+06CC) and `ھ`
  (U+06BE). NFKC keeps them as they are, so `أيام` never matches `أیام`.

A preview that widens the Arabic ranges, applies NFKC and folds those letters
matches the headings and keywords on their expected pages. Fixing the extractor
itself belongs to the upload pipeline, and this file is its failing test.

The single-Latin-word column cut also shows up: page 2 (`VPN`, `Wi-Fi`,
`Microsoft Teams`) loses about half its text to `arabic_column`.

## Questions

`questions.jsonl` has 15 questions: 10 answerable (4 `direct`, 3 `paraphrase`,
3 `colloquial`) and 5 `out_of_doc`. Each answerable question has an
`expected_section`, `expected_pages` (the PDF page) and `expected_keywords`.

**Retrieval hit:** a top-5 chunk whose text contains every expected keyword. Both
sides go through `evaluation_normalize`, so `٤٠٠` and `400` match. Keywords are
Arabic even where the document uses a Latin term, because PDF extraction may drop
Latin script.

## Limits

One document, 15 questions, one author who also wrote the pipeline. These numbers
are a smoke-level signal about the demo layer, not a benchmark.
