# Run 12 artefacts

Raw material for the rejection recorded in ADR-030, so the number can be
re-derived rather than taken on trust:

- `page_manifest.json` — the eight held-out pages, as rasterised (96 DPI,
  the scan's own resolution).
- `deepseek_output.json` — what the engine returned, unedited.
- `deepseek_timings.json` — ~190s a page, which is where ADR-030's ~7.2
  hours for 137 pages comes from.
- `deepseek_report.md` — per-page scoring and the rule applied to it, as
  measured. This is the file the verdict was reached on.
- `deepseek_report_after_ocr_text_fix.md` — the same output re-scored after
  the `ocr_text` furniture bug this run exposed was fixed. Kept beside the
  first rather than replacing it, because re-scoring with a ruler changed
  after seeing the result is the shape of post-hoc slicing this project
  refuses. Recall (46/53) and critical failures (3) are identical; only
  the invented count moves, 11 -> 7, as p094's leaked header now strips.
  The verdict is the same, which is why the fix was allowed to land.
- `prompt_replay.json` — evidence that Run 11 and Run 12 used one prompt.

Ground truth is `../law174_page_roles_ground_truth.json`. To reproduce:

```
python tasks.py ocr-gate evals/ocr/run12/deepseek_output.json deepseek \n  --gt evals/ocr/law174_page_roles_ground_truth.json --content
```

`qwen2.5vl:3b` has no artefacts here: it produced no output on this
machine, twice. That is recorded in EVAL.md as an incomplete measurement,
not as a score.
