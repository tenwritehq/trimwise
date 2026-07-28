# Evidence-sensitivity v1.2 protocol

## Status and scope

This is a **post-hoc robustness analysis**. Review identified that the v1.1
required-evidence scorer could accept an 80% bag of tokens collected anywhere in
the output. The v1.2 metrics were fixed before their aggregate results were
generated, using the same frozen 160 cases and saved compression rows. They
measure source-span survival; they do not establish semantic sufficiency,
downstream answer correctness, or general prompt-compression quality.

The canonical implementation is committed before execution. The pre-aggregate
manifest records that commit, Python and Unicode versions, scorer-file hashes,
input SHA-256 values, row identities, and annotation diagnostics. A v1.2 result
is valid only when its manifest predates its aggregate CSV.

## Inputs and retention rules

- Dataset: `data/position_controlled_160.jsonl` (160 cases).
- Compression rows: `results/position_controlled_160_results.jsonl`.
- Identity: `(case_id, method_id, budget, query_aware)`; every identity must be
  unique.
- Required spans are scored independently; a case passes only when **every**
  required span passes and its existing budget and prohibited-content checks
  pass. Failed compressor rows remain failures. Ordered-procedure scoring is
  unchanged.
- Blank required spans and required spans with no comparison tokens are invalid
  data. No case may be dropped for a new scorer failure.
- The analysis neither regenerates compressor outputs nor modifies any v1.1
  JSONL, CSV, manifest, or report.

## Metrics

All variants retain the existing budget and prohibited-content checks.

1. **Legacy v1.1 case pass** is retained only as a historical comparator: a
   normalized exact match or 80% bag-of-token recall anywhere in the output.
2. **Raw byte-for-byte span containment** remains a descriptive diagnostic in
   `exact_required_span_coverage`.
3. **Normalized contiguous required-span containment** is the strict primary
   source-span-survival metric. It case-folds, collapses whitespace, then checks
   whether the whole normalized reference is a contiguous substring of the
   normalized output. It is not byte-exact retention and is not semantic
   evidence preservation.
4. **Local ordered retention** is reported continuously as
   `local_ordered_required_span_recall`, and as 80% and 90% case-pass
   sensitivities. For reference tokens `R` of length `m` and output tokens `O`:

   `max_i LCS(R, O[i:i+m]) / m`

   Each window is token-indexed and end-exclusive. Its target length is exactly
   `m`; when fewer than `m` output tokens remain, the shorter suffix is scored.
   A threshold `t` passes exactly when `LCS >= ceil(t * m)`. LCS preserves order
   and cannot reuse repeated tokens. The continuous column is a mean of each
   row's per-required-span recalls, followed by the ordinary aggregate mean.

The 80% threshold keeps continuity with v1.1 while adding locality and order;
90% is a near-complete-retention sensitivity. Neither threshold is a headline
replacement for normalized contiguous containment.

## Normalization and tokenization

No Unicode normalization form is applied. Python `str.casefold()` is applied in
both reference and output; the manifest records the Python and Unicode database
versions. For normalized contiguous matching, all Unicode whitespace is split
and rejoined with one ASCII space. Punctuation is retained by this operation.

Local ordered retention tokenizes the case-folded string with the exact regular
expression `r"[\w./:+-]+"`. It therefore retains Unicode word characters,
digits, underscore, period, slash, colon, plus, and hyphen. Other punctuation
and whitespace delimit tokens. The same tokenizer is applied to both sides,
before local-window construction.

## Annotation diagnostics

Before aggregation, the manifest audits required-span token lengths, cases by
required-span count, normalized required spans that occur more than once in
their source, and required-span pairs that overlap by at least half of the
shorter span. Text containment cannot prove that a specific annotated location
survived when the same normalized text occurs elsewhere in the source; the
duplicate-occurrence count makes that limitation visible.

## Freeze and run order

From `benchmark/`, first commit the scorer, tests, README, and this protocol.
The manifest command refuses to run unless those exact files are tracked and
clean at `HEAD`.

```bash
uv run python scripts/build_evidence_sensitivity_manifest.py \
  --dataset data/position_controlled_160.jsonl \
  --input results/position_controlled_160_results.jsonl \
  --output data/manifests/evidence_sensitivity_v1_2_manifest.json

uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --output results/position_controlled_160_evidence_sensitivity_v1_2_summary.csv
```

Commit the generated manifest and summary as parallel v1.2 artifacts. Report
legacy, normalized contiguous, local-80, and local-90 outcomes regardless of
their ordering.
