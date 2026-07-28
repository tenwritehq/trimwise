# Trimwise context-selection benchmark

This is the single public protocol for Trimwise: a 160-case, position-controlled
context-selection evaluation. It measures whether a compressor preserves the exact source evidence
needed to answer a question under a token ceiling. The canonical configuration is
[`configs/position_controlled_160.yaml`](configs/position_controlled_160.yaml).

The dataset has 40 cases each with required evidence at the beginning, middle, end, or multiple
locations. The original 250-case corpus remains unchanged; the separate 160-case evaluation
contains 135 natural cases and 25 controlled relocations. The end stratum therefore contains 15
natural end cases and 25 relocated end cases. Public-source revisions and licenses are recorded in
[`data/manifests/source_manifest.csv`](data/manifests/source_manifest.csv); inspect
[`data/position_controlled_160_review.md`](data/position_controlled_160_review.md) before treating
the snapshot as frozen.

## Scope and fair comparisons

Compressors receive only the source context. Application instructions, schemas, and the final query
are assembled outside the compressed region. The benchmark reports two separate conditions:

- **Query-aware:** Trimwise Lexical and Hybrid, the LLMLingua GPT-2 token-pruning adapter, the
  LongLLMLingua GPT-2 single-context adapter, and the RECOMP NQ extractive sentence adapter each
  receive the same source and question. This is Trimwise’s primary use case.
- **Source-only:** Prefix, Head + tail, Trimwise Structural, questionless LLMLingua, and
  LLMLingua-2 receive only the source. These scores are not comparable to query-aware scores.

The LLMLingua GPT-2 adapters use `llmlingua==0.2.2` with `openai-community/gpt2`; the
token-pruning adapter uses token-level filtering, while the LongLLMLingua single-context adapter
receives each source as one context string. RECOMP here is a transparent **NQ extractive sentence
adapter**, not a reproduction of RECOMP’s complete end-to-end training and retrieval pipeline. It uses the released
`fangyuan/nq_extractive_compressor` checkpoint to rank complete source-backed sentences and greedily
packs them into the budget. Results should be interpreted as these adapters’ comparison, not as a
claim about every family configuration or RECOMP deployment.

## Install and validate the frozen dataset

Run from `benchmark/` with Python 3.12:

```bash
uv sync --python 3.12
uv run python scripts/build_position_controlled_160.py --check
```

The GPU runners pin `transformers==4.43.1` for LLMLingua 0.2.2 compatibility and resolve the
published Trimwise 0.2.0 release from PyPI. The first run downloads method models into
`cache/huggingface`.

## Run

The compressor requires CUDA. It reads NVIDIA telemetry, refuses to start when another pure CUDA
compute process is active, pauses at 70 C, resumes at 60 C, and stops after a 75 C event or a
30-minute cooldown timeout. It never changes LACT, power, clock, or fan settings.

Start with one case:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runners.run_compression --limit 1
```

Then resume the full compression run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runners.run_compression
```

Rows are keyed by `(case_id, method_id, budget, query_aware)`, so rerunning fills only missing
successful identities. Failures remain explicit rows.

The configured evaluators are GPT-5.4 Nano, GPT-5.4 Mini, and GPT-5.6 Luna, all with
`reasoning_effort: none`, a 128-token completion limit, and at most 16 concurrent requests. This
step is billable and resumable:

```bash
export OPENAI_API_KEY="..."
uv run python -m benchmark.runners.run_openai_qa
```

## Aggregate and publish results

```bash
uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_4_nano.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_4_mini.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_6_luna.jsonl \
  --output results/position_controlled_160_summary.csv

uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_4_nano.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_4_mini.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_6_luna.jsonl \
  --position-origin natural \
  --output results/position_controlled_160_natural_only_summary.csv

uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_4_nano.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_4_mini.jsonl \
  --qa-input results/position_controlled_160_qa_gpt_5_6_luna.jsonl \
  --position-origin controlled_relocation \
  --output results/position_controlled_160_relocated_end_summary.csv

uv run python scripts/paired_stats.py

uv run python -m benchmark.runners.plot \
  --input results/position_controlled_160_summary.csv \
  --natural-input results/position_controlled_160_natural_only_summary.csv \
  --relocated-end-input results/position_controlled_160_relocated_end_summary.csv \
  --output-dir reports
```

The public snapshot includes the canonical compression rows
(`results/position_controlled_160_results.jsonl`), the three canonical evaluator row files,
the aggregate CSVs, and the static report under `reports/`. Later direct-retrieval and component
studies publish their own raw rows, summaries, paired intervals, protocols, and reports alongside
the original snapshot. The raw rows let reviewers recompute the aggregates and inspect successes
and failures without rerunning GPU models or making API calls. Other result JSONL files remain
ignored as resumable local run state. Open `reports/index.html` for the available reports.
The strict-figure renderer also refreshes the detailed report served from the documentation site.

The later strict post-hoc sensitivity release additionally includes
`results/position_controlled_160_evidence_sensitivity_v1_2_summary.csv`, its natural-only,
relocated-only, and self-source-excluded companion CSVs, and
`results/position_controlled_160_evidence_sensitivity_v1_2_paired_stats.csv`. These are parallel
rescoring artifacts, not replacements for the immutable historical snapshot.

[`data/manifests/runtime_manifest.json`](data/manifests/runtime_manifest.json) records the
captured machine, package lock, model revisions, run settings, and SHA-256 hashes for this local
dataset and result snapshot. It is an integrity record released with the
[`paper-v1` tagged artifact](https://github.com/tenwritehq/trimwise/tree/paper-v1), not a portable
performance claim.

**Legacy bag-of-token case pass** requires all required evidence, no prohibited content, and budget
compliance. In the frozen result, a required span passes with normalized exact retention or at
least 80% bag-of-token recall anywhere in the complete output. It remains available to reproduce
the published snapshot, but can credit tokens scattered across unrelated excerpts.

The strict source-evidence analysis is an explicitly **post-hoc robustness analysis** over frozen
outputs. It measures source-span survival, not semantic sufficiency, downstream answer correctness,
or general prompt-compression quality. Its strict primary metric is **normalized contiguous
required-span containment**: the complete required span occurs after case-folding and whitespace
collapse. The accompanying local ordered 80% and 90% sensitivities require ordered retained tokens
inside one bounded output-token window. The legacy score and raw byte-for-byte containment remain
descriptive diagnostics. See the frozen
[strict metric protocol](data/manifests/evidence_sensitivity_v1_2_protocol.md) for the exact tokenizer,
normalization, empty-span handling, aggregation rule, annotation diagnostics, and artifact policy.

After committing the strict scorer and protocol, build its input manifest first. This is CPU-only and
refuses to run against an uncommitted scorer or protocol:

```bash
uv run python scripts/build_evidence_sensitivity_manifest.py \
  --dataset data/position_controlled_160.jsonl \
  --input results/position_controlled_160_results.jsonl \
  --output data/manifests/evidence_sensitivity_v1_2_manifest.json
```

Then generate the parallel strict sensitivity summary without compression or QA calls:

```bash
uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --output results/position_controlled_160_evidence_sensitivity_v1_2_summary.csv
```

Generate the predeclared sensitivity cohorts and paired intervals from those same saved rows:

```bash
uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --position-origin natural \
  --output results/position_controlled_160_evidence_sensitivity_v1_2_natural_only_summary.csv

uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --position-origin controlled_relocation \
  --output results/position_controlled_160_evidence_sensitivity_v1_2_relocated_only_summary.csv

uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --exclude-case-prefix real-trimwise- \
  --output results/position_controlled_160_evidence_sensitivity_v1_2_without_self_sources_summary.csv

uv run python scripts/paired_stats.py --metric-set strict-v1.2
uv run python scripts/render_evidence_sensitivity_figure.py
```

The first cohort uses 135 natural rows and is unbalanced by evidence position because only 15 have
natural end placement. The second contains the 25 controlled end relocations and is reported as a
construction diagnostic, not as a causal replacement for naturally occurring evidence position. The
third excludes every `real-trimwise-*` row. All four strict-score CSVs and the paired-bootstrap CSV are
derived from the original 6,400 saved compression rows; no GPU or API call is made.

### Strict source-span results

The strict primary metric is normalized contiguous required-span containment. Every required span
must survive as one contiguous normalized output passage; prohibited content and budget violations
still fail the case. It is a post-hoc robustness analysis fixed before its aggregate was inspected,
using the same saved outputs as the historical bag-of-token summary.

| Evaluated method or adapter | 128 | 256 | 512 | 1,024 |
| --- | ---: | ---: | ---: | ---: |
| Trimwise Lexical | **52.5%** | 60.0% | 61.9% | 66.2% |
| Trimwise Hybrid | 49.4% | **62.5%** | **66.9%** | **69.4%** |
| RECOMP NQ extractive sentence adapter | 27.5% | 30.6% | 35.0% | 35.0% |
| LLMLingua GPT-2 token-pruning adapter | 3.1% | 7.5% | 13.8% | 22.5% |
| LongLLMLingua GPT-2 single-context adapter | 0.6% | 4.4% | 6.9% | 16.2% |

The local ordered 80% and 90% sensitivities preserve this ordering at each budget. The static
[strict report](reports/evidence-sensitivity-v1-2/index.html) separates evidence survival from
feasible pass, natural placement from controlled relocation, and the self-source exclusion check.
Its complete results, continuous recall values, paired intervals, and the legacy metric are retained
in the released strict-score CSVs.

Render the committed strict figure from the frozen CSV with:

```bash
uv run python scripts/render_evidence_sensitivity_figure.py
```

## Direct retrieval baselines

The separate direct-retrieval experiment tests whether ordinary query-aware retrieval can match
Trimwise without its structural selection path. It adds fixed-window BM25, embedding top-*k*, and
embedding-MMR baselines over the same 160 cases and four budgets. The configuration fixes 128-token
windows, 32-token overlap, source-order reconstruction, the published Trimwise Hybrid default
encoder, and MMR relevance weight 0.7. It writes new rows rather than modifying the frozen
source-span-rescore JSONL:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runners.run_compression \
  --config configs/direct_retrieval_160.yaml

uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --input results/position_controlled_160_direct_retrieval_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --output results/position_controlled_160_direct_retrieval_summary.csv

uv run python scripts/paired_stats.py \
  --metric-set strict-direct-retrieval \
  --input results/position_controlled_160_results.jsonl \
  --input results/position_controlled_160_direct_retrieval_results.jsonl

uv run python scripts/render_direct_retrieval_report.py
```

This comparison uses the strict source-span metrics only; it makes no downstream QA calls. Its
protocol and results are separate from the post-hoc source-span rescore:
[`data/manifests/direct_retrieval_v1_protocol.md`](data/manifests/direct_retrieval_v1_protocol.md)
and [`reports/direct-retrieval-v1/index.html`](reports/direct-retrieval-v1/index.html).

## Hybrid component study

The separate exploratory component study holds Hybrid's encoder, lexical--semantic fusion,
source-order composition, token counter, and all remaining selection behavior fixed. It changes one
mechanism per variant: MMR becomes relevance-only, the adaptive evidence cutoff is removed, or
Markdown-aware segments become contiguous non-overlapping 128-token source windows. It writes
1,920 new compression outcomes and does not make downstream QA or API calls:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runners.run_compression \
  --config configs/ablation_160.yaml

uv run python -m benchmark.runners.aggregate \
  --input results/position_controlled_160_results.jsonl \
  --input results/position_controlled_160_ablation_results.jsonl \
  --dataset data/position_controlled_160.jsonl \
  --output results/position_controlled_160_ablation_summary.csv

uv run python scripts/paired_stats.py \
  --metric-set strict-ablation \
  --input results/position_controlled_160_results.jsonl \
  --input results/position_controlled_160_ablation_results.jsonl

uv run python scripts/render_ablation_report.py
```

This is a controlled exploratory study of three mechanisms on this fixed suite. It does not
establish component effects for every document type, encoder, or downstream task. Its frozen
protocol, hashes, results, and report are
[`data/manifests/ablation_v1_protocol.md`](data/manifests/ablation_v1_protocol.md) and
[`reports/ablation-v1/index.html`](reports/ablation-v1/index.html).

The aggregate reports evidence retention, output tokens, budget violations, latency, CUDA memory,
thermal events, and method failures. QA answer match is a normalized token-containment measure over
saved model continuations; it is not a human semantic-correctness evaluation.

## Limits

This is a position-controlled evaluation suite, not a naturally representative corpus or a claim
of universal production performance. Its 160 cases are balanced by evidence position but combine
135 natural placements with 25 controlled end relocations, controlled synthetic material, and
pinned public-source extracts; labels have source-span verification. The evaluation uses
one saved API completion per context and no independent human answer assessment. The component
study changes only three Hybrid mechanisms on the same author-annotated suite. Hardware-dependent
latency and memory measurements must be reported with the GPU, driver, and run date used for a
released snapshot.
