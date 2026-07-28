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

- **Query-aware:** Trimwise Lexical and Hybrid, LLMLingua, LongLLMLingua, and RECOMP each receive
  the same source and question. This is Trimwise’s primary use case.
- **Source-only:** Prefix, Head + tail, Trimwise Structural, questionless LLMLingua, and
  LLMLingua-2 receive only the source. These scores are not comparable to query-aware scores.

RECOMP here is a transparent **extractive adapter**, not a reproduction of RECOMP’s complete
end-to-end training and retrieval pipeline. It uses the released
`fangyuan/nq_extractive_compressor` checkpoint to rank complete source-backed sentences and greedily
packs them into the budget. Results should be interpreted as this adapter’s comparison, not as a
claim about every RECOMP deployment.

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
`results/position_controlled_160_summary.csv`,
`results/position_controlled_160_natural_only_summary.csv`,
`results/position_controlled_160_relocated_end_summary.csv`,
`results/position_controlled_160_paired_stats.csv`, and the static report under `reports/`.
The raw rows let reviewers recompute the aggregates and inspect successes and failures without
rerunning GPU models or making API calls. Other result JSONL files remain ignored as resumable local
run state. Open `reports/index.html`; it routes to separate query-aware and source-only reports.

[`data/manifests/runtime_manifest.json`](data/manifests/runtime_manifest.json) records the
captured machine, package lock, model revisions, run settings, and SHA-256 hashes for this local
dataset and result snapshot. It is an integrity record released with the
[`paper-v1` tagged artifact](https://github.com/tenwritehq/trimwise/tree/paper-v1), not a portable
performance claim.

**Case pass** requires all required evidence, no prohibited content, and budget compliance. A
required span passes with exact retention or at least 80% token recall. The aggregate reports
evidence retention, output tokens, budget violations, latency, CUDA memory, thermal events, and
method failures. QA answer match is a normalized token-containment measure over saved model
continuations; it is not a human semantic-correctness evaluation.

## Limits

This is a position-controlled evaluation suite, not a naturally representative corpus or a claim
of universal production performance. Its 160 cases are balanced by evidence position but combine
135 natural placements with 25 controlled end relocations, controlled synthetic material, and
pinned public-source extracts; labels have source-span verification. The evaluation uses
one saved API completion per context and has no component ablation. Hardware-dependent latency and
memory measurements must be reported with the GPU, driver, and run date used for a released snapshot.
