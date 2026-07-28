# Trimwise component-ablation protocol

This is a separate exploratory query-aware compression experiment over the
frozen 160-case suite. It does not modify the original 6,400 compression
outcomes, the strict v1.2 rescore, or the direct-retrieval follow-up.

## Question

Which observed outcomes of the complete Trimwise Hybrid configuration depend on
MMR diversity, the query-aware evidence-pool cutoff, and Markdown-aware source
segmentation?

## Fixed inputs and measurement

- Dataset: `data/position_controlled_160.jsonl` (160 cases).
- Budgets: 128, 256, 512, and 1,024 `o200k_base` tokens.
- Primary score: normalized contiguous required-span containment, with prohibited
  content absence and budget compliance required for case pass.
- Sensitivities: local ordered retention at 80% and 90%.
- Output: `results/position_controlled_160_ablation_results.jsonl`.
- No downstream QA or external API calls are made.

Each ablation changes one decision mechanism while retaining Hybrid's encoder,
lexical--semantic fusion, source-order composition, token counter, omission
marker, query, and the remaining selection behavior. CUDA uses the existing
thermal gate: it refuses foreign CUDA compute, pauses at 70 C, resumes at 60 C,
and records thermal metadata.

## Fixed ablations

1. **Hybrid without MMR** sets `mmr_lambda=1.0`, so selection is relevance-only
   and the diversity penalty has zero weight.
2. **Hybrid without evidence cutoff** retains every non-heading candidate rather
   than applying Hybrid's adaptive evidence-pool boundary.
3. **Hybrid over fixed windows** replaces Markdown-aware segmentation with
   non-overlapping, exact 128-token `o200k_base` source windows. Ranking,
   fusion, MMR, and composition otherwise remain unchanged.

## Analysis policy

The new outcomes are aggregated with the immutable canonical results to compare
each ablation against the complete Hybrid configuration at every budget. The
analysis reports all-case, position, task, feasibility, latency, and paired
bootstrap summaries. These are configuration ablations on one fixed suite; they
do not establish general causal effects outside the documented workload.

## Released input identities

| File | SHA-256 |
| --- | --- |
| `data/position_controlled_160.jsonl` | `9d9667127dc4461b000f78cd4b46e3e845c9d4030af09c353ee69a4bd81bfdab` |
| `results/position_controlled_160_results.jsonl` | `38248f76276c24db0ce2f9a8b3c74eeaaa7668e78862c0f2e1b4436c8a24d38d` |
| `configs/ablation_160.yaml` | `66956a05fe7962e466777937b7b6cbb0d0c13c5f3a09e13965aeb47e508c4738` |
| `benchmark/adapters/trimwise_adapter.py` | `4e2fcde1ea6a25539f25807c03b1803baf648ce4c0509a8f1c1f7673eb673eaa` |
| `results/position_controlled_160_ablation_results.jsonl` | `6b4fb95e4dbd7589dd97cd179b6a5df0cae23ae29000217d1521d51b4ccb9d89` |

The result file is expected to contain 1,920 successful outcomes: 160 cases
times four budgets times three ablations.

## Released derived identities

| File | SHA-256 |
| --- | --- |
| `results/position_controlled_160_ablation_summary.csv` | `ecc0899947ef58ae6bd0b9c3b740b139e60b6c68b8c29936005f840967718aea` |
| `results/position_controlled_160_ablation_paired_stats.csv` | `0bd9015896cb91fbe45044988d4fc6835f008e84b540adab426766b8f78e03d6` |
| `reports/ablation-v1/ablation_vs_hybrid.svg` | `6021ec4a25d8299a368de8ed88610827cf4b6f431452094afb6d46bc5c254343` |
