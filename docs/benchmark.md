---
title: "LLM Context Compression Benchmark | Trimwise Query-Aware Results"
description: "See Trimwise query-aware LLM context compression benchmark results: source-evidence case pass, token budgets, latency, and a reproducible 160-case evaluation."
---

# Query-Aware LLM Context Compression Benchmark

Trimwise is designed for **query-aware prompt assembly**: given a source and a
question, retain the exact source evidence most useful for that question under a strict output
budget. This evaluation tests that claim directly. Every method receives the same source and
question and must fit within a requested limit of 128, 256, 512, or 1,024 tokens.

The published v1.1 snapshot uses **legacy source-evidence case pass**. A case passes only when the
required source evidence survives, prohibited text is absent, and the finished output fits the
shared token counter. Its required-span check can accept 80% bag-of-token recall anywhere in the
output, so it measures a permissive source-evidence proxy—not general model intelligence or
generated-answer quality.

## Results at a glance

The v1.1 suite contains 160 cases: 40 each with required evidence at the beginning, middle, end,
or several locations in the source. Under its legacy metric, Trimwise Hybrid retained required
evidence more often than the evaluated query-aware comparators—LLMLingua, LongLLMLingua, and the
released-model extractive RECOMP adapter—at every tested budget.

| Method | 128 tokens | 256 tokens | 512 tokens | 1,024 tokens | Median warm trim time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trimwise Lexical | 53.8% | 61.3% | 62.5% | 66.3% | 6.4 ms |
| **Trimwise Hybrid** | 50.6% | **63.8%** | **66.9%** | **69.4%** | 42.8 ms |
| RECOMP extractive adapter | 33.1% | 38.1% | 48.1% | 55.6% | 126.2 ms |
| LLMLingua | 3.8% | 8.1% | 15.6% | 39.4% | 168.0 ms |
| LongLLMLingua | 0.6% | 5.0% | 8.1% | 23.1% | 340.7 ms |

These are observed all-case rates from one saved evaluation snapshot. Warm timing excludes model
loading and thermal cooldown and is specific to the recorded hardware.

![Legacy query-aware source-evidence case pass by output budget on the position-controlled 160-case benchmark.](https://raw.githubusercontent.com/tenwritehq/trimwise/paper-v1.1/assets/readme/query-aware-benchmark.svg)

### Budget compliance and speed

Trimwise Lexical and Hybrid and the RECOMP extractive adapter recorded no measured budget
violations under the shared counter. LLMLingua exceeded the requested limit in 30.6% of
query-aware calls; LongLLMLingua did so in 55.8% of calls. The chart below makes the practical
trade-off visible: higher is better for source-evidence case pass, and farther left is faster.

![Query-aware source-evidence case pass versus median trimming time for each token budget.](https://raw.githubusercontent.com/tenwritehq/trimwise/paper-v1.1/benchmark/reports/query-aware/utility_vs_latency.png)

## Downstream QA across three models

The benchmark also tests whether compressed evidence remains useful to downstream question-answering
models. This diagnostic covers 93 answerable cases and uses GPT-5.4 Nano, GPT-5.4 Mini, and
GPT-5.6 Luna. The dashed line in each panel is that evaluator's answer-match rate with the full,
uncompressed source; each solid line uses a compressed context.

![Downstream answer-match rate for GPT-5.4 Mini, GPT-5.4 Nano, and GPT-5.6 Luna at each context budget.](https://raw.githubusercontent.com/tenwritehq/trimwise/paper-v1.1/benchmark/reports/query-aware/answer_pass.png)

| Downstream evaluator | Full source | Trimwise Hybrid at 256 tokens |
| --- | ---: | ---: |
| GPT-5.4 Nano | 44.1% | 46.2% |
| GPT-5.4 Mini | 45.2% | 46.2% |
| GPT-5.6 Luna | 45.2% | 45.2% |

In this saved snapshot, Trimwise Hybrid remains at or near the full-source reference from 256
tokens upward for all three evaluators. The chart also shows the other compression methods. This
is a normalized token-containment answer-match diagnostic from one sampled continuation per
context—not a general measure of model capability or a human correctness study.

## What this benchmark establishes

Within the v1.1 legacy source-evidence evaluation, query-aware Trimwise configurations retained
the annotated evidence more often than the evaluated comparators at all four budgets. Hybrid was
the strongest observed configuration from 256 through 1,024 tokens; Lexical was strongest at 128
tokens and is substantially faster because it does not need embeddings.

This does **not** prove that Trimwise is universally best for every document, retrieval system,
language, target model, or summarization task. The RECOMP result is an extractive benchmark adapter
around the released checkpoint, not a reproduction of RECOMP's full retrieval and training
pipeline. Generated-answer results are reported separately as a diagnostic, because one sampled
answer and automatic token matching are weaker evidence than verified source retention.

## Dataset and reproducibility

The original 250-case corpus remains unchanged. The separate 160-case suite has 135 naturally
positioned examples and 25 source-preserving controlled relocations used to balance end-position
evidence. The natural-only sensitivity analysis excludes all relocations; it preserves the broad
ordering, but has only 15 naturally occurring end cases and is therefore not position-balanced.

The released artifact includes the dataset, source and runtime manifests, raw compression rows,
saved evaluator responses, aggregate CSVs, paired comparisons, and the complete query-aware and
source-only reports. The source-only condition is reported separately: it is a readable fallback
when no useful question exists, not Trimwise's primary benchmark claim.

A stricter v1.2 source-span analysis is intentionally separate and post-hoc: normalized contiguous
required-span containment is its primary metric, with local ordered 80% and 90% sensitivity
checks. It uses the same frozen outputs and does not establish semantic sufficiency or downstream
answer correctness. Its protocol and manifest will be published alongside its parallel summary,
not folded into the v1.1 figures.

- [Read the full benchmark protocol and reproduction steps](https://github.com/tenwritehq/trimwise/blob/paper-v1.1/benchmark/README.md)
- [Inspect the query-aware report and detailed figures](https://github.com/tenwritehq/trimwise/tree/paper-v1.1/benchmark/reports/query-aware)
- [Inspect the separate source-only report](https://github.com/tenwritehq/trimwise/tree/paper-v1.1/benchmark/reports/queryless)
- [Read the versioned paper and download the PDF](https://github.com/tenwritehq/trimwise/releases/tag/paper-v1.1)

For the selection methods behind these results, see [Research Foundations](research-foundations.md).
