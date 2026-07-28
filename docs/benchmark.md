---
title: "Query-Aware Context Compression Benchmark | Trimwise"
description: "Explore Trimwise query-aware context compression results: strict source-span survival, output-token budgets, latency, and a reproducible 160-case evaluation."
---

# Query-Aware Context Compression Benchmark

Trimwise is for prompt assembly when an application already has a source and a question: keep the
exact source passages most useful for that question inside a fixed output budget. This evaluation
tests that scoped task. Each method receives the same source, question, and 128-, 256-, 512-, or
1,024-token ceiling.

## Strict source-span results

The primary result is **normalized contiguous required-span containment**. A case passes only
when every annotated source span appears as one contiguous passage after case-folding and
whitespace collapse, prohibited text is absent, and the output fits the shared token counter. It
measures complete survival of the annotated source evidence—not semantic sufficiency, answer
quality, or general prompt-compression capability.

| Evaluated method or adapter | 128 tokens | 256 tokens | 512 tokens | 1,024 tokens | Median warm trim time |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Trimwise Lexical** | **52.5%** | 60.0% | 61.9% | 66.2% | 6.4 ms |
| **Trimwise Hybrid** | 49.4% | **62.5%** | **66.9%** | **69.4%** | 42.8 ms |
| RECOMP NQ extractive sentence adapter | 27.5% | 30.6% | 35.0% | 35.0% | 126.2 ms |
| LLMLingua GPT-2 token-pruning adapter | 3.1% | 7.5% | 13.8% | 22.5% | 168.0 ms |
| LongLLMLingua GPT-2 single-context adapter | 0.6% | 4.4% | 6.9% | 16.2% | 340.7 ms |

![Normalized contiguous required-span containment by query-aware output-token budget on 160 position-controlled cases.](assets/benchmark/normalized_contiguous_vs_budget.png)

[Open the detailed strict benchmark report](benchmark-report/index.html).

Trimwise Lexical is the highest observed method at 128 tokens. Trimwise Hybrid is highest from 256
through 1,024 tokens. The two local ordered-retention sensitivity checks—80% and 90% ordered
recall inside one reference-length output window—preserve that ordering at every tested budget.

The report also separates the 135 natural-placement cases from the 25 controlled end relocations.
On the natural-only sensitivity, Hybrid passes 63.0% at 256 tokens against 26.7% for the RECOMP NQ
extractive sentence adapter; its 70.4% versus 30.4% result at 1,024 tokens has the same ordering.
That natural subset is not position-balanced because it contains only 15 natural end cases. A
separate 150-case sensitivity removes every source excerpt taken from Trimwise itself; Hybrid still
passes 63.3% at 256 tokens and 67.3% at 1,024 tokens.

![Strict case pass is reported separately for natural placement and controlled relocation cohorts.](assets/benchmark/natural_vs_relocated.png)

### What this result means

The strict metric was specified after a review found that the original scorer could credit
bag-of-token matches scattered across an output. The strict protocol was frozen before the new
aggregates were inspected and re-scores the same saved compressor outputs; it does not rerun a GPU
model or an API evaluator. It is a post-hoc robustness analysis, not a new independent experiment.

The full sensitivity CSV includes normalized contiguous containment, local ordered 80% and 90%
case pass, continuous local ordered recall, exact byte containment, and the earlier bag-of-token
metric. The older metric remains available for historical reproducibility only; it is not the
headline result on this page.

## Direct retrieval follow-up

A separate follow-up asks a narrower question: does a complete Trimwise configuration retain more
annotated source evidence than ordinary fixed-window retrieval? It adds BM25, embedding top-*k*,
and embedding-MMR over the same 160 cases, questions, and four budgets. All use 128-token windows,
32-token overlap, source-order reconstruction, and the same strict source-span scorer. This is a
new compression experiment with 1,920 outcomes, not a rescore of the original 6,400 outcomes; it makes no
downstream LLM calls.

| Method | 128 tokens | 256 tokens | 512 tokens | 1,024 tokens |
| --- | ---: | ---: | ---: | ---: |
| **Trimwise Lexical** | **52.5%** | 60.0% | 61.9% | 66.2% |
| **Trimwise Hybrid** | 49.4% | **62.5%** | **66.9%** | **69.4%** |
| BM25 fixed-window retrieval | 43.8% | 51.9% | 59.4% | 56.9% |
| Embedding top-*k* fixed-window retrieval | 33.1% | 40.6% | 47.5% | 50.0% |
| Embedding-MMR fixed-window retrieval | 33.1% | 43.1% | 48.1% | 53.1% |

![Strict source-span case pass by output-token budget for Trimwise and direct fixed-window retrieval baselines.](assets/benchmark/direct_retrieval_vs_budget.svg)

At 256 tokens, Hybrid exceeds BM25 by 10.6 percentage points (paired bootstrap 95% interval
[+1.9, +19.4]). That aggregate is not universal across positions: BM25 and embedding MMR are
stronger on the middle-evidence stratum at 256 tokens, while Trimwise is stronger for beginning,
end, and multiple-span evidence. This is a comparison of complete configurations, not a component
ablation.

[Open the direct-retrieval report](benchmark-report/direct-retrieval.html).

## Exploratory component study

A separate component study asks a different question: which parts of the complete Hybrid
configuration matter on this fixed suite? Every variant receives the same source, question, four
budgets, encoder, lexical--semantic fusion, source-order reconstruction, token counter, and strict
scorer. It changes one decision at a time:

- **Relevance-only:** remove the MMR diversity penalty.
- **No evidence cutoff:** keep every non-heading candidate instead of applying the adaptive
  evidence boundary.
- **Fixed windows:** replace Markdown-aware source segments with contiguous, non-overlapping
  128-token windows.

| Hybrid configuration | 128 tokens | 256 tokens | 512 tokens | 1,024 tokens |
| --- | ---: | ---: | ---: | ---: |
| Complete Hybrid | 49.4% | **62.5%** | 66.9% | 69.4% |
| Relevance-only (no MMR) | **51.2%** | 61.9% | 66.9% | 69.4% |
| No adaptive evidence cutoff | 44.4% | 61.9% | **67.5%** | **72.5%** |
| Fixed 128-token source windows | 40.0% | 58.8% | 61.9% | 71.2% |

![Strict case pass by output-token budget for Hybrid and three one-component variants.](assets/benchmark/ablation_vs_hybrid.svg)

At 128 tokens, full Hybrid is 5.0 percentage points above the no-cutoff variant (paired 95%
interval [+1.9, +8.8]) and 9.4 points above fixed windows ([+1.3, +17.5]). Those are the two
supported low-budget findings. Relevance-only selection is 1.9 points above full Hybrid at that
budget, with interval [-5.6, +1.9]; MMR therefore shows no consistent measured benefit here. At
256 tokens and above, the paired intervals for the cutoff and fixed-window variants include zero,
so the small higher-budget differences are not a general ranking of the variants.

This exploratory study adds 1,920 compression outcomes and uses no downstream language-model or
API calls. It is controlled within this implementation and suite, not proof that the same component
effects will transfer to other document collections, encoders, or budgets.

[Open the component-study report](benchmark-report/ablation.html).

### Budget compliance and speed

Trimwise Lexical and Hybrid and the RECOMP NQ extractive sentence adapter recorded no measured
budget violations under the shared counter. The LLMLingua GPT-2 token-pruning adapter exceeded its
requested ceiling in 30.6% of query-aware calls; the LongLLMLingua GPT-2 single-context adapter did
so in 55.8%. Warm timing excludes cold model loading and thermal cooldown, and is specific to the
recorded GPU and software environment.

## Downstream answer diagnostic

The saved benchmark also asks whether a compressed context can support a short answer. This is a
separate, weaker diagnostic over 93 answerable cases with one continuation per context from
GPT-5.4 Nano, GPT-5.4 Mini, and GPT-5.6 Luna. It uses normalized answer matching; it is neither a
human correctness study nor proof that a model relied only on the context.

![Downstream answer-match rate for GPT-5.4 Mini, GPT-5.4 Nano, and GPT-5.6 Luna at each context budget.](https://raw.githubusercontent.com/tenwritehq/trimwise/paper-v1.4/benchmark/reports/query-aware/answer_pass.png)

| Downstream evaluator | Full source | Trimwise Hybrid at 256 tokens |
| --- | ---: | ---: |
| GPT-5.4 Nano | 44.1% | 46.2% |
| GPT-5.4 Mini | 45.2% | 46.2% |
| GPT-5.6 Luna | 45.2% | 45.2% |

## Scope and limits

This is an author-annotated, position-controlled 160-case suite: 40 cases each with evidence at
the beginning, middle, end, or multiple locations. It combines 135 natural placements with 25
source-preserving end relocations. A natural-only sensitivity analysis excludes the relocations;
its end stratum has only 15 cases, so it is not position-balanced.

The comparison is against the exact adapters named above, not every configuration in the
LLMLingua, LongLLMLingua, or RECOMP families. The RECOMP NQ extractive sentence adapter is built
around the released checkpoint, not a reproduction of RECOMP's full retrieval and training pipeline.
The relocation cohort is a source-preserving construction diagnostic: moving evidence also changes
its neighboring text and the remaining source composition, so it is not interpreted as a pure causal
effect of position.
The component study isolates only MMR, the evidence cutoff, and source segmentation within Hybrid.
It does not separately establish the effects of semantic retrieval, score fusion, source-order
composition, or the embedding model.
These results do not establish performance for every language, document type, private corpus,
production agent workflow, or downstream model.

## Reproduce or inspect

The frozen rows, dataset, method configuration, runtime information, source hashes, and scripts are
public. The strict-metric manifest records the exact input identities and the scorer/protocol commit; the
summary is sufficient to inspect every reported metric without making GPU or API calls.

- [Benchmark protocol and reproduction commands](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/README.md)
- [Strict metric protocol](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/data/manifests/evidence_sensitivity_v1_2_protocol.md)
- [Strict frozen input manifest](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/data/manifests/evidence_sensitivity_v1_2_manifest.json)
- [Strict complete sensitivity summary](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/results/position_controlled_160_evidence_sensitivity_v1_2_summary.csv)
- [Strict natural-only sensitivity](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/results/position_controlled_160_evidence_sensitivity_v1_2_natural_only_summary.csv)
- [Strict self-source-excluded sensitivity](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/results/position_controlled_160_evidence_sensitivity_v1_2_without_self_sources_summary.csv)
- [Strict paired bootstrap intervals](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/results/position_controlled_160_evidence_sensitivity_v1_2_paired_stats.csv)
- [Direct-retrieval protocol and output identities](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/data/manifests/direct_retrieval_v1_protocol.md)
- [Direct-retrieval results](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/results/position_controlled_160_direct_retrieval_summary.csv)
- [Component-study protocol and output identities](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/data/manifests/ablation_v1_protocol.md)
- [Component-study results](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/results/position_controlled_160_ablation_summary.csv)
- [Component-study paired intervals](https://github.com/tenwritehq/trimwise/blob/paper-v1.4/benchmark/results/position_controlled_160_ablation_paired_stats.csv)
- [Historical source-only diagnostic](https://github.com/tenwritehq/trimwise/tree/paper-v1.1/benchmark/reports)
- [Current paper and PDF](https://github.com/tenwritehq/trimwise/releases/tag/paper-v1.4)

For the selection methods behind these results, see [Research Foundations](research-foundations.md).
