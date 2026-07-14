# Trimwise roadmap

Version 1 deliberately stays extractive and dependency-light. Consider these only after real usage
or benchmarks demonstrate a need:

- Language detection and language-specific lexical tokenizers.
- A standalone multilingual semantic-quality and regression benchmark.
- Public tuning for the adaptive query cutoff or recall buffer, only if that benchmark shows one
  fixed distributional rule is not robust across embedding models and document types.
- Language-model self-information scoring inspired by Selective Context.
- Format-specific parsers for JSON, chat transcripts, and source code.
- Batch or streaming APIs and cross-call embedding caches.
- Approximate-neighbor MMR if profiling shows candidate selection is a bottleneck.

## Evidence coverage beyond MMR

MMR reduces representational redundancy; it does not prove factual diversity. Different wording
can make the same fact appear dissimilar, while distinct facts about the same entities can look
similar enough that one suppresses the other. Numbers, negation, dates, and relationships are
especially easy for vector similarity alone to mishandle.

The goal is therefore to improve source-grounded evidence coverage, not to claim a factual
diversity guarantee. Benchmark an internal selection objective shaped like:

```text
selection gain = relevance - similarity penalty + new evidence coverage
```

The coverage term could reward previously unselected:

- Markdown sections or query facets.
- Numbers, dates, URLs, identifiers, and code symbols.
- Rare query-relevant terms.
- Entity-value combinations such as `revenue + $4.2M` or `version + 3.12`.

Coverage must not rescue irrelevant material. In query-aware modes, only candidates admitted by
the relevance cutoff should be eligible for a coverage bonus. Any implementation should remain
extractive, deterministic, dependency-light, and composed from exact source spans. Keep its weight
internal until quality evidence shows that callers need to tune it.

Do not change the default selector from tests alone. Unit tests can establish budget compliance,
determinism, and source fidelity, but only a quality benchmark can show whether more useful facts
survive. Compare:

1. Existing MMR with exact fit checks during composition.
2. MMR marginal gain divided by exact composed budget cost.
3. MMR with lightweight evidence coverage.
4. Facility-location coverage selected by marginal gain per exact composed budget cost.
5. The best single candidate as a fallback for greedy selection.

The benchmark should include same-fact paraphrases, distinct facts sharing vocabulary, repeated
entities with different attributes, changed numbers or dates, negation, contradictions,
multilingual text, and technical identifiers. Measure labelled fact recall per retained token,
query relevance, redundancy, latency, and peak memory. Adopt a coverage term only when it improves
fact recall per token without materially degrading the other measures.

True atomic-fact comparison would require a heavier claim-extraction layer such as OpenIE, NLI, or
an LLM. That would add model cost, language-dependent errors, and a larger operational footprint,
so it remains outside the dependency-light core. Submodular summarization provides the relevant
importance/coverage/non-redundancy framework ([Lin and Bilmes, 2010](https://aclanthology.org/N10-1134/));
[FActScore](https://aclanthology.org/2023.emnlp-main.741/) illustrates how much more machinery is
needed to evaluate genuinely atomic factual claims.

Abstractive summarization, silent strategy fallback, and automatic installation of model runtimes
are not planned because they weaken source fidelity or operational predictability.
