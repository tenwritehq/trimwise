---
title: "Trimwise Strategies: Structural to Hybrid"
description: Compare Trimwise structural, lexical, semantic, and hybrid strategies to choose the right way to preserve useful evidence for each prompt.
---

# Choosing a Strategy

A strategy controls how Trimwise estimates which source fragments are worth keeping. It does not
change the core guarantees: retained fragments come from the original input, return in source
order, and are measured as a complete result before Trimwise accepts them.

Start with `auto` unless you know that exact wording, semantic meaning, or both must drive the
selection.

## Quick decision guide

| Your situation | Choose | Why |
| --- | --- | --- |
| You have no question and need a useful overview | `auto` or `structural` | Covers the document structure and favors representative material |
| The task contains an exact ID, name, error, URL, command, or phrase | `auto` with a query, or `lexical` | BM25 rewards exact query evidence without loading a model |
| The answer may use different words or another supported language | `semantic` | Embeddings compare meaning rather than only shared terms |
| Exact identifiers and paraphrased explanations both matter | `hybrid` | Combines lexical and semantic evidence before selection |

```python
from trimwise import Trimmer

trimmer = Trimmer()

# No task is known: preserve a broad overview.
overview = trimmer.trim(document, limit=500)

# An exact identifier matters: auto resolves to lexical.
incident = trimmer.trim(document, limit=500, query="ORION-774")

# The document may phrase the answer differently.
cause = trimmer.trim(
    document,
    limit=500,
    strategy="semantic",
    query="Why did customers lose access?",
)

# Preserve both the incident ID and related explanations.
combined = trimmer.trim(
    document,
    limit=500,
    strategy="hybrid",
    query="What caused incident ORION-774?",
)
```

Exact lowercase strings and `Strategy` enum values are both accepted. `lexical`, `semantic`, and
`hybrid` require a nonblank query. `structural` does not use a supplied query.

## What every strategy shares

Before strategy-specific scoring begins, Trimwise splits the input into exact source-backed units.
Markdown can produce headings, paragraphs, lists, blockquotes, tables, HTML blocks, and code
fences. Plain text remains source-backed and can fall back to complete sentences or lines when a
larger unit cannot fit.

Every candidate receives two kinds of ranking information:

1. A **primary score** from structural centrality, BM25, embeddings, or hybrid fusion.
2. A small **language-neutral signal** for heading context, URLs, numbers or dates, and code-like
   identifiers.

Trimwise min-max normalizes the primary scores and gives them 90% of the relevance weight. The four
binary signals share the remaining 10%. This small signal helps preserve operational details when
primary scores are close; it cannot make unrelated text genuinely relevant.

For ranking only, a candidate can see its nearest heading and its previous and next units from the
same section. The candidate itself is anchored first in that scoring text so direct evidence stays
distinguishable from inherited context. None of this extra context is silently copied into the
result: selected fragments still come from their exact source locations.

After relevance scoring, Trimwise uses a greedy Maximal Marginal Relevance objective:

```text
selection score = λ × relevance − (1 − λ) × greatest similarity to selected evidence
```

The default `λ` is `0.7`, so selection weighs relevance at 70% and the similarity penalty at 30%.
Structural and lexical modes measure similarity with TF-IDF vectors; semantic and hybrid modes use
embedding similarity. This follows the relevance-versus-redundancy idea introduced by
[Maximal Marginal Relevance](https://aclanthology.org/X98-1025/).

MMR discourages fragments that look repetitive. It does not inspect factual claims, so differently
worded copies of one fact may survive and distinct facts with similar wording may suppress one
another.

## `auto`: the lightweight default

`auto` resolves entirely from whether a usable query exists:

| Call | Resolved strategy |
| --- | --- |
| `trim(text, 500)` | `structural` |
| `trim(text, 500, query=None)` | `structural` |
| `trim(text, 500, query="   ")` | `structural` |
| `trim(text, 500, query="ORION-774")` | `lexical` |

The resolved value appears in `result.strategy`. Automatic mode never chooses `semantic` or
`hybrid`, even when FastEmbed or an embedding callback is available. This prevents a normal call
from unexpectedly downloading a model, invoking an external service, or paying semantic inference
cost.

Use `auto` when:

- You want a dependable default across calls with and without tasks.
- Exact query terms are a useful signal when a task is available.
- Avoiding an embedding dependency or model call matters.

Choose an explicit strategy when paraphrases or multilingual meaning are important enough to
justify semantic scoring.

## `structural`: cover a document without a query

Structural mode is designed for articles, reports, notes, and other sources where no single task
defines relevance.

It does not use a fixed positional ratio such as 50/25/25. Instead, it:

1. Builds TF-IDF vectors from tiktoken subword IDs for each candidate and its local context.
2. Averages those vectors into a document centroid.
3. Scores each candidate by cosine similarity to that centroid.
4. Protects the first and last complete units when their composed form fits.
5. Gives every Markdown section an equal provisional share of the remaining budget.
6. Redistributes unused room globally using live MMR relevance and similarity scores.

If the first and last units do not fit together, Trimwise tries the more salient anchor, breaking a
tie toward the beginning. Anchors provide orientation; centroid scoring and section shares prevent
the introduction from consuming the entire result.

Centroid salience comes from established extractive-summarization work such as
[Centroid-based summarization](https://aclanthology.org/W00-0403/). Trimwise adds exact source spans,
Markdown section allocation, hard output measurement, and source-order composition around that
scoring idea.

### Plain text without Markdown

Plain text still uses structural selection. When parsing produces one oversized paragraph,
Trimwise promotes complete sentences or source lines into separate candidates before ranking. This
allows material from the middle and end to compete with the opening.

If a single source line has no recognized sentence or line boundary, there is nothing smaller to
compare. Under a very tight limit, the final fallback is therefore the longest exact source prefix
that fits.

### When structural works well

- Creating a compact overview before clustering or classifying documents.
- Giving every article in a multi-source prompt broad coverage.
- Preserving decisions and conclusions when no task-specific query exists.
- Reducing long Markdown notes while keeping section context.

### Where structural can miss

Centrality estimates what resembles the document's overall vocabulary. A rare but critical fact
may score below repeated background material. The opening and closing anchors, section shares, and
fact-like signal reduce this risk but cannot eliminate it.

## `lexical`: preserve exact query evidence

Lexical mode uses BM25 with `k1=1.5` and `b=0.75`. It is the resolved `auto` behavior whenever a
nonblank query is supplied.

Before scoring, Trimwise normalizes ranking-only text with Unicode NFKC normalization, case
folding, and collapsed whitespace. It then uses tiktoken subword IDs rather than whitespace words.
That keeps lexical evidence available in languages such as Chinese and Japanese and avoids
discarding identifiers that ordinary word tokenizers may split poorly. Retained source text is
never normalized.

BM25 rewards query terms that occur in a candidate, gives more weight to terms that are rare across
the candidate set, and adjusts for candidate length. The implementation follows the Robertson
probabilistic relevance formulation described in
[BM25 and Beyond](https://doi.org/10.1561/1500000019).

```python
from trimwise import Trimmer

result = Trimmer().trim(
    document,
    limit=500,
    strategy="lexical",
    query="Where is error ORION-774 handled?",
)
```

### When lexical works well

- Incident IDs, ticket numbers, versions, dates, and product names.
- Function names, configuration keys, commands, and exact error messages.
- Questions where the source and query are likely to share important wording.
- Fast query-aware trimming without an embedding model.

### Where lexical can miss

BM25 does not know that “credentials were rejected” may answer “why did authentication fail?” when
the passages share few useful subwords. Use `semantic` or `hybrid` when differently worded evidence
must be found.

## `semantic`: preserve meaning and paraphrases

Semantic mode compares one query vector with one vector for every candidate passage using cosine
similarity. Candidate-to-candidate embedding similarity is then used for MMR repetition reduction.
This follows the sentence-vector approach established by
[Sentence-BERT](https://aclanthology.org/D19-1410/) and related
[multilingual sentence-embedding work](https://aclanthology.org/2020.emnlp-main.365/).

```python
from trimwise import Trimmer

result = Trimmer().trim(
    document,
    limit=500,
    strategy="semantic",
    query="Why were users unable to sign in?",
)
```

Semantic vectors can come from:

- Your synchronous embedding callback with `trim()` or `atrim()`.
- Your asynchronous embedding callback with `atrim()`.
- Trimwise-managed FastEmbed after installing `trimwise[semantic]` or
  `trimwise[semantic-gpu]`.

A supplied callback takes precedence over FastEmbed. Without a callback, the default multilingual
FastEmbed model is loaded lazily only when a semantic or hybrid call both needs trimming and reaches
ranking. Missing packages, model initialization, downloads, providers, and inference failures
raise `SemanticBackendError`; Trimwise never silently substitutes lexical scoring.

### When semantic works well

- Questions and answers that use different vocabulary.
- Conceptual evidence where exact keyword overlap is unreliable.
- Cross-language matching supported by the chosen embedding model.
- Applications that already have an embedding service or local model.

### Where semantic can miss

Semantic quality, supported languages, memory use, and latency depend on the embedding model. A
high cosine score is not proof that a passage answers the task, and changing models changes the
score distribution. The first Trimwise-managed call may also be much slower because it can download
and initialize the model.

## `hybrid`: preserve exact terms and broader meaning

Hybrid mode calculates both BM25 and semantic query scores for every candidate. When both score
rows are finite and contain meaningful variation, Trimwise min-max normalizes each row and combines
them with a fixed equal blend:

```text
hybrid score = 0.5 × normalized BM25 + 0.5 × normalized semantic similarity
```

This preserves score magnitude instead of reducing every usable result to rank positions. The
choice is informed by research comparing lexical-semantic fusion methods, including
[an analysis of hybrid fusion functions](https://arxiv.org/abs/2210.11934), but Trimwise's fixed
50/50 blend is a label-free default rather than a learned optimum for every dataset.

If either score row is constant or contains a non-finite value, normalized magnitude is not usable.
Trimwise then falls back to one-indexed Reciprocal Rank Fusion with `k=60`, following the stable
rank-combination idea from
[Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114). RRF is a defensive fallback, not
the normal hybrid path.

```python
from trimwise import Trimmer

result = Trimmer().trim(
    document,
    limit=500,
    strategy="hybrid",
    query="What caused incident ORION-774?",
)
```

Hybrid uses embedding similarity for MMR after fusion. It therefore has the same embedding
installation or callback requirement as semantic mode and also pays the lexical-scoring cost.

### When hybrid works well

- A precise incident ID plus a prose explanation of its cause.
- Product names or versions alongside semantically related recommendations.
- Technical questions where identifiers must survive but wording may vary.
- Tasks where missing either exact or paraphrased evidence is costly.

### Where hybrid can miss

An equal blend is deliberately simple and needs no training data, but it may not be ideal for your
domain. Trimwise does not expose a hybrid weight because a new public tuning knob should be backed
by representative quality evidence rather than guesswork.

## How query-aware selection decides when to stop

Lexical, semantic, and hybrid modes do not force the beginning or end into the output. After
ranking non-heading evidence, Trimwise:

1. Orders candidates by the strategy's primary query score.
2. Finds the clearest score drop while ignoring an extreme bottom-tail gap.
3. Keeps the candidates through that boundary plus a fixed five-candidate recall buffer.
4. Runs MMR only inside that bounded pool.
5. Tries to attach each selected passage's nearest heading when both still fit.

This score-shape rule is independent of the absolute scale, which matters because BM25 values and
embedding cosine distributions are not directly comparable. It adapts the principle of selecting
an evidence count per query from
[Adaptive-k context selection](https://aclanthology.org/2025.emnlp-main.1017/) without claiming to
implement that paper's exact thresholding system.

The recall buffer protects nearby candidates when the largest drop is too aggressive. It also
means the pool is not a strict “everything above this relevance score” filter. Once the pool is
exhausted, Trimwise stops—even if space remains—rather than scanning the weak tail merely to fill
the requested limit.

## Exact budgets and source order

Ranking order never becomes output order. Whenever Trimwise considers a candidate, it reconstructs
all currently selected fragments in their original source order, includes required separators and
any affordable omission markers or attached headings, and measures that complete proposal. The
candidate is accepted only if the result fits.

This guarantees `output_count <= limit`, but selection remains a greedy heuristic. It does not
solve a global knapsack problem that proves one large fragment is better or worse than several
smaller fragments with the same total cost. Under severe compression, benchmark the resulting LLM
answers on your own documents and tasks.

## Strategy comparison

| Property | `structural` | `lexical` | `semantic` | `hybrid` |
| --- | --- | --- | --- | --- |
| Query required | No; supplied query is ignored | Yes | Yes | Yes |
| Embeddings required | No | No | Yes | Yes |
| Primary relevance | TF-IDF centroid cosine | BM25 | Query-passage cosine | Equal normalized BM25/semantic blend, with RRF fallback |
| MMR similarity | TF-IDF cosine | TF-IDF cosine | Embedding cosine | Embedding cosine |
| Forces start/end anchors | When affordable | No | No | No |
| Gives sections an initial share | Yes | No | No | No |
| May stop below the limit | When no complete candidate fits | Yes, after bounded evidence is exhausted | Yes, after bounded evidence is exhausted | Yes, after bounded evidence is exhausted |
| Main runtime cost | Parsing and sparse scoring | Parsing and sparse scoring | Embedding inference | BM25 plus embedding inference |

## Practical recommendations

- Start with `auto` and inspect `result.strategy` in logs.
- Use `structural` for queryless overviews, not a made-up generic query.
- Use `lexical` when exact evidence is likely to appear in both query and source.
- Move to `semantic` only when paraphrases or multilingual matching materially matter.
- Use `hybrid` when exact identifiers and semantic evidence are both required.
- Reuse one `Trimmer` for Trimwise-managed FastEmbed so later calls reuse the loaded model.
- Evaluate answer quality at your real document lengths and compression ratios; research-backed
  components do not remove the need for task-specific testing.

## Continue exploring

- Return to the [Trimwise overview](index.md).
- Follow the [Getting Started guide](getting-started.md).
- Read the current [semantic model and callback guide](https://github.com/tenwritehq/trimwise#semantic-models).
- Review planned quality work in the [roadmap](https://github.com/tenwritehq/trimwise/blob/main/ROADMAP.md).
