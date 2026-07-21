---
title: How Trimwise Selects Useful Text
description: See how Trimwise measures input, maps source structure, ranks relevant passages, selects complementary evidence, and composes an exact-budget result.
---

# How Trimwise Works

Trimwise treats truncation as evidence selection under a hard budget. It does not rewrite,
summarize, or generate replacement text. Instead, it finds meaningful units in the supplied
source, estimates their usefulness, chooses complementary units, restores source order, and
measures the complete excerpt before returning it.

```text
source text
    │
    ├─ validate and measure
    ├─ segment into exact source spans
    ├─ score spans for salience or query relevance
    ├─ select useful, less-repetitive evidence
    ├─ compose in original order with affordable gap markers
    └─ remeasure ──> TrimResult
```

Trimwise operates only on the string you provide. It does not search, retrieve documents, query an
index, or create a RAG pipeline.

## The five-stage pipeline

Every strategy follows the same high-level path:

1. **Measure:** validate the request and return immediately when the source already fits.
2. **Segment:** map Markdown or plain text into complete, non-overlapping source units.
3. **Rank:** estimate usefulness with document centrality, BM25, embeddings, or hybrid fusion.
4. **Select:** balance relevance against similarity to evidence already retained.
5. **Compose:** restore source order, add affordable omission markers, and enforce the exact limit.

The strategy changes ranking and selection behavior. Measurement, source fidelity, composition,
fallback, and the final budget guarantee stay shared.

## Stage 1: validate and measure

Trimwise resolves public arguments before doing document work:

- `auto` becomes `structural` without a nonblank query.
- `auto` becomes `lexical` with a nonblank query.
- Explicit lexical, semantic, and hybrid calls require a nonblank query.
- Explicit structural calls ignore a supplied query for ranking.
- The text, limit, budget unit, strategy, query, and custom counter are validated.

The complete input is then measured in the requested unit:

| Unit | Measurement rule |
| --- | --- |
| Tokens | Tiktoken IDs from `o200k_base` by default, or the supplied custom counter |
| Words | `len(text.split())` |
| Characters | Python Unicode code points from `len(text)` |

If `limit == 0`, Trimwise returns empty text after measuring the input. If the input count is at
most the limit, it returns the original string exactly and stops before Markdown parsing, ranking,
callback invocation, or FastEmbed loading.

```python
from trimwise import Trimmer

source = "  # Heading\r\n\r\nExact text.  "
result = Trimmer().trim(source, len(source), unit="characters")

assert result.text == source
assert result.trimmed is False
```

This fast path matters for agentic pipelines that trim many already-small sources: a configured
semantic backend does not turn every call into an embedding call.

## Stage 2: map structure without rewriting source

For oversized input, Trimwise creates a fresh `markdown-it-py` parser for that call. It uses the
CommonMark rules plus table support. Parser block tokens provide source line ranges; Trimwise maps
those line ranges back to character offsets and slices the original string.

Each candidate stores:

```text
start offset  ─┐
end offset     ├─ exact source identity
source text   ─┘
block kind
section number
nearest preceding heading
```

The important invariant is:

```text
candidate.text == original_text[candidate.start:candidate.end]
```

No Markdown renderer is involved, so selected wording, whitespace, indentation, link syntax, list
markers, HTML, and code-fence language tags are not normalized on the way out.

### Recognized source units

| Source structure | Candidate behavior |
| --- | --- |
| ATX and Setext headings | Start a new numbered section and provide context to following units |
| Paragraphs | Become complete paragraph candidates |
| Ordered and unordered lists | Preserve list-backed source slices, including nested items |
| Blockquotes | Remain quoted source units |
| Tables | Remain complete table source blocks |
| Indented and fenced code | Remain code-backed candidates |
| HTML blocks | Remain exact HTML source slices |
| YAML-style front matter | A closed leading `---` block is retained as one raw unit |
| Reference definitions and other uncovered text | Become raw candidates rather than disappearing |

Whitespace-only input becomes one raw source unit. Empty input produces no candidates and normally
returns through the fitting-input path.

Parser spans can overlap or nest. Trimwise removes contained duplicates, prefers richer block
types for identical spans, clips unusual partial overlaps, and inserts raw units for nonblank
ranges that no parser block covered. The final candidate list is ordered and non-overlapping.

### Sections and headings

Every recognized heading starts a section. Following candidates remember the nearest heading until
another heading begins. That relationship supports two later behaviors:

- Structural mode can give each section an initial share of the budget.
- Query-aware modes can attach the selected passage's heading when both still fit.

### One enormous plain-text paragraph

Markdown may see a long plain-text document as one paragraph. A single candidate gives structural
ranking nothing to compare, so Trimwise expands that sole paragraph into independently rankable
complete sentences or source lines when boundaries exist.

Sentence boundaries include ordinary `.`, `!`, and `?`, plus the Unicode ellipsis, ideographic full
stop, and full-width exclamation and question marks used in CJK text. Source lines also provide
boundaries when punctuation is absent.

```python
from trimwise import Trimmer

source = (
    "Opening context. "
    "Repeated operating detail alpha beta. "
    "A decision was approved for September 2026. "
    "Repeated operating detail alpha gamma. "
    "Closing context."
)

result = Trimmer().trim(source, 85, unit="characters")

assert result.output_count <= 85
assert result.text.startswith("Opening context.")
assert result.text.endswith("Closing context.")
```

This expansion currently belongs to structural selection. If the source is one unpunctuated line,
there is still no smaller complete unit to rank; a tight budget eventually requires exact prefix
fallback.

## Stage 3A: build ranking context

A short candidate can be ambiguous in isolation. For scoring only, Trimwise gives each candidate a
small local window containing:

1. The candidate itself, placed first.
2. Its nearest section heading, when present.
3. The previous candidate from the same section.
4. The candidate in its source position.
5. The following candidate from the same section.

Duplicate indexes are removed from the window. The candidate is intentionally repeated at the
front when a larger window exists, preserving a stronger signal for words found directly in that
candidate rather than only in a neighbor.

For example, `It increased by 18%.` becomes easier to score when the ranking text also contains a
heading such as `## Database latency` and the adjacent explanation. Selection still retains only
exact source candidates; the contextual window is never emitted as synthetic output.

### Ranking-only normalization

Structural and lexical scoring normalize their ranking text with:

- Unicode NFKC normalization.
- Unicode case folding.
- Collapsed whitespace.
- One consistent leading space before tiktoken encoding.

The normalized text becomes tiktoken subword IDs. Using subwords instead of whitespace-only terms
keeps lexical evidence available in scripts such as Chinese and Japanese and handles technical
identifiers more consistently. Original source slices are never normalized.

## Stage 3B: estimate usefulness

The resolved strategy provides one primary score for every candidate:

| Strategy | Primary score | Research foundation |
| --- | --- | --- |
| Structural | Cosine similarity between candidate TF-IDF and the document centroid | [Centroid-based summarization](https://aclanthology.org/W00-0403/) |
| Lexical | Robertson BM25 with `k1=1.5` and `b=0.75` | [BM25 and Beyond](https://doi.org/10.1561/1500000019) |
| Semantic | Cosine similarity between normalized query and passage embeddings | [Sentence-BERT](https://aclanthology.org/D19-1410/) |
| Hybrid | Equal blend of normalized BM25 and semantic scores when both rows are usable | [Hybrid fusion analysis](https://arxiv.org/abs/2210.11934) |

These papers motivate individual ideas. Trimwise adapts them to source-backed fragments, exact
budgets, structural coverage, and deterministic composition rather than reproducing any complete
paper system.

### Structural centrality

Trimwise builds an L2-normalized sparse TF-IDF vector for every contextual candidate. Term
frequency is adjusted by candidate length; inverse document frequency rewards terms that do not
appear in every candidate. The mean of all candidate vectors becomes the document centroid, and
candidate-centroid cosine similarity estimates how representative each unit is.

Centrality is useful without a query, but it can undervalue a rare critical fact. Structural
anchors, section shares, MMR, and the fact-like signal reduce that risk without claiming to remove
it.

### BM25 relevance

Lexical mode compares query subword IDs with every contextual candidate using Robertson IDF and
BM25 length normalization. Rare matching terms contribute more than terms repeated throughout the
document, making IDs, error strings, dates, and names strong evidence.

BM25 matches lexical evidence; it does not understand that two differently worded sentences can
mean the same thing.

### Semantic relevance

Semantic mode obtains one query vector and one vector per contextual candidate. Trimwise converts
them to one `float32` matrix, validates dimensions and finite values, normalizes each row once, and
computes all query-passage cosine scores with a matrix-vector operation.

Semantic quality depends on the supplied embedding model. Similarity measures representation, not
truth, currency, authority, or answer completeness.

### Hybrid fusion

When both lexical and semantic score rows are finite and contain variation, Trimwise min-max
normalizes each row and blends them equally:

```text
hybrid = 0.5 × normalized BM25 + 0.5 × normalized semantic score
```

If either row is flat or non-finite, score magnitude cannot be compared meaningfully. Trimwise then
uses stable one-indexed Reciprocal Rank Fusion with `k=60`, following
[RRF](https://doi.org/10.1145/1571941.1572114). RRF is a defensive fallback, not the normal hybrid
path.

### Small structural and fact-like signal

After min-max normalization, the strategy's primary score receives 90% of final relevance. The
remaining 10% is the mean of four binary indicators:

- Candidate is a heading or belongs under one.
- Candidate contains a URL.
- Candidate contains a number or date-like value.
- Candidate contains a snake_case or camelCase-style identifier.

This language-neutral signal helps operational evidence survive when primary scores are close. It
does not override a large relevance difference, and it cannot determine whether a number, URL, or
identifier is useful to the task.

## Stage 4: choose complementary evidence

Trimwise selects candidates greedily. After each choice, it updates the greatest similarity between
every remaining candidate and anything already selected. The next choice maximizes:

```text
mmr_lambda × relevance − (1 − mmr_lambda) × maximum selected similarity
```

The default `mmr_lambda=0.7` gives relevance 70% of the objective and the similarity penalty 30%.
This follows the relevance-versus-redundancy idea from
[Maximal Marginal Relevance](https://aclanthology.org/X98-1025/).

Structural and lexical modes use TF-IDF cosine similarity for MMR. Semantic and hybrid modes use
nonnegative embedding cosine similarity and update all remaining similarities with vectorized
matrix operations.

MMR is a duplicate-control proxy, not factual analysis. Similar wording can describe separate
facts, while different wording can repeat one fact.

### Structural selection path

Structural mode selects in three phases:

1. **Protect anchors.** Try the first and last complete candidates together. If they do not fit,
   try the more relevant one, breaking equal-score ties toward the first.
2. **Cover sections.** Divide the remaining measured capacity equally across all Markdown sections
   and select fitting MMR candidates inside each provisional share.
3. **Redistribute globally.** Spend unused room on the strongest fitting candidates from any
   section.

The shares are provisional, not hard quotas. A section whose units are too large does not waste its
unused room; later global selection can use it elsewhere.

This is why structural mode is not a 50/25/25 slice. Position provides orientation through anchors,
while centrality, section coverage, similarity, and exact cost decide the rest.

### Query-aware selection path

Lexical, semantic, and hybrid modes do not force opening or closing anchors. They first remove
standalone heading candidates when non-heading evidence exists, then bound the candidate pool:

1. Order candidates by the strategy's primary query score.
2. Search the first 90% of score gaps for the largest drop, avoiding an extreme tail outlier.
3. Keep candidates through that boundary plus a five-candidate recall buffer.
4. Run MMR inside the bounded pool.
5. Try each selected passage with its nearest heading; if the pair is too large, try the passage
   alone.

The cutoff depends on score shape rather than an absolute BM25 or cosine threshold. This matters
because different models and queries produce different score scales. It adapts the central idea of
query-specific evidence counts from
[Adaptive-k context selection](https://aclanthology.org/2025.emnlp-main.1017/) without implementing
that paper's exact threshold method.

When the bounded pool is exhausted, query-aware selection stops even if capacity remains. The
limit means “at most,” not “fill with weak evidence.” The five-candidate buffer protects recall, so
the pool is not a strict relevance guarantee.

### Deterministic ties and complexity

Equal ranking outcomes break toward earlier source candidates. Given the same text, configuration,
counter behavior, and semantic vectors, Trimwise selection is deterministic.

The sparse MMR path keeps the straightforward exact `O(selected × candidates)` comparison loop.
Semantic and hybrid modes maintain the same exact objective but update remaining similarities in a
vectorized NumPy operation. Approximate-neighbor selection is intentionally deferred until
profiling representative workloads proves it necessary.

## Stage 5: compose within the exact budget

Ranking order never becomes output order. For every proposed addition, Trimwise:

1. Sorts all selected candidates by their original source position.
2. Inserts exact whitespace for untouched gaps or minimal separators for omitted nonblank gaps.
3. Composes the retained fragments without optional omission markers.
4. Measures that complete proposal.
5. Rejects the addition if retained content and required separators exceed the limit.
6. Otherwise, tries optional omission markers one gap at a time in source-gap order.
7. Carries the retained original-input ranges into the final result.

This trial composition accounts for the actual cost of headings, separators, and markers instead
of estimating candidate cost in isolation.

### Leading, internal, and trailing gaps

For every nonblank omitted range, Trimwise can place the configured marker—`[…omitted…]` by
default—before, between, or after retained fragments. Markers are attempted only after the retained
content already fits.

If a marker would exceed the limit, Trimwise leaves it out rather than discarding source evidence.
Internal nonblank gaps still receive a minimal newline separator so distant fragments do not run
together. Marker boundaries use no more than one blank line while preserving retained fragment
content and whitespace.

```python
from trimwise import TrimConfig, Trimmer

source = "FIRST important.\n\nMiddle filler.\n\nLAST important."
trimmer = Trimmer(TrimConfig(omission_marker="<cut>"))
result = trimmer.trim(source, 39, unit="characters")

assert result.output_count <= 39
```

Omission markers are the only configured explanatory text Trimwise adds. They indicate a source
gap; they are not summaries of what was removed.

## When no complete candidate fits

If selection cannot retain any complete candidate, Trimwise chooses the strongest candidate by
final relevance, breaking ties toward the earlier source position, and progressively shrinks it.

For ordinary text, fallback prefers:

1. The longest fitting complete paragraph prefix.
2. The longest fitting complete sentence or source-line prefix.
3. The longest exact source prefix accepted by the active measurer.

This preserves readable boundaries when possible without violating the limit.

### Closed and unclosed code fences

For a genuinely closed backtick or tilde fence, Trimwise checks that the closing marker uses the
same character and at least the opening marker's length. If the original opening-and-closing shell
fits, it removes whole body lines until the fenced fragment fits.

```python
from trimwise import Trimmer

source = "```python\n" + "print('large')\n" * 20 + "```\n"
result = Trimmer().trim(source, 35, unit="characters")

assert result.text.startswith("```python\n")
assert "\n```" in result.text[len("```python\n") :]
assert result.output_count <= 35
```

If even the fence shell is too large, the ordinary source-prefix fallback applies. A fence that
was unclosed in the source remains unclosed; Trimwise does not invent a closing marker.

### Exact prefix measurement

Character fallback can slice directly by code-point position. Word fallback uses complete
whitespace-delimited spans. Built-in token fallback uses tiktoken offsets when the encoded text
round-trips exactly; otherwise it scans source prefixes.

A custom token counter may be non-monotonic—for example, adding one character can merge several
tokens. Trimwise therefore scans candidate prefixes rather than assuming a binary-search boundary,
keeping the final fit exact at the cost of more callback calls on this rare path.

After content fallback, affordable leading and trailing omission markers are tried without removing
the retained fragment.

## Final measurement and result

Trimwise measures the finished string one final time. If an internal bug ever produced a result
above the requested limit, the implementation raises `RuntimeError` instead of returning an invalid
`TrimResult`.

The normal result records:

```text
text          final extractive excerpt
input_count   original measured size
output_count  final measured size
limit         caller-defined ceiling
unit          concrete budget unit
strategy      resolved concrete strategy
trimmed       whether output text differs from input
spans         ordered original-input ranges retained in the output
```

Span offsets follow Python string slicing: starts are inclusive and ends are exclusive. Adjacent
source-backed ranges are merged; overlapping ranges cannot arise from the nonoverlapping source
candidates. Generated omission markers and minimal separators have no source span.

The guarantee applies to `result.text`, not to the larger prompt assembled around it. Source labels,
instructions, examples, tool schemas, separators, and model output still need their own context
space.

## What source fidelity means

Retained candidate text always comes from exact source slices and returns in original order. The
complete result may differ from one contiguous source substring because Trimwise can:

- Select non-adjacent fragments.
- Add the configured omission marker where it fits.
- Add minimal newlines between separated fragments.
- Attach an original section heading to a selected query-aware passage.
- Shorten the strongest candidate through source-prefix fallback when no complete unit fits.

Trimwise never paraphrases retained text, synthesizes a transition, repairs malformed Markdown, or
combines distant facts into a new sentence.

## Research-backed components and product behavior

Research provides scoring and selection ideas; it does not provide Trimwise's complete product
contract.

| Research contribution | Trimwise adaptation |
| --- | --- |
| Centroid salience | Queryless ranking over contextual source candidates |
| BM25 | Model-free exact-query evidence over tiktoken subwords |
| Sentence embeddings | Optional semantic relevance and similarity |
| Hybrid fusion and RRF | Equal normalized score blend with rank-based defensive fallback |
| MMR | Greedy relevance-versus-similarity selection under exact composition checks |
| Adaptive evidence count | Score-shape candidate pool with a small recall buffer |

Trimwise adds Markdown source spans, section shares, anchors, fact-like signals, contextual scoring,
source-order composition, omission behavior, exact measurement, and progressive fallback. Those
adaptations are implementation choices tested by the package; citing the component research does
not prove that Trimwise is optimal for every downstream LLM task.

## Important algorithmic limits

- Usefulness scores do not understand factual truth, authority, recency, or completeness.
- MMR similarity discourages repeated-looking evidence but does not prove factual diversity.
- Greedy selection does not solve a global token-cost knapsack or facility-location optimum.
- Structural centroid scoring can undervalue rare but critical details.
- Lexical scoring can miss paraphrases; semantic scoring inherits model quality and bias.
- A single unpunctuated line eventually requires prefix truncation.
- Extractive output cannot synthesize two distant facts into one denser sentence.

At extreme compression ratios, a trained token-level or abstractive compressor may preserve more
task-relevant meaning per token. It also adds model cost and gives up Trimwise's exact-source
guarantee. Evaluate the final LLM answers—not only the selected excerpts—on representative data.

## Continue exploring

- Return to the [Trimwise overview](index.md).
- Follow the [Getting Started guide](getting-started.md).
- Compare ranking paths in [Choosing a Strategy](strategies.md).
- Configure models and callbacks in [Semantic Models and Async Usage](semantic-and-async.md).
- Review all public settings in [Configuration and API Reference](configuration-and-api.md).
- See deferred quality work in the [roadmap](https://github.com/tenwritehq/trimwise/blob/main/ROADMAP.md).
