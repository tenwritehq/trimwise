# Guarantees and Limitations

Trimwise is an extractive preprocessor for creating smaller evidence excerpts before prompt
assembly. It offers strong guarantees about measurement and source fidelity. It offers useful—but
necessarily heuristic—behavior for relevance, coverage, and repetition reduction.

The distinction matters:

| Category | Meaning |
| --- | --- |
| **Hard guarantee** | Enforced directly by validation, source spans, composition, and final measurement |
| **Deterministic policy** | Predictable behavior given the same input, configuration, counter, and semantic vectors |
| **Best-effort goal** | A ranking objective that can improve evidence selection but cannot prove correctness |

## Hard guarantees

### The returned text stays within its measured limit

Trimwise measures the complete output after selection, source-order composition, separators,
headings, and affordable omission markers have been applied. It returns only when:

```text
result.output_count <= result.limit
```

The guarantee uses the requested measurement rule:

- Tiktoken or your custom counter for token budgets.
- Whitespace-separated parts for word budgets.
- Python Unicode code points for character budgets.

```python
from trimwise import Trimmer

source = "First fact. Middle explanation. Final decision. " * 20

for unit, limit in (("tokens", 30), ("words", 20), ("characters", 100)):
    result = Trimmer().trim(source, limit, unit=unit)
    assert result.output_count <= limit
```

This guarantee applies to `result.text`, not to the larger prompt around it. Instructions, source
labels, separators, examples, tool definitions, output schemas, and the model's answer all need
their own context space.

### Input that already fits is returned exactly

After argument validation and input measurement, Trimwise returns a fitting source without parsing
Markdown, ranking candidates, or invoking an embedding backend.

```python
from trimwise import Trimmer

source = "  # Heading\r\n\r\nExact text.  "
result = Trimmer().trim(source, len(source), unit="characters")

assert result.text == source
assert result.input_count == result.output_count == len(source)
assert result.trimmed is False
```

Spaces, tabs, line endings, and Markdown syntax remain unchanged on this path.

### Retained fragments come from the source

Markdown parser line ranges are converted back into character offsets. Every selected candidate is
an exact slice of the input:

```text
candidate text = source[start:end]
```

Trimwise does not paraphrase selected fragments or render Markdown back into a normalized form.
Retained wording, indentation, link syntax, list markers, HTML, tables, and code-fence language tags
come from the original source.

### Retained fragments return in source order

Candidates are ranked by usefulness, but they are never emitted in ranking order. Before every
fit check, Trimwise sorts selected spans by their original position and composes them in that order.

This prevents a highly ranked conclusion from appearing before an earlier retained premise merely
because its score was higher. It does not guarantee that omitted context between the two was
unimportant.

### Content takes priority over omission markers

Trimwise first verifies that retained source fragments and required separators fit. It then tries
to add leading, internal, and trailing omission markers in source-gap order. A marker that would
break the budget is omitted rather than displacing retained evidence.

```python
from trimwise import TrimConfig, Trimmer

config = TrimConfig(omission_marker="marker-too-large")
result = Trimmer(config).trim("abcdef", 2, unit="characters")

assert result.text
assert "marker" not in result.text
assert result.output_count <= 2
```

A missing marker therefore does not mean that the excerpt is contiguous. Keep the original source
available when exact omitted ranges matter.

### Semantic failures do not silently change strategy

When explicit semantic or hybrid ranking needs embeddings, callback and FastEmbed failures raise a
chained `SemanticBackendError`. Trimwise does not quietly replace the requested strategy with
lexical scoring.

This makes operational and quality failures visible. A caller may deliberately catch the error and
issue a second call with another strategy, but that fallback is an application decision.

## Deterministic policies

### Automatic strategy resolution is predictable

| Request | Resolved behavior |
| --- | --- |
| `auto` without a nonblank query | Structural |
| `auto` with a nonblank query | Lexical |
| Explicit structural | Structural; supplied query is ignored for ranking |
| Explicit lexical, semantic, or hybrid | Same strategy; nonblank query required |

Automatic mode never loads an embedding model or invokes an embedding callback. Semantic work is
opt-in through an explicit semantic or hybrid strategy.

### Identical scoring inputs produce stable selection

Equal ranking outcomes break toward earlier source positions. With the same source, configuration,
counter results, and semantic vectors, candidate selection and composition are deterministic.

The qualification matters: an external callback, GPU provider, model service, or custom counter
can itself be nondeterministic. Trimwise cannot make unstable backend outputs stable.

### Query-aware selection treats the limit as a ceiling

Lexical, semantic, and hybrid modes bound their evidence pool around the clearest primary-score
drop and keep a five-candidate recall buffer. Once that pool is exhausted, selection stops rather
than scanning the weak tail just to fill remaining capacity.

The result may therefore be shorter than the limit. This is intentional, but it is not a guarantee
that every retained passage is relevant: the recall buffer can preserve borderline candidates,
and ranking signals can be wrong.

### Async execution has defined cancellation boundaries

`atrim()` moves synchronous work to worker threads. A native async embedding callback is awaited on
the calling event loop.

- Cancellation propagates into a currently awaited async callback.
- Cancellation stops waiting for worker-thread work.
- Python cannot forcibly terminate a worker thread that has already started.

The underlying synchronous counter, callback, or FastEmbed inference may continue after the
awaiting task is cancelled.

## Best-effort goals

### Structural mode aims for document-wide coverage

Structural selection tries to protect the first and last complete units when both fit, gives every
Markdown section an initial share, and redistributes unused room using document-centroid salience
and MMR.

This is broader than a fixed positional slice, but it is not a promise that every section, topic,
or important fact appears. A section may have no candidate small enough for its provisional share,
and a rare detail can score below material closer to the document's main vocabulary.

### Query-aware modes aim for relevant evidence

- Lexical mode uses BM25 for exact query evidence.
- Semantic mode uses embedding cosine similarity for related meaning.
- Hybrid mode normally blends normalized BM25 and semantic scores equally.
- MMR then penalizes similarity to evidence already selected.

These are relevance estimates. They do not understand correctness, authority, recency, causality,
negation, or whether the selected text fully answers the task.

### MMR aims to reduce repeated-looking evidence

Trimwise uses the Maximal Marginal Relevance objective introduced in
[diversity-aware reranking research](https://aclanthology.org/X98-1025/). Structural and lexical
modes compare TF-IDF vectors; semantic and hybrid modes compare embeddings.

MMR measures representational similarity, not atomic facts:

- The same fact written differently may look diverse.
- Different facts about the same entity may look redundant.
- Changed numbers, dates, or negation can be easy for similarity alone to mishandle.

Less repetition is therefore a useful goal, not a factual-diversity guarantee.

## What source fidelity does and does not mean

The retained fragments are verbatim, but the complete result is not necessarily one contiguous
substring. Trimwise may:

- Select non-adjacent source ranges.
- Add the configured omission marker where it fits.
- Add minimal newline separators between separated fragments.
- Attach an original section heading to a query-aware passage when both fit.
- Use an exact source prefix when no complete unit fits.

These operations preserve source-backed content but can still change interpretation. A genuine
sentence may depend on a qualification, definition, table header, warning, or negation that was not
selected. Verbatim does not mean context-complete.

For quoting, auditing, legal review, or high-stakes decisions, retain the original document and
source identity alongside the excerpt.

## Markdown and format limits

Trimwise understands CommonMark blocks plus tables and adds explicit handling for YAML-style front
matter and otherwise uncovered source ranges. It recognizes headings, paragraphs, nested lists,
blockquotes, tables, HTML blocks, code blocks, reference definitions, and fenced code.

It does not provide format-specific understanding for:

- JSON objects or schemas.
- Chat roles, turns, or tool messages.
- Programming-language syntax trees or symbol relationships.
- CSV dialects or spreadsheet semantics.
- XML document schemas.

Those inputs can still be supplied as text, but Trimwise may split or rank them without respecting
their domain relationships. V1 has no JSON, chat, source-code, CSV, or XML parser.

## Tiny-budget and indivisible-text limits

When no complete candidate fits, fallback prefers a complete paragraph, sentence, or source line
before taking the longest exact source prefix that fits.

This creates several boundaries:

- A single unpunctuated line has no smaller complete structural unit.
- A tiny token budget may end inside a natural-language idea.
- Character budgets count code points, not user-perceived grapheme clusters or encoded bytes.
- Word budgets use whitespace splitting, not linguistic word segmentation.
- Custom token counters may require many prefix measurements on the rare exact-fallback path.

For a genuinely closed code fence, Trimwise keeps the original opening and closing fence while
removing body lines only when the fence shell itself fits. If the shell is too large, ordinary
prefix fallback applies. An unclosed source fence remains unclosed; Trimwise does not invent syntax.

## Retrieval, reasoning, and generation limits

Trimwise does not:

- Search the web or a local corpus.
- Fetch documents or choose which documents enter the application.
- Query a keyword index or vector database.
- Chunk and store a retrieval collection.
- Resolve contradictions between sources.
- Verify claims, citations, dates, or calculations.
- Answer the query.
- Summarize, paraphrase, or combine distant facts into a new sentence.

In a RAG or agent system, retrieval decides which sources are available. Trimwise can then reduce
each supplied source before prompt assembly. The responsibilities are complementary, not
interchangeable.

## Prompt-injection and untrusted-source limits

Source fidelity means harmful or misleading source text is also preserved verbatim when selected.
Trimwise does not detect prompt injection, sanitize instructions embedded in evidence, classify
malicious content, or establish a trust boundary between sources and system instructions.

For agentic applications:

- Treat excerpts as untrusted evidence.
- Keep system and developer instructions separate from trimmed source text.
- Delimit and label each source clearly.
- Restrict tool permissions independently of prompt text.
- Preserve provenance so suspicious evidence can be traced and reviewed.

Smart trimming reduces context size; it is not a security control.

## Semantic-model limits

Semantic and hybrid behavior inherits the chosen embedding backend's properties:

- Supported languages and domains.
- Maximum input length and internal truncation behavior.
- Biases and semantic blind spots.
- Model download, initialization, memory, and inference cost.
- Provider availability, privacy, and data-retention rules.

The default multilingual FastEmbed model is approximately 220 MB. A cold call may include download
and initialization time; warm calls reuse the model held by that `Trimmer`. V1 does not cache
candidate embeddings across trim calls.

One `Trimmer` serializes its managed FastEmbed operations for model safety. Separate instances can
infer independently at the cost of additional model memory. Caller callbacks are not serialized;
the application owns their concurrency and rate limits.

Cosine similarity is not a confidence probability. Scores from one model should not be treated as
calibrated thresholds for another.

## Tokenizer and multilingual limits

The default token budget and lexical subwords use tiktoken's `o200k_base`. Subword IDs keep lexical
scoring useful for scripts without whitespace, and structural sentence detection recognizes common
Latin and CJK sentence punctuation.

Trimwise does not perform language detection or select language-specific tokenizers. Word budgets
still use whitespace splitting, which is not linguistically meaningful for every language.
Semantic multilingual quality depends on the embedding model, not on Trimwise alone.

Tiktoken may need network access the first time an encoding is placed in its local cache. Use a
custom counter when the hard budget must match another model's tokenizer, while remembering that
the configured tiktoken encoding still supplies structural and lexical ranking subwords.

## Selection is greedy, not globally optimal

Trimwise accepts a candidate only after composing and measuring the complete trial output. This
enforces the hard ceiling, but the selection objective does not solve a global knapsack problem.

A 300-token candidate with high relevance may be chosen where three complementary 100-token
candidates would have preserved more useful evidence—or the reverse. The current MMR pass also
does not implement facility-location coverage, claim extraction, or evidence-value combinations.

Submodular summarization research provides richer importance, coverage, and nonredundancy
objectives, such as [Lin and Bilmes, 2010](https://aclanthology.org/N10-1134/). Adopting one requires
Trimwise-specific benchmarks showing better fact recall per retained token without unacceptable
latency or complexity. It is therefore a roadmap item, not a current guarantee.

## Extreme compression

At a 10× or 20× compression ratio, a selected sentence can consume much of the entire budget.
Trimwise cannot compress inside that sentence without eventually falling back to a prefix, and it
cannot synthesize two distant facts into one dense statement.

A trained token compressor or abstractive summarizer may preserve more task-relevant meaning per
token under these conditions. The tradeoff is additional model cost and loss of the exact-source
guarantee, with possible grammatical damage or generated claims.

| Method | Main strength | Main tradeoff |
| --- | --- | --- |
| Prefix slicing | Minimal latency and complexity | Ignores everything after one position |
| Trimwise | Exact budgets, source-backed fragments, document-wide or query-aware selection | Cannot rewrite or globally optimize retained facts |
| Token-level model compression | Can remove low-value tokens throughout the prompt | Adds model cost and can damage wording or syntax |
| Abstractive summarization | Can synthesize very dense explanations | Generates new text and requires faithfulness evaluation |

The methods can be chained: use Trimwise to select evidence across a long source, then apply a
token-level compressor to the smaller excerpt. After the second step, Trimwise's whole-fragment and
source-layout guarantees no longer describe the final text.

## Caller responsibilities

Trimwise deliberately leaves several decisions with the application:

| Responsibility | What the caller should do |
| --- | --- |
| Complete prompt budget | Reserve room for labels, instructions, examples, tools, and model output |
| Source identity | Store document IDs, URLs, authors, timestamps, and access controls outside `TrimResult` |
| Relevance evaluation | Test downstream answers on representative documents and queries |
| Factual verification | Check claims against original sources when accuracy matters |
| Contradiction handling | Preserve and compare conflicting evidence explicitly |
| Embedding operations | Own callback caching, retries, timeouts, rate limits, privacy, and concurrency |
| Security | Treat selected source text as untrusted and enforce tool permissions separately |
| Tokenizer alignment | Supply a custom counter when the default encoding does not match the target model |

`TrimResult` intentionally does not expose source IDs, selected spans, scores, or embeddings in v1.
Keep provenance alongside each input before trimming several sources.

## How to evaluate Trimwise for your application

Unit tests can verify hard limits, determinism, source fidelity, and backend boundaries. They cannot
prove that the best evidence survived.

Evaluate with realistic:

- Document lengths, structures, languages, and repetition patterns.
- Queries containing exact identifiers and paraphrased concepts.
- Compression ratios and per-source budget allocations.
- Contradictions, changed numbers, dates, negation, and rare critical details.
- Cold and warm semantic calls.
- Final LLM answers, not only ranking scores.

Useful quality measures include labelled fact recall per retained token, query relevance,
redundancy, answer accuracy, latency, and peak memory. Research-backed components are a sound
starting point; representative downstream evaluation decides whether they work for your task.

## Continue exploring

- Return to the [Trimwise overview](index.md).
- Follow the [Getting Started guide](getting-started.md).
- Compare ranking tradeoffs in [Choosing a Strategy](strategies.md).
- Inspect the pipeline in [How Trimwise Works](how-it-works.md).
- Configure backends in [Semantic Models and Async Usage](semantic-and-async.md).
- Review the public contract in [Configuration and API Reference](configuration-and-api.md).
- Track deferred work in the [roadmap](https://github.com/tenwritehq/trimwise/blob/main/ROADMAP.md).
