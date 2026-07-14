# Trimwise

**Keep instructions intact. Smart-trim each source. Give the LLM higher-signal context.**

Trimwise creates high-signal excerpts from documents, already-fetched search results, logs, and tool
output before you assemble an LLM prompt. Keep system instructions, task instructions, and output
schemas unchanged; trim each evidence source independently; then place the resulting excerpts into
the prompt.

Each source is shortened to an exact token, word, or character budget while useful material is
selected from across it. Instead of returning only `text[:N]`, Trimwise can retain complete
sections, find passages relevant to a question, reduce repeated-looking evidence, and keep selected
text in its original order.

Trimwise is not a search, retrieval, or RAG system. It does not fetch documents or query an index,
database, or vector store; it only ranks and trims text your application passes in.

- **Lightweight by default:** the core uses BM25 and TF-IDF; embeddings are optional.
- **Exact budget:** every result is measured again before it is returned.
- **Source faithful:** retained fragments are copied from the input, not rewritten.
- **Markdown and plain text:** headings, paragraphs, lists, tables, code fences, and sentences can
  become selection units.
- **Sync and async:** use `trim()` or the non-blocking `atrim()` API.
- **Python 3.10-3.14:** ships with inline type information.

<details>
<summary><strong>Table of contents</strong></summary>

- [Main use case: high-signal excerpts for LLMs](#main-use-case-high-signal-excerpts-for-llms)
- [Why Trimwise instead of slicing or model-based compression?](#why-trimwise-instead-of-slicing-or-model-based-compression)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Async usage](#async-usage)
- [Configuration](#configuration)
- [Result object](#result-object)
- [Which strategy should I use?](#which-strategy-should-i-use)
- [How Trimwise works](#how-trimwise-works)
- [Research foundations](#research-foundations)
- [Budgets and tokenizers](#budgets-and-tokenizers)
- [Markdown and source fidelity](#markdown-and-source-fidelity)
- [Semantic models](#semantic-models)
- [Guarantees and limitations](#guarantees-and-limitations)
- [Project](#project)

</details>

## Main use case: high-signal excerpts for LLMs

Suppose an LLM must cluster 50 blog posts. Sending every full post may exceed the context window.
Keeping `post[:N]` gives every post an excerpt, but those excerpts contain only introductions.
Compressing the complete assembled prompt can also modify the clustering instructions unless they
are separately protected, or let one long source dominate the available evidence.

Trim each source independently, then build the prompt:

```python
from pathlib import Path

from trimwise import Trimmer

instructions = """\
Cluster the blog posts by their main topic.
Give each cluster a short name and list its source numbers.
Base the answer only on the supplied excerpts.
"""

paths = sorted(Path("posts").glob("*.md"))
trimmer = Trimmer()

excerpts = []
for source_number, path in enumerate(paths, start=1):
    post = path.read_text(encoding="utf-8")
    excerpt = trimmer.trim(post, limit=300, strategy="structural").text
    excerpts.append(f"## Source {source_number}: {path.name}\n{excerpt}")

prompt = instructions + "\n\n" + "\n\n".join(excerpts)
```

This pattern keeps the important layers separate:

- **Instructions remain exact.** Trimwise never sees or edits them.
- **Every source gets a ceiling.** One long blog post cannot consume every source token.
- **Each excerpt covers its own document.** Structural mode can keep evidence from the beginning,
  middle, and end instead of taking only the introduction.
- **Selection can follow the task.** For query-focused work, pass the same question or task as
  `query`, or choose `semantic` or `hybrid` when paraphrases matter.

The final prompt still includes labels, separators, and instructions, so reserve space for those
when choosing the per-source limit. Trimwise guarantees each excerpt's budget, not the size of the
prompt assembled around multiple excerpts.

## Why Trimwise instead of slicing or model-based compression?

There is no universally best way to shorten context. The useful choice depends on whether you care
most about speed, source fidelity, or maximum compression.

Keeping only the first _N_ tokens is fastest, but it assumes the beginning contains the best
information. Important conclusions, decisions, error messages, identifiers, and examples often
appear later. Trimwise scores units across the supplied input and returns complete source-backed
fragments:

```text
Prefix slicing:    [document beginning --------------------------] cut

Trimwise:          [opening context] [...omitted...] [relevant decision]
```

Model-based prompt compressors solve a different problem. Methods such as
[LLMLingua](https://aclanthology.org/2023.emnlp-main.825/),
[LongLLMLingua](https://aclanthology.org/2024.acl-long.91/),
[LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/), and
[Selective Context](https://arxiv.org/abs/2310.06201) can remove individual tokens or lexical
units inside a sentence. These methods are generally extractive at the token level, but they do not
preserve complete sentences or source blocks. The original LLMLingua paper reports up to 20x
compression with little performance loss on its evaluated datasets. That finer granularity can
retain more information at aggressive ratios, but it requires a language-model or trained-encoder
compression pass and may produce text that is harder for a person to read or trace to its source.

[RECOMP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bda88ed2892f5e61c9a9bf215c566913-Abstract-Conference.html)
goes further: it trains extractive and abstractive compressors for documents supplied by a
retrieval system. Its abstractive path can synthesize information from multiple sources, but
generated summaries no longer provide Trimwise's exact-source guarantee.

| Approach | What it removes | Extra model | Main tradeoff |
| --- | --- | --- | --- |
| Prefix slicing | Everything after one position | No | Fastest, but ignores the rest of the document |
| Trimwise | Whole blocks, sentences, or lines | No for structural or lexical use | Keeps readable source fragments, but cannot compress inside every selected sentence |
| LLMLingua-style compression | Tokens or lexical units throughout the prompt | Yes | Higher compression, but altered sentence structure and more runtime cost |
| Trained abstractive compression | Original wording and document structure | Yes | Can synthesize dense summaries, but loses exact-source fidelity and needs faithfulness checks |

Choose Trimwise when:

- Retained text must remain readable, auditable, and safe to quote.
- Markdown sections, lists, tables, code fences, dates, or identifiers should remain intact.
- You want an exact caller-defined budget without running another language model.
- Predictable dependencies and latency matter more than the highest possible compression ratio.

Choose a model-based compressor when extreme compression is more important than complete source
fragments, you can afford its model and runtime, and you can evaluate downstream answer quality on
your own data. The methods can also be chained: use Trimwise to select evidence from across a long
document, then apply token-level compression to that smaller result. After the second step, the
whole-fragment and source-layout guarantees no longer apply.

## Installation

Trimwise supports Python 3.10 through 3.14. The
[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
recommends installing third-party packages in a virtual environment so project dependencies stay
isolated.

### Create an isolated environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If your project already manages an environment, use that environment and skip this step.

### Choose an installation

| You need | Install | What becomes available |
| --- | --- | --- |
| Text trimming or your own embedding backend | `python -m pip install trimwise` | Every strategy; semantic and hybrid require your callback |
| Built-in CPU semantic model | `python -m pip install "trimwise[semantic]"` | Every strategy through Trimwise's local FastEmbed integration |
| Built-in NVIDIA GPU semantic model | `python -m pip install "trimwise[semantic-gpu]"` | Every strategy through Trimwise's CUDA-enabled FastEmbed integration |

The core installation includes Markdown parsing, tiktoken measurement, and NumPy vector scoring.
It does not install FastEmbed or an embedding model. Structural, lexical, and automatic trimming
work immediately; semantic and hybrid trimming also work when you provide an
[embedding callback](#semantic-models). `auto` remains lightweight whether a callback is configured
or not: it uses structural coverage without a query and lexical BM25 with one.

The semantic extras add FastEmbed so Trimwise can supply the embedding model for you. Model weights
are downloaded only when semantic or hybrid ranking is first requested without a callback. The
default multilingual model is approximately 220 MB and is then reused by that `Trimmer` instance.

Do not install `trimwise[semantic]` and `trimwise[semantic-gpu]` in the same environment. FastEmbed
documents that its CPU and GPU runtime packages conflict. GPU users must also provide compatible
CUDA and cuDNN libraries; see the [FastEmbed GPU guide](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/).

Tiktoken may need network access on first use while it places the configured encoding in its local
cache. Later structural and lexical calls reuse that cache.

### Verify the installation

```bash
python -c "from trimwise import Trimmer; r = Trimmer().trim('alpha beta gamma', 2, unit='words'); assert r.output_count <= 2; print(r.text)"
```

This smoke test uses only the lightweight core and confirms that the public API returns text within
the requested budget.

## Quick start

```python
from trimwise import Trimmer

document = """\
Incident report

The service became unavailable at 09:14. Initial checks focused on the network.

Investigation

The team traced the failure to an expired credential.

Decision

Credentials will now rotate automatically every 30 days.
"""

result = Trimmer().trim(
    document,
    limit=16,
    query="How often will credentials rotate?",
)

print(result.text)
print(result.output_count)
```

Example output from the 16-token budget:

```text
Incident report

Decision

Credentials will now rotate automatically every 30 days.
```

The default budget is tokens. Because a query was supplied, `auto` uses fast lexical BM25 scoring.
Without a query, `auto` uses structural coverage. Neither default path imports or downloads an
embedding model.

If the input already fits, Trimwise returns it byte-for-byte unchanged without parsing it.

## Async usage

```python
result = await trimmer.atrim(
    document,
    500,
    strategy="semantic",
    query="What are the main risks?",
)
```

With structural, lexical, FastEmbed, or a synchronous embedding callback, `atrim()` runs the
pipeline in a worker thread so the event loop remains responsive. With an asynchronous embedding
callback, parsing and CPU scoring still run in worker threads while the callback itself is awaited
on the calling event loop.

Cancellation propagates to a currently awaited asynchronous embedding callback. During worker
thread work, cancellation stops waiting for the result, but Python cannot forcibly terminate a
thread that has already started.

## Configuration

Most applications do not need configuration. `Trimmer()` uses balanced defaults and loads no
semantic model unless you explicitly select `semantic` or `hybrid` for text that needs trimming.
Create a `TrimConfig` when you need to change output formatting, tune relevance versus diversity,
match a particular tokenizer, or control Trimwise-managed FastEmbed. Embedding callbacks belong on
`Trimmer`, not in `TrimConfig`, because they are live runtime dependencies rather than immutable
settings.

| Setting | Default | Change it when | What to know |
| --- | --- | --- | --- |
| `omission_marker` | `[…omitted…]` | Your output needs a different visible gap marker | Trimwise inserts it only where source text was skipped and only when it fits; retained content has priority |
| `mmr_lambda` | `0.7` | You want to adjust the balance between query/document relevance and avoiding similar fragments | `1.0` favors relevance only; lower values apply a stronger similarity penalty; `0.7` is the recommended starting point |
| `token_encoding` | `o200k_base` | Token budgets should approximate a different tiktoken-supported model family | Controls built-in token counting and the subword terms used by structural and lexical scoring |
| `embedding_model` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | You use Trimwise-managed FastEmbed and need another supported model | Ignored when a callback supplies vectors; changing models affects speed, memory, languages, and scores |
| `embedding_batch_size` | `256` | FastEmbed uses too much memory, or your hardware can efficiently handle larger batches | Ignored for callbacks; smaller FastEmbed batches generally use less peak memory |
| `fastembed_options` | `{}` | FastEmbed needs provider-, threading-, cache-, or runtime-specific constructor options | Ignored for callbacks; put the FastEmbed model in `embedding_model`, not `model_name` here |

Configuration belongs to a `Trimmer` instance and is reused across its calls. The object is
immutable, and `fastembed_options` is defensively copied, so later changes to the original mapping
cannot silently change a running trimmer.

For example, customize only the parts that affect the output you want:

```python
from trimwise import TrimConfig, Trimmer

trimmer = Trimmer(
    TrimConfig(
        mmr_lambda=0.8,
        omission_marker="[...omitted...]",
    )
)
```

This trimmer slightly favors the most relevant fragments and uses an ASCII omission marker. It does
not enable semantic scoring or load a model; strategy selection still happens in each `trim()` or
`atrim()` call.

To configure semantic scoring, keep the model name separate from its runtime options:

```python
from trimwise import TrimConfig, Trimmer

semantic_trimmer = Trimmer(
    TrimConfig(
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_batch_size=128,
        fastembed_options={"cache_dir": ".fastembed_cache"},
    )
)
```

The smaller batch can reduce peak memory. The model is still loaded lazily on the first semantic or
hybrid trim that actually requires ranking; the same `Trimmer` then reuses it. See
[semantic models](#semantic-models) before changing models or runtime options.

All names, models, encodings, and omission markers must be nonblank. `embedding_batch_size` must be
a positive integer, `mmr_lambda` must be between `0` and `1`, and `fastembed_options` keys must be
strings. Invalid configuration fails immediately when `TrimConfig` is created.

For one-off token accounting, pass `token_counter=` to `trim()` or `atrim()` instead of creating a
new configuration. A custom counter changes budget measurement for that call; `token_encoding`
continues to provide the internal subword representation used for ranking.

## Result object

`trim()` and `atrim()` return an immutable `TrimResult`:

| Field | Meaning |
| --- | --- |
| `text` | Retained source text |
| `input_count` | Measured input size |
| `output_count` | Measured result size |
| `limit` | Requested maximum |
| `unit` | Unit used for the counts |
| `strategy` | Concrete strategy used after resolving `auto` |
| `trimmed` | Whether the returned text differs from the input |

## Which strategy should I use?

Start with the kind of information you need to preserve:

- **No question or task:** use `auto` or `structural` for a balanced document overview.
- **Exact names, IDs, error codes, or phrases:** use `auto` with a query, or choose `lexical`.
- **Paraphrases, concepts, or meaning across languages:** choose `semantic`.
- **Exact evidence and broader meaning both matter:** choose `hybrid`.

If you are unsure, use `auto`. It stays lightweight: without a query it resolves to `structural`;
with a query it resolves to `lexical`. It never loads an embedding model.

| Strategy | Choose it when | How it chooses source fragments | Cost and important behavior |
| --- | --- | --- | --- |
| `auto` | You want a safe default | Resolves to `structural` without a query and `lexical` with one | Core installation only; never loads FastEmbed |
| `structural` | You need a useful overview but have no specific question | Protects useful opening and closing context, spreads space across sections, favors content central to the document, and avoids near-duplicates | Core installation only; ignores a supplied query |
| `lexical` | The query contains exact evidence such as `ORION-774`, a function name, product name, or error message | Uses BM25 to find source fragments sharing important query terms, then favors complementary fragments | Core installation only; fast and deterministic, but does not understand paraphrases as well as `semantic` |
| `semantic` | The document may express the answer with different words, or the query and text may use different languages | Uses embedding similarity to compare meaning, then favors complementary fragments | Requires your [embedding callback](#semantic-models) or an optional FastEmbed installation |
| `hybrid` | You cannot afford to miss either an exact identifier or a differently worded explanation | Combines normalized BM25 and embedding scores before selecting complementary fragments | Requires your [embedding callback](#semantic-models) or FastEmbed; usually the most compute-intensive choice |

`lexical`, `semantic`, and `hybrid` require a nonblank query. Unlike positional truncation, these
query-aware strategies do not force the beginning or end into the result. They may also stop before
the limit when the remaining fragments are much weaker, so an “at most 500 tokens” budget does not
become 500 tokens of loosely related text. When space allows, Trimwise includes the nearest section
heading so an excerpt remains understandable.

Every strategy keeps selected text byte-for-byte from the source, emits fragments in source order,
and rechecks the complete result against the requested budget. Diversity scoring discourages
repeated wording or meaning; it is a useful duplicate-control signal, not proof that two fragments
contain different facts. Trimwise ranks only the string you provide—it does not search documents,
query an index, or perform RAG.

Here are the common choices in code. Exact lowercase strings and `Strategy` enum values are both
accepted.

```python
from trimwise import Trimmer

trimmer = Trimmer()

# No question: preserve a balanced, high-signal overview.
overview = trimmer.trim(document, limit=500)

# Exact identifier: auto resolves to the lightweight lexical strategy.
incident = trimmer.trim(document, limit=500, query="ORION-774")

# The answer may use different words or another language.
cause = trimmer.trim(
    document,
    limit=500,
    strategy="semantic",
    query="Why did authentication fail?",
)

# Preserve both the exact incident ID and semantically related explanations.
mixed = trimmer.trim(
    document,
    limit=500,
    strategy="hybrid",
    query="What caused incident ORION-774?",
)
```

## How Trimwise works

Trimwise follows the same high-level pipeline for every strategy:

1. **Measure:** return the original input immediately when it already fits.
2. **Segment:** split Markdown or plain text into source-backed units such as headings,
   paragraphs, lists, tables, code fences, sentences, or lines.
3. **Rank:** score those units using document centrality, BM25, embeddings, or both.
4. **Select:** prefer useful candidates while penalizing candidates similar to material already
   selected.
5. **Compose:** restore source order, add affordable omission markers, and measure the complete
   result again.

Structural mode does not use a fixed ratio such as 50/25/25. It protects fitting beginning and
ending units, gives each Markdown section a provisional share, and fills the remaining budget with
central, nonredundant material. A single oversized plain-text paragraph is expanded into complete
sentences or source lines before ranking when possible.

Query-aware modes first narrow the candidate pool around the strongest query evidence. They may
stop below the requested limit rather than padding the result with weakly related text. When a
selected passage has a section heading, Trimwise includes that heading when it fits.

For scoring only, a candidate can see its nearest section heading and neighboring source units.
The returned fragment is still the candidate's exact source text.

## Research foundations

Trimwise combines established ranking and extractive-summarization ideas. It adapts them to exact
budgets and source-backed fragments; it does not claim to reproduce every paper's full system.
Some cited work studies information retrieval, but Trimwise borrows only its scoring or selection
concepts—it does not retrieve documents.

| Concept | How Trimwise uses it | Research |
| --- | --- | --- |
| BM25 relevance | Fast lexical matching between a query and source units | [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019) |
| TF-IDF centroid | Queryless units are compared with the document's lexical center | Centroid-based queryless extraction is discussed in the original [MMR summarization work](https://aclanthology.org/X98-1025/) |
| Maximal Marginal Relevance | Each choice balances relevance against similarity to already selected units | [Using MMR for Diversity-Based Reranking](https://aclanthology.org/X98-1025/) |
| Hybrid score fusion | Usable BM25 and semantic scores are normalized and blended equally | [Hybrid fusion analysis](https://arxiv.org/abs/2210.11934) |
| Reciprocal Rank Fusion | RRF is a defensive fallback when a hybrid score row is flat or invalid | [Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114) |
| Adaptive evidence count | A query-specific score drop bounds evidence, with a small recall buffer | Inspired by [Adaptive-k context selection](https://aclanthology.org/2025.emnlp-main.1017/) |

MMR measures vector similarity, not factual truth. It discourages fragments that *look* redundant;
it cannot guarantee that every retained fragment contains a different fact. Improving factual
coverage without adding a heavy claim-extraction model is an active benchmark item in the
[roadmap](ROADMAP.md).

## Budgets and tokenizers

Trimwise supports tokens, whitespace-separated words, and Python character counts:

```python
from trimwise import BudgetUnit, Trimmer

trimmer = Trimmer()

by_tokens = trimmer.trim(document, 500)
by_words = trimmer.trim(document, 300, unit=BudgetUnit.WORDS)
by_characters = trimmer.trim(document, 2_000, unit="characters")
```

Token budgets use tiktoken's `o200k_base` encoding by default. Tiktoken may need network access on
first use while it populates its local encoding cache.

Use a custom counter when the destination model has a different tokenizer:

```python
result = trimmer.trim(
    document,
    500,
    token_counter=lambda value: len(my_tokenizer.encode(value)),
)
```

A custom token counter is valid only with token budgets. Trimwise remeasures every proposed final
result through that callback.

## Markdown and source fidelity

Trimwise recognizes CommonMark headings, paragraphs, nested lists, blockquotes, tables, HTML
blocks, fenced code, front matter, reference definitions, and otherwise uncovered source ranges.
It slices the original input instead of rendering Markdown again.

Skipped ranges receive the configured omission marker when it fits. Retained content always takes
priority over markers. If no complete unit fits, Trimwise progressively tries smaller complete
units and finally an exact fitting source prefix. Closed code fences keep their opening and closing
fences during fallback; genuinely unclosed fences remain unclosed.

## Semantic models

Semantic trimming finds relevant passages even when the document uses different words from your
query. Hybrid trimming combines that meaning-based matching with BM25 exact-term matching. Only
these two explicit strategies need embeddings.

| Choose | Install | Best when |
| --- | --- | --- |
| Your own embedding callback | `python -m pip install trimwise` | You already have a local model, hosted API, internal service, or embedding cache |
| Trimwise-managed FastEmbed on CPU | `python -m pip install "trimwise[semantic]"` | You want semantic trimming to work without connecting another model |
| Trimwise-managed FastEmbed on an NVIDIA GPU | `python -m pip install "trimwise[semantic-gpu]"` | You already have a compatible CUDA environment |

**Already have an embedding model or API?** Use the normal `trimwise` installation and pass a
callback. Both `semantic` and `hybrid` then work without installing a semantic extra, importing
FastEmbed, or downloading Trimwise's default model.

### Bring your own embeddings

A callback receives the query and the candidate passages that Trimwise wants to compare. Return
the query embedding followed by one embedding for each passage, in the same order. Trimwise handles
vector normalization and uses the scores to choose source fragments; the returned excerpt still
contains only exact text from the original document.

For a local model or blocking client, pass a synchronous callback:

```python
from collections.abc import Sequence

from trimwise import Trimmer


def embed(query: str, passages: Sequence[str]) -> tuple[object, Sequence[object]]:
    """Create embeddings with an existing model."""
    return model.encode_query(query), model.encode_document(list(passages))


trimmer = Trimmer(embedding_callback=embed)
result = trimmer.trim(
    document,
    500,
    strategy="semantic",
    query="What caused the outage?",
)
```

Adapt `encode_query()` and `encode_document()` to your model's API. Keeping the query separate is
useful for models that apply different instructions to queries and documents.

For an asynchronous embedding API, pass an async callback and call `atrim()`:

```python
from collections.abc import Sequence

from trimwise import Trimmer


async def aembed(query: str, passages: Sequence[str]) -> tuple[object, Sequence[object]]:
    """Create embeddings with an existing async client."""
    vectors = await client.embed([query, *passages])
    return vectors[0], vectors[1:]


trimmer = Trimmer(async_embedding_callback=aembed)
result = await trimmer.atrim(
    document,
    500,
    strategy="hybrid",
    query="What caused incident ORION-774?",
)
```

The callback can return Python sequences or NumPy arrays. For every call, make sure that:

- The first returned item is one nonempty query vector.
- The second returned item contains exactly one nonempty vector per supplied passage, in the same
  order.
- Every vector has the same number of finite numeric values.

Embed each passage exactly as received. Trimwise may include its nearby heading and neighboring
text to improve matching, but it never inserts that extra context into the final excerpt unless the
corresponding source fragment is selected. Invalid vectors and callback failures raise
`SemanticBackendError` instead of quietly switching to a less accurate strategy.

A few behaviors help avoid surprises:

- Your callback takes precedence over FastEmbed, even if a semantic extra is installed.
- A synchronous callback works with both `trim()` and `atrim()`. An async callback must be used
  through `atrim()` whenever semantic scoring is needed.
- Configure either a synchronous callback or an async callback, not both.
- `auto` remains lightweight: it chooses structural trimming without a query and lexical trimming
  with one. It does not automatically invoke your embedding callback.
- Trimwise does not call the callback when the input already fits, or when you choose `structural`
  or `lexical`.
- The same callback may receive concurrent calls. Reuse and protect your model or client as needed,
  and handle any API caching, rate limits, and retries there.

### Let Trimwise manage FastEmbed

If you do not already have an embedding provider, install one FastEmbed extra and use `Trimmer`
without a callback:

```python
from trimwise import Trimmer

result = Trimmer().trim(
    document,
    500,
    strategy="semantic",
    query="What caused the outage?",
)
```

Trimwise lazily loads the multilingual
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model. It is approximately 220 MB,
so the first semantic call may download the model and take noticeably longer; later calls on the
same `Trimmer` reuse it. Do not install the CPU and GPU FastEmbed extras together.

One `Trimmer` runs its FastEmbed work one call at a time for model safety. Separate `Trimmer`
instances can run in parallel, but each may consume additional model memory. FastEmbed loading,
download, and inference failures raise `SemanticBackendError`; Trimwise never silently replaces
semantic scoring with lexical scoring.

## Guarantees and limitations

Trimwise guarantees that the measured result does not exceed the requested limit and that retained
fragments come from the original source in source order.

It deliberately does not:

- Summarize, paraphrase, or combine distant facts into a new sentence.
- Guarantee factual diversity; MMR similarity is only a redundancy proxy.
- Load an embedding model for the default structural or lexical paths.
- Fill a query-aware budget with unrelated text merely because space remains.
- Parse JSON, chat transcripts, or source code with format-specific grammars.

A single unpunctuated source line has no smaller complete structural unit, so very small limits can
still require prefix truncation. At extreme compression ratios, an abstractive or trained
compressor may preserve more meaning per token, but it gives up Trimwise's exact-source guarantee.

## Project

- Read planned experiments and explicit deferrals in the [roadmap](ROADMAP.md).
- Report bugs or request features through [GitHub Issues](https://github.com/aakashH242/trimwise/issues).
- Trimwise is available under the [MIT License](LICENSE).
