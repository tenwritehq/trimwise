# Trimwise

**Create compact, high-signal excerpts for LLM prompts without rewriting the source.**

Trimwise is an extractive Python library for the step between collecting text and assembling a
prompt. Give it a document, a maximum size, and optionally the task you care about. It selects
useful fragments from across the document, returns them in source order, and guarantees that the
measured result stays within your token, word, or character limit.

This is especially useful when an agent has several sources but cannot place every source in the
context window. Instead of taking the first `N` characters from each article, report, note, or
search result, Trimwise gives each source a smarter evidence budget while leaving your system
prompt, instructions, examples, and output schema untouched.

```text
source 1 ──> Trimwise ──> high-signal excerpt 1 ┐
source 2 ──> Trimwise ──> high-signal excerpt 2 ├─> your prompt
source 3 ──> Trimwise ──> high-signal excerpt 3 ┘
                         + unchanged instructions
```

Trimwise compacts text you already have. It does **not** search the web, retrieve documents, query
an index, or replace a RAG system.

## Why smart trimming?

Prefix truncation is fast, but it assumes the beginning contains everything worth keeping. Real
documents often place conclusions, decisions, identifiers, error messages, examples, and updated
facts in the middle or at the end.

```text
text[:N]   [introduction and setup --------------------] cut

Trimwise   [opening context] […omitted…] [key evidence] […omitted…] [decision]
```

Trimwise remains extractive: it selects original source fragments rather than asking another model
to summarize them. This makes the result predictable, traceable, and safe to quote. It also avoids
the extra model call, latency, and faithfulness checks required by abstractive compression.

## Install

Trimwise supports Python 3.10 through 3.14.

| You need | With `pip` | With `uv` |
| --- | --- | --- |
| Structural, lexical, or your own embedding callback | `python -m pip install trimwise` | `uv add trimwise` |
| Trimwise-managed semantic model on CPU | `python -m pip install "trimwise[semantic]"` | `uv add "trimwise[semantic]"` |
| Trimwise-managed semantic model on NVIDIA GPU | `python -m pip install "trimwise[semantic-gpu]"` | `uv add "trimwise[semantic-gpu]"` |

The core installation is enough for the default structural and lexical paths. It also supports
semantic and hybrid trimming when you provide your own embedding callback. Install a semantic
extra only when you want Trimwise to create embeddings through FastEmbed. Do not install the CPU
and GPU FastEmbed variants together.

## Your first high-signal excerpt

```python
from trimwise import Trimmer

document = """
# Incident ORION-774

The service became unavailable shortly after the scheduled deployment.

## Investigation

Workers exhausted the database connection pool after a retry loop ignored backoff settings.
The alert initially pointed to elevated API latency.

## Resolution

The team disabled the retry loop and restored service at 14:32 UTC.
"""

result = Trimmer().trim(
    document,
    limit=60,
    query="What caused the outage and how was it resolved?",
)

print(result.text)
print(result.output_count, result.unit)
```

The default unit is tokens. Because this call includes a query, `auto` resolves to the lightweight
`lexical` strategy and uses BM25 to favor exact query evidence. Without a query, `auto` resolves to
`structural` and preserves a balanced, high-signal overview without loading an embedding model.

If the source already fits, Trimwise returns it unchanged without parsing Markdown, ranking
fragments, or invoking an embedding backend.

## Choose how evidence is selected

| Strategy | Use it when | What Trimwise does |
| --- | --- | --- |
| `auto` | You want a safe default | Uses `structural` without a query and `lexical` with one |
| `structural` | No specific question is available | Covers the document structure and favors material central to the document |
| `lexical` | Exact names, IDs, errors, URLs, or wording matter | Matches source units to the query with model-free BM25 scoring |
| `semantic` | The answer may be paraphrased or expressed in another supported language | Compares query and passage embeddings by semantic similarity |
| `hybrid` | Exact terms and semantic meaning both matter | Combines normalized BM25 and semantic scores, normally with an equal 50/50 blend |

`lexical`, `semantic`, and `hybrid` require a nonblank query. Semantic and hybrid strategies can
use either Trimwise-managed FastEmbed or a synchronous or asynchronous embedding callback supplied
by your application.

Structural mode does not divide text into a fixed ratio such as 50/25/25. It protects useful
opening and closing units when they fit, gives Markdown sections an initial share of the budget,
and redistributes unused room to central, nonredundant material. A single oversized plain-text
paragraph can be expanded into complete sentences or source lines before selection, allowing
useful content from its middle and end to compete.

Query-aware strategies narrow selection around the strongest evidence and may return less than the
requested maximum instead of filling spare space with weak matches. When a selected passage has a
nearby section heading, Trimwise includes that heading when the complete result still fits.

## Built from established research ideas

Trimwise combines established methods from information retrieval and extractive summarization to
answer one narrow question: **which original fragments deserve the limited space available in this
prompt?** It adapts the ideas below to source-backed spans, exact budgets, Markdown structure, and
deterministic output. It does not claim to reproduce any paper's complete system.

| Research idea | What it contributes to Trimwise | Foundation |
| --- | --- | --- |
| BM25 lexical relevance | Finds source units containing exact query terms, identifiers, and wording | [BM25 and the probabilistic relevance framework](https://doi.org/10.1561/1500000019) |
| TF-IDF centroid salience | Estimates which units best represent a document when no query exists | [Centroid-based summarization](https://aclanthology.org/W00-0403/) |
| Sentence embeddings | Finds paraphrases and related meaning when the selected model supports them | [Sentence-BERT](https://aclanthology.org/D19-1410/) and [multilingual sentence embeddings](https://aclanthology.org/2020.emnlp-main.365/) |
| Normalized score fusion | Lets hybrid mode preserve both exact and semantic evidence without discarding score strength | [Analysis of fusion functions for hybrid retrieval](https://arxiv.org/abs/2210.11934) |
| Maximal Marginal Relevance | Balances usefulness against similarity to fragments already selected | [MMR for diversity-based reranking](https://aclanthology.org/X98-1025/) |
| Adaptive evidence count | Lets query-aware trimming stop before weak evidence fills the remaining budget | [Adaptive-k context selection](https://aclanthology.org/2025.emnlp-main.1017/) |

In the current implementation, hybrid mode min-max normalizes usable lexical and semantic scores
and blends them equally. Reciprocal Rank Fusion with `k=60` is used only as a defensive fallback
when a score row cannot be compared meaningfully. Selection then applies MMR, using the configured
default of 70% relevance and 30% similarity reduction.

Trimwise adds prompt-specific behavior around those scoring methods:

- Candidates receive their nearest heading and same-section neighbors as ranking context, while
  the retained fragment remains the exact candidate text.
- Headings, URLs, numbers or dates, and code-like identifiers receive a small language-neutral
  signal so operational details are less likely to disappear when relevance scores are close.
- Structural and lexical selection use TF-IDF similarity for repetition reduction; semantic and
  hybrid selection use embedding similarity.
- Every proposed selection is restored to source order, composed with affordable headings,
  separators, and omission markers, and measured before it is accepted.

These algorithms estimate relevance and representational diversity; they do not determine factual
truth. MMR can discourage fragments that look repetitive, but it cannot guarantee that every
selected fragment contains a different fact.

## Source fidelity and exact budgets

Trimwise recognizes CommonMark headings, paragraphs, lists, blockquotes, tables, HTML blocks,
fenced code, YAML-style front matter, reference definitions, and otherwise uncovered source ranges.
Selected fragments are sliced from the original input rather than rendered back into Markdown.

You can rely on:

- **A hard caller-defined ceiling.** The finished output is measured again and cannot exceed the
  requested token, word, or character limit.
- **Verbatim retained fragments.** Original wording, indentation, links, list markers, code-fence
  language tags, and fragment order are preserved.
- **Visible gaps when affordable.** The configured omission marker shows where non-adjacent source
  ranges were joined, but retained evidence takes priority when the marker cannot fit.
- **Predictable fallback behavior.** Very small budgets prefer complete blocks, paragraphs,
  sentences, or lines before using the longest exact source prefix that fits.
- **Synchronous and asynchronous APIs.** Use `trim()` in blocking code and `atrim()` when the event
  loop must remain responsive.

Trimwise does not summarize, paraphrase, combine distant facts into a new sentence, or prove that
the retained text contains every fact required by your task. At extreme compression ratios, a
trained or abstractive compressor may preserve more meaning per token, but it gives up Trimwise's
exact-source guarantee.

## Project links

- [Source code](https://github.com/tenwritehq/trimwise)
- [Python package](https://pypi.org/project/trimwise/)
- [Issue tracker](https://github.com/tenwritehq/trimwise/issues)
- [Roadmap](https://github.com/tenwritehq/trimwise/blob/main/ROADMAP.md)
- [MIT license](https://github.com/tenwritehq/trimwise/blob/main/LICENSE)
