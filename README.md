# Trimwise

[![PyPI version](https://img.shields.io/pypi/v/trimwise.svg)](https://pypi.org/project/trimwise/)

> Keep the most useful parts of long text before adding it to an LLM prompt.

Trimwise creates compact, high-signal excerpts from documents, blog posts, search results, logs, and tool output. 
Instead of keeping only `text[:N]`, it can select complete
fragments from across the source, reduce obvious repetition, and return everything inside an exact
token, word, or character limit.

The result remains extractive: retained text comes from your input, keeps its original wording, and
appears in source order. Trimwise does not search the web, retrieve documents, query a vector
database, or rewrite your evidence.

[Documentation](https://trimwise.readthedocs.io/en/latest/) ·
[Getting started](https://trimwise.readthedocs.io/en/latest/getting-started/) ·
[API reference](https://trimwise.readthedocs.io/en/latest/api-reference/) ·
[PyPI](https://pypi.org/project/trimwise/)

## A typical use-case

Suppose an LLM must cluster many blog posts. Sending every full post may exceed the context window,
while `post[:N]` gives the model only introductions. Trim each post independently, then assemble the
prompt from the resulting excerpts:

```python
from pathlib import Path

from trimwise import Trimmer

instructions = """\
Cluster the blog posts by their main topic.
Give each cluster a short name and list its source numbers.
Base the answer only on the supplied excerpts.
"""

trimmer = Trimmer()
excerpts = []

for number, path in enumerate(sorted(Path("posts").glob("*.md")), start=1):
    source = path.read_text(encoding="utf-8")
    excerpt = trimmer.trim(source, limit=300, strategy="structural").text
    excerpts.append(f"## Source {number}: {path.name}\n{excerpt}")

prompt = instructions + "\n\n" + "\n\n".join(excerpts)
```

This keeps the prompt layers separate: instructions remain exact, every source gets its own
ceiling, and each excerpt can represent material from across its source. Labels, separators, and
instructions still consume space in the final prompt, so leave room for them when choosing each
source limit.

## Installation

Trimwise supports Python 3.10 through 3.14.

| What you need | `pip` | `uv` |
| --- | --- | --- |
| Structural, lexical, or your own embedding callback | `python -m pip install trimwise` | `uv add trimwise` |
| Trimwise-managed semantic models on CPU | `python -m pip install "trimwise[semantic]"` | `uv add "trimwise[semantic]"` |
| Trimwise-managed semantic models on NVIDIA GPU | `python -m pip install "trimwise[semantic-gpu]"` | `uv add "trimwise[semantic-gpu]"` |

The core installation includes Markdown parsing, token measurement, lexical ranking, and vector
scoring. It does not install FastEmbed or download an embedding model.

Do not install the CPU and GPU semantic extras together. GPU use also requires compatible CUDA and
cuDNN libraries. See [Semantic Models and Async Use](https://trimwise.readthedocs.io/en/latest/semantic-and-async/)
for callbacks, model loading, concurrency, and GPU details.

## Quick start

```python
from trimwise import Trimmer

document = """\
# Incident report

The service became unavailable at 09:14. Initial checks focused on the network.

## Root cause

The team traced the failure to an expired credential.

## Decision

Credentials will now rotate automatically every 30 days.
"""

result = Trimmer().trim(
    document,
    limit=24,
    query="What caused the outage and how will it be prevented?",
)

print(result.text)
print(result.output_count)  # Always <= 24
print(result.strategy)      # Strategy.LEXICAL: auto resolved from the query
print(result.spans)         # Original-input Python-string offsets
```

`auto` uses structural coverage when no query is supplied and fast lexical BM25 when a query is
present. Neither path loads an embedding model. If the original input already fits, Trimwise
returns it byte-for-byte unchanged. `result.spans` contains ordered, end-exclusive ranges into the
original Python string, so `document[span.start:span.end]` recovers each retained source fragment.
Generated omission markers and separators are not included in those ranges.

## How Trimwise compares with prompt compressors

Trimwise and model-based prompt compressors shorten text at different levels. Trimwise chooses
complete source fragments before prompt assembly. Methods such as LLMLingua can remove individual
tokens from an already assembled prompt, which can achieve much denser compression but may leave
text that is harder for people to read or trace.

| Approach | What it keeps or removes | Extra compression model | Best fit |
| --- | --- | --- | --- |
| Prefix slicing | Keeps only the beginning | No | Lowest possible overhead when missing later evidence is acceptable |
| Trimwise | Selects complete source blocks, sentences, or lines and restores source order | No for structural or lexical use | Readable, source-backed excerpts with an exact final budget |
| [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/) family | Removes tokens throughout a prompt; LongLLMLingua also uses the query and long-context position | Yes | Aggressive compression when downstream model performance matters more than human-readable excerpts |
| [Selective Context](https://arxiv.org/abs/2310.06201) | Removes low-self-information tokens, phrases, or sentences | Yes | Pruning predictable language using a causal language model |
| [RECOMP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bda88ed2892f5e61c9a9bf215c566913-Abstract-Conference.html) | Selects sentences or generates a summary from retrieved documents | Yes, with trained compressors | Compressing RAG results for a downstream task, including abstractive synthesis when allowed |

The LLMLingua family can preserve more task-relevant information per token at aggressive ratios.
Its remaining tokens still come from the prompt, but complete sentence and block boundaries are not
preserved. RECOMP's extractive path keeps selected sentences; its abstractive path can combine
information across documents but no longer returns only original wording.

Choose Trimwise when evidence must stay readable, source fragments must remain verbatim and ordered,
or adding another compression model is undesirable. Choose a model-based compressor when maximum
compression density is more important and you can evaluate its effect on your own downstream task.
The methods can also be chained: select broad evidence with Trimwise, then apply token-level
compression. After the second step, Trimwise's whole-fragment and source-layout guarantees no
longer describe the final prompt.

See the detailed [research comparison](https://trimwise.readthedocs.io/en/latest/research-foundations/#how-trimwise-compares-with-model-based-compression)
for the differences among LLMLingua, LongLLMLingua, LLMLingua-2, Selective Context, and RECOMP.

## Choose a strategy

| Strategy | Use it when | What it prioritizes |
| --- | --- | --- |
| `auto` | You want a safe default | `structural` without a query; `lexical` with one |
| `structural` | No question or task is available | Document centrality, section coverage, and fitting beginning/end units |
| `lexical` | Exact names, IDs, errors, URLs, or phrases matter | BM25 matches between the query and source fragments |
| `semantic` | The source may express the answer with different words or another supported language | Embedding similarity between the query and candidates |
| `hybrid` | Literal evidence and paraphrases both matter | An equal blend of normalized BM25 and semantic scores |

`lexical`, `semantic`, and `hybrid` require a nonblank query. Semantic and hybrid calls require
either your own embedding callback or one of the FastEmbed extras.

Query-aware strategies may stop below the requested limit when the remaining candidates appear
weakly related. The limit means “at most,” not “fill every token with progressively less useful
text.”

Read [Strategies](https://trimwise.readthedocs.io/en/latest/strategies/) for examples, scoring
behavior, and practical tradeoffs.

## Semantic trimming

When performing query-aware trimming, Trimwise uses an embedding model to find pieces most relevant to the query. You
can either provide the embedding model or let Trimwise manage its own model. You can configure the model Trimwise uses
using the configuration object.

### Let Trimwise manage the model

Install a semantic extra and request `semantic` or `hybrid` explicitly:

```python
from trimwise import Trimmer

result = Trimmer().trim(
    document,
    limit=500,
    strategy="hybrid",
    query="What caused incident ORION-774?",
)
```

The default multilingual model is downloaded and initialized on the first semantic call, then
reused by that `Trimmer`. Structural and lexical calls never load it.

### Bring your own embeddings

The core installation can use semantic and hybrid strategies without FastEmbed when you provide an
embedding callback. Use a synchronous callback with `trim()` or `atrim()`, or an asynchronous
callback with `atrim()` for an async embedding service.

Your callback receives the query separately from the candidate passages and returns one query
vector plus one same-dimension vector per passage. Trimwise validates and scores the vectors; your
model or client remains under your control.

See [Bring Your Own Embeddings](https://trimwise.readthedocs.io/en/latest/semantic-and-async/)
for complete synchronous and asynchronous examples.

## Budgets and configuration

Token budgets use Tiktoken and the `o200k_base` encoding by default:

```python
result = Trimmer().trim(document, limit=500)
```

Words and characters are available when tokens are not the unit you need:

```python
word_result = Trimmer().trim(document, limit=300, unit="words")
character_result = Trimmer().trim(document, limit=2_000, unit="characters")
```

Use `TrimConfig` for reusable behavior:

```python
from trimwise import TrimConfig, Trimmer

trimmer = Trimmer(
    TrimConfig(
        omission_marker="[content omitted]",
        mmr_lambda=0.75,
    )
)
```

You can also choose a Tiktoken encoding, FastEmbed model, inference batch size, and FastEmbed
options, or supply a custom token counter for token budgets. See
[Configuration and API](https://trimwise.readthedocs.io/en/latest/configuration-and-api/) for every
field, argument, validation rule, and result value.

## Async use

`atrim()` keeps parsing, measurement, ranking, model loading, inference, callbacks, and selection
from blocking the event loop:

```python
result = await Trimmer().atrim(
    document,
    limit=500,
    strategy="lexical",
    query="Which decision was approved?",
)
```

Cancellation stops waiting for the result but cannot terminate synchronous work already running in
a worker thread. Async embedding callbacks are awaited directly and can receive cancellation.

## What Trimwise guarantees

- The measured output never exceeds the requested limit.
- Input that already fits is returned byte-for-byte unchanged.
- Retained fragments use exact source text and appear in source order.
- Omission markers are added only when they fit; retained evidence takes priority.
- Closed code fences preserve their opening and closing fences during fallback when the fence shell
  itself fits.
- FastEmbed is never imported for structural, lexical, or already-fitting inputs.

Trimwise does not summarize, paraphrase, combine distant facts into a new sentence, or guarantee
that selected fragments contain different facts. MMR reduces vector similarity, which is a useful
proxy for repetition rather than factual proof. At extreme compression ratios, a trained or
abstractive compressor may preserve more answer-relevant information per token, but it gives up
Trimwise’s exact-source guarantee.

Read [Guarantees and Limitations](https://trimwise.readthedocs.io/en/latest/guarantees-and-limitations/)
for the precise boundaries.

## Research foundations

Trimwise adapts established methods to source-backed fragments and exact output budgets:

- BM25 for exact lexical relevance
- TF-IDF centroid similarity for queryless representativeness
- Sentence embeddings for semantic relevance
- Normalized convex fusion, with RRF as a fallback
- Maximal Marginal Relevance for reduced representational repetition
- Adaptive-k-inspired evidence boundaries for query-aware selection

These methods guide selection; they do not prove that every excerpt is optimal. The
[Research Foundations](https://trimwise.readthedocs.io/en/latest/research-foundations/) page maps
each idea to the current behavior, explains the user benefit, and states what the evidence does not
guarantee.

## Documentation

- [Getting Started](https://trimwise.readthedocs.io/en/latest/getting-started/)
- [Strategies](https://trimwise.readthedocs.io/en/latest/strategies/)
- [Semantic Models and Async Use](https://trimwise.readthedocs.io/en/latest/semantic-and-async/)
- [Configuration and API](https://trimwise.readthedocs.io/en/latest/configuration-and-api/)
- [How Trimwise Works](https://trimwise.readthedocs.io/en/latest/how-it-works/)
- [Guarantees and Limitations](https://trimwise.readthedocs.io/en/latest/guarantees-and-limitations/)
- [Research Foundations](https://trimwise.readthedocs.io/en/latest/research-foundations/)
- [API Reference](https://trimwise.readthedocs.io/en/latest/api-reference/)

Trimwise is available under the [MIT License](LICENSE).
