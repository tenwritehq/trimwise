---
title: Trim Many Sources with One Shared Limit
description: Let evidence from several sources compete for one Trimwise token, word, or character budget while keeping source identity and local spans.
---

# Many Sources, One Shared Limit

Use `trim_context()` when several sources must fit inside one evidence allowance. Trimwise considers
all of their passages together, so a source with stronger evidence can use more of the available
space. The result still contains one entry per input source, in the same order.

This is useful after retrieval, search, or tool calls have already chosen the sources. Trimwise does
not retrieve documents or copy your labels and metadata; it reduces the source strings you supply.

## A runnable core-install example

Lexical selection needs no embedding model or optional dependency:

```python
from trimwise import Trimmer

question = "Which retry loop ignored backoff settings, and when did service recover?"
records = [
    {
        "url": "https://example.test/incident",
        "text": (
            "Initial checks focused on the network. "
            "Workers later exhausted the database connection pool."
        ),
    },
    {
        "url": "https://example.test/follow-up",
        "text": (
            "The team disabled a retry loop that ignored backoff settings. "
            "Service recovered at 14:32 UTC."
        ),
    },
]

result = Trimmer().trim_context(
    [record["text"] for record in records],
    limit=18,
    unit="words",
    strategy="lexical",
    query=question,
)

for source in result.sources:
    record = records[source.source_index]
    print(record["url"])
    print(source.text or "(no excerpt fit)")

assert result.output_count == sum(source.output_count for source in result.sources)
assert result.output_count <= result.limit
```

`source_index` is the zero-based position of the original string. Use it to reconnect each excerpt
to caller-owned URLs, filenames, permissions, timestamps, or other metadata. Trimwise deliberately
does not copy or transform that information.

Some entries may contain `text=""`. This can happen when the shared limit is too small or other
sources have stronger evidence. The empty row remains present so indexes never shift.

## Choose the operation that matches the budget

| Operation | Inputs | Budget behavior | Result |
| --- | --- | --- | --- |
| `trim()` / `atrim()` | One source | One limit for that source | One `TrimResult` |
| `atrim_many()` | Independent requests | Each request keeps its own limit | One `TrimResult` per request |
| `trim_context()` / `atrim_context()` | Many sources | All sources share one limit | One input-aligned `ContextTrimResult` |

Use `atrim_many()` when every input has already been assigned its own allowance. Use the context
methods when passages should compete for the same allowance.

## Counts and prompt assembly

Each source string is measured independently with the selected unit and optional custom counter:

```text
result.input_count  = sum(source.input_count  for source in result.sources)
result.output_count = sum(source.output_count for source in result.sources)
result.output_count <= result.limit
```

The shared limit covers only the strings in `source.text`. It does not cover labels, URLs,
caller-added headings, instructions, separators, examples, tool definitions, an output schema, or
the model's answer.
Reserve room for those parts before choosing the limit.

Tokenizers can also count separately measured strings differently after they are joined. If the
completed prompt needs an exact token ceiling, assemble it, measure it with the target model's
tokenizer, and leave a safety margin or trim again with room reserved for prompt formatting.

## Result fields

`ContextTrimResult` reports the shared operation:

| Field | Meaning |
| --- | --- |
| `sources` | One `ContextSourceResult` per input source, in input order |
| `input_count` | Sum of independently measured source inputs |
| `output_count` | Sum of independently measured source outputs |
| `limit` and `unit` | Shared ceiling and its measurement rule |
| `strategy` | Concrete strategy after resolving `auto` |
| `trimmed` | Whether any source output differs from its input |

Each `ContextSourceResult` contains `source_index`, `text`, its own counts, `trimmed`, and local
`spans`. A span always indexes the corresponding original source:

```python
for source in result.sources:
    original = records[source.source_index]["text"]
    retained_ranges = [original[span.start : span.end] for span in source.spans]
```

Generated separators and omission markers do not have spans.

## Async semantic use

Use `atrim_context()` with a native async embedding client. The callback receives one query and the
passages needed for the whole context operation:

```python
from collections.abc import Sequence

from trimwise import Trimmer


async def embed(query: str, passages: Sequence[str]) -> tuple[object, Sequence[object]]:
    """Embed one shared query and its candidate passages."""
    return await client.embed_query(query), await client.embed_documents(list(passages))


result = await Trimmer(async_embedding_callback=embed).atrim_context(
    source_texts,
    limit=800,
    strategy="hybrid",
    query="Which recommendations are supported by the reports?",
    deduplicate=True,
)
```

The async callback runs on the calling event loop. Counting, parsing, ranking, and selection run in
worker threads. Cancellation propagates into the awaited callback; it cannot forcibly stop work
already running in a worker thread.

## Deduplication and queryless limits

`deduplicate=False` is the default. With `deduplicate=True`, Trimwise sends each exact repeated
passage string once during that operation and reuses its vector for every matching occurrence. It
does not remove a source or result row.

This option is best effort. Use it only when your backend returns the same vector for the same
query and passage regardless of batch position. It does not fuzzy-match similar wording, retain
vectors after the call, share work across calls, or collect unrelated requests in the background.

Without a query, `auto` uses structural selection. Trimwise gives candidate-bearing sources an
initial opportunity in input order, then spends remaining room on the strongest fitting material.
This is deterministic, but it is not an optimal allocation and does not guarantee that every
source contributes.

With a query, an oversized best-matching passage is shortened to fit instead of being dropped for
a weaker source that happens to fit whole. This fallback returns the shortened passage in its own
source row and leaves the other source rows empty.

Trimwise also does not resolve contradictions, verify claims, rank source authority, or copy source
metadata. Preserve the originals and provenance whenever those responsibilities matter.

## Continue exploring

- Follow the [Getting Started guide](getting-started.md).
- Review all method and result fields in [Configuration and API Reference](configuration-and-api.md).
- Connect embedding callbacks in [Semantic Models and Async Usage](semantic-and-async.md).
- Read the [hard guarantees and limitations](guarantees-and-limitations.md).
