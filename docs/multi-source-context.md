---
title: Trim Many Sources with One Shared Limit
description: Let evidence and caller-supplied source wrappers from several inputs compete for one measured token, word, or character budget.
---

# Many Sources, One Shared Limit

Use `trim_context()` when several sources must fit inside one evidence allowance. Trimwise considers
all of their passages together, so a source with stronger evidence can use more of the available
space. The result still contains one entry per input source, in the same order.

This is useful after retrieval, search, or tool calls have already chosen the sources. Trimwise does
not retrieve documents or define a metadata format. You can provide exact opening and closing text
for each source when labels, URLs, or balanced delimiters must count toward the same limit as the
evidence.

## A runnable core-install example

Lexical selection needs no embedding model or optional dependency:

```python
from trimwise import ContextSource, Trimmer

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
    [
        ContextSource(
            text=record["text"],
            prefix=f"--- Source: {record['url']} ---\n",
            suffix="\n--- End source ---",
        )
        for record in records
    ],
    limit=30,
    unit="words",
    strategy="lexical",
    query=question,
    separator="\n\n",
)

print(result.text)

assert result.text is not None
assert len(result.text.split()) == result.output_count
assert result.output_count <= result.limit
```

`ContextSource.text` is evidence. Its `prefix` and `suffix` are copied exactly around that source's
returned evidence, but only when the source contributes a nonempty excerpt. Neither wrapper half
influences which evidence wins or appears in source spans. The separator is copied only between
contributing sources.

The prefix and suffix are conditional as a pair. If the limit cannot afford both wrapper halves
and any evidence from a source, that source's result row is empty; Trimwise never returns a bare or
half-finished wrapper.

`source_index` is the zero-based input position. Use it to reconnect each result row to caller-owned
filenames, permissions, timestamps, or other metadata that does not belong in the rendered text.

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

## Choose what the limit covers

### Evidence-only results

Passing only strings and omitting `separator` preserves the original row-oriented behavior:

```text
result.input_count  = sum(source.input_count  for source in result.sources)
result.output_count = sum(source.output_count for source in result.sources)
result.output_count <= result.limit
result.text is None
```

Use this mode when your application will assemble and measure the final prompt itself. Any labels
or separators added later are outside Trimwise's limit.

### Prompt-ready context

Passing at least one `ContextSource`, or supplying `separator` explicitly, enables complete
rendering. In this mode:

```text
result.input_count = sum(source.input_count for source in result.sources)
measure(result.text) = result.output_count
result.output_count <= result.limit
```

`input_count` still measures evidence only. Each source row's `output_count` still measures only
that row's returned evidence and omission markers. The aggregate `output_count` measures the final
`result.text`, including emitted prefixes, suffixes, and separators, so it need not equal the sum
of row counts. A plain string can be mixed with `ContextSource`; it simply has no wrapper text.

Instructions, examples, tool definitions, an output schema, a fixed prompt header, and the model's
answer are still outside this limit. Reserve room for those surrounding parts.

Trimwise remeasures the complete rendered string because token counts are not always additive at
text boundaries. If the entire prompt needs an exact token ceiling, use the target model's tokenizer
as `token_counter`, reserve room for everything outside `result.text`, and measure the completed
prompt as a final application-level check.

## Result fields

`ContextTrimResult` reports the shared operation:

| Field | Meaning |
| --- | --- |
| `sources` | One `ContextSourceResult` per input source, in input order |
| `input_count` | Sum of independently measured source inputs |
| `output_count` | Complete rendered size in rendering mode; otherwise the sum of source outputs |
| `limit` and `unit` | Shared ceiling and its measurement rule |
| `strategy` | Concrete strategy after resolving `auto` |
| `trimmed` | Whether any source output differs from its input |
| `text` | Prompt-ready rendered context, or `None` for evidence-only string calls |

Each `ContextSourceResult` contains `source_index`, `text`, its own counts, `trimmed`, and local
`spans`. A span always indexes the corresponding original source:

```python
for source in result.sources:
    original = records[source.source_index]["text"]
    retained_ranges = [original[span.start : span.end] for span in source.spans]
```

Caller prefixes, suffixes, separators, and Trimwise-generated omission text do not have spans.

## Async semantic use

Use `atrim_context()` with a native async embedding client. The callback receives one query and the
passages needed for the whole context operation:

```python
from collections.abc import Sequence

from trimwise import ContextSource, Trimmer


async def embed(query: str, passages: Sequence[str]) -> tuple[object, Sequence[object]]:
    """Embed one shared query and its candidate passages."""
    return await client.embed_query(query), await client.embed_documents(list(passages))


result = await Trimmer(async_embedding_callback=embed).atrim_context(
    [
        ContextSource(
            text,
            prefix=f"--- Source {index + 1} ---\n",
            suffix="\n--- End source ---",
        )
        for index, text in enumerate(source_texts)
    ],
    limit=800,
    strategy="hybrid",
    query="Which recommendations are supported by the reports?",
    deduplicate=True,
    separator="\n\n",
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

Trimwise treats prefixes, suffixes, and separators as opaque text. It does not validate or escape
titles, URLs, or other caller values, so applications must handle untrusted metadata safely.
Trimwise also does not resolve contradictions, verify claims, or rank source authority. Preserve
the originals and provenance whenever those responsibilities matter.

## Continue exploring

- Follow the [Getting Started guide](getting-started.md).
- Review all method and result fields in [Configuration and API Reference](configuration-and-api.md).
- Connect embedding callbacks in [Semantic Models and Async Usage](semantic-and-async.md).
- Read the [hard guarantees and limitations](guarantees-and-limitations.md).
