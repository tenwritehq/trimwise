# Configuration and API Reference

Trimwise exposes a deliberately small public API. Most applications need only `Trimmer` and the
`text` field of its result. Use `TrimConfig` when the defaults do not match your tokenizer,
omission style, relevance-versus-repetition preference, or Trimwise-managed FastEmbed setup.

```python
from trimwise import TrimConfig, Trimmer

trimmer = Trimmer(
    TrimConfig(
        omission_marker="[...source omitted...]",
        mmr_lambda=0.8,
    )
)

result = trimmer.trim(
    document,
    limit=500,
    query="What caused the outage?",
)

prompt_ready_excerpt = result.text
```

## Public imports

Import supported objects directly from `trimwise`:

```python
from trimwise import (
    BudgetUnit,
    SemanticBackendError,
    Strategy,
    TrimConfig,
    TrimResult,
    Trimmer,
)
```

These are the package's six documented exports:

| Name | Purpose |
| --- | --- |
| `Trimmer` | Runs synchronous or asynchronous trimming and reuses semantic state |
| `TrimConfig` | Stores immutable reusable configuration |
| `TrimResult` | Reports the excerpt, measurements, resolved strategy, and whether text changed |
| `Strategy` | Enumerates automatic, structural, lexical, semantic, and hybrid selection |
| `BudgetUnit` | Enumerates token, word, and character measurement |
| `SemanticBackendError` | Reports callback or FastEmbed failures without hiding the original cause |

Internal modules and underscored names are not part of the compatibility promise. Trimwise ships a
`py.typed` marker, so type checkers can use its inline annotations from the installed package.

## `Trimmer`

Create a `Trimmer` once and reuse it across related calls:

```text
Trimmer(
    config: TrimConfig | None = None,
    *,
    embedding_callback: Callable | None = None,
    async_embedding_callback: Callable | None = None,
)
```

| Constructor argument | Default | Meaning |
| --- | --- | --- |
| `config` | `TrimConfig()` | Reusable measurement, ranking, marker, and FastEmbed settings |
| `embedding_callback` | `None` | Synchronous query-and-passage embedding backend |
| `async_embedding_callback` | `None` | Asynchronous query-and-passage embedding backend used through `atrim()` |

Only one embedding callback may be configured. Non-callable callback values raise `TypeError`, and
supplying both callback forms raises `ValueError`.

Reusing a `Trimmer` has little effect on structural and lexical calls. For Trimwise-managed
FastEmbed, it matters: the instance lazily caches one initialized model and serializes access to
that model. Caller callbacks are not serialized and remain responsible for their own client reuse,
caching, rate limits, and thread or task safety.

## `trim()`

Use `trim()` in synchronous code:

```text
trim(
    text: str,
    limit: int,
    *,
    unit: BudgetUnit | str = BudgetUnit.TOKENS,
    strategy: Strategy | str = Strategy.AUTO,
    query: str | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> TrimResult
```

| Argument | Meaning |
| --- | --- |
| `text` | The complete source string to reduce |
| `limit` | Maximum measured size of the returned text |
| `unit` | `tokens`, `words`, or `characters` |
| `strategy` | `auto`, `structural`, `lexical`, `semantic`, or `hybrid` |
| `query` | Optional task or question; required by lexical, semantic, and hybrid strategies |
| `token_counter` | Optional synchronous counter that replaces built-in token measurement for this call |

`text` must be a Python `str`. `limit` must be a non-boolean integer greater than or equal to zero.
Enum members or exact lowercase string values are accepted for `unit` and `strategy`; unsupported
values raise `ValueError`.

```python
from trimwise import BudgetUnit, Strategy, Trimmer

trimmer = Trimmer()

using_strings = trimmer.trim(
    document,
    500,
    unit="tokens",
    strategy="lexical",
    query="ORION-774",
)

using_enums = trimmer.trim(
    document,
    500,
    unit=BudgetUnit.TOKENS,
    strategy=Strategy.LEXICAL,
    query="ORION-774",
)

assert using_strings == using_enums
```

## `atrim()`

`atrim()` accepts the same arguments and returns the same `TrimResult`:

```text
async atrim(
    text: str,
    limit: int,
    *,
    unit: BudgetUnit | str = BudgetUnit.TOKENS,
    strategy: Strategy | str = Strategy.AUTO,
    query: str | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> TrimResult
```

```python
import asyncio

from trimwise import Trimmer


async def main() -> None:
    """Trim one source without blocking the event loop."""
    result = await Trimmer().atrim(
        document,
        limit=500,
        query="What caused the outage?",
    )
    print(result.text)


asyncio.run(main())
```

Without an async embedding callback, `atrim()` sends the synchronous pipeline to a worker thread.
With an async callback, preparation and CPU work use worker threads while the callback is awaited
on the calling event loop. Cancellation can propagate into an awaited async callback, but it cannot
terminate a worker thread that has already started. See
[Semantic Models and Async Usage](semantic-and-async.md) for the full execution model.

## Validation and fast paths

Every call follows this public order:

1. Parse `unit` and `strategy`.
2. Normalize and validate the query.
3. Validate the text, limit, and custom counter.
4. Measure the input.
5. Return an early result if the limit is zero or the input already fits.
6. Segment, rank, select, compose, and remeasure only when trimming is required.

This order means invalid arguments are rejected even when the source would fit. It also means a
valid fitting semantic call returns unchanged without importing FastEmbed or invoking a callback.

| Input condition | Result |
| --- | --- |
| `limit < 0` | Raises `ValueError` |
| `limit` is `True`, a float, or a string | Raises `TypeError` |
| `limit == 0` | Returns empty `text` after measuring the input |
| Input count is at most `limit` | Returns the original string exactly |
| Empty input with a positive limit | Returns the empty input unchanged |

The unchanged path preserves spaces and line endings exactly. `trimmed` is calculated from whether
`result.text != text`, not merely from whether the input was over budget.

## Strategy and query rules

`auto` resolves before trimming begins:

| Requested strategy and query | Resolved strategy |
| --- | --- |
| `auto` with no query | `structural` |
| `auto` with `None`, an empty string, or whitespace-only text | `structural` |
| `auto` with a nonblank query | `lexical` |
| Explicit `structural` | `structural`; any supplied query is ignored for ranking |
| Explicit `lexical`, `semantic`, or `hybrid` | Same strategy; a nonblank query is required |

Leading and trailing whitespace is stripped from a usable query before ranking. `result.strategy`
contains the resolved concrete strategy, never `auto`.

Automatic mode never chooses semantic or hybrid. This is a deliberate cost boundary: adding a
query does not unexpectedly invoke an embedding provider or download a model.

## Budget measurement

`unit` controls `input_count`, `output_count`, `limit`, fitting checks, and final enforcement:

| Unit | Exact rule |
| --- | --- |
| `tokens` | Number of tiktoken IDs from the configured encoding; `o200k_base` by default |
| `words` | `len(text.split())` |
| `characters` | `len(text)`, measured as Python Unicode code points |

Token encoding uses `disallowed_special=()`, so strings that look like special tokens are measured
as ordinary source text. Tiktoken may need network access the first time it places an encoding in
its local cache.

Even after individual fragments appear to fit, Trimwise composes the complete source-ordered
result with separators, headings, and affordable omission markers, then measures it again. The
returned `output_count` cannot exceed `limit`.

## Custom token counters

Use `token_counter` when the target model's tokenizer differs from the configured tiktoken
encoding:

```python
import tiktoken

from trimwise import Trimmer

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Measure text with the target model's tokenizer.

    Args:
        text: Proposed source or composed output.

    Returns:
        Nonnegative token count.
    """
    return len(encoding.encode(text, disallowed_special=()))


result = Trimmer().trim(
    document,
    limit=500,
    unit="tokens",
    token_counter=count_tokens,
)
```

The callback must be synchronous and return a nonnegative, non-boolean integer. It is valid only
with `unit="tokens"`. Trimwise uses it for input measurement, candidate fitting, fallback prefix
search, and the final result count.

The custom counter replaces public token measurement, not lexical tokenization. Structural and
lexical scoring—and the lexical half of hybrid scoring—still use `TrimConfig.token_encoding` to
create consistent subword IDs. Keep that encoding representative of the text even when a custom
counter owns the hard budget.

Because Trimwise does not assume a custom counter is monotonic, the rare final-prefix fallback may
measure multiple source prefixes to find the longest exact fit. Keep the callback inexpensive or
cache work inside it when this path matters.

## `TrimConfig`

`TrimConfig` is a frozen, slotted dataclass:

```text
TrimConfig(
    token_encoding: str = "o200k_base",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    fastembed_options: Mapping[str, Any] = {},
    embedding_batch_size: int = 256,
    mmr_lambda: float = 0.7,
    omission_marker: str = "[…omitted…]",
)
```

| Setting | Change it when | Important behavior |
| --- | --- | --- |
| `token_encoding` | Built-in token budgets or lexical subwords should use another tiktoken encoding | Used lazily for token measurement and structural or lexical ranking |
| `embedding_model` | Trimwise-managed FastEmbed should use another supported model | Ignored by caller-provided callbacks |
| `fastembed_options` | `TextEmbedding` needs provider, cache, or runtime options | Forwarded only during managed model initialization |
| `embedding_batch_size` | Managed passage inference needs a smaller memory footprint or different throughput tradeoff | Passed to `passage_embed()`; ignored by callbacks |
| `mmr_lambda` | Relevance should receive more or less weight than repetition reduction | `1.0` removes the similarity penalty; lower values strengthen it |
| `omission_marker` | The output needs another visible source-gap marker | Added only where source was omitted and only when it fits |

Most applications should keep the defaults. Configuration changes affect every trimming call made
through that `Trimmer`; per-call choices such as strategy, query, limit, unit, and custom token
counter remain method arguments.

### Configuration validation

| Setting | Accepted boundary |
| --- | --- |
| `token_encoding` | Nonblank string |
| `embedding_model` | Nonblank string |
| `omission_marker` | Nonblank string |
| `embedding_batch_size` | Positive non-boolean integer |
| `mmr_lambda` | Non-boolean integer or float from `0` through `1` |
| `fastembed_options` | Mapping with string keys and no `model_name` key |

Invalid text-like settings can raise `TypeError` or `ValueError` depending on whether their type or
content is wrong. Other invalid configuration boundaries raise the exception documented by their
validation rule.

### Configuration immutability

Trimwise defensively copies `fastembed_options` and exposes the copied top-level mapping as
read-only. The original mapping can be changed later without altering the stored keys or values at
that level.

```python
from trimwise import TrimConfig

options = {"threads": 2}
config = TrimConfig(fastembed_options=options)
options["threads"] = 8

assert config.fastembed_options["threads"] == 2
```

The defensive copy is shallow. Avoid mutating nested objects stored as option values after the
configuration is created.

## Relevance versus repetition with `mmr_lambda`

Trimwise's MMR selection objective is:

```text
mmr_lambda × relevance − (1 − mmr_lambda) × maximum selected similarity
```

The default `0.7` weighs relevance at 70% and the similarity penalty at 30%. `1.0` chooses by
relevance without an MMR similarity penalty. At `0.0`, relevance no longer influences the MMR
objective, so candidates are chosen only by least similarity with deterministic source-order ties;
that extreme is rarely a useful default.

MMR comes from established diversity-aware ranking research, but no single lambda is optimal for
every document and task. Change it only when evaluation on your own prompts shows that the default
keeps too much repeated-looking material or suppresses too much relevant evidence. See
[Choosing a Strategy](strategies.md) and the original
[MMR work](https://aclanthology.org/X98-1025/) for context.

## Omission markers

The default marker is `[…omitted…]`. Trimwise considers leading, internal, and trailing source gaps
in source order. A marker is added only when the complete output still fits; retained source text
always takes priority.

```python
from trimwise import TrimConfig, Trimmer

trimmer = Trimmer(TrimConfig(omission_marker="<source omitted>"))
result = trimmer.trim(
    "First fact.\n\nRepeated filler.\n\nFinal decision.",
    limit=38,
    unit="characters",
)
```

Trimwise may also add minimal newline separators around non-adjacent fragments. The source
fragments themselves remain unchanged.

## FastEmbed settings

These settings affect only Trimwise-managed FastEmbed. They have no effect when
`embedding_callback` or `async_embedding_callback` supplies vectors.

```python
from trimwise import TrimConfig, Trimmer

config = TrimConfig(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    embedding_batch_size=64,
    fastembed_options={"providers": ["CPUExecutionProvider"]},
)
trimmer = Trimmer(config)
```

FastEmbed is imported and the model is initialized lazily. `fastembed_options` is expanded into the
`TextEmbedding` constructor alongside `model_name=embedding_model`, which is why `model_name` is
rejected inside the options mapping. The batch size is supplied separately during passage
inference.

See [Semantic Models and Async Usage](semantic-and-async.md) for callback precedence, vector
validation, model reuse, locking, cancellation, and error stages.

## `TrimResult`

`TrimResult` is a frozen, slotted dataclass returned by both public methods:

| Field | Type | Meaning |
| --- | --- | --- |
| `text` | `str` | Prompt-ready extractive output |
| `input_count` | `int` | Original source size measured in `unit` |
| `output_count` | `int` | Returned text size measured in `unit` |
| `limit` | `int` | Caller-supplied maximum output size |
| `unit` | `BudgetUnit` | Concrete measurement unit |
| `strategy` | `Strategy` | Concrete strategy after resolving `auto` |
| `trimmed` | `bool` | Whether `text` differs from the input string |

```python
from trimwise import Trimmer

result = Trimmer().trim(
    "First fact. Middle detail. Final decision.",
    limit=5,
    unit="words",
)

assert result.output_count <= result.limit
print(result.text)
print(result.input_count, result.output_count, result.unit.value)
print(result.strategy.value, result.trimmed)
```

The result does not expose internal candidate spans, ranking scores, embeddings, or source IDs.
Keep source identity in your own application when trimming several documents.

## `Strategy`

`Strategy` is a string enum:

| Member | Value | Query required | Embeddings required |
| --- | --- | --- | --- |
| `Strategy.AUTO` | `"auto"` | No | Never selected automatically |
| `Strategy.STRUCTURAL` | `"structural"` | No | No |
| `Strategy.LEXICAL` | `"lexical"` | Yes | No |
| `Strategy.SEMANTIC` | `"semantic"` | Yes | Yes |
| `Strategy.HYBRID` | `"hybrid"` | Yes | Yes |

Read [Choosing a Strategy](strategies.md) for ranking, adaptive evidence selection, research
background, runtime cost, and limitations.

## `BudgetUnit`

`BudgetUnit` is a string enum:

| Member | Value |
| --- | --- |
| `BudgetUnit.TOKENS` | `"tokens"` |
| `BudgetUnit.WORDS` | `"words"` |
| `BudgetUnit.CHARACTERS` | `"characters"` |

Returned results always contain enum instances even when the call used a string value.

## `SemanticBackendError`

`SemanticBackendError` extends `RuntimeError` and represents optional semantic-backend failures.
The message identifies whether a caller callback or FastEmbed failed and includes the relevant
stage. The original exception is chained as `error.__cause__`.

```python
from trimwise import SemanticBackendError, Trimmer

try:
    result = Trimmer().trim(
        document,
        limit=500,
        strategy="semantic",
        query="What caused the outage?",
    )
except SemanticBackendError as error:
    print(error)
    print("Original backend error:", error.__cause__)
```

Trimwise does not silently fall back to lexical scoring after a semantic failure. Handle the error,
fix the backend, retry according to your application's policy, or deliberately issue a separate
call with another strategy.

## Exception guide

| Exception | Typical cause |
| --- | --- |
| `TypeError` | Non-string input, non-integer limit, non-callable callback, invalid query type, or required async callback used through `trim()` |
| `ValueError` | Negative limit, unsupported enum string, missing query, incompatible custom counter, invalid counter result, or invalid configuration |
| `SemanticBackendError` | Callback inference/output failure or FastEmbed import, initialization, inference, or vector failure |
| `asyncio.CancelledError` | The caller cancels an awaited `atrim()` task |

Argument and configuration errors are programming errors; fix the call rather than retrying it.
Backend errors may be transient or permanent depending on their chained cause.

## Fixed behavior that is not configuration

Trimwise intentionally does not expose every ranking constant. In v1, these remain fixed:

- BM25 uses `k1=1.5` and `b=0.75`.
- Primary relevance receives 90% of the relevance score; structural and fact-like signal receives
  10%.
- Hybrid normally uses an equal normalized lexical-semantic blend.
- Hybrid's defensive RRF fallback uses `k=60`.
- Query-aware selection uses a fixed five-candidate recall buffer around its score-drop boundary.

These are implementation choices, not promises that each value is universally optimal. Keeping
them private avoids a large tuning surface before Trimwise-shaped benchmarks justify it. Publicly
configure only the behavior your application can evaluate directly: budget measurement, omission
style, MMR balance, and the managed semantic backend.

## Continue exploring

- Return to the [Trimwise overview](index.md).
- Follow the [Getting Started guide](getting-started.md).
- Compare ranking behavior in [Choosing a Strategy](strategies.md).
- Configure embeddings with [Semantic Models and Async Usage](semantic-and-async.md).
