# Semantic Models and Async Usage

Semantic trimming helps when the query and the answer use different words. Hybrid trimming adds
that meaning-based signal to BM25 exact-term matching. Both strategies need embeddings, but they
do not require Trimwise to own the model: you can provide vectors from an existing local model,
hosted API, internal service, or cache with the normal core installation.

This page explains how to choose a semantic backend, connect it safely, understand the vector
contract, reuse models, handle concurrency and cancellation, and diagnose failures.

## Choose a semantic path

| Your situation | Install | Configure | Call |
| --- | --- | --- | --- |
| You already have a blocking model or client | `trimwise` | `embedding_callback=` | `trim()` or `atrim()` |
| You already have an async embedding client | `trimwise` | `async_embedding_callback=` | `atrim()` |
| You want Trimwise to run a local CPU model | `trimwise[semantic]` | No callback | `trim()` or `atrim()` |
| You want Trimwise to run a local NVIDIA GPU model | `trimwise[semantic-gpu]` | No callback | `trim()` or `atrim()` |

With `pip`:

```bash
python -m pip install trimwise
python -m pip install "trimwise[semantic]"
python -m pip install "trimwise[semantic-gpu]"
```

With `uv`:

```bash
uv add trimwise
uv add "trimwise[semantic]"
uv add "trimwise[semantic-gpu]"
```

Install only one of the CPU and GPU extras. Their FastEmbed runtime packages conflict, and the GPU
variant also requires compatible CUDA and cuDNN libraries.

If you are unsure whether embeddings are necessary, start with `auto`. It uses structural coverage
without a query and lexical BM25 with one. It never invokes a semantic backend.

## What semantic scoring does

Trimwise first turns the source into candidate passages. Each passage receives local ranking
context: its nearest section heading and its previous and next units from the same section. The
candidate itself is placed first so direct evidence remains distinguishable from neighboring
context.

The backend produces:

```text
one query vector
one passage vector × number of candidate passages
```

Trimwise converts the vectors to NumPy `float32`, stacks them into one matrix, validates their
shape and values, and L2-normalizes every row. Semantic relevance is the cosine similarity between
the query row and each passage row. During selection, vectorized passage-to-passage similarity
penalizes evidence that looks too similar to fragments already selected.

The returned excerpt is still extractive. Ranking context and embeddings decide which candidate
source spans survive; they are not inserted into the output or used to generate new wording.

This sentence-vector approach is grounded in research such as
[Sentence-BERT](https://aclanthology.org/D19-1410/) and
[multilingual sentence embeddings](https://aclanthology.org/2020.emnlp-main.365/). Trimwise adapts
the idea to Markdown-aware source spans, MMR selection, exact budgets, and source-order output; it
does not reproduce either paper's full system.

## Bring your own synchronous embeddings

Use a synchronous callback for a local model or blocking API client. It receives the query
separately from the passages because retrieval-oriented embedding models often apply different
instructions or encoders to queries and documents.

```python
from collections.abc import Sequence

from trimwise import Trimmer


def embed(
    query: str,
    passages: Sequence[str],
) -> tuple[object, Sequence[object]]:
    """Embed one query and its candidate passages.

    Args:
        query: Task or question supplied to Trimwise.
        passages: Context-enriched candidate text in source order.

    Returns:
        Query vector and one same-dimension vector per passage.
    """
    return model.encode_query(query), model.encode_documents(list(passages))


trimmer = Trimmer(embedding_callback=embed)
result = trimmer.trim(
    document,
    limit=500,
    strategy="semantic",
    query="What caused the outage?",
)
```

Adapt `encode_query()` and `encode_documents()` to the model or client you already use. The
callback takes precedence over FastEmbed even if a semantic extra is installed, so Trimwise does
not import or download its default model on this path.

A synchronous callback also works with `atrim()`. Trimwise runs the full synchronous pipeline,
including your callback, in a worker thread so it does not block the event loop.

## Bring your own asynchronous embeddings

Use an asynchronous callback for a network client with a native async API. Configure it on the
`Trimmer`, then call `atrim()` whenever semantic vectors are needed.

```python
import asyncio
from collections.abc import Sequence

from trimwise import Trimmer


async def embed(
    query: str,
    passages: Sequence[str],
) -> tuple[object, Sequence[object]]:
    """Embed one query and its passages through an async client.

    Args:
        query: Task or question supplied to Trimwise.
        passages: Context-enriched candidate text in source order.

    Returns:
        Query vector and one same-dimension vector per passage.
    """
    vectors = await client.embed([query, *passages])
    return vectors[0], vectors[1:]


async def main() -> None:
    """Create one hybrid excerpt without blocking the event loop."""
    trimmer = Trimmer(async_embedding_callback=embed)
    result = await trimmer.atrim(
        document,
        limit=500,
        strategy="hybrid",
        query="What caused incident ORION-774?",
    )
    print(result.text)


asyncio.run(main())
```

Trimwise prepares the source in a worker thread, awaits your callback on the calling event loop,
then moves vector normalization, ranking, selection, and final measurement back to a worker thread.
This lets the embedding client use its existing async session, connection pool, and cancellation
behavior.

Calling `trim()` with only an async callback raises `TypeError` if semantic ranking is actually
required. Use `atrim()` for that path. A fitting input can still return unchanged from `trim()`
because no vectors are needed.

Configure either `embedding_callback` or `async_embedding_callback`, never both.

## Callback output requirements

Return a two-item tuple:

```text
(query_vector, passage_vectors)
```

Trimwise accepts Python numeric sequences and NumPy-compatible arrays. The output must satisfy all
of these rules:

| Requirement | Why it matters |
| --- | --- |
| One query vector | Every call represents one task or question |
| Exactly one passage vector per supplied passage | Scores must map back to the correct source candidate |
| Original passage order | Trimwise aligns vector row `i` with candidate `i` |
| One nonempty dimension shared by every vector | Query and passage rows must fit one numeric matrix |
| Finite numeric values | NaN and infinity cannot produce dependable rankings |

Zero vectors are accepted and remain zero after normalization, producing zero cosine similarity.
If your model emits them unexpectedly, handle that in the callback or model configuration.

Trimwise materializes passage iterables, converts all rows to `float32`, validates the matrix, and
normalizes it once. A callback failure or invalid output raises a chained `SemanticBackendError`;
Trimwise does not quietly switch to lexical scoring.

### Embed the passages exactly as received

The strings passed to your callback are ranking-only representations, not necessarily one isolated
paragraph. A passage can include the candidate again plus its heading and same-section neighbors.
Embed each supplied string as-is and return vectors in the same order.

An external embedding service may therefore receive source content and nearby context. Apply the
same privacy, data residency, and retention rules you would use when sending the document itself to
that provider.

## What your callback owns

Trimwise validates callback output and performs scoring, but the callback remains your runtime
boundary. Your application owns:

- Model or client reuse.
- Authentication and connection pooling.
- Request batching performed inside the client.
- Caching and cache invalidation.
- Rate limiting, retry policy, and timeouts.
- Thread or task safety.
- Provider-specific input limits and privacy controls.

Trimwise does not serialize caller callbacks. Concurrent calls on the same `Trimmer` may invoke the
same callback concurrently. Add a semaphore or lock inside the callback only if the underlying
model or service requires one.

## Let Trimwise manage FastEmbed

If you do not already have an embedding backend, install one semantic extra and use `Trimmer`
without a callback:

```python
from trimwise import Trimmer

trimmer = Trimmer()
result = trimmer.trim(
    document,
    limit=500,
    strategy="semantic",
    query="What caused the outage?",
)
```

Trimwise lazily creates FastEmbed's `TextEmbedding` only when an oversized semantic or hybrid call
reaches ranking. It uses `query_embed()` for the query and `passage_embed()` for candidates, then
materializes both generators while holding the model lock.

The default model is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Its weights are approximately 220 MB. A cold call may include downloading weights and initializing
the model, so it can be much slower than later inference. The same `Trimmer` reuses its initialized
model; candidate embeddings themselves are recomputed on later trim calls because v1 has no
cross-call embedding cache.

### Configure managed inference

```python
from trimwise import TrimConfig, Trimmer

config = TrimConfig(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    embedding_batch_size=64,
    fastembed_options={"providers": ["CPUExecutionProvider"]},
)
trimmer = Trimmer(config)
```

| Setting | What it controls |
| --- | --- |
| `embedding_model` | FastEmbed model passed as `model_name` |
| `embedding_batch_size` | Batch size used for passage inference; the default is `256` |
| `fastembed_options` | Additional keyword arguments forwarded when `TextEmbedding` is created |

`embedding_model` must be nonblank, the batch size must be a positive integer, and
`fastembed_options` keys must be strings. Do not put `model_name` inside `fastembed_options`; use
`embedding_model` so there is one source of truth.

Model support, provider options, GPU compatibility, and cache locations are FastEmbed concerns.
Changing the model can change languages, dimensions, quality, memory use, latency, and score
distributions.

## FastEmbed concurrency and memory

Each `Trimmer` owns one lazy FastEmbed model and one lock. Model initialization, query inference,
passage inference, and generator materialization run one call at a time for that instance.

```text
same Trimmer:       call A ─────> call B ─────> call C
separate Trimmers:  call A ─────────────>
                    call B ─────────────>
```

Reuse one `Trimmer` when model memory matters and serialized inference is acceptable. Separate
instances can infer independently, but each may initialize and retain another model in memory.

The lock applies only to Trimwise-managed FastEmbed. Caller callbacks remain caller-managed and may
run concurrently.

## When semantic work is skipped

Trimwise delays semantic work until it is certain that embeddings are needed:

| Call | Backend invoked? |
| --- | --- |
| Input already fits, with a valid semantic or hybrid request | No; returns the input unchanged |
| `limit=0` | No; returns empty text |
| `auto` without a query | No; resolves to structural |
| `auto` with a query | No; resolves to lexical |
| Explicit `structural` or `lexical` | No |
| Oversized explicit `semantic` or `hybrid` | Yes |

Arguments are still validated before the fitting-input return. For example, an explicit semantic
call still requires a nonblank query even when its text would fit.

## Sync and async behavior

`trim()` and `atrim()` return the same `TrimResult` for the same deterministic backend and inputs.
Choose between them based on how your application schedules work:

| Backend or strategy | `trim()` | `atrim()` |
| --- | --- | --- |
| Structural or lexical | Runs on the calling thread | Runs in a worker thread |
| Synchronous callback | Runs on the calling thread | Runs in a worker thread |
| Trimwise-managed FastEmbed | Runs on the calling thread | Runs in a worker thread |
| Asynchronous callback | Unsupported when vectors are required | Awaited on the calling event loop; CPU stages use worker threads |

```python
import asyncio

from trimwise import Trimmer


async def main() -> None:
    """Trim several sources while keeping the event loop responsive."""
    trimmer = Trimmer()
    sources = [
        "First source with an opening fact and a later decision.",
        "Second source with a warning and a final recommendation.",
    ]
    results = await asyncio.gather(
        *(trimmer.atrim(source, 8, unit="words") for source in sources)
    )
    for result in results:
        print(result.text)


asyncio.run(main())
```

For CPU-only structural or lexical work, async calls can overlap at the worker-thread level. For
FastEmbed, calls sharing one `Trimmer` still wait on that instance's model lock.

## Cancellation

Cancellation behavior depends on what `atrim()` is awaiting:

- **Asynchronous embedding callback:** cancellation propagates into the callback. Its `finally`
  blocks and normal async cleanup can run.
- **Worker-thread work:** cancellation stops the calling task from waiting, but Python cannot
  forcibly terminate a thread that has already begun parsing, counting, running a synchronous
  callback, or performing FastEmbed inference.

Design synchronous callbacks with their own timeouts where possible. Cancelling the outer task is
not a substitute for stopping a blocking network request inside a worker.

## Error behavior

Semantic failures cross one stable public boundary: `SemanticBackendError`. The original exception
is preserved as `__cause__` for logs and debugging.

Caller callbacks report two broad stages:

| Error text contains | Check |
| --- | --- |
| `callback inference failed` | The model or client call, credentials, timeout, or provider |
| `callback output failed` | Vector count, dimensions, numeric conversion, NaN, or infinity |

Trimwise-managed FastEmbed identifies more specific stages:

| Stage | Typical checks |
| --- | --- |
| `import` | Install exactly one semantic extra |
| `initialization` | Model name, cache, provider, and runtime options |
| `query inference` | Model and provider compatibility; exactly one query vector |
| `passage inference` | Model, provider, batch size, and one vector per passage |
| `inference output` | Equal dimensions and finite numeric vectors |

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
    print("Backend cause:", error.__cause__)
```

Trimwise never silently falls back from semantic or hybrid to lexical. A quiet fallback would make
quality and production failures difficult to detect and could return evidence chosen by a strategy
the caller did not request.

## Semantic and hybrid research boundaries

Semantic mode uses normalized query-passage cosine scores. Hybrid mode normally min-max normalizes
usable BM25 and semantic score rows, then combines them equally. This is informed by
[hybrid fusion analysis](https://arxiv.org/abs/2210.11934), while the fixed 50/50 weight remains a
label-free default rather than a learned optimum for every domain.

If either hybrid score row is flat or non-finite, Trimwise uses
[Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114) with `k=60` as a defensive fallback.
After semantic or hybrid relevance scoring, MMR uses embedding similarity to discourage
repeated-looking passages.

Embeddings measure representational similarity, not truth. A semantically close passage can still
be wrong, outdated, or insufficient, and MMR does not prove that two selected passages contain
different facts. Evaluate downstream answer quality with your documents, model, queries, and
compression ratios.

## Practical checklist

- Start with lexical selection unless paraphrases or multilingual meaning are being missed.
- Reuse an embedding model or client rather than recreating it inside every callback call.
- Return exactly one passage vector per supplied string and preserve passage order.
- Keep vector dimensions equal and values finite.
- Use `atrim()` for native async clients.
- Add callback-side rate limits, retries, timeouts, caching, and synchronization when needed.
- Reuse one `Trimmer` for managed FastEmbed unless measured throughput justifies extra model memory.
- Separate cold model download and initialization time from warm inference when benchmarking.
- Log `SemanticBackendError.__cause__` without silently changing strategies.

## Continue exploring

- Return to the [Trimwise overview](index.md).
- Follow the [Getting Started guide](getting-started.md).
- Compare all selection modes in [Choosing a Strategy](strategies.md).
- Review planned semantic quality work in the [roadmap](https://github.com/tenwritehq/trimwise/blob/main/ROADMAP.md).
