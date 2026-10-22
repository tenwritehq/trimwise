---
title: LlamaIndex integration
description: Install llama-index-postprocessor-trimwise and trim retrieved TextNodes under one shared evidence budget using LlamaIndex's native postprocessor hook.
---

# Trimwise for LlamaIndex

`TrimwiseNodePostprocessor` gives retrieved plain `TextNode` evidence **one shared
limit**. It implements LlamaIndex's native `BaseNodePostprocessor` hook, so you can
use it directly on retrieval results or attach it to your query engine.

```text
retriever → NodeWithScore results → Trimwise postprocessor → answer synthesis
```

## Install

```bash
python -m pip install llama-index-postprocessor-trimwise==0.1.0
```

The adapter requires Python 3.10 or later, Trimwise `>=0.7,<0.8`, and LlamaIndex
Core `>=0.14.25,<0.15`. Installing it brings in those dependencies. Installing
core Trimwise alone does not install LlamaIndex or the PDF demo dependencies.

## Retrieve and trim without a model

This complete example uses a native `SummaryIndex`. Its default retriever returns
every node, so the example needs no embedding or answer model. It demonstrates
the shared budget and native wiring, not retrieval quality.

```python
from llama_index.core import SummaryIndex
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.postprocessor.trimwise import TrimwiseNodePostprocessor

question = "What is the retry deadline?"
nodes = [
    TextNode(
        id_="garden-1",
        text="Gardens have flowers and trees. " * 10,
        metadata={"source": "garden.pdf", "page": 1},
    ),
    TextNode(
        id_="manual-3",
        text="The retry deadline is thirty seconds.",
        metadata={"source": "manual.pdf", "page": 3},
    ),
]
index = SummaryIndex(nodes)
retrieved = index.as_retriever().retrieve(question)
processor = TrimwiseNodePostprocessor(limit=12, unit="words", strategy="lexical")
retained = processor.postprocess_nodes(retrieved, query_str=question)

assert sum(len(item.node.get_content(MetadataMode.NONE).split()) for item in retained) <= 12
assert any("thirty seconds" in item.node.get_content(MetadataMode.NONE) for item in retained)
for item in retained:
    print(item.node.get_content(MetadataMode.NONE))
    print(item.node.node_id, item.score, item.node.metadata["trimwise"])
```

Use `query_str` or a native `QueryBundle` to supply the question. The default unit
is tokens. With the default `auto` strategy, a nonblank question selects lexical
ranking; no question selects structural ranking.

Inside async code, use the native async entry point:

```python
retained = await processor.apostprocess_nodes(retrieved, query_str=question)
```

## Attach to a query engine

Given your existing `index` and explicitly configured answer model `llm`, attach
the processor through the native hook:

```python
query_engine = index.as_query_engine(llm=llm, node_postprocessors=[processor])
```

Choose a token budget for production, for example
`TrimwiseNodePostprocessor(limit=300, strategy="lexical")`. Supply your own
model-specific `token_counter` when the default `o200k_base` tokenizer is
unsuitable. You can also pass a reusable `trimmer=Trimmer(...)` with an embedding
callback for semantic or hybrid selection; FastEmbed remains optional.
Async callbacks run on the caller's event loop, and cancellation reaches them,
though already-running CPU worker work may finish. Validation and backend errors
propagate to the caller.

## Budget and native node behavior

One operation covers all retrieved nodes. The limit bounds the sum of
independently measured text outputs, including omission markers. It excludes
rendered metadata, labels, separators, text templates, instructions, and answer
space. Check the complete final prompt with your answer model's tokenizer;
concatenation can change token totals.

Only plain `TextNode` objects are supported. Image, index, multimodal, and other
node types raise `TypeError`. Inputs are never mutated. Returned `NodeWithScore`
objects are deep copies in input order, omitting sources with no retained text.
They preserve IDs, existing metadata, relationships, templates, and retrieval
scores. Scores describe retrieval of the original evidence and are not recomputed.
When text changes, stale embeddings and contiguous character offsets are cleared.

The reserved `node.metadata["trimwise"]` field holds the original `source_index`,
original character offsets, and retained `spans`. Span `start` is inclusive and
`end` is exclusive; both are Python string offsets within the original node's
plain text. They are not output offsets or PDF coordinates. Keep original nodes
for highlighting. Omissions can make retained spans disjoint.

Provenance is excluded from LLM and embedding metadata rendering. A pre-existing
`trimwise` key raises `ValueError`; use original retrieval results for each
question. These source spans do not establish answer correctness, enforce document
permissions, or sanitize hostile instructions. Runtime Trimmer instances and token
counters are excluded from Pydantic serialization; reattach callbacks when
reconstructing a processor from config.

## Try your own PDFs

The [native LlamaIndex PDF playground](https://github.com/tenwritehq/trimwise/blob/main/integrations/llama-index/examples/pdf-rag-playground/README.md)
uses `TextNode`, `VectorStoreIndex`, and this postprocessor. It compares answers
from full and trimmed evidence after one frozen retrieval, with source-span
inspection and optional human feedback. Generation and embeddings are CPU-only
through local Ollama and a local embedding model. Follow its separate environment
and model setup from a complete checkout; it is not part of the adapter wheel.

See [many sources, one shared limit](../multi-source-context.md) for the core API,
[the minimal native example](https://github.com/tenwritehq/trimwise/blob/main/integrations/llama-index/examples/minimal.py),
and [the package on PyPI](https://pypi.org/project/llama-index-postprocessor-trimwise/).
This is an independently maintained integration; an upstream listing or
endorsement is not implied.
