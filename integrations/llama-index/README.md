# Trimwise for LlamaIndex

`TrimwiseNodePostprocessor` trims retrieved plain `TextNode` evidence under one
shared allowance. It subclasses LlamaIndex's native `BaseNodePostprocessor` and
uses Trimwise's public sync/async context APIs. Core Trimwise has no LlamaIndex dependency.

Read the [Trimwise LlamaIndex guide](https://trimwise.readthedocs.io/en/latest/integrations/llama-index/)
for usage, async calls, and budget and source-span behavior.

## Install

```bash
python -m pip install llama-index-postprocessor-trimwise==0.1.0
```

Requires Python 3.10 or later. For the PDF playground, use the complete repository
checkout and its separate environment described below.

## Run from source

With a Python 3.10+ virtual environment activated, run from the repository root:

```bash
python -m pip install -e . -e './integrations/llama-index[test]'
python integrations/llama-index/examples/minimal.py
python -m pytest integrations/llama-index/tests -c integrations/llama-index/pyproject.toml
```

For the repository's UV-managed environment, use
`uv pip install --python .venv/bin/python -e . -e './integrations/llama-index[test]'`
and run the commands with `.venv/bin/python`.

The example creates a tiny `SummaryIndex`, whose default retriever returns every
node, then trims and prints the evidence and source provenance. It demonstrates
framework wiring, not retrieval quality. No embedding model, answer model,
model download, external inference API, or API key is needed. Trimwise may fetch
and cache its small tiktoken vocabulary on first use.

## PDF playground

The [standalone LlamaIndex PDF example](https://github.com/tenwritehq/trimwise/blob/main/integrations/llama-index/examples/pdf-rag-playground/README.md) accepts
user PDFs, performs native retrieval, and compares answers with and without this
integration. It has its own lockfile and CPU-only environment; demo dependencies
are kept out of this package. Run it from a complete repository checkout.

## Use after retrieval

```python
from llama_index.postprocessor.trimwise import TrimwiseNodePostprocessor

processor = TrimwiseNodePostprocessor(limit=300, strategy="lexical")
# retrieved = your_retriever.retrieve(question)
# retained = processor.postprocess_nodes(retrieved, query_str=question)
# retained = await processor.apostprocess_nodes(retrieved, query_str=question)
```

For an existing index and explicitly configured answer model, use the native hook:

```python
query_engine = index.as_query_engine(llm=llm, node_postprocessors=[processor])
```

`limit` is required. Optional fields are `unit` (`tokens`, `words`, `characters`),
`strategy`, `token_counter`, and `trimmer=Trimmer(...)`. Defaults are tokens and
automatic strategy: lexical with a nonblank question, structural without one.
Use `query_str` or a `QueryBundle`, following the base LlamaIndex API. Semantic and
hybrid strategies can use a caller-configured embedding callback on the Trimmer;
FastEmbed remains optional. Native asynchronous callbacks run on the invoking
event loop. Cancellation reaches those callbacks, though CPU worker work already
running may finish. Validation and backend errors propagate to the caller.

## Budget and provenance contract

- One operation covers all retrieved nodes. The budget bounds the **sum of
  independently measured text outputs**, including omission markers. It excludes
  LlamaIndex's rendered metadata, labels, separators, templates, instructions, and
  answer space. Measure the complete prompt separately with your model's tokenizer;
  concatenation can change token totals. `token_counter` overrides `o200k_base`.
- Only plain `TextNode` objects are supported. Image, index, multimodal, and other
  node types raise `TypeError` rather than silently losing non-text content.
- Returned `NodeWithScore` objects are deep copies in input order; empty evidence
  rows are omitted. IDs, existing metadata, relationships, text templates, and
  retrieval scores remain associated with their original node. Scores still
  describe retrieval of the original evidence; they are not recomputed.
- Provenance is stored in `node.metadata["trimwise"]`: `source_index`, original
  `start_char_idx` / `end_char_idx` under `original_start_char_idx` /
  `original_end_char_idx`, and `spans` containing inclusive `start` and exclusive
  `end` Python-string offsets **within the original input node's plain text**.
  Retained spans can be disjoint; omission markers have no source span. These are
  not offsets into the new output or coordinates on a PDF page. Keep original
  nodes for highlighting.
- When text changes, the copied node's contiguous character offsets and cached
  embedding are cleared because they no longer describe the excerpt. Unchanged
  text retains those fields. Input nodes are never modified.
- The `trimwise` metadata key is reserved; a collision raises `ValueError`.
  Start each question from original retrieved nodes. Provenance is excluded from
  both LLM and embedding metadata rendering; user metadata exclusions are preserved.
- Runtime Trimmer instances and token counters are excluded from Pydantic
  serialization. Reattach callbacks when reconstructing a processor from config.

Trimming does not establish answer correctness, enforce document permissions, or
sanitize hostile document instructions.

Runtime dependencies are `trimwise>=0.7,<0.8` and
`llama-index-core>=0.14.25,<0.15`. Core Trimwise has no LlamaIndex dependency.
Framework compatibility does not imply upstream listing.

## Publishing preparation

Its independent PyPI release workflow and exact Trusted Publisher settings are
documented in [the release guide](https://github.com/tenwritehq/trimwise/blob/main/integrations/RELEASING.md).
Merging changes or pushing an integration tag does not publish a release.
