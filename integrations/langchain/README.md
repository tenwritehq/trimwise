# Trimwise for LangChain

A small `RunnableLambda` that trims retrieved documents together under one evidence
limit. It uses Trimwise's public `trim_context()` / `atrim_context()` APIs. This is a
separate package; core Trimwise does not depend on LangChain.

Read the [Trimwise LangChain guide](https://trimwise.readthedocs.io/en/latest/integrations/langchain/)
for usage, async calls, and budget and source-span behavior.

## Install

```bash
python -m pip install langchain-trimwise==0.1.0
```

Requires Python 3.10 or later. For the PDF playground, use the complete repository
checkout and its separate environment described below.

## Run from source

With a Python 3.10+ virtual environment activated, run from the repository root:

```bash
python -m pip install -e . -e './integrations/langchain[test]'
python integrations/langchain/examples/minimal.py
python -m pytest integrations/langchain/tests -c integrations/langchain/pyproject.toml
```

For the repository's UV-managed environment, use
`uv pip install --python .venv/bin/python -e . -e './integrations/langchain[test]'`
and run the commands with `.venv/bin/python`.

The example uses an already-retrieved document fixture and prints a composed
LangChain prompt plus provenance. It does not parse PDFs, retrieve embeddings,
download models, generate answers, or need an API key. Trimwise may fetch and cache
its small tiktoken vocabulary on first use; later runs can reuse that cache.

## Use after your retriever

```python
from langchain_core.documents import Document
from langchain_trimwise import create_trimwise_runnable

compress = create_trimwise_runnable(limit=300, strategy="lexical")
inputs = {
    "query": "What is the retry deadline?",
    "documents": [
        Document(
            id="chunk-1",
            page_content="The retry deadline is thirty seconds.",
            metadata={"source": "manual.pdf", "page": 3},
        ),
    ],
}
documents = compress.invoke(inputs)
# In an async application: documents = await compress.ainvoke(inputs)
```

The input has `documents` (a sequence of LangChain `Document` objects) and `query`
(a string, or `None` for queryless selection). The output is a list of new documents
in input order. Sources with no retained text are omitted. Native `invoke`,
`ainvoke`, `batch`, `abatch`, and Runnable composition work normally; every batch
item gets its own independent shared budget.

Options are `limit`, `unit` (`tokens`, `words`, `characters`), `strategy`,
`token_counter`, and an optional reusable `trimmer=Trimmer(...)`. Defaults are
tokens and automatic strategy. A question makes automatic mode lexical; without a
question it is structural. Semantic/hybrid strategies can use a caller-configured
embedding callback on the supplied Trimmer; FastEmbed is optional and is not an
integration dependency. `ainvoke` uses native async callbacks on the caller's loop.
Cancellation propagates to async callbacks; already-running CPU worker work may
finish. Argument and backend errors propagate rather than silently falling back.

## Budget and provenance contract

- One Trimwise call covers all retrieved chunks. The limit bounds the **sum of
  independently measured evidence outputs**, including any omission markers.
- Labels, metadata, separators, prompt instructions, and answer space are outside
  this limit. Measure the final rendered prompt with the answer model's tokenizer;
  independently counted token totals can differ after concatenation. Supply
  `token_counter` when the default `o200k_base` tokenizer is unsuitable.
- Inputs are never mutated. Output documents keep their IDs and independent deep
  copies of all existing metadata, including nested values. Duplicate IDs are
  allowed; mapping uses input positions rather than IDs.
- `metadata["trimwise"]` contains `source_index` and `spans`, a list of
  `{"start": inclusive, "end": exclusive}` offsets in the original input chunk's
  Python string. These are not byte, PDF-coordinate, or output-text offsets.
  Inserted omission markers have no source span. Keep the original chunks to
  reconstruct highlighted evidence; retained spans may be disjoint.
- The `trimwise` metadata key is reserved. Existing values, including a previous
  adapter result, raise `ValueError`; pass the original retrieved documents for
  each new question rather than repeatedly trimming a prior result.
- This adapter does not render metadata. If your prompt formatter dumps all
  metadata, explicitly omit `trimwise` bookkeeping from that prompt.

Source spans demonstrate where text came from; they do not prove that a generated
answer is supported. Trimming is not a prompt-injection defense or a substitute for
your retriever's authorization checks.

Only `trimwise` and `langchain-core` are runtime dependencies. Core Trimwise does
not depend on this package. Compatibility is declared for LangChain Core
`>=1.6.10,<2` and Trimwise `>=0.7,<0.8`; framework compatibility does not imply upstream listing.

## PDF playground

The [standalone LangChain PDF example](https://github.com/tenwritehq/trimwise/blob/main/integrations/langchain/examples/pdf-rag-playground/README.md) accepts
user PDFs, performs native retrieval, and compares answers with and without this
integration. It has its own lockfile and CPU-only environment; demo dependencies
are kept out of this package. Run it from a complete repository checkout.


## Publishing preparation

Its independent PyPI release workflow and exact Trusted Publisher settings are
documented in [the release guide](https://github.com/tenwritehq/trimwise/blob/main/integrations/RELEASING.md).
Merging changes or pushing an integration tag does not publish a release.
