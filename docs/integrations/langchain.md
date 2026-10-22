---
title: LangChain integration
description: Install langchain-trimwise and apply one shared evidence budget to retrieved LangChain Documents with native Runnable composition and async support.
---

# Trimwise for LangChain

Use `langchain-trimwise` after retrieval to give all retrieved documents **one shared
evidence limit**. `create_trimwise_runnable()` returns a native LangChain
`RunnableLambda`; its output is a list of copied `Document` objects ready for your
context formatter.

```text
retriever → retrieved Documents → Trimwise Runnable → context formatter → answer model
```

## Install

```bash
python -m pip install langchain-trimwise==0.1.0
```

The adapter requires Python 3.10 or later, Trimwise `>=0.7,<0.8`, and LangChain Core
`>=1.6.10,<2`. Installing it brings in those dependencies. Installing core Trimwise
alone does not install LangChain or the PDF demo dependencies.

## Trim retrieved documents

This complete example uses a small fixture in place of a retriever. It does not
call an answer model or create embeddings. The word budget makes the shared limit
easy to inspect; the default unit for real applications is tokens.

```python
from langchain_core.documents import Document
from langchain_trimwise import create_trimwise_runnable

question = "What is the retry deadline?"
documents = [
    Document(
        id="garden-1",
        page_content="Gardens have flowers and trees. " * 10,
        metadata={"source": "garden.pdf", "page": 1},
    ),
    Document(
        id="manual-3",
        page_content="The retry deadline is thirty seconds.",
        metadata={"source": "manual.pdf", "page": 3},
    ),
]
compress = create_trimwise_runnable(limit=12, unit="words", strategy="lexical")
inputs = {"documents": documents, "query": question}
retained = compress.invoke(inputs)

assert sum(len(document.page_content.split()) for document in retained) <= 12
assert any("thirty seconds" in document.page_content for document in retained)
for document in retained:
    print(document.page_content)
    print(document.id, document.metadata["trimwise"])
```

The input requires `documents` and `query`. Use `query=None` with automatic or
structural selection when there is no question. The default `auto` strategy is
lexical with a nonblank question and structural without one.

## Compose with your retriever

Given an existing LangChain `retriever` that accepts a question and returns
Documents, use Runnable composition to supply the two input fields:

```python
from langchain_core.runnables import RunnablePassthrough

pipeline = {"documents": retriever, "query": RunnablePassthrough()} | compress
retained = pipeline.invoke(question)
```

Then pass the retained Documents to your existing formatter and answer model.
`batch()` and `abatch()` work normally; each input gets its own independent shared
budget. Inside async code, use:

```python
retained = await compress.ainvoke(inputs)
```

## Budget, metadata, and source spans

The limit is the sum of independently measured output passage sizes, including
omission markers. It excludes labels, metadata, separators, instructions, and the
answer allowance. Measure the complete rendered prompt with the answer model's
tokenizer; separately counted token totals can change after concatenation.
`token_counter=` accepts a synchronous custom token counter when the default
`o200k_base` tokenizer is unsuitable.

Inputs are not mutated. Outputs preserve IDs and deep-copy existing metadata,
remain in input order, and omit sources with no retained text. The reserved
`metadata["trimwise"]` field contains the original `source_index` and a list of
`{"start": ..., "end": ...}` spans. These are inclusive-start, exclusive-end Python
string offsets into the original input Document's text, not the new excerpt or
coordinates on a PDF page. Keep the original documents for highlighting.

A pre-existing `trimwise` metadata key raises `ValueError`. Start each question
from the original retrieval, and omit this bookkeeping field if your formatter
serializes metadata into the answer prompt. Source spans establish provenance;
they do not establish answer correctness or defend against document instructions.

For semantic or hybrid selection, pass a reusable `trimmer=Trimmer(...)` configured
with your embedding callback. FastEmbed remains optional. Async callbacks run on
the calling event loop through `ainvoke`; cancellation reaches those callbacks,
though CPU worker work already running may finish. Validation and backend errors
propagate to the caller.

## Try your own PDFs

The [native LangChain PDF playground](https://github.com/tenwritehq/trimwise/blob/main/integrations/langchain/examples/pdf-rag-playground/README.md)
uses LangChain Documents, Chroma retrieval, and this Runnable. It compares one
frozen retrieval with and without trimming, with answer, token, and source-span
inspection. Generation and embeddings are CPU-only through local Ollama and a
local embedding model. Follow its separate environment and model setup from a
complete repository checkout; the app is not part of the adapter wheel.

See [many sources, one shared limit](../multi-source-context.md) for the core API,
[the minimal example](https://github.com/tenwritehq/trimwise/blob/main/integrations/langchain/examples/minimal.py)
for prompt composition, and [the package on PyPI](https://pypi.org/project/langchain-trimwise/).
This is an independently maintained integration; an upstream listing or
endorsement is not implied.
