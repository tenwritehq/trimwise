# LlamaIndex PDF RAG playground

A standalone local example for the **LlamaIndex Trimwise integration**. Upload your
own text PDFs and compare full retrieved evidence with one shared-budget Trimwise
selection. Both answers use Qwen3-4B-Instruct-2507 Q4_K_M through local Ollama.
**Generation and embeddings are CPU-only. There is no GPU selector.**

The native pipeline in [llamaindex_rag/pipeline.py](llamaindex_rag/pipeline.py) is:

```text
LlamaIndex TextNodes → VectorStoreIndex retriever → TrimwiseNodePostprocessor
```

One frozen retrieval feeds both conditions. The baseline includes every retrieved
chunk in full; the trimmed condition invokes this framework's integration once.
Both use the same question, model, instructions, tokenizer, and answer allowance.
Getting the same supported answer from less context is the intended outcome.
Inspect the removed evidence to find cases where compression loses information.
This is an integration showcase, not a general accuracy or speed benchmark.

## Run from this repository

Requirements: Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com/download). Clone the complete repository: the examples
share private PDF/UI helpers under `integrations/_example_support/`. This app's
isolated environment installs only LlamaIndex and its own integration, plus those
framework-neutral helpers. The helpers are not published, and no demo dependency
is added to core Trimwise or the integration package's runtime requirements.

Pull the model once:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

If Ollama is not running, start a CPU-only loopback server in another terminal:

```bash
CUDA_VISIBLE_DEVICES=-1 GGML_VK_VISIBLE_DEVICES=-1 OLLAMA_VULKAN=0 \
  OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 ollama serve
```

Then, from the repository root:

```bash
cd integrations/llama-index/examples/pdf-rag-playground
uv sync --locked
uv run python app.py
```

Open **http://127.0.0.1:7861**. Choose **Load the example**, then **Compare answers**,
or upload PDFs and choose **Index PDFs**. The two native apps use different ports,
so they can run together. Each uses separate session indexes and temporary storage.
Every Ollama preload and generation request explicitly sets `num_gpu=0`, including
when an already-running Ollama server supports GPU execution.

The first index downloads the CPU `BAAI/bge-small-en-v1.5` embedding model. First
comparison downloads the pinned Qwen tokenizer, without unquantized model weights.
The Ollama Q4 model is about 2.5 GB. Public model downloads require internet; cached
PDF processing and inference stay local. CPU latency depends on input/output size,
model load, and cache reuse. There are no GPU timing promises for this CPU-only app.

## Inspect the comparison

- Answer comparison: full and trimmed contexts, answers, citations, actual prompt
  and output usage, completion reasons, and observed timings.
- Evidence inspector: exact retained/removed source spans, pages, and chunk offsets.
- Run details: fixed framework identity, model/tokenizer revisions, source hashes,
  measured contexts, and separate model acquisition time.
- Evaluate & export: optional human feedback, normalized exact reference matching,
  exact quote checks, and JSON export. Content is excluded by default; explicitly
  enable the checkbox to include question, answers, filenames, and source text.

The Trimwise budget measures the sum of evidence passage tokens. Source labels,
filenames, separators, instructions, and reserved answer tokens are additional.
The main metrics count complete rendered contexts. Both full native chat prompts
are checked against the 4,096-token context with answer space reserved. If either
cannot fit, neither answer runs: reduce top-k or chunk size and re-index. Runtime
usage must match the complete native prompt count; silent truncation is rejected.

Citation labels and source spans establish provenance, not factual correctness.
Reference matching is normalized exact matching, not semantic grading. Timings
include retrieval/preparation/compression/generation and exclude model acquisition;
baseline generates first, so cache and order effects matter.

## Local data and limits

Select 1–5 PDFs, at most 10 MiB and 100 pages each, 500,000 extracted characters
total, and 1,000 chunks. Encrypted, malformed, and image-only PDFs are rejected;
partially textless files report skipped pages. OCR is out of scope. Parsing uses
a disposable subprocess with a 25-second timeout and a Unix memory limit.

Indexing atomically replaces the full selection. Exact duplicate PDF content is
indexed once. Failed indexing keeps the previous index; changed files or chunk
settings require re-indexing. Reset clears the current index, results, and exports;
Gradio upload/download copies expire within an hour or when the app exits. Original
PDFs and downloaded model caches are preserved.

This is a trusted local workbench without authentication. It binds to loopback,
disables public Gradio sharing, analytics, and LangSmith tracing, and does not send
evidence to a cloud model. The temporary Gradio cache is shared process storage,
not an authenticated boundary between local users. Do not expose it publicly.

## Verify

Adapter tests and the model-free minimal example are documented in the
[integration README](../../README.md). The shared PDF/session/comparison suite
runs against both real native vector indexes with explicitly synthetic embeddings
and answers. It also checks that all inference requests remain CPU-only:

```bash
# From this directory; the shared development environment includes both frameworks.
uv run --project ../../../_example_support/pdf-rag-playground --extra dev \
  pytest -c ../../../_example_support/pdf-rag-playground/pyproject.toml \
  ../../../_example_support/pdf-rag-playground/tests -q

# Optional real CPU run: all five reviewed public-fixture questions, failures retained.
uv run python smoke.py --output results/local-smoke.json
```

The original synthetic MIT-licensed three-page PDF and five frozen questions are
shared between both examples, including an unanswerable case. They were labelled
before model comparisons. Regenerate the fixture only from the shared helper's
`scripts/build_fixture.py` using its development environment.
