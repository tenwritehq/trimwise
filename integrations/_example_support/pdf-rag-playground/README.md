# Shared PDF playground support

Private helpers used by the independently runnable
[LangChain](../../langchain/examples/pdf-rag-playground/README.md) and
[LlamaIndex](../../llama-index/examples/pdf-rag-playground/README.md) examples.

This package owns PDF validation/chunking, CPU embeddings, a framework-neutral
Evidence record, comparison accounting, the CPU-only Ollama backend, session
lifecycle, escaped presentation, and the Gradio UI. Native retrieval and Trimwise
adaptation live in each integration's example. Runtime dependencies include
neither framework. The development suite installs both to exercise the same
contracts against real native vector indexes with model doubles.

From this directory:

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check . ../../langchain/examples/pdf-rag-playground ../../llama-index/examples/pdf-rag-playground
uv run ruff format --check . ../../langchain/examples/pdf-rag-playground ../../llama-index/examples/pdf-rag-playground
uv run python scripts/build_fixture.py
```

The support package is a repository-local editable dependency, not a new public
SDK or a package to publish. Its common suite covers both frameworks. A neutral
record is used only at the display/accounting boundary; the native examples call
real framework retrieval and their own installed Trimwise integration.
