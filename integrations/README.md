# Framework integrations

Two independent Python packages live alongside core Trimwise:

| Framework | Package | Entry point | CPU-only PDF example |
| --- | --- | --- | --- |
| [LangChain](langchain/README.md) | `langchain-trimwise` | `create_trimwise_runnable()` | [Run LangChain](langchain/examples/pdf-rag-playground/README.md) |
| [LlamaIndex](llama-index/README.md) | `llama-index-postprocessor-trimwise` | `TrimwiseNodePostprocessor` | [Run LlamaIndex](llama-index/examples/pdf-rag-playground/README.md) |

Each applies one shared evidence budget **after retrieval** using Trimwise's
public APIs. Each has its own dependencies, tests, runnable example, and build
configuration. Neither framework is added to core Trimwise's dependencies.
See the package READMEs for PyPI installation, local examples, and exact behavior.

Build locally from the repository root with `python -m build integrations/langchain`
or `python -m build integrations/llama-index`. Each package's `dist/` contains its
own wheel and source distribution. SDK source distributions include the minimal
model-free example. The PDF apps need the complete repository checkout for their
shared helpers. The [release guide](RELEASING.md) describes the separate, manual Trusted Publishing
workflow. Each adapter has its own version and release tag.

PDF extraction, the UI, and comparison accounting are shared privately under
`_example_support/pdf-rag-playground`; each example owns its native retrieval and
Trimwise adaptation. Neither app imports the other framework. Both currently force
CPU inference and have no GPU control.
