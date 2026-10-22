"""Launch the standalone native LlamaIndex PDF example on CPU."""

from rag_playground.app import launch

if __name__ == "__main__":
    from llamaindex_rag.pipeline import build_corpus

    launch(build_corpus, "LlamaIndex", 7861)
