"""Launch the standalone native LangChain PDF example on CPU."""

from rag_playground.app import launch

if __name__ == "__main__":
    from langchain_rag.pipeline import build_corpus

    launch(build_corpus, "LangChain", 7860)
