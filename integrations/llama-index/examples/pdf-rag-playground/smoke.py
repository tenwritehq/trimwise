"""Record all public fixture cases through native LlamaIndex retrieval on CPU."""

from llamaindex_rag.pipeline import build_corpus
from rag_playground.smoke import run

if __name__ == "__main__":
    run(build_corpus, "LlamaIndex")
