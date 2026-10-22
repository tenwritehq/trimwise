"""Record all public fixture cases through native LangChain retrieval on CPU."""

from langchain_rag.pipeline import build_corpus
from rag_playground.smoke import run

if __name__ == "__main__":
    run(build_corpus, "LangChain")
