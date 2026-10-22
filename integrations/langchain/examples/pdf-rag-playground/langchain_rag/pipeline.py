"""Use LangChain Documents, Chroma retrieval, and the real Trimwise runnable."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_trimwise import create_trimwise_runnable
from langsmith import tracing_context
from rag_playground.documents import LoadedPDFs
from rag_playground.models import Evidence, RunSettings
from rag_playground.retrieval import Embeddings, local_embeddings


class NativeEmbeddings(LangChainEmbeddings):
    """Expose the CPU embedding model through LangChain's native interface."""

    def __init__(self, model: Embeddings) -> None:
        """Use an already acquired model without owning its cache or device policy."""
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a passage batch with the local model."""
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed the retrieval query with query-specific preprocessing."""
        return self.model.embed_query(text)


def native_document(evidence: Evidence) -> Document:
    """Copy source text, ID, and metadata into a native LangChain Document."""
    return Document(
        id=evidence.id, page_content=evidence.page_content, metadata=deepcopy(evidence.metadata)
    )


def source_evidence(document: Document) -> Evidence:
    """Copy a native result into the display record without sharing mutable metadata."""
    return Evidence(document.page_content, deepcopy(document.metadata), document.id)


@dataclass
class Corpus:
    """Own one local Chroma collection and execute the LangChain integration.

    Attributes:
        loaded: Exact page-backed source chunks.
        store: Native LangChain Chroma vector store.
        index_seconds: Complete index construction time.
        embedding_load_seconds: Separate model acquisition time.
        framework: Fixed name exported with comparison results.
    """

    loaded: LoadedPDFs
    store: Chroma
    index_seconds: float
    embedding_load_seconds: float
    framework: str = "LangChain"
    _retrieved: tuple[Document, ...] = field(default_factory=tuple, repr=False)

    def retrieve(self, question: str, top_k: int) -> tuple[Evidence, ...]:
        """Invoke LangChain's retriever once and freeze independent result copies."""
        if not 1 <= top_k <= 12:
            raise ValueError("Top-k must be between 1 and 12.")
        retriever = self.store.as_retriever(
            search_kwargs={"k": min(top_k, len(self.loaded.chunks))}
        )
        with tracing_context(enabled=False):
            documents = retriever.invoke(question)
        self._retrieved = tuple(deepcopy(documents))
        return tuple(source_evidence(document) for document in self._retrieved)

    def compress(
        self,
        documents: tuple[Evidence, ...],
        request: tuple[str, RunSettings],
        token_counter: Callable[[str], int],
    ) -> tuple[Evidence, ...]:
        """Invoke the installed langchain-trimwise runnable once for all retrieved Documents."""
        question, settings = request
        native = deepcopy(self._retrieved)
        if len(documents) != len(native) or any(
            item.id != source.id or item.page_content != source.page_content
            for item, source in zip(documents, native, strict=True)
        ):
            raise ValueError("Compression must use the current frozen retrieval.")
        for item, source in zip(documents, native, strict=True):
            source.metadata = deepcopy(item.metadata)
        runnable = create_trimwise_runnable(
            settings.budget, strategy=settings.strategy, token_counter=token_counter
        )
        with tracing_context(enabled=False):
            retained = runnable.invoke({"documents": list(native), "query": question})
        return tuple(source_evidence(document) for document in retained)

    def close(self) -> None:
        """Delete this collection while preserving all other browser sessions."""
        self.store.delete_collection()


def build_corpus(loaded: LoadedPDFs, embeddings: Embeddings | None = None) -> Corpus:
    """Build a complete replacement Chroma index, cleaning up a failed partial build."""
    started = perf_counter()
    model = embeddings if embeddings is not None else local_embeddings()
    acquired = perf_counter() - started
    store = Chroma(
        collection_name=f"trimwise_{uuid4().hex}",
        embedding_function=NativeEmbeddings(model),
        client_settings=Settings(anonymized_telemetry=False),
    )
    try:
        with tracing_context(enabled=False):
            store.add_documents(
                [native_document(item) for item in loaded.chunks],
                ids=[str(item.id) for item in loaded.chunks],
            )
    except BaseException:
        store.delete_collection()
        raise
    return Corpus(loaded, store, perf_counter() - started, acquired)
