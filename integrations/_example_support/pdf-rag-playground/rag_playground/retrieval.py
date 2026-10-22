"""Share CPU embeddings and define the boundary owned by each native framework."""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import lru_cache
from typing import Protocol, cast

from fastembed import TextEmbedding

from rag_playground.documents import LoadedPDFs
from rag_playground.models import Evidence, RunSettings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbeddings:
    """Serialize CPU-only embedding work shared by either native framework."""

    def __init__(self) -> None:
        """Load public model weights locally, with no inference service or remote code."""
        self.model = TextEmbedding(EMBEDDING_MODEL, threads=2, cuda=False)
        self.lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a local passage batch.

        Args:
            texts: Extracted PDF evidence.

        Returns:
            One dense vector per passage in input order.
        """
        with self.lock:
            return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Use the embedding model's query-specific preprocessing.

        Args:
            text: User's retrieval question.

        Returns:
            Dense query vector.
        """
        with self.lock:
            return cast(list[float], next(iter(self.model.query_embed(text))).tolist())


@lru_cache(maxsize=1)
def local_embeddings() -> LocalEmbeddings:
    """Reuse only model weights, never user documents or retrieval results.

    Returns:
        Process-local CPU embedding adapter.
    """
    return LocalEmbeddings()


class Embeddings(Protocol):
    """Accept a local embedding model or an explicitly supplied test double."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a passage batch into equally dimensioned vectors."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the model's query preprocessing."""
        ...


class Corpus(Protocol):
    """Keep retrieval and Trimwise adaptation inside the chosen framework.

    Attributes:
        framework: Name of the native framework used by this corpus.
        loaded: Exact original source chunks and corpus identity.
        index_seconds: Wall time spent constructing the index.
        embedding_load_seconds: Time spent acquiring the local embedding model.
    """

    framework: str
    loaded: LoadedPDFs
    index_seconds: float
    embedding_load_seconds: float

    def retrieve(self, question: str, top_k: int) -> tuple[Evidence, ...]:
        """Return independently copied evidence from one native retrieval."""
        ...

    def compress(
        self,
        documents: tuple[Evidence, ...],
        request: tuple[str, RunSettings],
        token_counter: Callable[[str], int],
    ) -> tuple[Evidence, ...]:
        """Apply the framework's Trimwise integration under one shared evidence budget."""
        ...

    def close(self) -> None:
        """Release only this corpus's native index resources."""
        ...


CorpusBuilder = Callable[[LoadedPDFs, Embeddings | None], Corpus]
