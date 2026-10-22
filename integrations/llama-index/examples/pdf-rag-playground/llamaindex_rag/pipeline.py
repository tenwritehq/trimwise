"""Use LlamaIndex TextNodes, VectorStoreIndex, and the real Trimwise postprocessor."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
from llama_index.postprocessor.trimwise import TrimwiseNodePostprocessor
from pydantic import PrivateAttr
from rag_playground.documents import LoadedPDFs
from rag_playground.models import Evidence, RunSettings
from rag_playground.retrieval import Embeddings, local_embeddings


class NativeEmbeddings(BaseEmbedding):
    """Expose CPU embeddings through LlamaIndex's native embedding interface."""

    _local_model: Embeddings = PrivateAttr()

    def __init__(self, model: Embeddings) -> None:
        """Bind one explicit local model, avoiding LlamaIndex's default remote embeddings."""
        super().__init__(model_name="BAAI/bge-small-en-v1.5", embed_batch_size=128)
        self._local_model = model

    def _get_query_embedding(self, query: str) -> list[float]:
        """Use the local model's query-specific preprocessing."""
        return self._local_model.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Offload local query embedding when a native caller uses async retrieval."""
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Embed one passage through the same local passage model."""
        return self._local_model.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Keep native index construction batched instead of invoking the model per node."""
        return self._local_model.embed_documents(texts)


def native_node(evidence: Evidence) -> TextNode:
    """Copy exact evidence into a native TextNode, excluding metadata from its text."""
    return TextNode(
        id_=evidence.id,
        text=evidence.page_content,
        metadata=deepcopy(evidence.metadata),
        excluded_embed_metadata_keys=list(evidence.metadata),
        excluded_llm_metadata_keys=list(evidence.metadata),
    )


def source_evidence(item: NodeWithScore) -> Evidence:
    """Preserve the native node ID, text, metadata, and retrieval score for inspection."""
    return Evidence(
        item.node.get_content(MetadataMode.NONE),
        deepcopy(item.node.metadata),
        item.node.node_id,
        item.score,
    )


@dataclass
class Corpus:
    """Own one in-memory native vector index without a global Settings override.

    Attributes:
        loaded: Exact page-backed source chunks.
        index: Native LlamaIndex vector index and its private storage context.
        index_seconds: Complete index construction time.
        embedding_load_seconds: Separate model acquisition time.
        framework: Fixed name exported with comparison results.
    """

    loaded: LoadedPDFs
    index: VectorStoreIndex
    index_seconds: float
    embedding_load_seconds: float
    framework: str = "LlamaIndex"
    _retrieved: tuple[NodeWithScore, ...] = field(default_factory=tuple, repr=False)

    def retrieve(self, question: str, top_k: int) -> tuple[Evidence, ...]:
        """Retrieve native NodeWithScore results once and copy their exact source text."""
        if not 1 <= top_k <= 12:
            raise ValueError("Top-k must be between 1 and 12.")
        retriever = self.index.as_retriever(similarity_top_k=min(top_k, len(self.loaded.chunks)))
        self._retrieved = tuple(deepcopy(retriever.retrieve(question)))
        return tuple(source_evidence(item) for item in self._retrieved)

    def compress(
        self,
        documents: tuple[Evidence, ...],
        request: tuple[str, RunSettings],
        token_counter: Callable[[str], int],
    ) -> tuple[Evidence, ...]:
        """Postprocess the whole frozen retrieval through the installed LlamaIndex adapter."""
        question, settings = request
        native = deepcopy(self._retrieved)
        if len(documents) != len(native) or any(
            item.id != source.node.node_id
            or item.page_content != source.node.get_content(MetadataMode.NONE)
            for item, source in zip(documents, native, strict=True)
        ):
            raise ValueError("Compression must use the current frozen retrieval.")
        for item, source in zip(documents, native, strict=True):
            source.node.metadata = deepcopy(item.metadata)
        processor = TrimwiseNodePostprocessor(
            limit=settings.budget, strategy=settings.strategy, token_counter=token_counter
        )
        retained = processor.postprocess_nodes(
            list(native),
            query_str=question,
        )
        return tuple(source_evidence(item) for item in retained)

    def close(self) -> None:
        """Remove this index's nodes and vectors without touching another session's index."""
        self.index.delete_nodes(
            [str(item.id) for item in self.loaded.chunks], delete_from_docstore=True
        )


def build_corpus(loaded: LoadedPDFs, embeddings: Embeddings | None = None) -> Corpus:
    """Build a private native VectorStoreIndex using explicitly supplied CPU embeddings."""
    started = perf_counter()
    model = embeddings if embeddings is not None else local_embeddings()
    acquired = perf_counter() - started
    index = VectorStoreIndex(
        nodes=[native_node(item) for item in loaded.chunks], embed_model=NativeEmbeddings(model)
    )
    return Corpus(loaded, index, perf_counter() - started, acquired)
