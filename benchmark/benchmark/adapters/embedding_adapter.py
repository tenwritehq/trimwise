"""FastEmbed ranking baseline with exact output-budget accounting."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from benchmark.utils.tokens import count_tokens, token_prefix

from .base import CompressionResult, TimedAdapter, release_model_resources
from .bm25_adapter import _chunks


class EmbeddingMMRAdapter(TimedAdapter):
    """Select query-relevant chunks with embedding MMR."""

    query_aware = True
    model_backed = True

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 128,
        overlap: int = 32,
        mmr_lambda: float = 0.7,
        source_order: bool = True,
        cache_dir: str | None = None,
        use_gpu: bool = False,
    ) -> None:
        """Configure the embedding model, chunking, and MMR selection.

        Args:
            model_name: FastEmbed model identifier.
            chunk_size: Token count in each candidate window.
            overlap: Shared token count between consecutive windows.
            mmr_lambda: Relevance weight in maximal marginal relevance.
            source_order: Whether to restore selected windows to input order.
            cache_dir: Optional model-cache location.
            use_gpu: Whether FastEmbed may use CUDA.
        """
        super().__init__()
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.mmr_lambda = mmr_lambda
        self.source_order = source_order
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        self.method_id = (
            "embedding_mmr_source_order" if source_order else "embedding_mmr_rank_order"
        )
        self._model = None

    def _load(self) -> Any:
        """Lazily construct and return the configured embedding backend."""
        if self._model is None:
            started = time.perf_counter_ns()
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                cuda=self.use_gpu,
            )
            self._model_load_ms = (time.perf_counter_ns() - started) / 1_000_000
            self._model_loaded = True
        return self._model

    def close(self) -> None:
        """Drop the embedding model and release cached accelerator memory."""
        self._model = None
        self._model_loaded = False
        release_model_resources()

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        chunks = _chunks(context, self.chunk_size, self.overlap)
        if not chunks:
            return CompressionResult(self.method_id, "")
        model = self._load()
        passages = [chunk.text for chunk in chunks]
        vectors = np.asarray(list(model.embed([query, *passages])), dtype=np.float32)
        vectors = self._normalize(vectors)
        q = vectors[0]
        passages = vectors[1:]
        relevance = passages @ q
        remaining = set(range(len(chunks)))
        selected: list[int] = []
        max_similarity = np.zeros(len(chunks), dtype=np.float32)
        marker = "\n[…omitted…]\n"
        while remaining:
            chosen = max(
                remaining,
                key=lambda i: (
                    self.mmr_lambda * float(relevance[i])
                    - (1 - self.mmr_lambda) * float(max_similarity[i]),
                    -i,
                ),
            )
            candidate_order = [*selected, chosen]
            output_order = sorted(candidate_order) if self.source_order else candidate_order
            candidate = marker.join(chunks[i].text for i in output_order)
            if count_tokens(candidate) > budget:
                break
            selected.append(chosen)
            remaining.remove(chosen)
            if remaining:
                max_similarity = np.maximum(max_similarity, passages @ passages[chosen])
        output_order = sorted(selected) if self.source_order else selected
        output = marker.join(chunks[i].text for i in output_order)
        if not output and chunks:
            output = token_prefix(chunks[int(np.argmax(relevance))].text, budget)
        return CompressionResult(
            self.method_id,
            output,
            metadata={"selected": selected, "model_name": self.model_name},
        )
