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

    def _compose_selected(self, selected: dict[int, str]) -> str:
        """Compose selected windows in the configured output order.

        Args:
            selected: Candidate-window text keyed by source index.

        Returns:
            Source-ordered or rank-ordered windows separated by omission markers.
        """
        marker = "\n[…omitted…]\n"
        output_order = sorted(selected) if self.source_order else list(selected)
        return marker.join(selected[index] for index in output_order)

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
        selected: dict[int, str] = {}
        max_similarity = np.zeros(len(chunks), dtype=np.float32)
        while remaining:
            chosen = max(
                remaining,
                key=lambda i: (
                    self.mmr_lambda * float(relevance[i])
                    - (1 - self.mmr_lambda) * float(max_similarity[i]),
                    -i,
                ),
            )
            candidate = self._compose_selected({**selected, chosen: chunks[chosen].text})
            if count_tokens(candidate) > budget:
                remaining_tokens = budget - count_tokens(self._compose_selected(selected))
                if selected:
                    remaining_tokens -= count_tokens("\n[…omitted…]\n")
                prefix = token_prefix(chunks[chosen].text, remaining_tokens)
                if prefix:
                    selected[chosen] = prefix
                break
            selected[chosen] = chunks[chosen].text
            remaining.remove(chosen)
            if remaining:
                max_similarity = np.maximum(max_similarity, passages @ passages[chosen])
        output = self._compose_selected(selected)
        if not output and chunks:
            output = token_prefix(chunks[int(np.argmax(relevance))].text, budget)
        return CompressionResult(
            self.method_id,
            output,
            metadata={"selected": list(selected), "model_name": self.model_name},
        )


class EmbeddingTopKAdapter(EmbeddingMMRAdapter):
    """Select fixed embedding-ranked chunks without a diversity penalty."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 128,
        overlap: int = 32,
        source_order: bool = True,
        cache_dir: str | None = None,
        use_gpu: bool = False,
    ) -> None:
        """Configure fixed-window embedding retrieval without MMR.

        Args:
            model_name: FastEmbed model identifier.
            chunk_size: Token count in each candidate window.
            overlap: Shared token count between consecutive windows.
            source_order: Whether to restore selected windows to input order.
            cache_dir: Optional model-cache location.
            use_gpu: Whether FastEmbed may use CUDA.
        """
        super().__init__(
            model_name=model_name,
            chunk_size=chunk_size,
            overlap=overlap,
            mmr_lambda=1.0,
            source_order=source_order,
            cache_dir=cache_dir,
            use_gpu=use_gpu,
        )
        suffix = "source_order" if source_order else "rank_order"
        self.method_id = f"embedding_topk_{suffix}"
