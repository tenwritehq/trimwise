"""Adapter for the released RECOMP extractive sentence scorer."""

from __future__ import annotations

import re
import time
from typing import Any

from benchmark.utils.tokens import count_tokens

from .base import CompressionResult, TimedAdapter, release_model_resources

RECOMP_BATCH_SIZE = 16


def split_sentences(text: str) -> list[str]:
    """Split plain or Markdown-like context into source-backed sentences."""
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?。\uFF01\uFF1F])\s+|\n{2,}", text)
        if part.strip()
    ]


def _mean_pool(token_embeddings: Any, attention_mask: Any) -> Any:
    """Mean-pool transformer token embeddings using the attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class RecompAdapter(TimedAdapter):
    """Rank complete sentences with a RECOMP extractive model."""

    model_backed = True
    query_aware = True

    def __init__(
        self,
        model_name: str = "fangyuan/nq_extractive_compressor",
        device: str = "cuda",
    ) -> None:
        """Configure the RECOMP checkpoint and target device.

        Args:
            model_name: Extractive RECOMP checkpoint identifier.
            device: PyTorch device receiving the encoder.
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.method_id = "recomp_extractive"
        self._model = None
        self._tokenizer = None

    def _load(self) -> tuple[Any, Any]:
        """Lazily load and return the tokenizer and encoder."""
        if self._model is None:
            started = time.perf_counter_ns()
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self._model_load_ms = (time.perf_counter_ns() - started) / 1_000_000
            self._model_loaded = True
        return self._tokenizer, self._model

    def close(self) -> None:
        """Drop the RECOMP encoder and release accelerator memory."""
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        release_model_resources()

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        sentences = split_sentences(context)
        if not sentences or count_tokens(context) <= budget:
            return CompressionResult(self.method_id, context)
        tokenizer, model = self._load()
        import torch

        with torch.inference_mode():
            query_inputs = tokenizer(
                [query], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            query_outputs = model(**query_inputs)
            query_vector = _mean_pool(
                query_outputs.last_hidden_state,
                query_inputs["attention_mask"],
            )[0]
            sentence_vectors = []
            for start in range(0, len(sentences), RECOMP_BATCH_SIZE):
                sentence_inputs = tokenizer(
                    sentences[start : start + RECOMP_BATCH_SIZE],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                sentence_outputs = model(**sentence_inputs)
                sentence_vectors.append(
                    _mean_pool(
                        sentence_outputs.last_hidden_state,
                        sentence_inputs["attention_mask"],
                    ).cpu()
                )
            scores = torch.cat(sentence_vectors) @ query_vector.cpu()
        ranked = sorted(range(len(sentences)), key=lambda index: float(scores[index]), reverse=True)
        selected: list[str] = []
        for index in ranked:
            candidate = "\n".join([*selected, sentences[index]])
            if count_tokens(candidate) <= budget:
                selected.append(sentences[index])
        output = "\n".join(selected)
        return CompressionResult(
            self.method_id,
            output,
            metadata={"model_name": self.model_name, "selected_sentences": len(selected)},
        )
