"""Adapter for the released Selective Context self-information compressor."""

from __future__ import annotations

import time
from typing import Any

from benchmark.utils.tokens import count_tokens

from .base import CompressionResult, TimedAdapter, release_model_resources


class SelectiveContextAdapter(TimedAdapter):
    """Compress context with Selective Context's phrase-level information filter."""

    model_backed = True
    query_aware = False

    def __init__(self, model_name: str = "gpt2", lang: str = "en") -> None:
        """Configure the Selective Context language model and language.

        Args:
            model_name: Selective Context model identifier.
            lang: Source-language code accepted by Selective Context.
        """
        super().__init__()
        self.model_name = model_name
        self.lang = lang
        self.method_id = "selective_context"
        self._compressor = None

    def _load(self) -> Any:
        """Lazily construct and return the Selective Context compressor."""
        if self._compressor is None:
            started = time.perf_counter_ns()
            from selective_context import SelectiveContext

            self._compressor = SelectiveContext(model_type=self.model_name, lang=self.lang)
            self._model_load_ms = (time.perf_counter_ns() - started) / 1_000_000
            self._model_loaded = True
        return self._compressor

    def close(self) -> None:
        """Drop Selective Context's language model and release accelerator memory."""
        self._compressor = None
        self._model_loaded = False
        release_model_resources()

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        input_tokens = count_tokens(context)
        if input_tokens <= budget:
            return CompressionResult(self.method_id, context)
        reduce_ratio = 1 - budget / input_tokens
        compressor = self._load()
        tokenizer = compressor.tokenizer
        chunk_size = max(1, compressor.max_token_length - 1)
        token_ids = tokenizer.encode(context, add_special_tokens=False)
        chunks = [
            tokenizer.decode(
                token_ids[start : start + chunk_size], clean_up_tokenization_spaces=False
            )
            for start in range(0, len(token_ids), chunk_size)
        ]
        outputs = []
        for chunk in chunks:
            _, reduced = compressor(chunk, reduce_ratio=reduce_ratio)
            outputs.append(" ".join(reduced) if isinstance(reduced, list) else reduced)
        output = "".join(outputs)
        return CompressionResult(
            self.method_id,
            output,
            metadata={"model_name": self.model_name, "reduce_ratio": reduce_ratio},
        )
