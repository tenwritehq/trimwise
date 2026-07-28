"""Shared lifecycle handling for the LLMLingua family."""

from __future__ import annotations

import time
from typing import Any

from .base import CompressionResult, TimedAdapter, release_model_resources


class LLMLinguaBase(TimedAdapter):
    """Load one LLMLingua-family compressor and release it between methods."""

    learned = True
    model_backed = True

    def __init__(
        self,
        model_name: str,
        device_map: str = "cpu",
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Configure one reusable LLMLingua prompt compressor.

        Args:
            model_name: Hugging Face model identifier.
            device_map: Device placement requested by the benchmark profile.
            model_config: Optional model-loading settings.
        """
        super().__init__()
        self.model_name = model_name
        self.device_map = device_map
        self.model_config = dict(model_config or {})
        self._compressor = None

    def _load(self, *, use_llmlingua2: bool = False) -> Any:
        """Load the configured compressor and preserve exact code-token spacing.

        Args:
            use_llmlingua2: Whether to initialize LLMLingua-2 mode.

        Returns:
            The initialized LLMLingua prompt compressor.
        """
        if self._compressor is None:
            started = time.perf_counter_ns()
            from llmlingua import PromptCompressor

            self._compressor = PromptCompressor(
                model_name=self.model_name,
                device_map=self.device_map,
                model_config=self.model_config,
                use_llmlingua2=use_llmlingua2,
            )
            self._compressor.tokenizer.clean_up_tokenization_spaces = False
            self._model_load_ms = (time.perf_counter_ns() - started) / 1_000_000
            self._model_loaded = True
        return self._compressor

    def close(self) -> None:
        """Drop the compressor and release CUDA allocations before the next method."""
        self._compressor = None
        self._model_loaded = False
        release_model_resources()

    @staticmethod
    def _result(method_id: str, response: dict[str, Any], **metadata: Any) -> CompressionResult:
        """Convert one LLMLingua response into the benchmark result shape.

        Args:
            method_id: Benchmark method identifier.
            response: LLMLingua response mapping.
            **metadata: Adapter metadata included in the result row.

        Returns:
            A normalized compression result.
        """
        return CompressionResult(
            method_id,
            str(response.get("compressed_prompt", "")),
            metadata={
                "origin_tokens_reported": response.get("origin_tokens"),
                "compressed_tokens_reported": response.get("compressed_tokens"),
                "ratio_reported": response.get("ratio"),
                **metadata,
            },
        )
