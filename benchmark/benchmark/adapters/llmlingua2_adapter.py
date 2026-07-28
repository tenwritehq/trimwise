"""Context-only adapter for the released LLMLingua-2 classifier."""

from __future__ import annotations

from typing import Any

from .base import CompressionResult
from .llmlingua_common import LLMLinguaBase


class LLMLingua2Adapter(LLMLinguaBase):
    """Task-agnostic LLMLingua-2 token classifier on context only."""

    method_id = "llmlingua2"
    query_aware = False

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device_map: str = "cpu",
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Configure the released LLMLingua-2 model and execution device.

        Args:
            model_name: LLMLingua-2 checkpoint identifier.
            device_map: Transformers device-map setting.
            model_config: Optional checkpoint loading options.
        """
        super().__init__(model_name, device_map, model_config)

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        """Compress context without exposing the query to LLMLingua-2."""
        compressor = self._load(use_llmlingua2=True)
        response = compressor.compress_prompt(
            context=[context],
            target_token=budget,
            use_context_level_filter=False,
            use_token_level_filter=True,
            force_tokens=["\n", "?"],
            force_reserve_digit=True,
            chunk_end_tokens=[".", "\n"],
        )
        return self._result(self.method_id, response, model_name=self.model_name)
