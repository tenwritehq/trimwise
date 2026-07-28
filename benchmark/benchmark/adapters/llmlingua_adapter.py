"""Adapter for the original LLMLingua compression mode."""

from __future__ import annotations

from typing import Any

from .base import CompressionResult
from .llmlingua_common import LLMLinguaBase


class LLMLinguaAdapter(LLMLinguaBase):
    """Original LLMLingua, applied only to the document/context region."""

    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        device_map: str = "cpu",
        model_config: dict[str, Any] | None = None,
        query_aware: bool = True,
    ) -> None:
        """Configure the LLMLingua model used for compression.

        Args:
            model_name: Hugging Face model identifier.
            device_map: Device placement passed to LLMLingua.
            model_config: Optional model-loading options.
            query_aware: Whether compression may use the benchmark question.
        """
        super().__init__(model_name, device_map, model_config)
        self.query_aware = query_aware
        self.method_id = "llmlingua" if query_aware else "llmlingua_queryless"

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        """Compress one context string with optional question-conditioned token pruning.

        Args:
            context: Source context to compress.
            query: Question used for ranking, or an empty string for source-only compression.
            budget: Target token budget.
            seed: Reserved benchmark seed.

        Returns:
            The compressed benchmark result.
        """
        compressor = self._load(use_llmlingua2=False)
        response = compressor.compress_prompt(
            context=context,
            instruction="",
            question=query,
            target_token=budget,
            use_context_level_filter=False,
            use_sentence_level_filter=False,
            use_token_level_filter=True,
            condition_in_question="after",
            reorder_context="original",
            rank_method="llmlingua",
            add_instruction=False,
            concate_question=False,
        )
        return self._result(self.method_id, response, model_name=self.model_name)
