"""Adapter for LongLLMLingua compression mode."""

from __future__ import annotations

from typing import Any

from .base import CompressionResult
from .llmlingua_common import LLMLinguaBase


class LongLLMLinguaAdapter(LLMLinguaBase):
    """LongLLMLingua query-conditioned context selection and token pruning."""

    method_id = "longllmlingua"
    query_aware = True

    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        device_map: str = "cpu",
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Configure the LongLLMLingua model used for compression.

        Args:
            model_name: Hugging Face model identifier.
            device_map: Device placement passed to LLMLingua.
            model_config: Optional model-loading options.
        """
        super().__init__(model_name, device_map, model_config)

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        """Compress one context string with query-conditioned long-context pruning.

        Args:
            context: Source context to compress.
            query: Question used for ranking.
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
            use_context_level_filter=True,
            use_sentence_level_filter=False,
            use_token_level_filter=True,
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua",
            add_instruction=False,
            concate_question=False,
        )
        return self._result(self.method_id, response, model_name=self.model_name)
