"""Adapter for Trimwise's four public strategies."""

from __future__ import annotations

from typing import Any

from benchmark.utils.tokens import count_tokens

from .base import CompressionResult, TimedAdapter, release_model_resources


class TrimwiseAdapter(TimedAdapter):
    """Run one Trimwise strategy using the benchmark's token counter."""

    def __init__(
        self,
        strategy: str = "lexical",
        native_semantic: bool = True,
        cache_dir: str | None = None,
        use_gpu: bool = False,
    ) -> None:
        """Configure a Trimwise strategy and optional FastEmbed runtime.

        Args:
            strategy: Public Trimwise strategy name.
            native_semantic: Retained benchmark setting for native semantic behavior.
            cache_dir: Optional FastEmbed cache location.
            use_gpu: Whether FastEmbed may use CUDA.
        """
        super().__init__()
        allowed = {"structural", "lexical", "semantic", "hybrid"}
        if strategy not in allowed:
            raise ValueError(f"strategy must be one of {sorted(allowed)}")
        self.strategy_name = strategy
        self.query_aware = strategy in {"lexical", "semantic", "hybrid"}
        self.native_semantic = native_semantic
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        self.method_id = f"trimwise_{strategy}"
        self._trimmer = None
        self.model_backed = strategy in {"semantic", "hybrid"}

    def _load(self) -> Any:
        """Lazily construct and return the configured Trimwise trimmer."""
        if self._trimmer is None:
            from trimwise import TrimConfig, Trimmer

            fastembed_options = {}
            if self.cache_dir is not None:
                fastembed_options["cache_dir"] = self.cache_dir
            if self.use_gpu:
                fastembed_options["cuda"] = True
            self._trimmer = Trimmer(
                TrimConfig(token_encoding="o200k_base", fastembed_options=fastembed_options)
            )
        return self._trimmer

    def close(self) -> None:
        """Drop the managed Trimmer so its optional embedding backend can be collected."""
        self._trimmer = None
        self._model_loaded = False
        release_model_resources()

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        from trimwise import BudgetUnit, Strategy

        trimmer = self._load()
        kwargs = {
            "unit": BudgetUnit.TOKENS,
            "strategy": Strategy(self.strategy_name),
            "token_counter": count_tokens,
        }
        if self.query_aware:
            kwargs["query"] = query
        result = trimmer.trim(context, budget, **kwargs)
        if self.model_backed:
            self._model_loaded = True
        return CompressionResult(
            self.method_id,
            result.text,
            metadata={
                "input_count": result.input_count,
                "output_count": result.output_count,
                "limit": result.limit,
                "strategy": result.strategy.value,
                "unit": result.unit.value,
            },
        )
