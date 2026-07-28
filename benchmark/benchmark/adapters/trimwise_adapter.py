"""Adapter for Trimwise's four public strategies."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from benchmark.utils.tokens import count_tokens
from trimwise import Trimmer, TrimResult
from trimwise.ranking import CandidateRanking
from trimwise.segmentation import Segment
from trimwise.trimmer import _PreparedTrim, _RankingRequest, _TrimArguments

from .base import CompressionResult, TimedAdapter, release_model_resources

_FIXED_WINDOW_TOKENS = 128
_ABLATIONS = frozenset({"no_mmr", "no_evidence_cutoff", "fixed_windows"})


def _fixed_window_segments(text: str, token_encoding: str) -> list[Segment]:
    """Split source text into non-overlapping, exact token-window candidates.

    Args:
        text: Original source string.
        token_encoding: Tiktoken encoding used to define fixed windows.

    Returns:
        Source-backed segments with one fixed-size window per candidate.

    Raises:
        ValueError: If token decoding cannot be mapped losslessly to the source.
    """
    import tiktoken

    encoding = tiktoken.get_encoding(token_encoding)
    token_ids = encoding.encode(text, disallowed_special=())
    if not token_ids:
        return []
    decoded, offsets = encoding.decode_with_offsets(token_ids)
    if decoded != text:
        raise ValueError("fixed-window ablation requires lossless source token offsets")
    segments = []
    for index, start_token in enumerate(range(0, len(token_ids), _FIXED_WINDOW_TOKENS)):
        end_token = min(start_token + _FIXED_WINDOW_TOKENS, len(token_ids))
        start = offsets[start_token]
        end = offsets[end_token] if end_token < len(offsets) else len(text)
        segments.append(Segment(index, start, end, text[start:end], "window", 0, None))
    return segments


class _UnboundedCandidateRanking(CandidateRanking):
    """Disable only the query-aware evidence-pool cutoff for an ablation."""

    def adaptive_indexes(self, candidates: set[int]) -> set[int]:
        """Retain every eligible non-heading candidate.

        Args:
            candidates: Candidate indexes available to query-aware selection.

        Returns:
            A defensive copy of every supplied candidate index.
        """
        return set(candidates)


def _unbounded_ranking(ranking: CandidateRanking) -> CandidateRanking:
    """Return one ranking whose adaptive pool includes all candidates.

    Args:
        ranking: Ordinary Trimwise candidate ranking.

    Returns:
        Equivalent ranking except for the adaptive evidence-pool boundary.
    """
    return _UnboundedCandidateRanking(
        ranking.primary,
        ranking.relevance,
        ranking.similarity,
        ranking.maximum_similarity_factory,
        ranking.maximum_similarity_update,
    )


class _NoEvidenceCutoffTrimmer(Trimmer):
    """Replace only Hybrid's adaptive evidence-pool boundary for an ablation."""

    def _rank(self, request: _RankingRequest) -> CandidateRanking:
        """Build the ordinary ranking and retain its full candidate set.

        Args:
            request: Trimwise's internal ranking request.

        Returns:
            An ordinary candidate ranking with no adaptive evidence cutoff.
        """
        return _unbounded_ranking(super()._rank(request))


class _FixedWindowTrimmer(Trimmer):
    """Replace only Markdown-aware segmentation with fixed source windows."""

    def _prepare(self, arguments: _TrimArguments) -> TrimResult | _PreparedTrim:
        """Prepare normally, then substitute fixed source windows for long input.

        Args:
            arguments: Trimwise's internal call arguments.

        Returns:
            An ordinary early result or a prepared fixed-window selection request.
        """
        prepared = super()._prepare(arguments)
        if isinstance(prepared, TrimResult):
            return prepared
        return replace(
            prepared,
            segments=_fixed_window_segments(arguments.text, self.config.token_encoding),
        )


def _ablation_trimmer_class(ablation: str | None) -> type[Trimmer]:
    """Choose the benchmark-only Trimmer subclass for one ablation.

    Args:
        ablation: Optional controlled component removal.

    Returns:
        Trimmer-compatible class implementing the requested ablation.
    """
    if ablation == "no_evidence_cutoff":
        return _NoEvidenceCutoffTrimmer
    if ablation == "fixed_windows":
        return _FixedWindowTrimmer
    return Trimmer


class TrimwiseAdapter(TimedAdapter):
    """Run one Trimwise strategy using the benchmark's token counter."""

    def __init__(
        self,
        strategy: str = "lexical",
        native_semantic: bool = True,
        cache_dir: str | None = None,
        use_gpu: bool = False,
        ablation: str | None = None,
    ) -> None:
        """Configure a Trimwise strategy and optional FastEmbed runtime.

        Args:
            strategy: Public Trimwise strategy name.
            native_semantic: Retained benchmark setting for native semantic behavior.
            cache_dir: Optional FastEmbed cache location.
            use_gpu: Whether FastEmbed may use CUDA.
            ablation: Optional benchmark-only Hybrid component removal.
        """
        super().__init__()
        allowed = {"structural", "lexical", "semantic", "hybrid"}
        if strategy not in allowed:
            raise ValueError(f"strategy must be one of {sorted(allowed)}")
        if ablation not in _ABLATIONS | {None}:
            raise ValueError(f"ablation must be one of {sorted(_ABLATIONS)}")
        if ablation is not None and strategy != "hybrid":
            raise ValueError("Trimwise ablations require the hybrid strategy")
        self.strategy_name = strategy
        self.query_aware = strategy in {"lexical", "semantic", "hybrid"}
        self.native_semantic = native_semantic
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        self.ablation = ablation
        self.method_id = f"trimwise_{strategy}" + (f"_{ablation}" if ablation else "")
        self._trimmer = None
        self.model_backed = strategy in {"semantic", "hybrid"}

    def _load(self) -> Any:
        """Lazily construct and return the configured Trimwise trimmer."""
        if self._trimmer is None:
            from trimwise import TrimConfig

            fastembed_options = {}
            if self.cache_dir is not None:
                fastembed_options["cache_dir"] = self.cache_dir
            if self.use_gpu:
                fastembed_options["cuda"] = True
            config = TrimConfig(
                token_encoding="o200k_base",
                fastembed_options=fastembed_options,
                mmr_lambda=1.0 if self.ablation == "no_mmr" else 0.7,
            )
            trimmer_class = _ablation_trimmer_class(self.ablation)
            self._trimmer = trimmer_class(config)
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
                "ablation": self.ablation,
                "mmr_lambda": 1.0 if self.ablation == "no_mmr" else 0.7,
                "adaptive_evidence_cutoff": self.ablation != "no_evidence_cutoff",
                "candidate_segmentation": (
                    "fixed_128_token_windows"
                    if self.ablation == "fixed_windows"
                    else "markdown_aware"
                ),
            },
        )
