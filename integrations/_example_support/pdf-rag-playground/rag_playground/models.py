"""Define comparison measurements separately from mutable model clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    """Carry exact text and provenance between native pipelines and the shared UI.

    Attributes:
        page_content: Exact extracted or compressed evidence text.
        metadata: Source coordinates and adapter provenance.
        id: Stable chunk identity.
        score: Native retrieval score, when the framework supplies one.
    """

    page_content: str
    metadata: dict[str, Any]
    id: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class RunSettings:
    """Keep the user-selected comparison settings together.

    Attributes:
        top_k: Maximum retrieved chunks, shared by both conditions.
        budget: Shared evidence-only token ceiling for Trimwise.
        output_tokens: Maximum generated tokens for both conditions.
        strategy: Trimwise ranking strategy.
    """

    top_k: int = 6
    budget: int = 256
    output_tokens: int = 160
    strategy: str = "lexical"

    def __post_init__(self) -> None:
        """Validate control values before retrieval, compression, or model calls."""
        if not 1 <= self.top_k <= 12 or not 16 <= self.budget <= 4096:
            raise ValueError("Choose top-k 1-12 and an evidence budget of 16-4096 tokens.")
        if not 32 <= self.output_tokens <= 512:
            raise ValueError("Answer allowance must be 32-512 tokens.")
        if self.strategy not in {"lexical", "structural", "auto"}:
            raise ValueError("This playground supports lexical, structural, or automatic trimming.")


@dataclass(frozen=True)
class Generation:
    """Record a real model's answer, usage, and completion state.

    Attributes:
        text: Untrusted generated answer text.
        model: Actual model ID reported or loaded.
        prompt_tokens: Actual input usage reported by the local runtime.
        output_tokens: Actual generated token usage reported by the local runtime.
        seconds: Observed generation duration for this condition.
        status: Runtime completion reason, including output-limited answers.
        usage_kind: Provenance of the token measurements.
    """

    text: str
    model: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    status: str
    usage_kind: str


@dataclass
class Comparison:
    """Freeze one retrieval and retain both successes and failures for inspection.

    Attributes:
        framework: Native retrieval and compression framework.
        run_id: Unique local comparison identity.
        question: User question used for both conditions.
        settings: Shared comparison settings.
        corpus_fingerprint: Source/chunking identity of the indexed corpus.
        original: Frozen original retrieved chunks in retrieval order.
        compressed: Evidence returned by the selected native Trimwise integration.
        contexts: Complete rendered context for each condition, including labels.
        context_tokens: Measured rendered context token counts.
        prompt_estimates: Full prompt sizes checked before generation.
        retrieval_seconds: One shared retrieval observation.
        compression_seconds: Trimwise operation duration.
        model_load_seconds: Separate answer model acquisition/load duration.
        model_info: Nonsecret model, tokenizer, revision, and runtime information.
        generations: Actual answer results, if available.
        errors: Safe condition-specific failure descriptions.
        feedback: Optional human assessment, never an invented accuracy score.
        evaluation: Optional reference/evidence checks with explicit definitions.
    """

    framework: str
    run_id: str
    question: str
    settings: RunSettings
    corpus_fingerprint: str
    original: tuple[Evidence, ...]
    compressed: tuple[Evidence, ...]
    contexts: dict[str, str]
    context_tokens: dict[str, int]
    prompt_estimates: dict[str, int]
    retrieval_seconds: float
    compression_seconds: float
    model_load_seconds: float
    model_info: dict[str, Any]
    totals_seconds: dict[str, float] = field(default_factory=dict)
    generations: dict[str, Generation] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    feedback: str = "Unrated"
    evaluation: dict[str, Any] = field(default_factory=dict)
