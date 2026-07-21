"""Define Trimwise's public value objects and validation rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class Strategy(str, Enum):
    """Select the evidence-ranking method used during truncation."""

    AUTO = "auto"
    STRUCTURAL = "structural"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class BudgetUnit(str, Enum):
    """Define how an input and output budget is measured."""

    TOKENS = "tokens"
    WORDS = "words"
    CHARACTERS = "characters"


class SemanticBackendError(RuntimeError):
    """Report an optional semantic backend failure without hiding its cause."""


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Identify a retained range in the original input string.

    Attributes:
        start: Inclusive Python-string offset.
        end: Exclusive Python-string offset.
    """

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TrimConfig:
    """Configure reusable measurement, ranking, and omission behavior.

    Attributes:
        token_encoding: Tiktoken encoding used for token budgets and lexical ranking.
        embedding_model: FastEmbed model used by semantic strategies.
        fastembed_options: Additional keyword arguments for ``TextEmbedding``.
        embedding_batch_size: Passage batch size used during inference.
        mmr_lambda: Balance between relevance and diversity.
        omission_marker: Text inserted where source content was omitted.
    """

    token_encoding: str = "o200k_base"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fastembed_options: Mapping[str, Any] = field(default_factory=dict)
    embedding_batch_size: int = 256
    mmr_lambda: float = 0.7
    omission_marker: str = "[…omitted…]"

    def __post_init__(self) -> None:
        """Validate configuration and freeze a defensive options copy.

        Raises:
            TypeError: If FastEmbed options are not a mapping.
            ValueError: If a configured value is outside its supported domain.
        """
        if not isinstance(self.token_encoding, str):
            raise TypeError("token_encoding must be a string")
        if not self.token_encoding.strip():
            raise ValueError("token_encoding must not be blank")
        if not isinstance(self.embedding_model, str):
            raise TypeError("embedding_model must be a string")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be blank")
        if not isinstance(self.omission_marker, str):
            raise TypeError("omission_marker must be a string")
        if not self.omission_marker.strip():
            raise ValueError("omission_marker must not be blank")
        if (
            isinstance(self.embedding_batch_size, bool)
            or not isinstance(self.embedding_batch_size, int)
            or self.embedding_batch_size <= 0
        ):
            raise ValueError("embedding_batch_size must be a positive integer")
        if (
            isinstance(self.mmr_lambda, bool)
            or not isinstance(self.mmr_lambda, (int, float))
            or not 0 <= self.mmr_lambda <= 1
        ):
            raise ValueError("mmr_lambda must be between 0 and 1")
        if not isinstance(self.fastembed_options, Mapping):
            raise TypeError("fastembed_options must be a mapping")

        options = dict(self.fastembed_options)
        if any(not isinstance(key, str) for key in options):
            raise ValueError("fastembed_options keys must be strings")
        if "model_name" in options:
            raise ValueError("model_name belongs in embedding_model")
        object.__setattr__(self, "fastembed_options", MappingProxyType(options))


@dataclass(frozen=True, slots=True)
class TrimResult:
    """Describe the resolved strategy and measured truncation result.

    Attributes:
        text: Extractive output that fits the requested limit.
        input_count: Measured size of the original input.
        output_count: Measured size of the returned output.
        limit: Requested maximum output size.
        unit: Unit used for all counts.
        strategy: Concrete strategy used after resolving ``auto``.
        trimmed: Whether the output differs from the input.
        spans: Ordered, nonoverlapping ranges retained from the original input.
    """

    text: str
    input_count: int
    output_count: int
    limit: int
    unit: BudgetUnit
    strategy: Strategy
    trimmed: bool
    spans: tuple[SourceSpan, ...] = ()
