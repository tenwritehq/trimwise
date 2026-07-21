"""Verify Trimwise's public contract, validation, and budget guarantees."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

import pytest

import trimwise
from trimwise import BudgetUnit, SourceSpan, Strategy, TrimConfig, Trimmer


def test_public_exports_are_intentionally_small() -> None:
    """Expose only the seven documented public objects."""
    assert trimwise.__all__ == [
        "BudgetUnit",
        "SemanticBackendError",
        "SourceSpan",
        "Strategy",
        "TrimConfig",
        "TrimResult",
        "Trimmer",
    ]


def test_default_config_is_immutable() -> None:
    """Defensively freeze FastEmbed options and all config fields."""
    config = TrimConfig(fastembed_options={"threads": 2})
    assert isinstance(config.fastembed_options, MappingProxyType)
    with pytest.raises(TypeError):
        config.fastembed_options["threads"] = 3  # type: ignore[index]


@pytest.mark.parametrize("field", ["embedding_callback", "async_embedding_callback"])
def test_trimmer_rejects_noncallable_embedding_callbacks(field: str) -> None:
    """Reject invalid callback dependencies during construction.

    Args:
        field: Constructor callback field under test.
    """
    with pytest.raises(TypeError, match=f"{field} must be callable"):
        Trimmer(**{field: 1})  # type: ignore[arg-type]


def test_trimmer_rejects_two_embedding_callbacks() -> None:
    """Require callers to choose one semantic callback execution model."""

    def embed(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Return one synchronous embedding batch."""
        return [1.0], [[1.0]]

    async def aembed(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Return one asynchronous embedding batch."""
        return [1.0], [[1.0]]

    with pytest.raises(ValueError, match="only one embedding callback"):
        Trimmer(embedding_callback=embed, async_embedding_callback=aembed)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("token_encoding", " ", ValueError),
        ("token_encoding", 1, TypeError),
        ("embedding_model", "", ValueError),
        ("embedding_model", None, TypeError),
        ("omission_marker", "\n", ValueError),
        ("omission_marker", 1, TypeError),
        ("embedding_batch_size", 0, ValueError),
        ("embedding_batch_size", True, ValueError),
        ("embedding_batch_size", "2", ValueError),
        ("mmr_lambda", -0.1, ValueError),
        ("mmr_lambda", 1.1, ValueError),
        ("mmr_lambda", True, ValueError),
        ("mmr_lambda", "0.7", ValueError),
        ("fastembed_options", [], TypeError),
        ("fastembed_options", {1: "bad"}, ValueError),
        ("fastembed_options", {"model_name": "bad"}, ValueError),
    ],
)
def test_config_rejects_invalid_values(
    field: str,
    value: Any,
    error_type: type[Exception],
) -> None:
    """Reject every invalid configuration boundary.

    Args:
        field: Config field under test.
        value: Invalid field value.
        error_type: Expected validation exception.
    """
    with pytest.raises(error_type):
        TrimConfig(**{field: value})


def test_short_input_is_returned_exactly() -> None:
    """Avoid parsing or rewriting text that already fits."""
    source = "  # Heading\r\n\r\nExact text.  "
    result = Trimmer().trim(source, len(source), unit="characters")
    assert result.text == source
    assert result.input_count == result.output_count == len(source)
    assert result.strategy is Strategy.STRUCTURAL
    assert result.trimmed is False
    assert result.spans == (SourceSpan(0, len(source)),)


def test_auto_with_query_resolves_to_lexical() -> None:
    """Choose lexical retrieval without loading semantic dependencies."""
    result = Trimmer().trim("short", 10, unit="characters", query="task")
    assert result.strategy is Strategy.LEXICAL


def test_zero_limit_returns_empty_measured_result() -> None:
    """Return an empty output while still reporting the original size."""
    result = Trimmer().trim("abc", 0, unit=BudgetUnit.CHARACTERS)
    assert result.text == ""
    assert result.input_count == 3
    assert result.output_count == 0
    assert result.trimmed is True
    assert result.spans == ()


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"limit": -1}, ValueError),
        ({"limit": True}, TypeError),
        ({"limit": 1.5}, TypeError),
        ({"text": 4}, TypeError),
        ({"unit": "bytes"}, ValueError),
        ({"strategy": "first"}, ValueError),
        ({"query": 4}, TypeError),
        ({"token_counter": 4}, TypeError),
    ],
)
def test_public_call_rejects_invalid_arguments(
    kwargs: dict[str, Any],
    error_type: type[Exception],
) -> None:
    """Validate public trust-boundary types and values.

    Args:
        kwargs: Invalid argument override.
        error_type: Expected exception type.
    """
    arguments: dict[str, Any] = {"text": "abc", "limit": 2, "unit": "characters"}
    arguments.update(kwargs)
    with pytest.raises(error_type):
        Trimmer().trim(**arguments)


@pytest.mark.parametrize("strategy", ["lexical", "semantic", "hybrid"])
def test_query_aware_strategies_require_a_query(strategy: str) -> None:
    """Reject missing and whitespace-only retrieval queries.

    Args:
        strategy: Query-aware strategy under test.
    """
    with pytest.raises(ValueError, match="requires a nonblank query"):
        Trimmer().trim("long input", 2, unit="characters", strategy=strategy, query="  ")


def test_structural_strategy_ignores_query() -> None:
    """Keep explicit structural behavior queryless even when query text is supplied."""
    result = Trimmer().trim("short", 10, unit="characters", strategy="structural", query="x")
    assert result.strategy is Strategy.STRUCTURAL


def test_custom_counter_controls_token_measurement() -> None:
    """Use the callback for both input and final output counts."""

    def counter(text: str) -> int:
        """Count source characters as model tokens."""
        return len(text)

    result = Trimmer().trim("abcdef", 3, token_counter=counter)
    assert result.input_count == 6
    assert result.output_count <= 3


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "1"])
def test_custom_counter_output_is_validated(invalid: object) -> None:
    """Reject negative, boolean, and noninteger callback results.

    Args:
        invalid: Unsupported callback return value.
    """

    def counter(_: str) -> Any:
        """Return the configured invalid value."""
        return invalid

    with pytest.raises(ValueError, match="nonnegative integer"):
        Trimmer().trim("abc", 2, token_counter=counter)


def test_custom_counter_is_token_only() -> None:
    """Reject a token callback for word and character budgets."""
    with pytest.raises(ValueError, match="only valid for token"):
        Trimmer().trim("abc", 2, unit="words", token_counter=len)


def test_word_budget_counts_whitespace_delimited_words() -> None:
    """Measure words with the documented whitespace rule."""
    result = Trimmer().trim("one  two\nthree", 2, unit="words")
    assert result.output_count <= 2
    assert result.input_count == 3


def test_character_budget_counts_unicode_code_points() -> None:
    """Count multilingual text as Python code points rather than bytes."""
    source = "你好世界"
    result = Trimmer().trim(source, 3, unit="characters")
    assert result.output_count <= 3
    assert len(result.text) <= 3
    assert result.spans == (SourceSpan(0, 3),)
    assert source[result.spans[0].start : result.spans[0].end] == result.text


def test_token_budget_accepts_special_token_looking_text() -> None:
    """Treat special-token-looking input as ordinary source text."""
    result = Trimmer().trim("hello <|endoftext|> world", 100)
    assert result.text == "hello <|endoftext|> world"


def test_lexical_trim_retains_query_evidence_and_source_order() -> None:
    """Prefer exact query evidence while emitting retained blocks in source order."""
    source = (
        "# Start\n\nRoutine introduction with repeated filler words.\n\n"
        "# Launch\n\nProject Zephyr launched on 2026-07-14.\n\n"
        "# Owner\n\nThe owner is Aakash and the escalation URL is https://example.com/ops.\n"
    )
    result = Trimmer().trim(
        source,
        100,
        unit="characters",
        strategy="lexical",
        query="When did Project Zephyr launch?",
    )
    assert "# Launch" in result.text
    assert "2026-07-14" in result.text
    assert result.output_count <= 100


def test_query_aware_trim_stops_before_filling_the_limit() -> None:
    """Leave capacity unused after the adaptively relevant candidate pool is exhausted."""
    paragraphs = ["Routine gardening notes with no project details."] * 20
    paragraphs[10] = "Project Zephyr launched on July 14, 2026."
    paragraphs[-1] = "Unrelated closing instructions for the cafeteria."
    source = "\n\n".join(paragraphs)
    result = Trimmer().trim(
        source,
        len(source) - 1,
        unit="characters",
        strategy="lexical",
        query="When did Project Zephyr launch?",
    )
    assert (
        "Project Zephyr launched on July 14, 2026." in result.text,
        result.output_count < result.limit,
        "Unrelated closing instructions" not in result.text,
    ) == (True, True, True)


def test_scoring_context_is_not_emitted() -> None:
    """Retain each exact source fragment once despite richer ranking text."""
    source = (
        "# Installation\n\nRun this command.\n\nUse Orion afterward.\n\n"
        "# Removal\n\nDelete every generated file.\n"
    )
    result = Trimmer(TrimConfig(omission_marker="<cut>")).trim(
        source,
        45,
        unit="characters",
        strategy="lexical",
        query="installation command",
    )
    assert result.text.count("# Installation") == 1
    assert result.text.count("Run this command.") == 1
    assert result.output_count <= 45


def test_custom_marker_is_used_when_it_fits() -> None:
    """Insert the configured marker for an affordable omitted gap."""
    source = "FIRST important.\n\nMiddle filler that is not useful.\n\nLAST important."
    result = Trimmer(TrimConfig(omission_marker="<cut>")).trim(
        source,
        42,
        unit="characters",
        strategy="structural",
    )
    assert "<cut>" in result.text
    assert result.output_count <= 42
    assert result.spans == (
        SourceSpan(0, len("FIRST important.\n")),
        SourceSpan(source.index("LAST"), len(source)),
    )


def test_adjacent_retained_segments_merge_across_copied_whitespace() -> None:
    """Return one maximal span for contiguous source-backed output."""
    source = "# Ignore\n\nnoise filler words.\n\n# Target\n\ntarget detail here.\n\ntrailing noise."
    result = Trimmer(TrimConfig(omission_marker="<cut>")).trim(
        source,
        30,
        unit="characters",
        strategy="lexical",
        query="target detail",
    )
    start = source.index("# Target")
    assert result.spans == (SourceSpan(start, start + len(result.text)),)
    assert source[result.spans[0].start : result.spans[0].end] == result.text


def test_content_wins_when_marker_cannot_fit() -> None:
    """Omit an oversized marker rather than discarding measurable content."""
    config = TrimConfig(omission_marker="marker-too-large")
    result = Trimmer(config).trim("abcdef", 2, unit="characters")
    assert result.text
    assert "marker" not in result.text
    assert result.output_count <= 2
