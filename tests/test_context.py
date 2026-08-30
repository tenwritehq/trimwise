"""Verify shared-budget multi-source validation, selection, and composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from trimwise import (
    BudgetUnit,
    ContextSourceResult,
    ContextTrimResult,
    SourceSpan,
    Strategy,
    TrimConfig,
    TrimInput,
    Trimmer,
)
from trimwise.ranking import CandidateRanking


def test_context_result_types_are_frozen_and_slotted() -> None:
    """Keep public context values immutable and free of instance dictionaries."""
    result = Trimmer().trim_context(["abc"], 3, unit="characters")
    assert isinstance(result, ContextTrimResult)
    assert isinstance(result.sources[0], ContextSourceResult)
    assert not hasattr(result, "__dict__")
    assert not hasattr(result.sources[0], "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.limit = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.sources[0].text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "sources",
    [
        "bare source",
        cast(Sequence[str], iter(["arbitrary iterable"])),
        cast(Sequence[str], object()),
    ],
)
def test_context_requires_an_explicit_nonstring_sequence(sources: Sequence[str]) -> None:
    """Reject ambiguous collections instead of consuming arbitrary iterables.

    Args:
        sources: Invalid public source collection.
    """
    with pytest.raises(TypeError, match="sources must be a sequence"):
        Trimmer().trim_context(sources, 1)


def test_context_rejects_nonstring_source_elements() -> None:
    """Reject a source collection whose positions cannot map to strings."""
    with pytest.raises(TypeError, match="only strings"):
        Trimmer().trim_context(["valid", cast(str, 3)], 1)


@pytest.mark.parametrize("method_name", ["trim_context", "atrim_context"])
@pytest.mark.asyncio
async def test_context_rejects_nonboolean_deduplication(method_name: str) -> None:
    """Reject truthy values that could accidentally enable deduplication.

    Args:
        method_name: Synchronous or asynchronous public context method.
    """
    method = getattr(Trimmer(), method_name)
    if method_name == "atrim_context":
        with pytest.raises(TypeError, match="deduplicate must be a bool"):
            await method([], 0, deduplicate=cast(bool, 1))
    else:
        with pytest.raises(TypeError, match="deduplicate must be a bool"):
            method([], 0, deduplicate=cast(bool, 1))


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"limit": -1}, ValueError),
        ({"limit": True}, TypeError),
        ({"limit": 1.5}, TypeError),
        ({"unit": "bytes"}, ValueError),
        ({"strategy": "first"}, ValueError),
        ({"query": 4}, TypeError),
        ({"token_counter": 4}, TypeError),
        ({"unit": "words", "token_counter": len}, ValueError),
    ],
)
def test_context_reuses_public_argument_validation(
    kwargs: dict[str, Any],
    error_type: type[Exception],
) -> None:
    """Reject invalid arguments before taking an empty or fitting fast path.

    Args:
        kwargs: Invalid public argument override.
        error_type: Expected validation exception.
    """
    arguments: dict[str, Any] = {"sources": [], "limit": 1}
    arguments.update(kwargs)
    with pytest.raises(error_type):
        Trimmer().trim_context(**arguments)


@pytest.mark.parametrize("strategy", ["lexical", "semantic", "hybrid"])
def test_context_query_aware_strategies_require_a_query(strategy: str) -> None:
    """Reject query-aware intent with no usable operation-wide query.

    Args:
        strategy: Query-aware strategy under test.
    """
    with pytest.raises(ValueError, match="requires a nonblank query"):
        Trimmer().trim_context([], 0, strategy=strategy, query="  ")


def test_empty_context_returns_validated_empty_result() -> None:
    """Represent an empty source collection without inventing output rows."""
    result = Trimmer().trim_context([], 7, unit="words", query="find this")
    assert result.sources == ()
    assert result.input_count == result.output_count == 0
    assert result.limit == 7
    assert result.unit is BudgetUnit.WORDS
    assert result.strategy is Strategy.LEXICAL
    assert result.trimmed is False


def test_context_fitting_fast_path_preserves_every_source_position() -> None:
    """Return aggregate-fitting sources exactly with full local spans."""
    sources = ["alpha", "", "beta"]
    result = Trimmer().trim_context(sources, 9, unit="characters")
    assert [source.source_index for source in result.sources] == [0, 1, 2]
    assert [source.text for source in result.sources] == sources
    assert [source.spans for source in result.sources] == [
        (SourceSpan(0, 5),),
        (),
        (SourceSpan(0, 4),),
    ]
    assert result.input_count == result.output_count == 9
    assert result.trimmed is False


def test_all_empty_sources_keep_input_aligned_zero_rows() -> None:
    """Preserve every empty source without inventing counts or spans."""
    result = Trimmer().trim_context(["", "", ""], 1, unit="characters")
    assert [source.source_index for source in result.sources] == [0, 1, 2]
    assert [source.text for source in result.sources] == ["", "", ""]
    assert [source.output_count for source in result.sources] == [0, 0, 0]
    assert [source.spans for source in result.sources] == [(), (), ()]
    assert result.input_count == result.output_count == 0
    assert result.trimmed is False


def test_zero_limit_measures_inputs_and_preserves_empty_rows() -> None:
    """Return one empty output per source without invoking selection."""
    result = Trimmer().trim_context(["alpha", "", "beta"], 0, unit="characters")
    assert [source.text for source in result.sources] == ["", "", ""]
    assert [source.output_count for source in result.sources] == [0, 0, 0]
    assert [source.trimmed for source in result.sources] == [True, False, True]
    assert result.input_count == 9
    assert result.output_count == 0
    assert result.trimmed is True


@pytest.mark.parametrize(
    ("unit", "limit"),
    [("characters", 10), ("words", 3), ("tokens", 3)],
)
def test_context_aggregate_counts_equal_source_count_sums(unit: str, limit: int) -> None:
    """Measure each source independently and enforce the summed ceiling.

    Args:
        unit: Character or word measurement mode.
        limit: Shared aggregate limit.
    """
    result = Trimmer().trim_context(
        ["alpha beta gamma", "delta epsilon zeta"],
        limit,
        unit=unit,
    )
    assert result.input_count == sum(source.input_count for source in result.sources)
    assert result.output_count == sum(source.output_count for source in result.sources)
    assert result.output_count <= result.limit


def test_context_custom_counter_measures_sources_separately() -> None:
    """Use one caller counter for independent source and output measurements."""

    def count_words(text: str) -> int:
        """Count whitespace-delimited words for a token-budget test.

        Args:
            text: One independently measured string.

        Returns:
            Number of whitespace-delimited words.
        """
        return len(text.split())

    result = Trimmer().trim_context(
        ["one two", "three four"],
        3,
        token_counter=count_words,
    )
    assert result.input_count == 4
    assert result.output_count == sum(source.output_count for source in result.sources)
    assert result.output_count <= 3


def test_individually_fitting_sources_still_share_one_limit() -> None:
    """Trim a collection whose individual members fit but whose sum does not."""
    result = Trimmer().trim_context(["alpha", "beta"], 5, unit="characters")
    assert [source.text for source in result.sources] == ["alpha", ""]
    assert result.output_count == 5
    assert result.sources[1].trimmed is True


def test_later_lexical_evidence_displaces_earlier_filler() -> None:
    """Let relevant evidence win regardless of its source position."""
    result = Trimmer().trim_context(
        ["early filler\n\nmore filler", "target answer\n\nnoise"],
        14,
        unit="characters",
        strategy="lexical",
        query="target",
    )
    assert result.sources[0].text == ""
    assert result.sources[1].text == "target answer\n"


def test_queryless_source_round_is_deterministic_and_then_fills_globally() -> None:
    """Give sources an ordered opportunity before spending remaining space."""
    sources = ["A1.\n\nA2.", "B1.\n\nB2."]
    first = Trimmer().trim_context(sources, 12, unit="characters", strategy="structural")
    second = Trimmer().trim_context(sources, 12, unit="characters", strategy="structural")
    assert first == second
    assert all(source.text for source in first.sources)
    assert first.output_count <= 12
    joined = "".join(source.text for source in first.sources)
    assert sum(label in joined for label in ("A1", "A2", "B1", "B2")) == 3


def test_queryless_source_round_tries_a_lower_ranked_fitting_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Try a source's smaller complete evidence after its strongest unit is too large.

    Args:
        monkeypatch: Pytest attribute replacement helper.
    """

    def rank_candidates(*_: object) -> CandidateRanking:
        """Rank the oversized candidate ahead of a smaller candidate.

        Args:
            *_: Ignored structural ranking inputs.

        Returns:
            Fixed source-order scores with inert similarity.
        """
        return CandidateRanking((3.0, 2.0, 1.0), (3.0, 2.0, 1.0), lambda _a, _b: 0.0)

    monkeypatch.setattr("trimwise.trimmer.rank_structural", rank_candidates)
    result = Trimmer().trim_context(
        ["a", "long long\n\nb"],
        2,
        unit="characters",
        strategy="structural",
    )
    assert [source.text for source in result.sources] == ["a", "b"]


def test_queryless_source_round_attaches_only_an_affordable_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a source heading with its body without displacing another fitting body.

    Args:
        monkeypatch: Pytest attribute replacement helper.
    """

    def rank_candidates(*_: object) -> CandidateRanking:
        """Prefer each body over its heading with deterministic source order.

        Args:
            *_: Ignored structural ranking inputs.

        Returns:
            Fixed body-first relevance with inert similarity.
        """
        scores = (0.0, 1.0, 0.0, 0.9)
        return CandidateRanking(scores, scores, lambda _a, _b: 0.0)

    monkeypatch.setattr("trimwise.trimmer.rank_structural", rank_candidates)
    result = Trimmer().trim_context(
        ["# A\n\nbody a", "# B\n\nbody b"],
        17,
        unit="characters",
        strategy="structural",
    )
    assert result.sources[0].text == "# A\n\nbody a"
    assert result.sources[1].text == "body b"


def test_tiny_queryless_budget_does_not_guarantee_every_source() -> None:
    """Leave later sources empty when the first fitting contribution spends the budget."""
    result = Trimmer().trim_context(["alpha", "beta", "gamma"], 5, unit="characters")
    assert [source.text for source in result.sources] == ["alpha", "", ""]


@pytest.mark.parametrize(
    ("strategy", "query"),
    [("structural", None), ("lexical", "target")],
)
def test_one_active_source_matches_ordinary_trim(strategy: str, query: str | None) -> None:
    """Route one candidate-bearing source through established ordinary behavior.

    Args:
        strategy: Structural or lexical selection policy.
        query: Optional query for lexical ranking.
    """
    source = "first fact.\n\ntarget fact.\n\nlast fact."
    trimmer = Trimmer()
    ordinary = trimmer.trim(
        source,
        18,
        unit="characters",
        strategy=strategy,
        query=query,
    )
    context = trimmer.trim_context(
        ["", source, ""],
        18,
        unit="characters",
        strategy=strategy,
        query=query,
    )
    active = context.sources[1]
    assert (active.text, active.input_count, active.output_count, active.trimmed, active.spans) == (
        ordinary.text,
        ordinary.input_count,
        ordinary.output_count,
        ordinary.trimmed,
        ordinary.spans,
    )


def test_context_spans_are_local_ordered_and_source_backed() -> None:
    """Keep every retained range inside its own original source string."""
    sources = ["alpha one\n\nomega one", "alpha two\n\nomega two"]
    result = Trimmer(TrimConfig(omission_marker="...")).trim_context(
        sources,
        20,
        unit="characters",
        strategy="structural",
    )
    for source_result in result.sources:
        previous_end = 0
        for span in source_result.spans:
            assert 0 <= span.start < span.end <= len(sources[source_result.source_index])
            assert span.start >= previous_end
            assert sources[source_result.source_index][span.start : span.end].strip()
            assert sources[source_result.source_index][span.start : span.end] in source_result.text
            previous_end = span.end


def test_context_markers_follow_source_order_within_the_aggregate_budget() -> None:
    """Add affordable markers deterministically after all retained content fits."""
    result = Trimmer(TrimConfig(omission_marker="...")).trim_context(
        ["alpha\n\nomega", "beta\n\ngamma"],
        15,
        unit="characters",
        strategy="structural",
    )
    assert [source.text for source in result.sources] == ["alpha\n\n...", "beta\n"]
    assert result.output_count == result.limit == 15


def test_content_is_kept_when_omission_markers_do_not_fit() -> None:
    """Spend the aggregate budget on source evidence before optional markers."""
    result = Trimmer(TrimConfig(omission_marker="[very long omission]")).trim_context(
        ["alpha\n\nomega", "beta\n\ngamma"],
        11,
        unit="characters",
        strategy="structural",
    )
    assert result.output_count <= 11
    assert "[very long omission]" not in "".join(source.text for source in result.sources)
    assert all(source.text for source in result.sources)


def test_global_fallback_uses_the_strongest_source_row() -> None:
    """Place one exact prefix only in the source with strongest query evidence."""
    result = Trimmer().trim_context(
        ["irrelevant material without breaks", "target evidence without breaks"],
        6,
        unit="characters",
        strategy="lexical",
        query="target",
    )
    assert result.sources[0].text == ""
    assert result.sources[1].text == "target"
    assert result.sources[1].spans == (SourceSpan(0, 6),)


def test_balanced_fence_fallback_stays_in_its_source_row() -> None:
    """Reuse ordinary balanced-fence fallback for one active context source."""
    source = "```py\none\ntwo\n```\n"
    trimmer = Trimmer()
    ordinary = trimmer.trim(source, 13, unit="characters")
    context = trimmer.trim_context(["", source], 13, unit="characters")
    assert context.sources[0].text == ""
    assert context.sources[1].text == ordinary.text
    assert context.sources[1].spans == ordinary.spans


def test_duplicate_sources_remain_distinct_result_rows() -> None:
    """Preserve caller identity even when source strings are exact duplicates."""
    source = "same source text"
    result = Trimmer().trim_context([source, source], 8, unit="characters")
    assert len(result.sources) == 2
    assert [row.source_index for row in result.sources] == [0, 1]
    assert sum(bool(row.text) for row in result.sources) == 1


@pytest.mark.asyncio
async def test_atrim_many_keeps_independent_limits_while_context_shares_one() -> None:
    """Keep batch trimming semantics distinct from shared-context semantics."""
    trimmer = Trimmer()
    independent = await trimmer.atrim_many(
        [
            TrimInput("alpha", 5, unit="characters"),
            TrimInput("beta", 5, unit="characters"),
        ]
    )
    shared = await trimmer.atrim_context(["alpha", "beta"], 5, unit="characters")
    assert [result.text for result in independent] == ["alpha", "beta"]
    assert sum(source.output_count for source in shared.sources) <= 5
    assert sum(bool(source.text) for source in shared.sources) == 1
