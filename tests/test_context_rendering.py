"""Verify budgeted wrappers and complete shared-context rendering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from trimwise import ContextSource, SourceSpan, TrimConfig, Trimmer
from trimwise.measurement import Measurer
from trimwise.models import BudgetUnit


def _target_vectors(
    _: str,
    passages: Sequence[str],
) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
    """Align only passages containing target evidence with the query.

    Args:
        _: Query text, which is fixed for these deterministic tests.
        passages: Evidence-only strings submitted for embedding.

    Returns:
        One query vector and one vector per passage.
    """
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    vectors = [
        np.asarray(
            [1.0, 0.0] if "target" in passage else [0.0, 1.0],
            dtype=np.float32,
        )
        for passage in passages
    ]
    return query, vectors


def test_context_source_is_frozen_and_slotted() -> None:
    """Keep caller-owned evidence and its output prefix immutable."""
    source = ContextSource("evidence", prefix="Source: A\n")
    assert not hasattr(source, "__dict__")
    with pytest.raises(FrozenInstanceError):
        source.prefix = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "source",
    [
        ContextSource(cast(str, 1)),
        ContextSource("evidence", prefix=cast(str, 1)),
    ],
)
def test_context_source_fields_require_strings(source: ContextSource) -> None:
    """Reject malformed wrapper values before measurement or selection.

    Args:
        source: Context input containing one invalid field.
    """
    with pytest.raises(TypeError, match="ContextSource"):
        Trimmer().trim_context([source], 0)


def test_context_separator_requires_text() -> None:
    """Reject a separator whose exact output cannot be defined."""
    with pytest.raises(TypeError, match="separator must be a string or None"):
        Trimmer().trim_context([], 0, separator=cast(str, 1))


def test_wrapped_fitting_sources_return_complete_rendered_context() -> None:
    """Render unchanged evidence with prefixes only on nonempty source rows."""
    sources = [
        ContextSource("alpha", prefix="A: "),
        ContextSource("", prefix="EMPTY: "),
        ContextSource("beta", prefix="B: "),
    ]
    expected = "A: alpha\nB: beta"
    result = Trimmer().trim_context(
        sources,
        len(expected),
        unit="characters",
        separator="\n",
    )
    assert result.text == expected
    assert result.output_count == len(expected)
    assert [source.text for source in result.sources] == ["alpha", "", "beta"]
    assert [source.output_count for source in result.sources] == [5, 0, 4]
    assert [source.spans for source in result.sources] == [
        (SourceSpan(0, 5),),
        (),
        (SourceSpan(0, 4),),
    ]
    assert result.trimmed is False


def test_plain_sources_without_separator_keep_legacy_accounting() -> None:
    """Preserve existing independently measured row semantics by default."""
    result = Trimmer().trim_context(["alpha", "beta"], 9, unit="characters")
    assert result.text is None
    assert result.output_count == sum(source.output_count for source in result.sources)


def test_explicit_separator_renders_plain_source_rows() -> None:
    """Allow complete rendering without requiring per-source prefixes."""
    result = Trimmer().trim_context(
        ["alpha", "beta"],
        13,
        unit="characters",
        separator="\n--\n",
    )
    assert result.text == "alpha\n--\nbeta"
    assert result.output_count == 13


@pytest.mark.parametrize("unit", list(BudgetUnit))
def test_complete_rendering_is_remeasured_in_each_builtin_unit(unit: BudgetUnit) -> None:
    """Measure the final joined string rather than adding piece counts.

    Args:
        unit: Built-in measurement rule used for the complete rendering.
    """
    text = "A: alpha beta\nB: gamma"
    result = Trimmer().trim_context(
        [ContextSource("alpha beta", "A: "), ContextSource("gamma", "B: ")],
        100,
        unit=unit,
        separator="\n",
    )
    measurer = Measurer(unit, "o200k_base", None)
    assert result.text == text
    assert result.output_count == measurer.count(text)


def test_custom_counter_measures_the_complete_rendering() -> None:
    """Charge prefixes and separators through the caller's exact counter."""
    measured: list[str] = []

    def count_characters(text: str) -> int:
        """Record and count every string supplied by Trimwise.

        Args:
            text: String being measured.

        Returns:
            Number of Python code points.
        """
        measured.append(text)
        return len(text)

    result = Trimmer().trim_context(
        [ContextSource("aaaa", "P:"), ContextSource("b", "Q:")],
        8,
        token_counter=count_characters,
        separator="|",
    )
    assert result.text == "P:aaaa"
    assert result.output_count == count_characters(result.text)
    assert "P:aaaa|Q:b" in measured


def test_nonadditive_custom_counter_sees_the_join_boundary() -> None:
    """Reject a second source when its rendered boundary exceeds the custom limit."""

    def count_with_separator_surcharge(text: str) -> int:
        """Make a joined rendering cost more than its independently measured parts.

        Args:
            text: Complete string being measured.

        Returns:
            Character count plus a visible join surcharge.
        """
        return len(text) + (10 if "|" in text else 0)

    result = Trimmer().trim_context(
        [ContextSource("a", "P:"), ContextSource("b", "Q:")],
        7,
        token_counter=count_with_separator_surcharge,
        separator="|",
    )
    assert result.text == "P:a"
    assert result.output_count == count_with_separator_surcharge(result.text)


def test_wrapper_text_never_enters_semantic_passages_or_spans() -> None:
    """Keep provenance outside embedding input and source-backed ranges."""
    batches: list[list[str]] = []

    def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Capture evidence passages and return deterministic vectors.

        Args:
            query: Shared semantic query.
            passages: Evidence-only ranking passages.

        Returns:
            One query vector and one vector per passage.
        """
        batches.append(list(passages))
        return _target_vectors(query, passages)

    result = Trimmer(embedding_callback=embed).trim_context(
        [
            ContextSource("unrelated material", "target-label: "),
            ContextSource("target evidence", "source-b: "),
        ],
        25,
        unit="characters",
        strategy="semantic",
        query="target",
        separator="\n",
    )
    assert all(
        "target-label" not in passage and "source-b" not in passage for passage in batches[0]
    )
    assert result.text is not None and result.text.startswith("source-b: ")
    assert result.sources[1].spans == (SourceSpan(0, len("target evidence")),)


def test_prefix_text_never_influences_lexical_ranking() -> None:
    """Rank evidence rather than a query term appearing only in a prefix."""
    result = Trimmer().trim_context(
        [
            ContextSource("unrelated material", "target: "),
            ContextSource("target evidence", "source: "),
        ],
        23,
        unit="characters",
        strategy="lexical",
        query="target",
        separator="\n",
    )
    assert result.text == "source: target evidence"
    assert [source.text for source in result.sources] == ["", "target evidence"]


def test_wrapper_activation_cost_can_exclude_a_weaker_source() -> None:
    """Spend a shared limit on stronger evidence when another prefix will not fit."""
    evidence = ["target answer.", "target filler."]
    plain = Trimmer().trim_context(
        evidence,
        29,
        unit="characters",
        strategy="lexical",
        query="target",
        separator="\n",
    )
    wrapped = Trimmer().trim_context(
        [ContextSource(evidence[0], "A: "), ContextSource(evidence[1], "B: ")],
        29,
        unit="characters",
        strategy="lexical",
        query="target",
        separator="\n",
    )
    assert all(source.text for source in plain.sources)
    assert wrapped.text == "A: target answer."
    assert [source.text for source in wrapped.sources] == ["target answer.", ""]


def test_unaffordable_strong_prefix_does_not_block_a_fitting_source() -> None:
    """Try weaker evidence when the strongest source cannot emit its whole prefix."""
    result = Trimmer().trim_context(
        [
            ContextSource("target " * 20, "X" * 20),
            ContextSource("other", "O:"),
        ],
        10,
        unit="characters",
        strategy="lexical",
        query="target",
        separator="\n",
    )
    assert result.text == "O:other"
    assert [source.text for source in result.sources] == ["", "other"]


def test_fallback_reserves_room_for_the_contributing_source_prefix() -> None:
    """Bound an oversized best match using the complete rendered limit."""
    relevant = "target " * 100
    prefix = "Source: A\n"
    result = Trimmer().trim_context(
        [ContextSource(relevant, prefix), ContextSource("other", "Source: B\n")],
        20,
        unit="characters",
        strategy="lexical",
        query="target",
        separator="\n\n",
    )
    assert result.text == prefix + relevant[: 20 - len(prefix)]
    assert result.output_count == 20
    assert result.sources[0].spans == (SourceSpan(0, 20 - len(prefix)),)
    assert result.sources[1].text == ""


def test_balanced_fence_fallback_charges_its_prefix() -> None:
    """Keep a complete code-fence shell after reserving its source prefix."""
    source = "```py\none\ntwo\n```\n"
    result = Trimmer().trim_context(
        [ContextSource(source, "S:\n")],
        13,
        unit="characters",
    )
    assert result.text == "S:\n```py\n```\n"
    assert result.sources[0].text == "```py\n```\n"
    assert result.sources[0].spans == (SourceSpan(0, 6), SourceSpan(14, 18))


def test_omission_marker_is_kept_only_when_prefixed_output_fits() -> None:
    """Remeasure optional omission text together with its contributing prefix."""
    source = ContextSource("first\n\nmiddle\n\nlast", "S:")
    trimmer = Trimmer(TrimConfig(omission_marker="[...]"))
    without_marker = trimmer.trim_context([source], 19, unit="characters")
    with_marker = trimmer.trim_context([source], 20, unit="characters")
    assert without_marker.text == "S:first\n\nlast"
    assert "[...]" not in without_marker.text
    assert with_marker.text == "S:first\n\n[...]\n\nlast"
    assert with_marker.output_count == 20


@pytest.mark.parametrize(
    ("sources", "limit"),
    [
        ([], 5),
        ([ContextSource("", "unused")], 5),
        ([ContextSource("evidence", "unused")], 0),
        ([ContextSource("evidence", "12345")], 5),
    ],
)
def test_empty_or_unaffordable_rendering_never_returns_a_bare_wrapper(
    sources: list[ContextSource],
    limit: int,
) -> None:
    """Return an empty rendering unless source evidence can accompany its prefix.

    Args:
        sources: Empty, blank, zero-budget, or prefix-exhausted inputs.
        limit: Complete rendered-output limit.
    """
    result = Trimmer().trim_context(sources, limit, unit="characters", separator="\n")
    assert result.text == ""
    assert result.output_count == 0
    assert all(not source.text for source in result.sources)


@pytest.mark.asyncio
async def test_sync_and_async_wrapped_semantic_results_match() -> None:
    """Keep complete rendering identical across callback execution models."""

    async def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Return the synchronous test vectors asynchronously.

        Args:
            query: Shared semantic query.
            passages: Evidence-only ranking passages.

        Returns:
            One query vector and one vector per passage.
        """
        return _target_vectors(query, passages)

    sources = [
        ContextSource("other fact", "A: "),
        ContextSource("target fact", "B: "),
    ]
    synchronous = Trimmer(embedding_callback=_target_vectors).trim_context(
        sources,
        18,
        unit="characters",
        strategy="semantic",
        query="target",
        separator="\n",
    )
    asynchronous = await Trimmer(async_embedding_callback=embed).atrim_context(
        sources,
        18,
        unit="characters",
        strategy="semantic",
        query="target",
        separator="\n",
    )
    assert asynchronous == synchronous
