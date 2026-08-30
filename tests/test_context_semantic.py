"""Verify shared-context semantic batching, deduplication, and async behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from trimwise import SemanticBackendError, Trimmer
from trimwise.semantic import _SemanticVectors, normalize_callback_output


def _target_vectors(
    _: str,
    passages: Sequence[str],
) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
    """Align candidates whose own text contains the target term.

    Args:
        _: Query text, which is fixed for these deterministic tests.
        passages: Contextual candidate strings.

    Returns:
        One query vector and one vector per passage.
    """
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    vectors = [
        np.asarray(
            [1.0, 0.0] if "target" in passage.split("\n\n", maxsplit=1)[0] else [0.0, 1.0],
            dtype=np.float32,
        )
        for passage in passages
    ]
    return query, vectors


def test_sync_context_semantic_uses_one_operation_wide_batch() -> None:
    """Send every oversized context candidate through one callback invocation."""
    batches: list[list[str]] = []

    def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Record one batch and return deterministic vectors.

        Args:
            query: Shared semantic query.
            passages: Candidates from every source.

        Returns:
            One query vector and one vector per passage.
        """
        batches.append(list(passages))
        return _target_vectors(query, passages)

    result = Trimmer(embedding_callback=embed).trim_context(
        ["other one\n\ntarget one", "other two\n\ntarget two"],
        12,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    assert len(batches) == 1
    assert len(batches[0]) == 4
    assert result.output_count <= 12
    assert any("target" in source.text for source in result.sources)


def test_exact_deduplication_reduces_passages_without_changing_results() -> None:
    """Embed repeated contextual strings once and restore every candidate occurrence."""
    batches: list[list[str]] = []

    def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Capture first-seen passages and return deterministic vectors.

        Args:
            query: Shared semantic query.
            passages: Possibly deduplicated contextual strings.

        Returns:
            One query vector and one vector per passage.
        """
        batches.append(list(passages))
        return _target_vectors(query, passages)

    source = "other fact\n\ntarget fact\n\nlast fact"
    trimmer = Trimmer(embedding_callback=embed)
    ordinary = trimmer.trim_context(
        [source, source],
        20,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    deduplicated = trimmer.trim_context(
        [source, source],
        20,
        unit="characters",
        strategy="semantic",
        query="target",
        deduplicate=True,
    )
    assert len(batches[0]) == 6
    assert len(batches[1]) == 3
    assert batches[1] == list(dict.fromkeys(batches[0]))
    assert deduplicated == ordinary


def test_managed_semantic_context_reuses_exact_deduplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the same exact-string batching to Trimwise-managed embeddings.

    Args:
        monkeypatch: Pytest attribute replacement helper.
    """
    batches: list[list[str]] = []
    trimmer = Trimmer()

    def embed(query: str, passages: list[str]) -> _SemanticVectors:
        """Record managed-backend passages and return normalized test vectors.

        Args:
            query: Shared semantic query.
            passages: Deduplicated candidate strings.

        Returns:
            Normalized semantic vectors.
        """
        batches.append(passages)
        return normalize_callback_output(_target_vectors(query, passages), len(passages))

    monkeypatch.setattr(trimmer._semantic, "embed", embed)
    source = "other fact\n\ntarget fact\n\nlast fact"
    result = trimmer.trim_context(
        [source, source],
        20,
        unit="characters",
        strategy="semantic",
        query="target",
        deduplicate=True,
    )
    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert any("target" in source_result.text for source_result in result.sources)


@pytest.mark.parametrize("strategy", ["semantic", "hybrid"])
def test_semantic_context_competition_can_favor_a_later_source(strategy: str) -> None:
    """Let operation-wide semantic relevance displace earlier filler.

    Args:
        strategy: Semantic-only or lexical-semantic fusion.
    """
    result = Trimmer(embedding_callback=_target_vectors).trim_context(
        ["early filler\n\nmore filler", "target answer\n\nnoise"],
        14,
        unit="characters",
        strategy=strategy,
        query="target",
    )
    assert result.sources[0].text == ""
    assert result.sources[1].text == "target answer\n"


def test_contextual_passages_never_mix_source_neighbors_or_headings() -> None:
    """Keep ranking-only headings and neighbors inside their original source."""
    batches: list[list[str]] = []

    def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Capture contextual strings and return deterministic vectors.

        Args:
            query: Shared semantic query.
            passages: Source-isolated contextual candidates.

        Returns:
            One query vector and one vector per passage.
        """
        batches.append(list(passages))
        return _target_vectors(query, passages)

    Trimmer(embedding_callback=embed).trim_context(
        [
            "# Alpha unique\n\ntarget alpha body\n\nalpha neighbor",
            "# Beta unique\n\ntarget beta body\n\nbeta neighbor",
        ],
        20,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    assert len(batches) == 1
    assert all(
        not ("Alpha unique" in passage and "Beta unique" in passage) for passage in batches[0]
    )
    assert all(
        not ("alpha neighbor" in passage and "beta neighbor" in passage) for passage in batches[0]
    )


@pytest.mark.parametrize("failure", ["inference", "output"])
def test_sync_context_semantic_reports_staged_callback_failures(failure: str) -> None:
    """Preserve stable inference and output error stages for context calls.

    Args:
        failure: Callback stage to fail.
    """

    def embed(_: str, __: Sequence[str]) -> tuple[object, list[object]]:
        """Raise or return too few vectors for the selected stage.

        Args:
            _: Ignored query.
            __: Ignored passage collection.

        Returns:
            Malformed vectors for output-stage validation.

        Raises:
            RuntimeError: When testing inference failure translation.
        """
        if failure == "inference":
            raise RuntimeError("backend unavailable")
        return [1.0], []

    with pytest.raises(SemanticBackendError, match=f"callback {failure} failed"):
        Trimmer(embedding_callback=embed).trim_context(
            ["target fact\n\nother fact"],
            8,
            unit="characters",
            strategy="semantic",
            query="target",
        )


def test_context_skips_semantic_backend_on_nonsemantic_and_fast_paths() -> None:
    """Avoid embedding work when ranking does not require semantic vectors."""
    calls = 0

    def reject(_: str, __: Sequence[str]) -> tuple[object, list[object]]:
        """Fail if a supposedly skipped semantic path invokes the callback.

        Args:
            _: Ignored query.
            __: Ignored passages.

        Raises:
            AssertionError: Always, because these paths must skip inference.
        """
        nonlocal calls
        calls += 1
        raise AssertionError("semantic callback should have been skipped")

    trimmer = Trimmer(embedding_callback=reject)
    trimmer.trim_context(["short", "fit"], 8, unit="characters", strategy="semantic", query="x")
    trimmer.trim_context(["long source"], 0, unit="characters", strategy="semantic", query="x")
    trimmer.trim_context(["long source", "more"], 5, unit="characters", strategy="structural")
    trimmer.trim_context(
        ["long source", "more"],
        5,
        unit="characters",
        strategy="lexical",
        query="source",
    )
    assert calls == 0


def test_sync_context_rejects_async_only_backend_only_when_inference_is_needed() -> None:
    """Allow fast paths but reject oversized synchronous semantic inference."""

    async def embed(_: str, passages: Sequence[str]) -> tuple[object, list[object]]:
        """Return placeholder vectors if asynchronous inference is reached.

        Args:
            _: Ignored query.
            passages: Candidate strings.

        Returns:
            One query vector and one vector per passage.
        """
        return [1.0], [[1.0] for _ in passages]

    trimmer = Trimmer(async_embedding_callback=embed)
    fitting = trimmer.trim_context(
        ["short"],
        5,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    assert fitting.sources[0].text == "short"
    with pytest.raises(TypeError, match=r"use atrim_context\(\)"):
        trimmer.trim_context(
            ["target fact\n\nother fact"],
            8,
            unit="characters",
            strategy="semantic",
            query="target",
        )


@pytest.mark.asyncio
async def test_async_context_callback_runs_on_calling_event_loop() -> None:
    """Await caller-owned semantic work on the active event loop."""
    calling_loop = asyncio.get_running_loop()
    callback_loop: asyncio.AbstractEventLoop | None = None

    async def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Record loop ownership and return deterministic vectors.

        Args:
            query: Shared semantic query.
            passages: Operation-wide contextual candidates.

        Returns:
            One query vector and one vector per passage.
        """
        nonlocal callback_loop
        callback_loop = asyncio.get_running_loop()
        return _target_vectors(query, passages)

    result = await Trimmer(async_embedding_callback=embed).atrim_context(
        ["other fact\n\ntarget fact", "more filler"],
        12,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    assert callback_loop is calling_loop
    assert any("target" in source.text for source in result.sources)


@pytest.mark.asyncio
async def test_async_context_snapshots_sources_before_callback_suspension() -> None:
    """Isolate worker preparation from caller mutations during async inference."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Pause after source preparation to allow a caller mutation.

        Args:
            query: Shared semantic query.
            passages: Candidates prepared from the source snapshot.

        Returns:
            One query vector and one vector per passage.
        """
        entered.set()
        await release.wait()
        return _target_vectors(query, passages)

    sources = ["other fact\n\ntarget fact"]
    task = asyncio.create_task(
        Trimmer(async_embedding_callback=embed).atrim_context(
            sources,
            8,
            unit="characters",
            strategy="semantic",
            query="target",
        )
    )
    await entered.wait()
    sources[0] = "replacement"
    release.set()
    result = await task
    assert result.sources[0].input_count == len("other fact\n\ntarget fact")
    assert "replacement" not in result.sources[0].text


@pytest.mark.asyncio
async def test_async_context_cancellation_propagates_and_runs_cleanup() -> None:
    """Cancel the caller-owned callback and let its cleanup complete."""
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def embed(_: str, __: Sequence[str]) -> tuple[object, list[object]]:
        """Wait indefinitely and expose cancellation cleanup.

        Args:
            _: Ignored query.
            __: Ignored candidates.

        Returns:
            Unreachable placeholder vectors.
        """
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return [1.0], []

    task = asyncio.create_task(
        Trimmer(async_embedding_callback=embed).atrim_context(
            ["target fact\n\nother fact"],
            8,
            unit="characters",
            strategy="semantic",
            query="target",
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_sync_and_async_context_results_match_for_deterministic_vectors() -> None:
    """Produce equivalent shared selections across callback execution models."""

    async def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Return the same deterministic vectors as the synchronous callback.

        Args:
            query: Shared semantic query.
            passages: Operation-wide contextual candidates.

        Returns:
            One query vector and one vector per passage.
        """
        return _target_vectors(query, passages)

    sources = ["other one\n\ntarget one", "other two\n\ntarget two"]
    synchronous = Trimmer(embedding_callback=_target_vectors).trim_context(
        sources,
        12,
        unit="characters",
        strategy="semantic",
        query="target",
        deduplicate=True,
    )
    asynchronous = await Trimmer(async_embedding_callback=embed).atrim_context(
        sources,
        12,
        unit="characters",
        strategy="semantic",
        query="target",
        deduplicate=True,
    )
    assert asynchronous == synchronous


@pytest.mark.asyncio
async def test_one_active_async_context_source_matches_atrim() -> None:
    """Preserve ordinary async selection when empty rows surround one source."""

    async def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Return deterministic vectors for ordinary and context calls.

        Args:
            query: Shared semantic query.
            passages: Contextual candidates.

        Returns:
            One query vector and one vector per passage.
        """
        return _target_vectors(query, passages)

    source = "other fact\n\ntarget fact\n\nlast fact"
    trimmer = Trimmer(async_embedding_callback=embed)
    ordinary = await trimmer.atrim(
        source,
        18,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    context = await trimmer.atrim_context(
        ["", source, ""],
        18,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    active = context.sources[1]
    assert (active.text, active.output_count, active.spans) == (
        ordinary.text,
        ordinary.output_count,
        ordinary.spans,
    )


@pytest.mark.asyncio
async def test_async_context_rejects_malformed_vectors_after_await() -> None:
    """Validate async passage-vector counts before worker-side ranking."""

    async def embed(_: str, __: Sequence[str]) -> tuple[object, list[object]]:
        """Return no passage vectors for an oversized semantic request.

        Args:
            _: Ignored query.
            __: Ignored candidates.

        Returns:
            Deliberately malformed callback output.
        """
        return [1.0], []

    with pytest.raises(SemanticBackendError, match="callback output failed"):
        await Trimmer(async_embedding_callback=embed).atrim_context(
            ["target fact\n\nother fact"],
            8,
            unit="characters",
            strategy="semantic",
            query="target",
        )
