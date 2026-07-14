"""Verify optional semantic error boundaries, caching, locking, and async use."""

from __future__ import annotations

import asyncio
import gc
import threading
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from trimwise import SemanticBackendError, TrimConfig, Trimmer
from trimwise.semantic import SemanticEmbedder


def _callback_vectors(
    _: str,
    passages: Sequence[str],
) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
    """Represent target-bearing candidates as aligned with the query.

    Args:
        _: Query text, which does not affect this deterministic fake.
        passages: Contextual candidate texts.

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


class _FakeModel:
    """Return deterministic vectors and expose concurrent inference count."""

    instances: ClassVar[int] = 0
    active: ClassVar[int] = 0
    maximum_active: ClassVar[int] = 0
    last_options: ClassVar[dict[str, Any]] = {}
    last_passages: ClassVar[list[str]] = []
    guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, **options: Any) -> None:
        """Record model construction while accepting backend options."""
        type(self).instances += 1
        type(self).last_options = options

    def query_embed(self, _: str) -> Iterator[NDArray[np.float32]]:
        """Yield one fixed query representation."""
        yield np.asarray([1.0, 0.0], dtype=np.float32)

    def passage_embed(
        self,
        passages: list[str],
        **_: Any,
    ) -> Iterator[NDArray[np.float32]]:
        """Yield relevant vectors while tracking simultaneous generators.

        Args:
            passages: Candidate source texts.
            **_: Ignored inference options.

        Yields:
            Deterministic passage vectors.
        """
        type(self).last_passages = passages
        with type(self).guard:
            type(self).active += 1
            type(self).maximum_active = max(type(self).maximum_active, type(self).active)
        try:
            for passage in passages:
                candidate = passage.split("\n\n", maxsplit=1)[0]
                values = [1.0, 0.0] if "target" in candidate else [0.0, 1.0]
                yield np.asarray(values, dtype=np.float32)
        finally:
            with type(self).guard:
                type(self).active -= 1


@pytest.fixture(autouse=True)
def reset_fake_model() -> None:
    """Reset fake backend class state between tests."""
    _FakeModel.instances = 0
    _FakeModel.active = 0
    _FakeModel.maximum_active = 0
    _FakeModel.last_options = {}
    _FakeModel.last_passages = []


def test_semantic_backend_converts_vectors_and_caches_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain backend arrays while constructing one model."""
    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    embedder = SemanticEmbedder(Trimmer().config)
    first = embedder.embed("query", ["target", "other"])
    second = embedder.embed("query", ["target"])
    assert (first.query_scores(), second.query_scores(), _FakeModel.instances) == (
        [1.0, 0.0],
        [1.0],
        1,
    )


def test_semantic_backend_retains_one_normalized_float32_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep query and passage embeddings in one normalized float32 matrix."""
    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    vectors = SemanticEmbedder(Trimmer().config).embed("query", ["target", "other"])
    norms = np.linalg.norm(vectors.matrix, axis=1)
    assert (vectors.matrix.shape, vectors.matrix.dtype, np.allclose(norms, 1.0)) == (
        (3, 2),
        np.dtype("float32"),
        True,
    )


def test_zero_semantic_vectors_remain_valid() -> None:
    """Preserve zero vectors as zero instead of producing non-finite values."""
    zero = np.zeros(2, dtype=np.float32)
    vectors = SemanticEmbedder(Trimmer().config)._normalize_vectors(zero, [zero])
    assert vectors.query_scores() == [0.0]


def test_semantic_strategy_uses_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select semantically aligned evidence without a real model download."""
    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    source = "Other material that is verbose.\n\nThe target fact is retained here.\n"
    result = Trimmer().trim(
        source,
        35,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    assert "target fact" in result.text


def test_semantic_backend_receives_scoring_context_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embed heading and neighbors without changing retained candidate source.

    Args:
        monkeypatch: Pytest attribute replacement helper.
    """
    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    source = (
        "# Installation\n\nRun this command.\n\n"
        "It was discontinued in 2024.\n\nUse Orion instead.\n\n"
        "# Removal\n\nDelete files.\n"
    )

    Trimmer().trim(source, 35, unit="characters", strategy="semantic", query="2024")

    context = _FakeModel.last_passages[2]
    assert context.startswith("It was discontinued in 2024.")
    assert "# Installation" in context
    assert "Run this command." in context
    assert "Use Orion instead." in context
    assert "# Removal" not in context


def test_provider_options_reach_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward explicit GPU or provider options without interpreting them."""
    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    config = TrimConfig(fastembed_options={"cuda": True, "threads": 2})
    SemanticEmbedder(config).embed("query", ["target"])
    assert _FakeModel.last_options == {
        "model_name": config.embedding_model,
        "cuda": True,
        "threads": 2,
    }


@pytest.mark.parametrize(
    "fastembed_options",
    [{}, {"providers": ["CUDAExecutionProvider"]}],
    ids=["cpu", "gpu"],
)
def test_inference_does_not_force_process_garbage_collection(
    monkeypatch: pytest.MonkeyPatch,
    fastembed_options: dict[str, Any],
) -> None:
    """Avoid scanning the host process heap after managed inference.

    Args:
        monkeypatch: Pytest patch helper.
        fastembed_options: Provider configuration under test.
    """

    def reject_collection() -> int:
        """Fail if inference requests process-wide garbage collection.

        Raises:
            AssertionError: Always, because backend calls must not force collection.
        """
        raise AssertionError("managed inference forced process-wide garbage collection")

    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    monkeypatch.setattr(gc, "collect", reject_collection)
    SemanticEmbedder(TrimConfig(fastembed_options=fastembed_options)).embed("query", ["target"])


def test_hybrid_strategy_uses_both_rankers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise lexical-semantic score fusion through the public hybrid strategy."""
    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    source = "Other material.\n\nThe target fact is here.\n\nMore unrelated material.\n"
    result = Trimmer().trim(
        source,
        32,
        unit="characters",
        strategy="hybrid",
        query="target fact",
    )
    assert result.output_count <= 32


@pytest.mark.parametrize("strategy", ["semantic", "hybrid"])
def test_sync_embedding_callback_overrides_fastembed(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    """Use caller vectors without importing the installed semantic backend.

    Args:
        monkeypatch: Pytest attribute replacement helper.
        strategy: Semantic strategy under test.
    """

    def fail_import(_: str) -> None:
        """Fail if an explicit callback does not take precedence."""
        raise AssertionError("unexpected FastEmbed import")

    monkeypatch.setattr("trimwise.semantic.import_module", fail_import)
    source = "Other material that is verbose.\n\nThe target fact is retained here.\n"
    result = Trimmer(embedding_callback=_callback_vectors).trim(
        source,
        35,
        unit="characters",
        strategy=strategy,
        query="target",
    )
    assert "target fact" in result.text


def test_sync_embedding_callback_receives_contextual_passages() -> None:
    """Give callbacks ranking context while retaining exact candidate source."""
    received: tuple[str, Sequence[str]] | None = None

    def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Capture callback inputs and return deterministic vectors.

        Args:
            query: Normalized semantic query.
            passages: Contextual scoring texts.

        Returns:
            Deterministic query and passage vectors.
        """
        nonlocal received
        received = query, passages
        return _callback_vectors(query, passages)

    source = "# Install\n\nPrevious context.\n\nThe target fact.\n\nFollowing context.\n"
    result = Trimmer(embedding_callback=embed).trim(
        source,
        25,
        unit="characters",
        strategy="semantic",
        query="  target  ",
    )
    assert received is not None
    query, passages = received
    assert (query, "# Install" in passages[2], "target fact" in result.text) == (
        "target",
        True,
        True,
    )


def test_callback_failure_is_staged_and_chained() -> None:
    """Translate caller backend failures without mentioning FastEmbed."""

    def fail(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Simulate a failing caller-provided model."""
        raise RuntimeError("service unavailable")

    with pytest.raises(SemanticBackendError, match="embedding callback inference") as captured:
        Trimmer(embedding_callback=fail).trim(
            "one\n\ntwo",
            2,
            unit="characters",
            strategy="semantic",
            query="q",
        )
    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "passages",
    [
        [],
        [[1.0], [1.0]],
        [[float("nan"), 0.0], [1.0, 0.0]],
    ],
    ids=["count", "dimension", "nonfinite"],
)
def test_callback_output_is_validated(passages: list[list[float]]) -> None:
    """Reject malformed caller vectors through one stable error boundary.

    Args:
        passages: Invalid passage vector rows to return.
    """

    def embed(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Return the configured malformed vectors."""
        return [1.0, 0.0], passages

    with pytest.raises(SemanticBackendError, match="embedding callback output"):
        Trimmer(embedding_callback=embed).trim(
            "one\n\ntwo",
            2,
            unit="characters",
            strategy="semantic",
            query="q",
        )


def test_callback_output_stops_after_first_excess_vector() -> None:
    """Bound malformed callback iterables by the known passage count."""
    produced = 0

    def passage_vectors() -> Iterator[list[float]]:
        """Yield vectors and fail if validation consumes beyond the first excess row.

        Yields:
            Reusable one-dimensional passage vectors.

        Raises:
            AssertionError: If validation requests a fourth row.
        """
        nonlocal produced
        while True:
            produced += 1
            if produced > 3:
                raise AssertionError("callback output was over-consumed")
            yield [1.0]

    def embed(_: str, __: Sequence[str]) -> tuple[list[float], Iterator[list[float]]]:
        """Return one query vector and a malformed excessive passage stream."""
        return [1.0], passage_vectors()

    with pytest.raises(SemanticBackendError, match="embedding callback output"):
        Trimmer(embedding_callback=embed).trim(
            "one\n\ntwo",
            2,
            unit="characters",
            strategy="semantic",
            query="q",
        )
    assert produced == 3


def test_short_semantic_input_does_not_call_sync_embedding_callback() -> None:
    """Preserve the unchanged fast path before caller inference."""

    def fail(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Fail if a fitting input reaches semantic inference."""
        raise AssertionError("unexpected embedding callback")

    result = Trimmer(embedding_callback=fail).trim(
        "short",
        10,
        unit="characters",
        strategy="semantic",
        query="q",
    )
    assert result.text == "short"


def test_auto_query_does_not_call_embedding_callback() -> None:
    """Keep automatic query handling on the lightweight lexical strategy."""

    def fail(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Fail if automatic mode unexpectedly enables semantic scoring."""
        raise AssertionError("unexpected embedding callback")

    result = Trimmer(embedding_callback=fail).trim(
        "alpha material\n\ntarget evidence",
        15,
        unit="characters",
        query="target",
    )
    assert result.strategy.value == "lexical"


def test_short_semantic_input_does_not_import_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delay the optional import until semantic ranking is necessary."""

    def fail_import(_: str) -> None:
        """Fail if the short-input path attempts an import."""
        raise AssertionError("unexpected import")

    monkeypatch.setattr("trimwise.semantic.import_module", fail_import)
    result = Trimmer().trim("short", 10, unit="characters", strategy="semantic", query="q")
    assert result.text == "short"


def test_missing_semantic_extra_has_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate a missing optional package into the public staged error."""

    def fail_import(_: str) -> None:
        """Simulate an unavailable optional dependency."""
        raise ImportError("missing")

    monkeypatch.setattr("trimwise.semantic.import_module", fail_import)
    with pytest.raises(SemanticBackendError, match=r"install trimwise\[semantic\]") as captured:
        Trimmer().trim("one\n\ntwo", 2, unit="characters", strategy="semantic", query="q")
    assert isinstance(captured.value.__cause__, ImportError)


def test_backend_import_runtime_failure_is_staged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap provider loader failures that are not ordinary missing imports."""

    def fail_import(_: str) -> None:
        """Simulate a native provider loader failure."""
        raise OSError("provider DLL failed to load")

    monkeypatch.setattr("trimwise.semantic.import_module", fail_import)
    with pytest.raises(SemanticBackendError, match="FastEmbed import failed") as captured:
        Trimmer().trim("one\n\ntwo", 2, unit="characters", strategy="semantic", query="q")
    assert isinstance(captured.value.__cause__, OSError)


@pytest.mark.parametrize(
    "failure",
    ["init", "query", "passage", "count", "dimension", "nonfinite"],
)
def test_semantic_failures_are_staged_and_chained(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Wrap initialization and every inference failure without fallback.

    Args:
        monkeypatch: Pytest patch helper.
        failure: Backend stage to break.
    """

    class BrokenModel(_FakeModel):
        """Fail at the requested semantic backend stage."""

        def __init__(self, **kwargs: Any) -> None:
            """Optionally fail during model initialization."""
            if failure == "init":
                raise RuntimeError("init")
            super().__init__(**kwargs)

        def query_embed(self, query: str) -> Iterator[NDArray[np.float32]]:
            """Optionally fail while materializing the query vector."""
            if failure == "query":
                raise RuntimeError("query")
            yield from super().query_embed(query)

        def passage_embed(
            self,
            passages: list[str],
            **kwargs: Any,
        ) -> Iterator[NDArray[np.float32]]:
            """Optionally fail or return the wrong passage count."""
            if failure == "passage":
                raise RuntimeError("passage")
            if failure == "count":
                yield np.asarray([1.0, 0.0], dtype=np.float32)
                return
            if failure == "dimension":
                yield from (np.asarray([1.0], dtype=np.float32) for _ in passages)
                return
            if failure == "nonfinite":
                yield from (np.asarray([float("nan"), 0.0], dtype=np.float32) for _ in passages)
                return
            yield from super().passage_embed(passages, **kwargs)

    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=BrokenModel),
    )
    with pytest.raises(SemanticBackendError) as captured:
        Trimmer().trim("one\n\ntwo\n\nthree", 2, unit="characters", strategy="semantic", query="q")
    assert captured.value.__cause__ is not None


def test_empty_query_vector_is_a_staged_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a backend that yields no query representation."""

    class EmptyQueryModel(_FakeModel):
        """Return an empty query generator."""

        def query_embed(self, _: str) -> Iterator[NDArray[np.float32]]:
            """Yield no query vectors."""
            return
            yield np.asarray([], dtype=np.float32)

    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=EmptyQueryModel),
    )
    with pytest.raises(SemanticBackendError, match="query inference"):
        Trimmer().trim("one\n\ntwo", 2, unit="characters", strategy="semantic", query="q")


def test_multiple_query_vectors_are_a_staged_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a backend that returns more than one vector for one query."""
    produced = 0

    class MultipleQueryModel(_FakeModel):
        """Return two query vectors for one query."""

        def query_embed(self, _: str) -> Iterator[NDArray[np.float32]]:
            """Yield two vectors and fail if validation requests a third.

            Yields:
                Two invalid query vectors.

            Raises:
                AssertionError: If validation consumes beyond the known invalid count.
            """
            nonlocal produced
            while True:
                produced += 1
                if produced > 2:
                    raise AssertionError("query output was over-consumed")
                yield np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=MultipleQueryModel),
    )
    with pytest.raises(SemanticBackendError, match="query inference"):
        Trimmer().trim("one\n\ntwo", 2, unit="characters", strategy="semantic", query="q")
    assert produced == 2


def test_excess_passage_vectors_stop_after_first_unexpected_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound malformed managed passage iterables by the supplied passage count.

    Args:
        monkeypatch: Pytest attribute replacement helper.
    """
    produced = 0

    class ExcessPassageModel(_FakeModel):
        """Return one more passage vector than requested."""

        def passage_embed(
            self,
            passages: list[str],
            **_: Any,
        ) -> Iterator[NDArray[np.float32]]:
            """Yield one excess row and fail if validation keeps consuming.

            Args:
                passages: Candidate source texts.
                **_: Ignored inference options.

            Yields:
                Valid-shaped passage vectors including one excess row.

            Raises:
                AssertionError: If validation requests a second excess row.
            """
            nonlocal produced
            while True:
                produced += 1
                if produced > len(passages) + 1:
                    raise AssertionError("passage output was over-consumed")
                yield np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=ExcessPassageModel),
    )
    with pytest.raises(SemanticBackendError, match="passage inference"):
        SemanticEmbedder(Trimmer().config).embed("query", ["one", "two"])
    assert produced == 3


@pytest.mark.asyncio
async def test_async_and_sync_results_match() -> None:
    """Return the same deterministic structural result from both APIs."""
    source = "First fact.\n\nMiddle filler.\n\nLast fact."
    trimmer = Trimmer()
    synchronous = trimmer.trim(source, 30, unit="characters")
    asynchronous = await trimmer.atrim(source, 30, unit="characters")
    assert asynchronous == synchronous


@pytest.mark.asyncio
async def test_async_callback_runs_outside_event_loop() -> None:
    """Allow the event loop to progress while a synchronous callback blocks."""
    started = threading.Event()
    release = threading.Event()

    def blocking_counter(text: str) -> int:
        """Block the worker until the event loop releases it."""
        started.set()
        release.wait(timeout=2)
        return len(text)

    task = asyncio.create_task(Trimmer().atrim("abcdef", 3, token_counter=blocking_counter))
    await asyncio.to_thread(started.wait, 1)
    assert not task.done()
    release.set()
    result = await task
    assert result.output_count <= 3


@pytest.mark.asyncio
async def test_sync_embedding_callback_runs_outside_event_loop() -> None:
    """Keep synchronous caller inference away from the event-loop thread."""
    event_loop_thread = threading.get_ident()
    callback_thread: int | None = None

    def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Record the worker thread used for caller inference.

        Args:
            query: Semantic query.
            passages: Contextual candidate texts.

        Returns:
            Deterministic query and passage vectors.
        """
        nonlocal callback_thread
        callback_thread = threading.get_ident()
        return _callback_vectors(query, passages)

    source = "other block\n\ntarget block\n\nthird block"
    await Trimmer(embedding_callback=embed).atrim(
        source,
        20,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    assert callback_thread is not None and callback_thread != event_loop_thread


@pytest.mark.asyncio
async def test_async_embedding_callback_runs_on_calling_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Await a caller's asynchronous client on its existing event loop.

    Args:
        monkeypatch: Pytest attribute replacement helper.
    """
    event_loop = asyncio.get_running_loop()
    callback_loop: asyncio.AbstractEventLoop | None = None

    async def embed(
        query: str,
        passages: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
        """Record native async execution and return deterministic vectors.

        Args:
            query: Semantic query.
            passages: Contextual candidate texts.

        Returns:
            Deterministic query and passage vectors.
        """
        nonlocal callback_loop
        callback_loop = asyncio.get_running_loop()
        await asyncio.sleep(0)
        return _callback_vectors(query, passages)

    def fail_import(_: str) -> None:
        """Fail if FastEmbed is imported despite the async callback."""
        raise AssertionError("unexpected FastEmbed import")

    monkeypatch.setattr("trimwise.semantic.import_module", fail_import)
    source = "other block\n\ntarget block\n\nthird block"
    result = await Trimmer(async_embedding_callback=embed).atrim(
        source,
        20,
        unit="characters",
        strategy="semantic",
        query="target",
    )
    assert (callback_loop is event_loop, "target" in result.text) == (True, True)


@pytest.mark.asyncio
async def test_async_embedding_callback_failure_is_staged_and_chained() -> None:
    """Translate asynchronous caller failures through the public error boundary."""

    async def fail(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Simulate an unavailable asynchronous embedding service."""
        raise RuntimeError("service unavailable")

    with pytest.raises(SemanticBackendError, match="embedding callback inference") as captured:
        await Trimmer(async_embedding_callback=fail).atrim(
            "one\n\ntwo",
            2,
            unit="characters",
            strategy="semantic",
            query="q",
        )
    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_async_embedding_callback_is_skipped_for_fitting_input() -> None:
    """Return unchanged input before awaiting caller inference."""

    async def fail(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Fail if the unchanged fast path invokes the callback."""
        raise AssertionError("unexpected embedding callback")

    result = await Trimmer(async_embedding_callback=fail).atrim(
        "short",
        10,
        unit="characters",
        strategy="semantic",
        query="q",
    )
    assert result.text == "short"


@pytest.mark.asyncio
async def test_async_embedding_callback_is_skipped_for_structural_ranking() -> None:
    """Keep a configured semantic dependency out of structural calls."""

    async def fail(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Fail if structural ranking invokes the callback."""
        raise AssertionError("unexpected embedding callback")

    result = await Trimmer(async_embedding_callback=fail).atrim(
        "First fact.\n\nMiddle material.\n\nLast fact.",
        25,
        unit="characters",
        strategy="structural",
    )
    assert result.strategy.value == "structural"


@pytest.mark.asyncio
async def test_async_embedding_callback_cancellation_propagates() -> None:
    """Cancel native caller inference instead of leaving it behind."""
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def embed(
        _: str,
        __: Sequence[str],
    ) -> tuple[list[float], list[list[float]]]:
        """Wait until cancellation and expose callback cleanup."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return [1.0], [[1.0]]

    source = "other block\n\ntarget block\n\nthird block"
    task = asyncio.create_task(
        Trimmer(async_embedding_callback=embed).atrim(
            source,
            20,
            unit="characters",
            strategy="semantic",
            query="target",
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(stopped.wait(), timeout=1)


def test_sync_trim_rejects_required_async_embedding_callback() -> None:
    """Direct synchronous callers to the API that can await their backend."""

    async def embed(
        _: str,
        passages: Sequence[str],
    ) -> tuple[list[float], list[list[float]]]:
        """Return vectors only through an asynchronous interface."""
        return [1.0], [[1.0] for _ in passages]

    with pytest.raises(TypeError, match=r"use atrim\(\)"):
        Trimmer(async_embedding_callback=embed).trim(
            "one\n\ntwo",
            2,
            unit="characters",
            strategy="semantic",
            query="q",
        )


def test_short_sync_trim_ignores_async_embedding_callback() -> None:
    """Return fitting input before requiring an asynchronous backend."""

    async def fail(_: str, __: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Fail if a fitting input attempts asynchronous inference."""
        raise AssertionError("unexpected embedding callback")

    result = Trimmer(async_embedding_callback=fail).trim(
        "short",
        10,
        unit="characters",
        strategy="semantic",
        query="q",
    )
    assert result.text == "short"


@pytest.mark.asyncio
async def test_cancellation_stops_waiting_for_worker() -> None:
    """Cancel awaiting without claiming to terminate the underlying thread."""
    started = threading.Event()
    release = threading.Event()

    def blocking_counter(text: str) -> int:
        """Hold the worker long enough for the awaiting task to be cancelled."""
        started.set()
        release.wait(timeout=2)
        return len(text)

    task = asyncio.create_task(Trimmer().atrim("abcdef", 3, token_counter=blocking_counter))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()


@pytest.mark.asyncio
async def test_same_trimmer_serializes_semantic_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limit one trimmer instance to one active FastEmbed generator."""
    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    trimmer = Trimmer()
    source = "other block\n\ntarget block\n\nthird block"
    await asyncio.gather(
        trimmer.atrim(source, 20, unit="characters", strategy="semantic", query="target"),
        trimmer.atrim(source, 20, unit="characters", strategy="semantic", query="target"),
    )
    assert _FakeModel.maximum_active == 1
    assert _FakeModel.instances == 1


@pytest.mark.asyncio
async def test_separate_trimmers_have_independent_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow callers to trade extra model memory for independent instances."""
    monkeypatch.setattr(
        "trimwise.semantic.import_module",
        lambda _: SimpleNamespace(TextEmbedding=_FakeModel),
    )
    source = "other block\n\ntarget block\n\nthird block"
    await asyncio.gather(
        Trimmer().atrim(source, 20, unit="characters", strategy="semantic", query="target"),
        Trimmer().atrim(source, 20, unit="characters", strategy="semantic", query="target"),
    )
    assert _FakeModel.instances == 2
