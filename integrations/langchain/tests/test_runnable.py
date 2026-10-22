"""Exercise real LangChain Runnable composition and Trimwise source mapping."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
import tiktoken
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_trimwise import TrimwiseInput, create_trimwise_runnable

from trimwise import SemanticBackendError, Trimmer


@pytest.fixture
def documents() -> list[Document]:
    """Provide repeated IDs, an empty source, nested metadata, and later evidence.

    Returns:
        Retrieved documents with intentionally nonunique identities.
    """
    return [
        Document(
            id="same",
            page_content="Gardens have flowers. " * 12,
            metadata={"source": "garden.pdf", "nested": {"page": 1}},
        ),
        Document(id="empty", page_content="", metadata={"page": 2}),
        Document(
            id="same",
            page_content="The retry deadline is thirty seconds.",
            metadata={"source": "manual.pdf", "page": 3},
        ),
    ]


def test_shared_budget_and_source_mapping(documents: list[Document]) -> None:
    """Find later evidence while preserving exact identities and local spans.

    Args:
        documents: Retrieved documents including duplicate IDs and an empty row.
    """
    inputs: TrimwiseInput = {"documents": documents, "query": "retry deadline seconds"}
    output = create_trimwise_runnable(10, unit="words").invoke(inputs)
    assert sum(len(document.page_content.split()) for document in output) <= 10
    assert any(document.metadata["source"] == "manual.pdf" for document in output)
    indexes = [document.metadata["trimwise"]["source_index"] for document in output]
    assert indexes == sorted(indexes)
    for document in output:
        provenance = document.metadata["trimwise"]
        original = documents[provenance["source_index"]]
        assert document.id == original.id
        assert {key: value for key, value in document.metadata.items() if key != "trimwise"} == (
            original.metadata
        )
        for span in provenance["spans"]:
            assert original.page_content[span["start"] : span["end"]] in document.page_content


def test_copies_nested_metadata_without_mutation(documents: list[Document]) -> None:
    """Allow callers to edit returned metadata without changing indexed documents.

    Args:
        documents: Original document objects.
    """
    before = [document.model_dump() for document in documents]
    output = create_trimwise_runnable(1000).invoke({"documents": documents, "query": None})
    output[0].metadata["nested"]["page"] = 99
    assert [document.model_dump() for document in documents] == before


@pytest.mark.parametrize("limit", [0, 1, 8, 40])
def test_native_and_custom_token_budgets(limit: int, documents: list[Document]) -> None:
    """Respect one summed token allowance, including a caller-supplied tokenizer.

    Args:
        limit: Shared token ceiling.
        documents: Original retrieved sources.
    """
    encoding = tiktoken.get_encoding("o200k_base")

    def count(text: str) -> int:
        """Measure the selected model's tokens.

        Args:
            text: Evidence to measure.

        Returns:
            Token count without surrounding prompt text.
        """
        return 2 * len(encoding.encode(text))

    for counter in (None, count):
        runnable = create_trimwise_runnable(limit, token_counter=counter)
        output = runnable.invoke({"documents": documents, "query": None})
        measure = counter or (lambda text: len(tiktoken.get_encoding("o200k_base").encode(text)))
        assert sum(measure(document.page_content) for document in output) <= limit


@pytest.mark.parametrize("texts, limit", [([], 20), (["", ""], 20), (["hello"], 0)])
def test_empty_results(texts: list[str], limit: int) -> None:
    """Return no placeholder documents when no evidence contributes.

    Args:
        texts: Empty or budget-excluded sources.
        limit: Shared ceiling.
    """
    output = create_trimwise_runnable(limit).invoke(
        {
            "documents": [Document(page_content=text) for text in texts],
            "query": None,
        }
    )
    assert output == []


def test_metadata_collision_is_explicit() -> None:
    """Protect existing metadata and prevent silently replacing earlier provenance."""
    document = Document(page_content="evidence", metadata={"trimwise": "caller-owned"})
    with pytest.raises(ValueError, match="reserved"):
        create_trimwise_runnable(20).invoke({"documents": [document], "query": None})
    assert document.metadata == {"trimwise": "caller-owned"}


@pytest.mark.parametrize("limit, unit", [(-1, "tokens"), (True, "tokens"), (10, "bad")])
def test_invalid_options_propagate(limit: int, unit: str) -> None:
    """Keep Trimwise's argument validation even when no documents are present.

    Args:
        limit: Evidence allowance, possibly invalid.
        unit: Measurement unit, possibly invalid.
    """
    with pytest.raises((TypeError, ValueError)):
        create_trimwise_runnable(limit, unit=unit).invoke({"documents": [], "query": None})


def test_runnable_composition_and_batch(documents: list[Document]) -> None:
    """Compose using native LangChain operators and keep batch calls independent.

    Args:
        documents: A shared caller-owned retrieval result.
    """

    def contents(items: list[Document]) -> list[str]:
        """Extract evidence from the next step in a chain.

        Args:
            items: Trimmed documents.

        Returns:
            Retained evidence strings.
        """
        return [item.page_content for item in items]

    chain = create_trimwise_runnable(10, unit="words") | RunnableLambda(contents)
    inputs: TrimwiseInput = {"documents": documents, "query": "retry deadline"}
    expected = chain.invoke(inputs)
    assert chain.batch([inputs, inputs]) == [expected, expected]


async def test_async_parity_and_batch(documents: list[Document]) -> None:
    """Make native ainvoke and abatch agree with synchronous processing.

    Args:
        documents: Retrieved sources reused across requests.
    """
    runnable = create_trimwise_runnable(30, unit="characters")
    inputs: TrimwiseInput = {"documents": documents, "query": "retry deadline"}
    expected = runnable.invoke(inputs)
    assert await runnable.ainvoke(inputs) == expected
    assert await runnable.abatch([inputs, inputs]) == [expected, expected]


async def test_async_callback_dispatch_and_sync_rejection(documents: list[Document]) -> None:
    """Await real async embedding callbacks on the invoking event loop.

    Args:
        documents: Oversized retrieved evidence that requires ranking.
    """
    loop = asyncio.get_running_loop()
    calls: list[Sequence[str]] = []

    async def embed(query: str, passages: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Supply deterministic test vectors without a model or external service.

        Args:
            query: Question being ranked.
            passages: Context-rich passages requested by Trimwise.

        Returns:
            One query vector and one vector per passage.
        """
        assert asyncio.get_running_loop() is loop
        assert query == "retry deadline"
        calls.append(passages)
        return [1.0, 0.0], [[1.0, 0.0] for _ in passages]

    runnable = create_trimwise_runnable(
        10,
        unit="words",
        strategy="semantic",
        trimmer=Trimmer(async_embedding_callback=embed),
    )
    inputs: TrimwiseInput = {"documents": documents, "query": "retry deadline"}
    await runnable.ainvoke(inputs)
    assert len(calls) == 1
    with pytest.raises(TypeError, match="async"):
        runnable.invoke(inputs)


async def test_cancellation_reaches_callback(documents: list[Document]) -> None:
    """Cancel pending embedding work without mutating source documents.

    Args:
        documents: Oversized sources requiring semantic ranking.
    """
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def embed(query: str, passages: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Block deterministically until the caller cancels.

        Args:
            query: Retrieval question.
            passages: Candidate evidence.

        Raises:
            CancelledError: When the request is cancelled.
        """
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    runnable = create_trimwise_runnable(
        1,
        strategy="semantic",
        trimmer=Trimmer(async_embedding_callback=embed),
    )
    task = asyncio.create_task(runnable.ainvoke({"documents": documents, "query": "retry"}))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert all("trimwise" not in document.metadata for document in documents)


def test_backend_errors_propagate(documents: list[Document]) -> None:
    """Expose backend failures rather than returning a misleading successful result.

    Args:
        documents: Oversized retrieved sources.
    """

    def fail(query: str, passages: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Simulate an unavailable caller-owned embedding service.

        Args:
            query: Retrieval question.
            passages: Candidate evidence.

        Raises:
            RuntimeError: Always, to exercise the error boundary.
        """
        raise RuntimeError("service unavailable")

    runnable = create_trimwise_runnable(
        1,
        strategy="semantic",
        trimmer=Trimmer(embedding_callback=fail),
    )
    with pytest.raises(SemanticBackendError):
        runnable.invoke({"documents": documents, "query": "retry"})
