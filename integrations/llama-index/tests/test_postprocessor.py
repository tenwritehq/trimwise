"""Exercise native LlamaIndex postprocessing with real shared-budget trimming."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
import tiktoken
from llama_index.core import SummaryIndex
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import (
    ImageNode,
    MetadataMode,
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.postprocessor.trimwise import TrimwiseNodePostprocessor

from trimwise import SemanticBackendError, Trimmer


@pytest.fixture
def nodes() -> list[NodeWithScore]:
    """Provide source offsets, embeddings, duplicate IDs, and a later useful node.

    Returns:
        Retrieved nodes with scores and caller-owned metadata.
    """
    return [
        NodeWithScore(
            node=TextNode(
                id_="same",
                text="Gardens have flowers. " * 12,
                metadata={"source": "garden.pdf", "nested": {"page": 1}},
                start_char_idx=100,
                end_char_idx=352,
                embedding=[0.0, 1.0],
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="document-1")},
            ),
            score=0.9,
        ),
        NodeWithScore(node=TextNode(id_="empty", text=""), score=None),
        NodeWithScore(
            node=TextNode(
                id_="same",
                text="The retry deadline is thirty seconds.",
                metadata={"source": "manual.pdf", "page": 3},
            ),
            score=0.8,
        ),
    ]


def test_shared_budget_and_provenance(nodes: list[NodeWithScore]) -> None:
    """Keep later evidence, score/identity alignment, and source-local spans.

    Args:
        nodes: Retrieved nodes with intentionally duplicate IDs.
    """
    output = TrimwiseNodePostprocessor(limit=10, unit="words").postprocess_nodes(
        nodes,
        query_str="retry deadline seconds",
    )
    assert sum(len(item.node.get_content(MetadataMode.NONE).split()) for item in output) <= 10
    assert any(item.node.metadata["source"] == "manual.pdf" for item in output)
    indexes = [item.node.metadata["trimwise"]["source_index"] for item in output]
    assert indexes == sorted(indexes)
    for item in output:
        original = nodes[item.node.metadata["trimwise"]["source_index"]]
        assert (item.node.node_id, item.score) == (original.node.node_id, original.score)
        assert item.node.relationships == original.node.relationships
        for span in item.node.metadata["trimwise"]["spans"]:
            assert original.node.get_content(MetadataMode.NONE)[span["start"] : span["end"]] in (
                item.node.get_content(MetadataMode.NONE)
            )


def test_copy_and_derived_fields(nodes: list[NodeWithScore]) -> None:
    """Leave indexed nodes untouched and invalidate changed-text offsets/embeddings.

    Args:
        nodes: Original indexed evidence.
    """
    before = [item.model_dump() for item in nodes]
    processor = TrimwiseNodePostprocessor(limit=20, unit="characters")
    output = processor.postprocess_nodes(nodes[:1])
    node = output[0].node
    assert isinstance(node, TextNode)
    assert (node.start_char_idx, node.end_char_idx, node.embedding) == (None, None, None)
    assert node.metadata["trimwise"]["original_start_char_idx"] == 100
    assert node.metadata["trimwise"]["original_end_char_idx"] == 352
    node.metadata["nested"]["page"] = 99
    assert [item.model_dump() for item in nodes] == before


def test_fitting_text_keeps_derived_fields(nodes: list[NodeWithScore]) -> None:
    """Keep valid existing offsets and embeddings when text fits unchanged.

    Args:
        nodes: Original indexed evidence.
    """
    output = TrimwiseNodePostprocessor(limit=1000).postprocess_nodes(nodes[:1])
    node = output[0].node
    assert isinstance(node, TextNode)
    assert (node.start_char_idx, node.end_char_idx, node.embedding) == (100, 352, [0.0, 1.0])


def test_provenance_excluded_from_prompt_and_embedding(nodes: list[NodeWithScore]) -> None:
    """Prevent internal span bookkeeping from entering LLM or embedding inputs.

    Args:
        nodes: Evidence containing user metadata.
    """
    output = TrimwiseNodePostprocessor(limit=1000).postprocess_nodes(nodes)
    for item in output:
        assert "trimwise" not in item.node.get_content(MetadataMode.LLM)
        assert "trimwise" not in item.node.get_content(MetadataMode.EMBED)
        assert item.node.metadata["source"] in item.node.get_content(MetadataMode.LLM)


@pytest.mark.parametrize("limit", [0, 1, 8, 40])
def test_custom_token_budget(limit: int, nodes: list[NodeWithScore]) -> None:
    """Use the supplied tokenizer across all evidence rows together.

    Args:
        limit: Shared token ceiling.
        nodes: Original retrieved sources.
    """
    encoding = tiktoken.get_encoding("o200k_base")

    def count(text: str) -> int:
        """Measure evidence using a model-specific tokenizer.

        Args:
            text: Evidence string.

        Returns:
            Number of encoded tokens.
        """
        return 2 * len(encoding.encode(text))

    processor = TrimwiseNodePostprocessor(limit=limit, token_counter=count)
    output = processor.postprocess_nodes(nodes)
    assert sum(count(item.node.get_content(MetadataMode.NONE)) for item in output) <= limit


@pytest.mark.parametrize("limit", [-1, True])
def test_invalid_budget(limit: int) -> None:
    """Reject negative and boolean ceilings at construction.

    Args:
        limit: Invalid evidence allowance.
    """
    with pytest.raises(ValueError):
        TrimwiseNodePostprocessor(limit=limit)


def test_empty_nodes() -> None:
    """Return an empty result without needing a query or embedding backend."""
    processor = TrimwiseNodePostprocessor(limit=10)
    assert processor.postprocess_nodes([]) == []
    assert processor.postprocess_nodes([NodeWithScore(node=TextNode(text=""))]) == []


def test_collision_and_unsupported_nodes() -> None:
    """Reject metadata collisions and image nodes before backend processing."""
    processor = TrimwiseNodePostprocessor(limit=10)
    with pytest.raises(TypeError, match="plain TextNode"):
        processor.postprocess_nodes([NodeWithScore(node=ImageNode(text="image caption"))])
    with pytest.raises(ValueError, match="reserved"):
        processor.postprocess_nodes(
            [
                NodeWithScore(node=TextNode(text="evidence", metadata={"trimwise": "owned"})),
            ]
        )


def test_native_index_retrieval(nodes: list[NodeWithScore]) -> None:
    """Consume a real LlamaIndex retriever without an embedding or answer model.

    Args:
        nodes: Original plain text nodes.
    """
    index = SummaryIndex([item.node for item in nodes])
    retrieved = index.as_retriever().retrieve("retry deadline")
    processor = TrimwiseNodePostprocessor(limit=10, unit="words")
    output = processor.postprocess_nodes(retrieved, QueryBundle("retry deadline"))
    assert any("thirty seconds" in item.node.get_content(MetadataMode.NONE) for item in output)
    assert all("trimwise" not in item.node.metadata for item in retrieved)


async def test_native_query_engine_hook(nodes: list[NodeWithScore]) -> None:
    """Run the native sync/async query-engine hook with a clearly mocked answer model.

    Args:
        nodes: Retrieved fixture evidence; no generation quality is being evaluated.
    """
    index = SummaryIndex([nodes[0].node, nodes[2].node.model_copy(update={"id_": "manual"})])
    processor = TrimwiseNodePostprocessor(limit=10, unit="words")
    engine = RetrieverQueryEngine.from_args(
        index.as_retriever(),
        llm=MockLLM(),
        node_postprocessors=[processor],
    )
    query = QueryBundle("retry deadline")
    sync_nodes = engine.retrieve(query)
    assert await engine.aretrieve(query) == sync_nodes
    assert sum(len(item.node.get_content(MetadataMode.NONE).split()) for item in sync_nodes) <= 10
    assert any("thirty seconds" in item.node.get_content(MetadataMode.NONE) for item in sync_nodes)


def test_existing_metadata_exclusions_and_config_round_trip(nodes: list[NodeWithScore]) -> None:
    """Preserve user exclusions and reconstruct declarative configuration safely.

    Args:
        nodes: Original text nodes used to check rendering after reconstruction.
    """
    node = nodes[0].node
    node.excluded_llm_metadata_keys = ["nested", "trimwise"]
    node.excluded_embed_metadata_keys = ["source"]
    processor = TrimwiseNodePostprocessor(limit=1000)
    saved = processor.to_dict()
    assert saved["class_name"] == "TrimwiseNodePostprocessor"
    assert "trimmer" not in saved
    assert "token_counter" not in saved
    restored = TrimwiseNodePostprocessor.from_dict(saved)
    retained = restored.postprocess_nodes(nodes[:1])[0].node
    assert retained.excluded_llm_metadata_keys == ["nested", "trimwise"]
    assert retained.excluded_embed_metadata_keys == ["source", "trimwise"]
    assert node.excluded_embed_metadata_keys == ["source"]


async def test_async_semantic_success(nodes: list[NodeWithScore]) -> None:
    """Finish concurrent async callback requests without shared result state.

    Args:
        nodes: Original oversized sources reused by both requests.
    """
    loop = asyncio.get_running_loop()
    calls = []

    async def embed(query: str, passages: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Return controlled vectors on the caller's loop, without external inference.

        Args:
            query: Retrieval question.
            passages: Candidate evidence.

        Returns:
            One query vector and a vector for each passage.
        """
        assert asyncio.get_running_loop() is loop
        calls.append(query)
        return [1.0, 0.0], [[1.0, 0.0] for _ in passages]

    processor = TrimwiseNodePostprocessor(
        limit=10,
        unit="words",
        strategy="semantic",
        trimmer=Trimmer(async_embedding_callback=embed),
    )
    first, second = await asyncio.gather(
        processor.apostprocess_nodes(nodes, query_str="retry"),
        processor.apostprocess_nodes(nodes, query_str="retry"),
    )
    assert calls == ["retry", "retry"]
    assert first == second
    first[0].node.metadata["local"] = True
    assert "local" not in second[0].node.metadata


async def test_async_parity(nodes: list[NodeWithScore]) -> None:
    """Keep native asynchronous postprocessing equivalent to the sync path.

    Args:
        nodes: Shared retrieved evidence.
    """
    processor = TrimwiseNodePostprocessor(limit=30, unit="characters")
    expected = processor.postprocess_nodes(nodes, query_str="retry deadline")
    assert await processor.apostprocess_nodes(nodes, query_str="retry deadline") == expected


async def test_async_callback_and_cancellation(nodes: list[NodeWithScore]) -> None:
    """Use native async dispatch and propagate cancellation through the framework.

    Args:
        nodes: Oversized retrieved evidence.
    """
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def embed(query: str, passages: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Wait for cancellation on the original event loop.

        Args:
            query: Retrieval question.
            passages: Candidate evidence.

        Raises:
            CancelledError: When the awaiting caller cancels.
        """
        assert asyncio.get_running_loop() is loop
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    processor = TrimwiseNodePostprocessor(
        limit=1,
        strategy="semantic",
        trimmer=Trimmer(async_embedding_callback=embed),
    )
    task = asyncio.create_task(processor.apostprocess_nodes(nodes, query_str="retry"))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    with pytest.raises(TypeError, match="async"):
        processor.postprocess_nodes(nodes, query_str="retry")


def test_callback_receives_evidence_only_once(nodes: list[NodeWithScore]) -> None:
    """Batch all passages into one backend call and exclude user metadata from ranking.

    Args:
        nodes: Oversized retrieved evidence with metadata.
    """
    calls: list[Sequence[str]] = []

    def embed(query: str, passages: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Return controlled vectors for testing orchestration rather than model quality.

        Args:
            query: Retrieval question.
            passages: Candidate evidence without source metadata.

        Returns:
            Query and passage vectors.
        """
        calls.append(passages)
        return [1.0, 0.0], [[1.0, 0.0] for _ in passages]

    processor = TrimwiseNodePostprocessor(
        limit=10,
        unit="words",
        strategy="semantic",
        trimmer=Trimmer(embedding_callback=embed),
    )
    processor.postprocess_nodes(nodes, query_str="retry")
    assert len(calls) == 1
    assert all("garden.pdf" not in passage for passage in calls[0])


def test_backend_error_propagates(nodes: list[NodeWithScore]) -> None:
    """Expose malformed callback output as a stable Trimwise backend error.

    Args:
        nodes: Oversized evidence that triggers callback validation.
    """

    def invalid(query: str, passages: Sequence[str]) -> tuple[list[float], list[list[float]]]:
        """Return deliberately invalid vectors.

        Args:
            query: Retrieval question.
            passages: Requested evidence passages.

        Returns:
            Invalid empty vectors for exercising validation.
        """
        return [], []

    processor = TrimwiseNodePostprocessor(
        limit=1,
        strategy="semantic",
        trimmer=Trimmer(embedding_callback=invalid),
    )
    with pytest.raises(SemanticBackendError):
        processor.postprocess_nodes(nodes, query_str="retry")
