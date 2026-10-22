"""Apply Trimwise after retrieval without mutating indexed LlamaIndex text nodes."""

from __future__ import annotations

from collections.abc import Callable

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from trimwise import BudgetUnit, ContextTrimResult, Strategy, Trimmer


class TrimwiseNodePostprocessor(BaseNodePostprocessor):
    """Share one evidence budget across retrieved plain TextNode objects.

    Attributes:
        limit: Nonnegative maximum sum of independently measured evidence sizes.
        unit: Token, word, or character allowance.
        strategy: Ranking strategy passed to Trimwise.
        trimmer: Reusable engine, optionally configured with an embedding callback.
        token_counter: Optional model-specific synchronous token counter.
    """

    limit: int = Field(ge=0, strict=True)
    unit: BudgetUnit | str = BudgetUnit.TOKENS
    strategy: Strategy | str = Strategy.AUTO
    trimmer: Trimmer = Field(default_factory=Trimmer, exclude=True)
    token_counter: Callable[[str], int] | None = Field(default=None, exclude=True)

    @classmethod
    def class_name(cls) -> str:
        """Identify this postprocessor in LlamaIndex component descriptions.

        Returns:
            The stable component name.
        """
        return "TrimwiseNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Trim retrieved evidence synchronously using one shared operation.

        Args:
            nodes: Original retrieved plain text nodes and their scores.
            query_bundle: Optional retrieval question.

        Returns:
            Copied contributing nodes with original-source provenance.
        """
        result = self.trimmer.trim_context(
            _texts(nodes),
            self.limit,
            unit=self.unit,
            strategy=self.strategy,
            query=query_bundle.query_str if query_bundle is not None else None,
            token_counter=self.token_counter,
        )
        return _nodes(nodes, result)

    async def _apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Keep async embedding callbacks on the caller's event loop.

        Args:
            nodes: Original retrieved plain text nodes and their scores.
            query_bundle: Optional retrieval question.

        Returns:
            Copied contributing nodes with original-source provenance.
        """
        result = await self.trimmer.atrim_context(
            _texts(nodes),
            self.limit,
            unit=self.unit,
            strategy=self.strategy,
            query=query_bundle.query_str if query_bundle is not None else None,
            token_counter=self.token_counter,
        )
        return _nodes(nodes, result)


def _texts(nodes: list[NodeWithScore]) -> list[str]:
    """Validate supported nodes and read text without metadata rendering.

    Args:
        nodes: Retrieved nodes before any mutation or backend invocation.

    Returns:
        Plain source evidence in input order.

    Raises:
        TypeError: If a node is not a plain TextNode.
        ValueError: If metadata already uses the reserved trimwise key.
    """
    texts = []
    for item in nodes:
        if type(item.node) is not TextNode:
            raise TypeError("Trimwise supports plain TextNode objects only")
        if "trimwise" in item.node.metadata:
            raise ValueError("metadata key 'trimwise' is reserved for trimming provenance")
        texts.append(item.node.text)
    return texts


def _nodes(nodes: list[NodeWithScore], result: ContextTrimResult) -> list[NodeWithScore]:
    """Copy selected nodes, retaining identity and clearing stale derived fields.

    Args:
        nodes: Original retrieved text nodes, aligned with the result.
        result: One shared-budget Trimwise result.

    Returns:
        Independent node copies in input order, preserving retrieval scores.
    """
    output = []
    for source in result.sources:
        if not source.text:
            continue
        item = nodes[source.source_index].model_copy(deep=True)
        node = item.node
        assert isinstance(node, TextNode)
        node.metadata["trimwise"] = {
            "source_index": source.source_index,
            "spans": [{"start": span.start, "end": span.end} for span in source.spans],
            "original_start_char_idx": node.start_char_idx,
            "original_end_char_idx": node.end_char_idx,
        }
        node.set_content(source.text)
        if source.trimmed:
            # Disjoint excerpts no longer have one contiguous range or the original embedding.
            node.start_char_idx = None
            node.end_char_idx = None
            node.embedding = None
        for excluded in (node.excluded_llm_metadata_keys, node.excluded_embed_metadata_keys):
            if "trimwise" not in excluded:
                excluded.append("trimwise")
        output.append(item)
    return output
