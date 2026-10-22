"""Translate retrieved LangChain documents into a single Trimwise operation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from trimwise import BudgetUnit, ContextTrimResult, Strategy, Trimmer


class TrimwiseInput(TypedDict):
    """Supply retrieved documents and a question, or None for queryless trimming.

    Attributes:
        documents: Retrieved documents in their desired output order.
        query: Retrieval question; None selects queryless automatic behavior.
    """

    documents: Sequence[Document]
    query: str | None


def _texts(documents: Sequence[Document]) -> list[str]:
    """Read evidence while protecting the reserved provenance metadata key.

    Args:
        documents: Original, caller-owned retrieved documents.

    Returns:
        Document text in the original order.

    Raises:
        ValueError: If a document already uses the reserved trimwise key.
    """
    if any("trimwise" in document.metadata for document in documents):
        raise ValueError("metadata key 'trimwise' is reserved for trimming provenance")
    return [document.page_content for document in documents]


def _documents(documents: Sequence[Document], result: ContextTrimResult) -> list[Document]:
    """Copy contributing documents and attach spans relative to their original text.

    Args:
        documents: Original retrieved documents, aligned with the result.
        result: One shared-budget result produced by Trimwise.

    Returns:
        Independent document copies in input order, without empty evidence rows.
    """
    output = []
    for source in result.sources:
        if not source.text:
            continue
        document = documents[source.source_index].model_copy(deep=True)
        document.page_content = source.text
        document.metadata["trimwise"] = {
            "source_index": source.source_index,
            "spans": [{"start": span.start, "end": span.end} for span in source.spans],
        }
        output.append(document)
    return output


def create_trimwise_runnable(
    limit: int,
    *,
    trimmer: Trimmer | None = None,
    unit: BudgetUnit | str = BudgetUnit.TOKENS,
    strategy: Strategy | str = Strategy.AUTO,
    token_counter: Callable[[str], int] | None = None,
) -> RunnableLambda[TrimwiseInput, list[Document]]:
    """Create a sync/async Runnable that shares one evidence budget across documents.

    Args:
        limit: Maximum sum of the individually measured output evidence sizes.
        trimmer: Reusable Trimwise instance, including any embedding callbacks.
        unit: Token, word, or character allowance, as accepted by Trimwise.
        strategy: Ranking strategy, as accepted by Trimwise.
        token_counter: Optional model-specific synchronous token counter.

    Returns:
        A Runnable accepting documents and query and returning copied documents.
        Labels, separators, metadata, and surrounding prompt text are outside the limit.
        Argument validation and backend failures propagate from Trimwise on invocation.
    """
    engine = trimmer if trimmer is not None else Trimmer()

    def compress(inputs: TrimwiseInput) -> list[Document]:
        """Apply one synchronous shared-budget operation.

        Args:
            inputs: Retrieved documents and their question.

        Returns:
            Copied documents with retained evidence and original-source spans.
        """
        documents = inputs["documents"]
        result = engine.trim_context(
            _texts(documents),
            limit,
            unit=unit,
            strategy=strategy,
            query=inputs["query"],
            token_counter=token_counter,
        )
        return _documents(documents, result)

    async def acompress(inputs: TrimwiseInput) -> list[Document]:
        """Use Trimwise's async dispatch, including native async embedding callbacks.

        Args:
            inputs: Retrieved documents and their question.

        Returns:
            Copied documents with retained evidence and original-source spans.
        """
        documents = inputs["documents"]
        result = await engine.atrim_context(
            _texts(documents),
            limit,
            unit=unit,
            strategy=strategy,
            query=inputs["query"],
            token_counter=token_counter,
        )
        return _documents(documents, result)

    return RunnableLambda(compress, afunc=acompress, name="trimwise")
