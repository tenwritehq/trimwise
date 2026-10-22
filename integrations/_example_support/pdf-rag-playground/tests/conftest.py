"""Provide explicit model-free doubles; real model runs are recorded separately."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from rag_playground.documents import LoadedPDFs
from rag_playground.models import Evidence, Generation
from rag_playground.retrieval import CorpusBuilder


class TestEmbeddings:
    """Use keyword vectors to verify native wiring without measuring model quality."""

    __test__ = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed test strings with the same controlled query rule.

        Args:
            texts: Fixture passages.

        Returns:
            Two-dimensional test vectors.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Separate retry evidence from unrelated gardening text.

        Args:
            text: Fixture passage or question.

        Returns:
            A deterministic, nonzero test vector.
        """
        return [1.0, 0.0] if "retry" in text.lower() else [0.0, 1.0]


class TestBackend:
    """Record exact generation inputs without performing inference or fabricating scores."""

    __test__ = False

    def __init__(self, context_limit: int = 4096) -> None:
        """Start an independent test answer backend.

        Args:
            context_limit: Test prompt-plus-answer allowance.
        """
        self.context_limit = context_limit
        self.info: dict[str, Any] = {"backend": "test double", "model": "no-real-model"}
        self.calls: list[tuple[str, str, int]] = []
        self.fail = False

    def count(self, text: str) -> int:
        """Use character counts as deliberately different test token measurements.

        Args:
            text: Source or context text.

        Returns:
            Character count used only as a synthetic token counter.
        """
        return len(text)

    def prompt_count(self, question: str, context: str) -> int:
        """Add a fixed framing allowance in the test oracle.

        Args:
            question: Test question.
            context: Rendered evidence.

        Returns:
            Controlled full-prompt measurement.
        """
        return 40 + len(question) + len(context)

    def generate(self, question: str, context: str, output_tokens: int) -> Generation:
        """Record a call and return a clearly mocked answer or injected failure.

        Args:
            question: Exact question supplied by the pipeline.
            context: Exact condition context.
            output_tokens: Shared answer allowance.

        Returns:
            A labelled test-only answer.

        Raises:
            RuntimeError: When fault injection is enabled.
        """
        self.calls.append((question, context, output_tokens))
        if self.fail:
            raise RuntimeError("secret-key-sentinel must never reach a UI or export")
        return Generation(
            "A mocked answer [S1].",
            "test-only",
            self.prompt_count(question, context),
            10,
            0.01,
            "complete",
            "test counts",
        )


@pytest.fixture(params=["langchain_rag.pipeline", "llamaindex_rag.pipeline"])
def builder(request: pytest.FixtureRequest) -> CorpusBuilder:
    """Exercise each shared contract through both actual native framework pipelines.

    Args:
        request: Parametrized native pipeline module.

    Returns:
        Native index factory without any model downloads.
    """
    return import_module(request.param).build_corpus


@pytest.fixture
def corpus(builder: CorpusBuilder) -> Any:
    """Create a real isolated native index using only controlled test embeddings.

    Args:
        builder: Native framework index factory.

    Yields:
        Local corpus with stable IDs, page metadata, and a long distractor.
    """
    chunks = []
    for index, text in enumerate(
        ("Gardens have flowers. " * 30, "The retry deadline is thirty seconds.")
    ):
        digest = hashlib.sha256(text.encode()).hexdigest()
        chunks.append(
            Evidence(
                id=digest,
                page_content=text,
                metadata={
                    "filename": f"source-{index}.pdf",
                    "document_sha256": digest,
                    "page": index + 1,
                    "page_start": 0,
                    "page_end": len(text),
                },
            )
        )
    loaded = LoadedPDFs(tuple(chunks), (), "test-corpus")
    active = builder(loaded, TestEmbeddings())
    try:
        yield active
    finally:
        active.close()


@pytest.fixture
def runtime_root(tmp_path: Path) -> Path:
    """Allocate an example-owned directory for session isolation tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Private test runtime root.
    """
    return tmp_path
