"""Verify the original LLMLingua adapters receive one context string."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from benchmark.adapters.llmlingua_adapter import LLMLinguaAdapter
from benchmark.adapters.llmlingua_common import LLMLinguaBase
from benchmark.adapters.long_llmlingua_adapter import LongLLMLinguaAdapter


class RecordingCompressor:
    """Record compression calls without loading LLMLingua model weights."""

    def __init__(self) -> None:
        """Initialize the recorded context value."""
        self.context: object | None = None

    def compress_prompt(self, **kwargs: object) -> dict[str, Any]:
        """Record a call and return the minimum LLMLingua-shaped response.

        Args:
            **kwargs: Prompt-compressor arguments.

        Returns:
            A response accepted by the benchmark adapter.
        """
        self.context = kwargs["context"]
        return {"compressed_prompt": "compressed"}


class LoadingCompressor:
    """Expose tokenizer cleanup settings without loading model weights."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the tokenizer setting observed by the shared adapter.

        Args:
            **kwargs: Prompt-compressor construction settings.
        """
        del kwargs
        self.tokenizer = SimpleNamespace(clean_up_tokenization_spaces=True)


@pytest.mark.parametrize("adapter_factory", [LLMLinguaAdapter, LongLLMLinguaAdapter])
def test_original_llmlingua_adapters_pass_a_context_string(
    adapter_factory: Callable[[], LLMLinguaBase],
) -> None:
    """Ensure the LLMLingua 0.2.2 API receives text rather than a list."""
    adapter = adapter_factory()
    compressor = RecordingCompressor()
    adapter._compressor = compressor

    adapter.compress("source text", "query", 12, 3)

    assert compressor.context == "source text"


def test_llmlingua_loader_preserves_code_import_spacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disable tokenizer cleanup that breaks LLMLingua prefix matching on code imports."""
    monkeypatch.setitem(
        sys.modules, "llmlingua", SimpleNamespace(PromptCompressor=LoadingCompressor)
    )
    adapter = LongLLMLinguaAdapter()

    compressor = adapter._load()

    assert compressor.tokenizer.clean_up_tokenization_spaces is False
