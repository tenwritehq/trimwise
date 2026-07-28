"""Build configured benchmark compression adapters."""

from __future__ import annotations

from typing import Any

from .base import CompressorAdapter
from .bm25_adapter import BM25Adapter
from .embedding_adapter import EmbeddingMMRAdapter
from .llmlingua2_adapter import LLMLingua2Adapter
from .llmlingua_adapter import LLMLinguaAdapter
from .long_llmlingua_adapter import LongLLMLinguaAdapter
from .positional import HeadTailAdapter, PrefixAdapter
from .recomp_adapter import RecompAdapter
from .selective_context_adapter import SelectiveContextAdapter
from .trimwise_adapter import TrimwiseAdapter


def build_adapter(spec: dict[str, Any]) -> CompressorAdapter:
    """Build the configured compressor adapter.

    Args:
        spec: Method configuration from a benchmark YAML file.

    Returns:
        Adapter ready to compress benchmark context.

    Raises:
        ValueError: If the method name is not supported.
    """
    name = str(spec["name"])
    if name in {"prefix", "naive_first_n"}:
        return PrefixAdapter()
    if name == "head_tail":
        return HeadTailAdapter()
    if name == "bm25":
        return BM25Adapter(
            chunk_size=int(spec.get("chunk_size", 128)),
            overlap=int(spec.get("overlap", 32)),
            source_order=bool(spec.get("source_order", True)),
        )
    if name == "embedding_mmr":
        return EmbeddingMMRAdapter(
            model_name=str(
                spec.get(
                    "model_name",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                )
            ),
            chunk_size=int(spec.get("chunk_size", 128)),
            overlap=int(spec.get("overlap", 32)),
            mmr_lambda=float(spec.get("mmr_lambda", 0.7)),
            source_order=bool(spec.get("source_order", True)),
            cache_dir=spec.get("cache_dir"),
            use_gpu=bool(spec.get("use_gpu", False)),
        )
    if name.startswith("trimwise_"):
        return TrimwiseAdapter(
            name.removeprefix("trimwise_"),
            cache_dir=spec.get("cache_dir"),
            use_gpu=bool(spec.get("use_gpu", False)),
        )
    if name in {"llmlingua", "llmlingua_queryless"}:
        return LLMLinguaAdapter(
            str(spec.get("model_name", "openai-community/gpt2")),
            str(spec.get("device_map", "cpu")),
            {"cache_dir": spec["cache_dir"]} if spec.get("cache_dir") else None,
            query_aware=name == "llmlingua",
        )
    if name == "longllmlingua":
        return LongLLMLinguaAdapter(
            str(spec.get("model_name", "openai-community/gpt2")),
            str(spec.get("device_map", "cpu")),
            {"cache_dir": spec["cache_dir"]} if spec.get("cache_dir") else None,
        )
    if name == "llmlingua2":
        return LLMLingua2Adapter(
            str(
                spec.get(
                    "model_name",
                    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                )
            ),
            str(spec.get("device_map", "cpu")),
            {"cache_dir": spec["cache_dir"]} if spec.get("cache_dir") else None,
        )
    if name == "selective_context":
        return SelectiveContextAdapter(
            str(spec.get("model_name", "gpt2")),
            str(spec.get("lang", "en")),
        )
    if name == "recomp_extractive":
        return RecompAdapter(
            str(spec.get("model_name", "fangyuan/nq_extractive_compressor")),
            str(spec.get("device", "cuda")),
        )
    raise ValueError(f"unknown method: {name}")
