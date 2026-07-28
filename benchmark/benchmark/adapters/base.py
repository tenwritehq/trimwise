"""Shared adapter contracts and per-call measurements."""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class CompressionResult:
    """Store one compressor output, status, timing, and diagnostics."""

    method_id: str
    output: str
    status: str = "success"
    latency_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CompressorAdapter(Protocol):
    """Describe the interface used by the compression runner."""

    method_id: str
    query_aware: bool

    def compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        """Compress source context under a token budget."""
        ...

    def close(self) -> None:
        """Release model resources owned by the adapter."""
        ...


class TimedAdapter:
    """Time adapter calls and capture optional GPU memory measurements."""

    method_id = "base"
    query_aware = False
    model_backed = False

    def __init__(self) -> None:
        """Initialize lifecycle fields shared by model-backed adapters."""
        self._model_loaded = False
        self._model_load_ms: float | None = None

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        """Implement one unhandled compression call in a concrete adapter."""
        raise NotImplementedError

    def close(self) -> None:
        """Release non-model resources and encourage prompt cleanup."""
        gc.collect()

    def compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        """Run one compression call and attach lifecycle and memory metadata."""
        started = time.perf_counter_ns()
        cold_call = self.model_backed and not getattr(self, "_model_loaded", False)
        _reset_gpu_peak_memory()
        try:
            result = self._compress(context, query, budget, seed)
        except MemoryError as exc:
            result = CompressionResult(
                method_id=self.method_id,
                output="",
                status="out_of_memory",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            )
        except Exception as exc:  # Every failed invocation must remain auditable.
            result = CompressionResult(
                method_id=self.method_id,
                output="",
                status="runtime_failure",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            )
        result.latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        result.metadata.update(
            {
                "cold_call": cold_call,
                "model_load_ms": getattr(self, "_model_load_ms", None),
                **_gpu_memory_metadata(),
                **clear_cuda_cache(),
            }
        )
        return result


def _reset_gpu_peak_memory() -> None:
    """Reset CUDA peak counters when a CUDA runtime is available."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _gpu_memory_metadata() -> dict[str, Any]:
    """Return measured CUDA peak memory, or an explicit unavailable marker."""
    try:
        import torch
    except ImportError:
        return {"gpu_available": False}
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    try:
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return {"gpu_available": True, "gpu_sync_error": str(exc)}
    return {
        "gpu_available": True,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1_048_576,
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 1_048_576,
    }


def clear_cuda_cache() -> dict[str, Any]:
    """Collect temporary tensors and return cached CUDA memory to the driver."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return {"gpu_cache_cleared": False}
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except RuntimeError as exc:
            return {"gpu_cache_cleared": False, "gpu_cache_error": str(exc)}
        return {
            "gpu_cache_cleared": True,
            "gpu_allocated_after_clear_mb": torch.cuda.memory_allocated() / 1_048_576,
            "gpu_reserved_after_clear_mb": torch.cuda.memory_reserved() / 1_048_576,
        }
    return {"gpu_cache_cleared": False}


def release_model_resources() -> None:
    """Collect released model objects and return cached CUDA memory to the driver."""
    clear_cuda_cache()
