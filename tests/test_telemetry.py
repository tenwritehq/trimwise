"""Verify optional OpenTelemetry tracing and its privacy contract."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from importlib.metadata import version

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from trimwise import ContextSource, TrimInput, Trimmer, telemetry


@pytest.fixture
def span_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """Capture Trimwise spans without changing the process-global provider.

    Args:
        monkeypatch: Pytest patch helper.

    Yields:
        Isolated provider and in-memory exporter.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_TRACER", provider.get_tracer("trimwise", version("trimwise")))
    yield provider, exporter
    provider.shutdown()


def test_trim_span_is_parented_and_records_safe_result_attributes(
    span_recorder: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """Describe a successful trim without recording source or query text.

    Args:
        span_recorder: Isolated provider and captured-span exporter.
    """
    provider, exporter = span_recorder
    app_tracer = provider.get_tracer("example-app")
    with app_tracer.start_as_current_span("answer.request") as parent:
        parent_id = parent.get_span_context().span_id
        result = Trimmer().trim(
            "Keep this. Remove the rest of this source.",
            10,
            unit="characters",
            strategy="auto",
            query="private question",
        )

    spans = {span.name: span for span in exporter.get_finished_spans()}
    span = spans["trimwise.trim"]
    assert span.parent is not None and span.parent.span_id == parent_id
    assert span.instrumentation_scope is not None
    assert span.instrumentation_scope.name == "trimwise"
    assert span.instrumentation_scope.version == version("trimwise")
    assert span.attributes == {
        "trimwise.limit": 10,
        "trimwise.unit": "characters",
        "trimwise.strategy.requested": "auto",
        "trimwise.strategy.resolved": "lexical",
        "trimwise.input.count": 42,
        "trimwise.output.count": result.output_count,
        "trimwise.trimmed": True,
    }
    assert "private question" not in str(span.attributes)


def test_failure_span_omits_exception_content(
    span_recorder: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """Record a bounded error type without exception events or secret values.

    Args:
        span_recorder: Isolated provider and captured-span exporter.
    """
    _, exporter = span_recorder
    secret = "customer-secret-91"

    def failing_counter(_: str) -> int:
        """Raise a message that telemetry must never retain."""
        raise RuntimeError(secret)

    with pytest.raises(RuntimeError, match=secret):
        Trimmer().trim(
            secret,
            2,
            strategy="auto",
            query=secret,
            token_counter=failing_counter,
        )

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
    assert span.events == ()
    assert span.attributes == {
        "trimwise.limit": 2,
        "trimwise.unit": "tokens",
        "trimwise.strategy.requested": "auto",
        "error.type": "builtins.RuntimeError",
    }
    assert secret not in str(span.attributes)


def test_validation_failure_drops_invalid_attribute_values(
    span_recorder: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """Reject unsafe enum-like values without copying them into telemetry.

    Args:
        span_recorder: Isolated provider and captured-span exporter.
    """
    _, exporter = span_recorder
    invalid_unit = "private-unit-value"

    with pytest.raises(ValueError, match="unsupported budget unit"):
        Trimmer().trim("private source", 2, unit=invalid_unit)

    (span,) = exporter.get_finished_spans()
    assert span.attributes == {
        "trimwise.limit": 2,
        "trimwise.strategy.requested": "auto",
        "error.type": "builtins.ValueError",
    }
    assert invalid_unit not in str(span.attributes)


def test_context_span_records_source_count(
    span_recorder: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """Describe shared-context work with one bounded source count.

    Args:
        span_recorder: Isolated provider and captured-span exporter.
    """
    _, exporter = span_recorder
    result = Trimmer().trim_context(
        [ContextSource("alpha beta"), ContextSource("gamma delta")],
        3,
        unit="words",
    )

    (span,) = exporter.get_finished_spans()
    assert span.name == "trimwise.trim_context"
    assert span.attributes is not None
    assert span.attributes["trimwise.source.count"] == 2
    assert span.attributes["trimwise.input.count"] == result.input_count
    assert span.attributes["trimwise.output.count"] == result.output_count


@pytest.mark.asyncio
async def test_async_callback_span_keeps_trim_parent(
    span_recorder: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """Keep application callback spans below the asynchronous Trimwise span.

    Args:
        span_recorder: Isolated provider and captured-span exporter.
    """
    provider, exporter = span_recorder
    app_tracer = provider.get_tracer("example-app")

    async def embed(
        _: str,
        passages: Sequence[str],
    ) -> tuple[list[float], list[list[float]]]:
        """Create one application-owned child span while returning valid vectors."""
        with app_tracer.start_as_current_span("embedding.request"):
            return [1.0], [[1.0] for _ in passages]

    await Trimmer(async_embedding_callback=embed).atrim(
        "Alpha evidence. Beta evidence. Gamma evidence.",
        12,
        unit="characters",
        strategy="semantic",
        query="Alpha",
    )

    spans = {span.name: span for span in exporter.get_finished_spans()}
    trim_span = spans["trimwise.trim"]
    embed_span = spans["embedding.request"]
    assert embed_span.parent is not None
    assert embed_span.parent.span_id == trim_span.context.span_id


@pytest.mark.asyncio
async def test_batch_emits_one_operation_span_without_item_spans(
    span_recorder: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """Keep batch tracing useful without creating one span per item.

    Args:
        span_recorder: Isolated provider and captured-span exporter.
    """
    _, exporter = span_recorder
    results = await Trimmer().atrim_many(
        [
            TrimInput("short", 10, unit="characters"),
            TrimInput("longer source", 4, unit="characters"),
        ]
    )

    (span,) = exporter.get_finished_spans()
    assert span.name == "trimwise.trim_many"
    assert span.attributes == {
        "trimwise.batch.size": 2,
        "trimwise.trimmed": any(result.trimmed for result in results),
    }


@pytest.mark.asyncio
async def test_cancelled_async_call_closes_error_span(
    span_recorder: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """End the operation span when cancellation reaches an async callback.

    Args:
        span_recorder: Isolated provider and captured-span exporter.
    """
    _, exporter = span_recorder
    started = asyncio.Event()

    async def embed(
        _: str,
        passages: Sequence[str],
    ) -> tuple[list[float], list[list[float]]]:
        """Wait indefinitely so the test can cancel the public call."""
        started.set()
        await asyncio.Event().wait()
        return [1.0], [[1.0] for _ in passages]

    task = asyncio.create_task(
        Trimmer(async_embedding_callback=embed).atrim(
            "Alpha evidence. Beta evidence. Gamma evidence.",
            12,
            unit="characters",
            strategy="semantic",
            query="Alpha",
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == "asyncio.exceptions.CancelledError"
    assert span.events == ()
