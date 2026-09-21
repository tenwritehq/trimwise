"""Create privacy-safe OpenTelemetry spans around public trimming operations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from trimwise.models import BudgetUnit, ContextTrimResult, Strategy, TrimResult

_VERSION: str | None
try:
    _VERSION = version("trimwise")
except PackageNotFoundError:
    _VERSION = None

_TRACER = trace.get_tracer("trimwise", _VERSION)
_UNITS = {unit.value for unit in BudgetUnit}
_STRATEGIES = {strategy.value for strategy in Strategy}


def _request_attributes(
    limit: object,
    unit: object,
    strategy: object,
) -> dict[str, str | int]:
    """Keep only validated, bounded request values for a span.

    Args:
        limit: Requested output limit.
        unit: Requested measurement unit.
        strategy: Requested ranking strategy.

    Returns:
        Safe attributes available before the trimming call completes.
    """
    attributes: dict[str, str | int] = {}
    if isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0:
        attributes["trimwise.limit"] = limit

    unit_value = unit.value if isinstance(unit, BudgetUnit) else unit
    if isinstance(unit_value, str) and unit_value in _UNITS:
        attributes["trimwise.unit"] = unit_value

    strategy_value = strategy.value if isinstance(strategy, Strategy) else strategy
    if isinstance(strategy_value, str) and strategy_value in _STRATEGIES:
        attributes["trimwise.strategy.requested"] = strategy_value
    return attributes


@contextmanager
def _span(name: str, attributes: Mapping[str, str | int]) -> Iterator[Span]:
    """Create a span that records bounded failure types without exception content.

    Args:
        name: Stable public operation name.
        attributes: Validated request attributes.

    Yields:
        Current OpenTelemetry span for result attributes.

    Raises:
        BaseException: Re-raises any trimming failure unchanged.
    """
    with _TRACER.start_as_current_span(
        name,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as error:
            error_type = type(error)
            span.set_attribute(
                "error.type",
                f"{error_type.__module__}.{error_type.__qualname__}",
            )
            span.set_status(Status(StatusCode.ERROR))
            raise


def _record_result(span: Span, result: TrimResult | ContextTrimResult) -> None:
    """Add common bounded result values to a completed operation span.

    Args:
        span: Span that represents the public operation.
        result: Successful single-source or shared-context result.
    """
    span.set_attribute("trimwise.strategy.resolved", result.strategy.value)
    span.set_attribute("trimwise.unit", result.unit.value)
    span.set_attribute("trimwise.limit", result.limit)
    span.set_attribute("trimwise.input.count", result.input_count)
    span.set_attribute("trimwise.output.count", result.output_count)
    span.set_attribute("trimwise.trimmed", result.trimmed)
    if isinstance(result, ContextTrimResult):
        span.set_attribute("trimwise.source.count", len(result.sources))


def _record_batch_result(span: Span, results: Sequence[TrimResult]) -> None:
    """Add bounded aggregate values to a completed batch span.

    Args:
        span: Span that represents the batch operation.
        results: Ordered successful batch results.
    """
    span.set_attribute("trimwise.batch.size", len(results))
    span.set_attribute("trimwise.trimmed", any(result.trimmed for result in results))
