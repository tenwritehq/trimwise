"""Reconstruct exact source fragments, omission markers, spans, and fallbacks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from trimwise.measurement import Measurer
from trimwise.models import SourceSpan
from trimwise.segmentation import Segment

if TYPE_CHECKING:
    from trimwise.selection import _SelectionContext

_OPENING_FENCE_PATTERN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})")
_CLOSING_FENCE_PATTERN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[ \t]*(?:\r?\n)?$")
_PARAGRAPH_BOUNDARY_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")
_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"""(?:[.!?](?:["')\]]*)?(?:[^\S\r\n]+|\r?\n|$)|"""
    r"""[\u2026\u3002\uff01\uff1f](?:["')\]]*)?(?:[^\S\r\n]+|\r?\n|(?=\S)|$))"""
)
_NON_WHITESPACE_PATTERN = re.compile(r"\S")


@dataclass(frozen=True, slots=True)
class _ComposedOutput:
    """Pair composed text with its maximal original-input ranges."""

    text: str
    spans: tuple[SourceSpan, ...]


@dataclass(frozen=True, slots=True)
class _OutputPiece:
    """Represent fixed output text with an optional marked replacement."""

    fallback: str
    marked: str | None = None


@dataclass(frozen=True, slots=True)
class _SourceContext:
    """Provide one source's fallback measurement and marker settings."""

    source: str
    measurer: Measurer
    limit: int
    marker: str


def _compose(
    context: _SelectionContext,
    indexes: set[int],
) -> tuple[_ComposedOutput, ...] | None:
    """Compose input-aligned outputs and add every affordable gap marker.

    Args:
        context: Source and exact budget settings.
        indexes: Candidate indexes to retain.

    Returns:
        Fitting source outputs, or ``None`` when retained content alone is too large.
    """
    if not indexes:
        return tuple(_ComposedOutput("", ()) for _ in context.sources)

    retained: list[list[Segment]] = [[] for _ in context.sources]
    for index in sorted(indexes):
        retained[context.source_indexes[index]].append(context.segments[index])

    piece_groups: list[list[_OutputPiece]] = []
    current_groups: list[list[str]] = []
    outputs: list[_ComposedOutput] = []
    for source, segments in zip(context.sources, retained, strict=True):
        if not segments:
            piece_groups.append([])
            current_groups.append([])
            outputs.append(_ComposedOutput("", ()))
            continue
        pieces = _output_pieces(source, context.marker, segments)
        current = [piece.fallback for piece in pieces]
        piece_groups.append(pieces)
        current_groups.append(current)
        outputs.append(_ComposedOutput("".join(current), _source_spans(source, segments)))

    total_count = _output_count(context.measurer, outputs)
    if total_count > context.limit:
        return None
    for source_index, pieces in enumerate(piece_groups):
        current = current_groups[source_index]
        for piece_index, piece in enumerate(pieces):
            if piece.marked is None:
                continue
            fallback = current[piece_index]
            current[piece_index] = piece.marked
            candidate = "".join(current)
            previous = outputs[source_index]
            candidate_count = (
                total_count
                - _text_count(context.measurer, previous.text)
                + _text_count(context.measurer, candidate)
            )
            if candidate_count <= context.limit:
                outputs[source_index] = _ComposedOutput(candidate, previous.spans)
                total_count = candidate_count
            else:
                current[piece_index] = fallback
    return tuple(outputs)


def _output_count(measurer: Measurer, outputs: Iterable[_ComposedOutput]) -> int:
    """Sum independent source-output measurements.

    Args:
        measurer: Shared unit and optional token counter.
        outputs: Input-aligned source outputs.

    Returns:
        Aggregate count with empty source rows defined as zero.
    """
    return sum(_text_count(measurer, output.text) for output in outputs)


def _text_count(measurer: Measurer, text: str) -> int:
    """Measure nonempty output while keeping empty result rows at zero.

    Args:
        measurer: Shared output measurer.
        text: One source's complete output.

    Returns:
        Independent output count.
    """
    return measurer.count(text) if text else 0


def _source_spans(source: str, segments: list[Segment]) -> tuple[SourceSpan, ...]:
    """Combine retained segments across source whitespace copied into the output.

    Args:
        source: Original input string.
        segments: Retained source segments in order.

    Returns:
        Maximal ordered source-backed ranges.
    """
    first = segments[0]
    start = first.start if _NON_WHITESPACE_PATTERN.search(source, 0, first.start) else 0
    end = first.end
    spans: list[SourceSpan] = []
    for segment in segments[1:]:
        if _NON_WHITESPACE_PATTERN.search(source, end, segment.start):
            spans.append(SourceSpan(start, end))
            start = segment.start
        end = segment.end
    if not _NON_WHITESPACE_PATTERN.search(source, end):
        end = len(source)
    spans.append(SourceSpan(start, end))
    return tuple(spans)


def _output_pieces(
    source: str,
    marker: str,
    segments: list[Segment],
) -> list[_OutputPiece]:
    """Describe fixed fragments and optional marker-bearing source gaps.

    Args:
        source: Original source string.
        marker: Configured omission text.
        segments: Retained segments in source order.

    Returns:
        Alternating source and gap pieces.
    """
    pieces: list[_OutputPiece] = []
    first = segments[0]
    if first.start:
        leading_has_content = _NON_WHITESPACE_PATTERN.search(source, 0, first.start)
        marked = marker + _newlines_before(first.text)
        pieces.append(
            _OutputPiece(
                "" if leading_has_content else source[: first.start],
                marked if leading_has_content else None,
            )
        )
    pieces.append(_OutputPiece(first.text))

    for previous, segment in pairwise(segments):
        gap_has_content = _NON_WHITESPACE_PATTERN.search(
            source,
            previous.end,
            segment.start,
        )
        if gap_has_content:
            pieces.append(
                _OutputPiece(
                    _plain_separator(previous.text, segment.text),
                    _marked_separator(previous.text, marker, segment.text),
                )
            )
        else:
            pieces.append(_OutputPiece(source[previous.end : segment.start]))
        pieces.append(_OutputPiece(segment.text))

    last = segments[-1]
    if last.end < len(source):
        trailing_has_content = _NON_WHITESPACE_PATTERN.search(source, last.end)
        marked = _newlines_after(last.text) + marker
        pieces.append(
            _OutputPiece(
                "" if trailing_has_content else source[last.end :],
                marked if trailing_has_content else None,
            )
        )
    return pieces


def _plain_separator(left: str, right: str) -> str:
    """Create at most one blank line between separated source fragments.

    Args:
        left: Retained fragment before an omitted gap.
        right: Retained fragment after an omitted gap.

    Returns:
        Minimal newline separator.
    """
    needed = max(0, 2 - _trailing_newlines(left) - _leading_newlines(right))
    return "\n" * needed


def _marked_separator(left: str, marker: str, right: str) -> str:
    """Surround an internal omission marker with bounded blank lines.

    Args:
        left: Retained fragment before the marker.
        marker: Configured omission text.
        right: Retained fragment after the marker.

    Returns:
        Marker and required boundary newlines.
    """
    return _newlines_after(left) + marker + _newlines_before(right)


def _newlines_after(text: str) -> str:
    """Supply enough newlines for one blank line after text.

    Args:
        text: Text immediately before a boundary.

    Returns:
        Zero to two newline characters.
    """
    return "\n" * max(0, 2 - _trailing_newlines(text))


def _newlines_before(text: str) -> str:
    """Supply enough newlines for one blank line before text.

    Args:
        text: Text immediately after a boundary.

    Returns:
        Zero to two newline characters.
    """
    return "\n" * max(0, 2 - _leading_newlines(text))


def _trailing_newlines(text: str) -> int:
    """Count at most two trailing newline characters.

    Args:
        text: Boundary text.

    Returns:
        Capped trailing newline count.
    """
    return min(2, len(text) - len(text.rstrip("\n")))


def _leading_newlines(text: str) -> int:
    """Count at most two leading newline characters.

    Args:
        text: Boundary text.

    Returns:
        Capped leading newline count.
    """
    return min(2, len(text) - len(text.lstrip("\n")))


def _fallback_output(context: _SelectionContext) -> tuple[_ComposedOutput, ...]:
    """Retain a measurable prefix of the strongest indivisible candidate.

    Args:
        context: Source, ranking, and budget settings.

    Returns:
        Input-aligned outputs with at most one source-derived fragment.
    """
    if not context.segments:
        return tuple(_ComposedOutput("", ()) for _ in context.sources)
    index = max(
        range(len(context.segments)),
        key=lambda candidate: (context.ranking.relevance[candidate], -candidate),
    )
    return _fallback_candidate_output(context, index)


def _fallback_candidate_output(
    context: _SelectionContext,
    index: int,
) -> tuple[_ComposedOutput, ...]:
    """Retain a measurable prefix of one chosen indivisible candidate.

    Args:
        context: Source, ranking, and budget settings.
        index: Candidate selected for bounded-prefix fallback.

    Returns:
        Input-aligned outputs with the fragment in its original source row.
    """
    empty = tuple(_ComposedOutput("", ()) for _ in context.sources)
    segment = context.segments[index]
    source_index = context.source_indexes[index]
    source_context = _SourceContext(
        context.sources[source_index],
        context.measurer,
        context.limit,
        context.marker,
    )
    fragment = _fitting_segment(source_context, segment)
    if not fragment.text:
        return empty
    outputs = list(empty)
    outputs[source_index] = _add_fallback_markers(source_context, segment, fragment)
    return tuple(outputs)


def _fitting_segment(context: _SourceContext, segment: Segment) -> _ComposedOutput:
    """Shrink one segment while retaining balanced closed fences where possible.

    Args:
        context: Measurement and limit settings.
        segment: Strongest complete candidate.

    Returns:
        Fitting source-derived text and original-input ranges.
    """
    if segment.kind != "fence":
        return _fitting_segment_prefix(context, segment)
    lines = segment.text.splitlines(keepends=True)
    if len(lines) < 2 or not _matching_fences(lines[0], lines[-1]):
        return _fitting_segment_prefix(context, segment)
    opening = lines[0]
    closing = lines[-1]
    shell = opening + closing
    if context.measurer.count(shell) > context.limit:
        return _fitting_segment_prefix(context, segment)
    body = "".join(lines[1:-1])
    endpoints = _line_endpoints(body)
    for end in reversed(endpoints):
        candidate = opening + body[:end] + closing
        if context.measurer.count(candidate) <= context.limit:
            prefix_end = segment.start + len(opening) + end
            spans = (
                SourceSpan(segment.start, prefix_end),
                SourceSpan(segment.end - len(closing), segment.end),
            )
            return _ComposedOutput(candidate, spans)
    spans = (
        SourceSpan(segment.start, segment.start + len(opening)),
        SourceSpan(segment.end - len(closing), segment.end),
    )
    return _ComposedOutput(shell, spans)


def _fitting_segment_prefix(
    context: _SourceContext,
    segment: Segment,
) -> _ComposedOutput:
    """Fit one exact segment prefix and adjust its source range.

    Args:
        context: Measurement and limit settings.
        segment: Oversized source candidate.

    Returns:
        Fitting prefix and its original-input range.
    """
    text = _fitting_plain_prefix(context, segment.text)
    spans = (SourceSpan(segment.start, segment.start + len(text)),) if text else ()
    return _ComposedOutput(text, spans)


def _fitting_plain_prefix(context: _SourceContext, text: str) -> str:
    """Prefer complete structural boundaries before an arbitrary source prefix.

    Args:
        context: Measurement and limit settings.
        text: Oversized candidate source.

    Returns:
        Largest preferred complete prefix, or the longest measurable prefix.
    """
    paragraphs = (match.end() for match in _PARAGRAPH_BOUNDARY_PATTERN.finditer(text))
    prefix = _fitting_boundary_prefix(context, text, paragraphs)
    if prefix:
        return prefix
    prefix = _fitting_boundary_prefix(context, text, _complete_unit_endpoints(text))
    return prefix or context.measurer.fitting_prefix(text, context.limit)


def _fitting_boundary_prefix(
    context: _SourceContext,
    text: str,
    endpoints: Iterable[int],
) -> str:
    """Find the longest fitting prefix at preferred source boundaries.

    Args:
        context: Measurement and limit settings.
        text: Oversized candidate source.
        endpoints: Exclusive candidate boundary offsets.

    Returns:
        Longest fitting boundary prefix, or an empty string when none fits.
    """
    for end in sorted(set(endpoints), reverse=True):
        if 0 < end < len(text) and context.measurer.count(text[:end]) <= context.limit:
            return text[:end]
    return ""


def _line_endpoints(text: str) -> list[int]:
    """Return exclusive ends for complete source lines.

    Args:
        text: Source text to inspect.

    Returns:
        Ordered line-end offsets.
    """
    endpoints: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        offset += len(line)
        endpoints.append(offset)
    return endpoints


def _complete_unit_endpoints(text: str) -> list[int]:
    """Return ordered sentence and source-line ends.

    Args:
        text: Source text to inspect.

    Returns:
        Unique exclusive offsets for every complete unit.
    """
    return sorted(
        {
            *(match.end() for match in _SENTENCE_BOUNDARY_PATTERN.finditer(text)),
            *_line_endpoints(text),
        }
    )


def _matching_fences(opening: str, closing: str) -> bool:
    """Check whether two source lines form a compatible fenced block.

    Args:
        opening: First fence line.
        closing: Last fence line.

    Returns:
        Whether the closing marker matches the opener's character and length.
    """
    opening_match = _OPENING_FENCE_PATTERN.match(opening)
    closing_match = _CLOSING_FENCE_PATTERN.match(closing)
    if opening_match is None or closing_match is None:
        return False
    opening_marker = opening_match.group(1)
    closing_marker = closing_match.group(1)
    return opening_marker[0] == closing_marker[0] and len(closing_marker) >= len(opening_marker)


def _add_fallback_markers(
    context: _SourceContext,
    segment: Segment,
    fragment: _ComposedOutput,
) -> _ComposedOutput:
    """Add affordable leading and trailing markers around fallback content.

    Args:
        context: Source, marker, and measurement settings.
        segment: Candidate from which the fragment was derived.
        fragment: Fitting candidate content and source ranges.

    Returns:
        Fitting fragment with every affordable outer omission marker.
    """
    output = fragment.text
    if context.source[: segment.start].strip():
        candidate = context.marker + _newlines_before(output) + output
        if context.measurer.count(candidate) <= context.limit:
            output = candidate
    has_trailing_omission = fragment.text != segment.text or bool(
        context.source[segment.end :].strip()
    )
    if has_trailing_omission:
        candidate = output + _newlines_after(output) + context.marker
        if context.measurer.count(candidate) <= context.limit:
            output = candidate
    return _ComposedOutput(output, fragment.spans)
