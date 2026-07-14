"""Convert Markdown or plain text into non-overlapping source-backed units."""

from __future__ import annotations

from dataclasses import dataclass, replace

from markdown_it import MarkdownIt

_BLOCK_TYPES = {
    "code_block": "code",
    "fence": "fence",
    "heading_open": "heading",
    "hr": "rule",
    "html_block": "html",
    "paragraph_open": "paragraph",
    "table_open": "table",
}
_CONTAINER_TYPES = {
    "blockquote_open": "blockquote",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
}
_CONTAINER_END_TYPES = {token_type.replace("_open", "_close") for token_type in _CONTAINER_TYPES}
_KIND_PRIORITY = {"heading": 0, "fence": 1, "table": 2, "code": 3, "html": 4}


@dataclass(frozen=True, slots=True)
class Segment:
    """Represent one candidate with an exact span in the original source.

    Attributes:
        index: Stable position in the candidate sequence.
        start: Inclusive source character offset.
        end: Exclusive source character offset.
        text: Exact source slice for the span.
        kind: Structural Markdown block kind.
        section: Heading-delimited section number.
        heading_index: Nearest preceding heading candidate, if present.
    """

    index: int
    start: int
    end: int
    text: str
    kind: str
    section: int
    heading_index: int | None


def segment_text(text: str) -> list[Segment]:
    """Split source text into complete Markdown-aware candidate units.

    Args:
        text: Whole source string.

    Returns:
        Ordered non-overlapping source segments.
    """
    if not text:
        return []
    if not text.strip():
        return [Segment(0, 0, len(text), text, "raw", 0, None)]

    line_offsets = _line_offsets(text)
    parser = MarkdownIt("commonmark").enable("table")
    front_matter_end = _front_matter_end(text)
    mapped_blocks = [(0, front_matter_end, "raw")] if front_matter_end is not None else []
    containers: list[str] = []
    for token in parser.parse(text):
        if token.type in _CONTAINER_TYPES:
            containers.append(_CONTAINER_TYPES[token.type])
            continue
        if token.type in _CONTAINER_END_TYPES:
            containers.pop()
            continue
        if token.type not in _BLOCK_TYPES or token.map is None:
            continue
        start_line, end_line = token.map
        if front_matter_end is not None and line_offsets[start_line] < front_matter_end:
            continue
        kind = _BLOCK_TYPES[token.type]
        if kind == "paragraph" and containers:
            kind = containers[-1]
        mapped_blocks.append((line_offsets[start_line], line_offsets[end_line], kind))

    spans = _deduplicate_spans(mapped_blocks)
    complete_spans = _with_uncovered_source(text, spans)
    return _assign_sections(text, complete_spans)


def _front_matter_end(text: str) -> int | None:
    """Find a leading YAML-style front matter block.

    Args:
        text: Original source text.

    Returns:
        Exclusive closing-line offset, or ``None`` when no closed block exists.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None
    offset = len(lines[0])
    for line in lines[1:]:
        offset += len(line)
        if line.strip() in {"---", "..."}:
            return offset
    return None


def _line_offsets(text: str) -> list[int]:
    """Map Markdown line numbers to source character offsets.

    Args:
        text: Original source text.

    Returns:
        Offset for every line start plus the final source end.
    """
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _deduplicate_spans(blocks: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Remove duplicate and nested parser spans while preferring richer blocks.

    Args:
        blocks: Parser-derived source spans.

    Returns:
        Ordered, non-overlapping spans.
    """
    unique: dict[tuple[int, int], str] = {}
    for start, end, kind in blocks:
        existing = unique.get((start, end))
        if existing is None or _KIND_PRIORITY.get(kind, 99) < _KIND_PRIORITY.get(existing, 99):
            unique[(start, end)] = kind

    chosen: list[tuple[int, int, str]] = []
    for (start, end), kind in sorted(unique.items()):
        if chosen and start < chosen[-1][1]:
            previous = chosen[-1]
            if end <= previous[1]:
                continue
            start = previous[1]
        if start < end:
            chosen.append((start, end, kind))
    return chosen


def _with_uncovered_source(
    text: str,
    spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Add nonblank source ranges not represented by parser block tokens.

    Args:
        text: Original source text.
        spans: Ordered parser-backed spans.

    Returns:
        Parser spans plus raw uncovered ranges.
    """
    complete: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end, kind in spans:
        if text[cursor:start].strip():
            complete.append((cursor, start, "raw"))
        complete.append((start, end, kind))
        cursor = end
    if text[cursor:].strip():
        complete.append((cursor, len(text), "raw"))
    return complete


def _assign_sections(
    text: str,
    spans: list[tuple[int, int, str]],
) -> list[Segment]:
    """Attach heading-delimited section metadata to candidate spans.

    Args:
        text: Original source text.
        spans: Complete ordered source spans.

    Returns:
        Indexed segments with heading relationships.
    """
    segments: list[Segment] = []
    section = 0
    heading_index: int | None = None
    for index, (start, end, kind) in enumerate(spans):
        if kind == "heading":
            section += 1
            heading_index = index
        segment = Segment(index, start, end, text[start:end], kind, section, heading_index)
        if kind == "heading":
            segment = replace(segment, heading_index=None)
        segments.append(segment)
    return segments
