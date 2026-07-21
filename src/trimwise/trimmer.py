"""Orchestrate validation, ranking, selection, composition, and async use."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import cast

from trimwise.measurement import Measurer, TokenCounter
from trimwise.models import BudgetUnit, SourceSpan, Strategy, TrimConfig, TrimResult
from trimwise.ranking import (
    CandidateRanking,
    _contextual_ranking_texts,
    rank_hybrid,
    rank_lexical,
    rank_semantic,
    rank_structural,
)
from trimwise.segmentation import Segment, segment_text
from trimwise.semantic import (
    AsyncEmbeddingCallback,
    EmbeddingCallback,
    EmbeddingOutput,
    SemanticEmbedder,
    _SemanticVectors,
    embed_with_callback,
    invoke_async_embedding_callback,
    normalize_callback_output,
)

_OPENING_FENCE_PATTERN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})")
_CLOSING_FENCE_PATTERN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[ \t]*(?:\r?\n)?$")
_PARAGRAPH_BOUNDARY_PATTERN = re.compile(r"(?:\r?\n[ \t]*){2,}")
_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"""(?:[.!?](?:["')\]]*)?(?:[^\S\r\n]+|\r?\n|$)|"""
    r"""[\u2026\u3002\uff01\uff1f](?:["')\]]*)?(?:[^\S\r\n]+|\r?\n|(?=\S)|$))"""
)
_NON_WHITESPACE_PATTERN = re.compile(r"\S")


@dataclass(frozen=True, slots=True)
class _TrimRequest:
    """Collect one validated public call for internal processing."""

    text: str
    limit: int
    unit: BudgetUnit
    strategy: Strategy
    query: str | None
    token_counter: TokenCounter | None


@dataclass(frozen=True, slots=True)
class _TrimArguments:
    """Collect one unvalidated public call before worker-thread processing."""

    text: str
    limit: int
    unit: BudgetUnit | str
    strategy: Strategy | str
    query: str | None
    token_counter: TokenCounter | None


@dataclass(frozen=True, slots=True)
class _PreparedTrim:
    """Hold measured and segmented input until ranking is available."""

    request: _TrimRequest
    input_count: int
    segments: list[Segment]
    measurer: Measurer


@dataclass(frozen=True, slots=True)
class _RankingRequest:
    """Group the inputs needed to rank one segmented document."""

    segments: list[Segment]
    strategy: Strategy
    query: str | None
    measurer: Measurer


@dataclass(frozen=True, slots=True)
class _ComposedOutput:
    """Pair composed text with its maximal original-input ranges."""

    text: str
    spans: tuple[SourceSpan, ...]


@dataclass(frozen=True, slots=True)
class _SelectionContext:
    """Provide immutable source, budget, and ranking data to selection."""

    source: str
    segments: list[Segment]
    ranking: CandidateRanking
    measurer: Measurer
    limit: int
    marker: str
    mmr_lambda: float


@dataclass(slots=True)
class _SelectionState:
    """Track accepted and eligible candidates during greedy selection."""

    context: _SelectionContext
    remaining: set[int]
    maximum_similarities: Sequence[float]
    selected: set[int] = field(default_factory=set)
    output: _ComposedOutput = field(default_factory=lambda: _ComposedOutput("", ()))

    def track_mmr_selection(self, selected_index: int) -> None:
        """Update remaining candidates after one MMR selection.

        Args:
            selected_index: Newly retained main candidate.
        """
        ranking = self.context.ranking
        if ranking.maximum_similarity_update is not None:
            ranking.maximum_similarity_update(self.maximum_similarities, selected_index)
            return
        maximum_similarities = cast(MutableSequence[float], self.maximum_similarities)
        # ponytail: exact O(selected x candidates); use approximate neighbors only after profiling.
        for index in self.remaining:
            maximum_similarities[index] = max(
                maximum_similarities[index],
                ranking.similarity(index, selected_index),
            )


@dataclass(frozen=True, slots=True)
class _OutputPiece:
    """Represent fixed output text with an optional marked replacement."""

    fallback: str
    marked: str | None = None


class Trimmer:
    """Reuse trimming configuration and an optional semantic backend."""

    def __init__(
        self,
        config: TrimConfig | None = None,
        *,
        embedding_callback: EmbeddingCallback | None = None,
        async_embedding_callback: AsyncEmbeddingCallback | None = None,
    ) -> None:
        """Create a trimmer with validated reusable dependencies.

        Args:
            config: Optional custom configuration.
            embedding_callback: Optional synchronous query-and-passage embedder.
            async_embedding_callback: Optional asynchronous query-and-passage embedder.

        Raises:
            TypeError: If a supplied callback is not callable.
            ValueError: If both callback execution models are supplied.
        """
        if embedding_callback is not None and not callable(embedding_callback):
            raise TypeError("embedding_callback must be callable or None")
        if async_embedding_callback is not None and not callable(async_embedding_callback):
            raise TypeError("async_embedding_callback must be callable or None")
        if embedding_callback is not None and async_embedding_callback is not None:
            raise ValueError("only one embedding callback may be supplied")
        self.config = config or TrimConfig()
        self._embedding_callback = embedding_callback
        self._async_embedding_callback = async_embedding_callback
        self._semantic = SemanticEmbedder(self.config)

    def trim(
        self,
        text: str,
        limit: int,
        *,
        unit: BudgetUnit | str = BudgetUnit.TOKENS,
        strategy: Strategy | str = Strategy.AUTO,
        query: str | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> TrimResult:
        """Retain high-signal source fragments within an exact budget.

        Args:
            text: Whole source string to trim.
            limit: Maximum output size in ``unit``.
            unit: Token, whitespace-word, or code-point character budget.
            strategy: Structural, lexical, semantic, hybrid, or automatic ranking.
            query: Task or question required by query-aware strategies.
            token_counter: Optional synchronous token measurement callback.

        Returns:
            Measured extractive trimming result.

        Raises:
            TypeError: If an argument is invalid or only an async embedder is available.
            ValueError: If an argument value or strategy/query combination is invalid.
            SemanticBackendError: If an explicitly requested semantic backend fails.
        """
        return self._trim(_TrimArguments(text, limit, unit, strategy, query, token_counter))

    async def atrim(
        self,
        text: str,
        limit: int,
        *,
        unit: BudgetUnit | str = BudgetUnit.TOKENS,
        strategy: Strategy | str = Strategy.AUTO,
        query: str | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> TrimResult:
        """Run CPU and synchronous work outside the event loop.

        A configured asynchronous embedding callback is awaited on the calling event loop.
        Cancellation propagates to that callback, but cannot stop a worker thread already running.

        Args:
            text: Whole source string to trim.
            limit: Maximum output size in ``unit``.
            unit: Token, whitespace-word, or code-point character budget.
            strategy: Structural, lexical, semantic, hybrid, or automatic ranking.
            query: Task or question required by query-aware strategies.
            token_counter: Optional synchronous token measurement callback.

        Returns:
            Measured extractive trimming result.

        Raises:
            TypeError: If an argument has an unsupported type.
            ValueError: If an argument value or strategy/query combination is invalid.
            SemanticBackendError: If an explicitly requested semantic backend fails.
        """
        arguments = _TrimArguments(text, limit, unit, strategy, query, token_counter)
        callback = self._async_embedding_callback
        if callback is None:
            return await asyncio.to_thread(self._trim, arguments)

        prepared = await asyncio.to_thread(self._prepare, arguments)
        if isinstance(prepared, TrimResult):
            return prepared
        if prepared.request.strategy not in {Strategy.SEMANTIC, Strategy.HYBRID}:
            return await asyncio.to_thread(self._complete, prepared)

        query_text = prepared.request.query or ""
        passages = await asyncio.to_thread(_contextual_ranking_texts, prepared.segments)
        output = await invoke_async_embedding_callback(callback, query_text, passages)
        return await asyncio.to_thread(self._complete_with_embedding_output, prepared, output)

    def _trim(self, arguments: _TrimArguments) -> TrimResult:
        """Run one synchronous call through preparation, ranking, and selection.

        Args:
            arguments: Unvalidated public call values.

        Returns:
            Measured extractive trimming result.
        """
        prepared = self._prepare(arguments)
        if isinstance(prepared, TrimResult):
            return prepared
        return self._complete(prepared)

    def _prepare(self, arguments: _TrimArguments) -> TrimResult | _PreparedTrim:
        """Validate, measure, and segment without invoking a semantic backend.

        Args:
            arguments: Unvalidated public call values.

        Returns:
            An early result or prepared long input awaiting ranking.
        """
        resolved_unit = _parse_unit(arguments.unit)
        resolved_strategy, normalized_query = _resolve_strategy(
            _parse_strategy(arguments.strategy),
            arguments.query,
        )
        request = _TrimRequest(
            arguments.text,
            arguments.limit,
            resolved_unit,
            resolved_strategy,
            normalized_query,
            arguments.token_counter,
        )
        _validate_request(request)
        measurer = Measurer(
            resolved_unit,
            self.config.token_encoding,
            arguments.token_counter,
        )
        input_count = measurer.count(arguments.text)
        if arguments.limit == 0:
            return _result(
                _ComposedOutput("", ()), input_count, request, resolved_strategy, measurer
            )
        if input_count <= arguments.limit:
            spans = (SourceSpan(0, len(arguments.text)),) if arguments.text else ()
            output = _ComposedOutput(arguments.text, spans)
            return _result(output, input_count, request, resolved_strategy, measurer)

        segments = segment_text(arguments.text)
        if resolved_strategy is Strategy.STRUCTURAL:
            segments = _expand_structural_plaintext(segments)
        return _PreparedTrim(request, input_count, segments, measurer)

    def _complete(self, prepared: _PreparedTrim) -> TrimResult:
        """Rank and select one prepared input through a synchronous backend.

        Args:
            prepared: Validated, measured, and segmented input.

        Returns:
            Measured extractive trimming result.
        """
        request = _ranking_request(prepared)
        return self._select(prepared, self._rank(request))

    def _complete_with_embedding_output(
        self,
        prepared: _PreparedTrim,
        output: EmbeddingOutput,
    ) -> TrimResult:
        """Normalize asynchronous callback output before ranking and selection.

        Args:
            prepared: Validated, measured, and segmented input.
            output: Caller-provided query and passage vectors.

        Returns:
            Measured extractive trimming result.
        """
        vectors = normalize_callback_output(output, len(prepared.segments))
        ranking = _rank_with_semantic_vectors(_ranking_request(prepared), vectors)
        return self._select(prepared, ranking)

    def _rank(self, request: _RankingRequest) -> CandidateRanking:
        """Dispatch to the resolved ranking algorithm without a strategy hierarchy.

        Args:
            request: Segments and resolved ranking inputs.

        Returns:
            Strategy-specific candidate ranking.
        """
        if request.strategy is Strategy.STRUCTURAL:
            return rank_structural(request.segments, request.measurer)
        if request.strategy is Strategy.LEXICAL:
            return rank_lexical(request.segments, request.query or "", request.measurer)

        if self._async_embedding_callback is not None:
            raise TypeError("trim() cannot use async_embedding_callback; use atrim()")
        query = request.query or ""
        passages = _contextual_ranking_texts(request.segments)
        if self._embedding_callback is not None:
            vectors = embed_with_callback(self._embedding_callback, query, passages)
        else:
            vectors = self._semantic.embed(query, passages)
        return _rank_with_semantic_vectors(request, vectors)

    def _select(
        self,
        prepared: _PreparedTrim,
        ranking: CandidateRanking,
    ) -> TrimResult:
        """Select, compose, and measure ranked source fragments.

        Args:
            prepared: Validated, measured, and segmented input.
            ranking: Strategy-specific candidate scores and similarity behavior.

        Returns:
            Final measured extractive result.
        """
        request = prepared.request
        context = _SelectionContext(
            request.text,
            prepared.segments,
            ranking,
            prepared.measurer,
            request.limit,
            self.config.omission_marker,
            self.config.mmr_lambda,
        )
        output = (
            _select_structural(context)
            if request.strategy is Strategy.STRUCTURAL
            else _select_query_aware(context)
        )
        if output is None:
            output = _fallback_output(context)
        return _result(
            output,
            prepared.input_count,
            request,
            request.strategy,
            prepared.measurer,
        )


def _ranking_request(prepared: _PreparedTrim) -> _RankingRequest:
    """Build ranking inputs from one prepared trim.

    Args:
        prepared: Validated, measured, and segmented input.

    Returns:
        Strategy-specific ranking request.
    """
    request = prepared.request
    return _RankingRequest(
        prepared.segments,
        request.strategy,
        request.query,
        prepared.measurer,
    )


def _rank_with_semantic_vectors(
    request: _RankingRequest,
    vectors: _SemanticVectors,
) -> CandidateRanking:
    """Apply semantic vectors to semantic or hybrid candidate ranking.

    Args:
        request: Semantic or hybrid ranking request.
        vectors: Normalized query and passage matrix.

    Returns:
        Semantic ranking or lexical-semantic fusion.
    """
    semantic = rank_semantic(request.segments, vectors)
    if request.strategy is Strategy.SEMANTIC:
        return semantic
    lexical = rank_lexical(request.segments, request.query or "", request.measurer)
    return rank_hybrid(request.segments, lexical, semantic)


def _parse_unit(value: BudgetUnit | str) -> BudgetUnit:
    """Normalize a public budget unit.

    Args:
        value: Enum member or exact lowercase value.

    Returns:
        Normalized unit.

    Raises:
        ValueError: If the unit is unsupported.
    """
    try:
        return BudgetUnit(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported budget unit: {value!r}") from error


def _parse_strategy(value: Strategy | str) -> Strategy:
    """Normalize a public ranking strategy.

    Args:
        value: Enum member or exact lowercase value.

    Returns:
        Normalized strategy.

    Raises:
        ValueError: If the strategy is unsupported.
    """
    try:
        return Strategy(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported strategy: {value!r}") from error


def _resolve_strategy(strategy: Strategy, query: str | None) -> tuple[Strategy, str | None]:
    """Resolve automatic behavior and validate query-aware intent.

    Args:
        strategy: Requested public strategy.
        query: Optional raw task or question.

    Returns:
        Concrete strategy and stripped query.

    Raises:
        TypeError: If query is not text.
        ValueError: If a query-aware strategy has no usable query.
    """
    if query is not None and not isinstance(query, str):
        raise TypeError("query must be a string or None")
    normalized_query = query.strip() if query is not None else None
    has_query = bool(normalized_query)
    if strategy is Strategy.AUTO:
        return (Strategy.LEXICAL, normalized_query) if has_query else (Strategy.STRUCTURAL, None)
    if strategy in {Strategy.LEXICAL, Strategy.SEMANTIC, Strategy.HYBRID} and not has_query:
        raise ValueError(f"strategy {strategy.value!r} requires a nonblank query")
    return strategy, normalized_query


def _validate_request(request: _TrimRequest) -> None:
    """Validate runtime types and cross-field budget constraints.

    Args:
        request: Normalized trim request.

    Raises:
        TypeError: If text, limit, or callback has an unsupported type.
        ValueError: If the limit is negative or callback unit is incompatible.
    """
    if not isinstance(request.text, str):
        raise TypeError("text must be a string")
    if isinstance(request.limit, bool) or not isinstance(request.limit, int):
        raise TypeError("limit must be an integer")
    if request.limit < 0:
        raise ValueError("limit must not be negative")
    if request.token_counter is not None and not callable(request.token_counter):
        raise TypeError("token_counter must be callable or None")
    if request.token_counter is not None and request.unit is not BudgetUnit.TOKENS:
        raise ValueError("token_counter is only valid for token budgets")


def _result(
    output: _ComposedOutput,
    input_count: int,
    request: _TrimRequest,
    strategy: Strategy,
    measurer: Measurer,
) -> TrimResult:
    """Measure and construct a result while enforcing the hard limit.

    Args:
        output: Composed output and original-input ranges.
        input_count: Measured original size.
        request: Validated public request.
        strategy: Resolved concrete strategy.
        measurer: Configured output measurer.

    Returns:
        Immutable public result.

    Raises:
        RuntimeError: If internal composition exceeded the requested limit.
    """
    output_count = measurer.count(output.text)
    if output_count > request.limit:
        raise RuntimeError("internal composition exceeded the requested limit")
    return TrimResult(
        output.text,
        input_count,
        output_count,
        request.limit,
        request.unit,
        strategy,
        output.text != request.text,
        output.spans,
    )


def _new_selection_state(context: _SelectionContext) -> _SelectionState:
    """Initialize mutable state for one greedy selection pass.

    Args:
        context: Immutable selection inputs.

    Returns:
        Empty state with every candidate eligible.
    """
    return _SelectionState(
        context,
        set(range(len(context.segments))),
        context.ranking.new_maximum_similarities(),
    )


def _expand_structural_plaintext(segments: list[Segment]) -> list[Segment]:
    """Make complete units inside one plain-text paragraph rankable.

    Args:
        segments: Markdown-aware source candidates.

    Returns:
        Sentence- or line-level candidates when the sole paragraph can be split.
    """
    if len(segments) != 1 or segments[0].kind != "paragraph":
        return segments
    segment = segments[0]
    boundaries = [0, *_complete_unit_endpoints(segment.text)]
    if len(boundaries) == 2:
        return segments
    return [
        Segment(
            index,
            segment.start + start,
            segment.start + end,
            segment.text[start:end],
            segment.kind,
            segment.section,
            segment.heading_index,
        )
        for index, (start, end) in enumerate(pairwise(boundaries))
    ]


def _select_structural(context: _SelectionContext) -> _ComposedOutput | None:
    """Select anchors, per-section evidence, then global structural evidence.

    Args:
        context: Queryless selection inputs.

    Returns:
        Fitting composed output, or ``None`` when no complete unit fits.
    """
    state = _new_selection_state(context)
    _seed_anchors(state)
    _fill_section_shares(state)
    _fill_remaining(state, _try_add)
    return state.output if state.selected else None


def _select_query_aware(context: _SelectionContext) -> _ComposedOutput | None:
    """Select adaptively bounded evidence and attach its heading when affordable.

    Args:
        context: Query-aware selection inputs.

    Returns:
        Fitting composed output, or ``None`` when no complete unit fits.
    """
    state = _new_selection_state(context)
    evidence = {index for index in state.remaining if context.segments[index].kind != "heading"}
    if evidence:
        state.remaining = evidence
    state.remaining = context.ranking.adaptive_indexes(state.remaining)
    _fill_remaining(state, _try_add_with_heading)
    return state.output if state.selected else None


def _seed_anchors(state: _SelectionState) -> None:
    """Protect fitting first and last complete structural candidates.

    Args:
        state: Mutable structural selection state.
    """
    if not state.remaining:
        return
    first = min(state.remaining)
    last = max(state.remaining)
    if first == last:
        _try_add(state, first)
        return

    anchors = {first, last}
    output = _compose(state.context, anchors)
    if output is not None:
        state.selected.update(anchors)
        state.remaining.difference_update(anchors)
        state.output = output
        state.track_mmr_selection(first)
        state.track_mmr_selection(last)
        return
    relevance = state.context.ranking.relevance
    preferred = first if relevance[first] >= relevance[last] else last
    _try_add(state, preferred)


def _fill_section_shares(state: _SelectionState) -> None:
    """Spend an equal provisional content budget in each remaining section.

    Args:
        state: Mutable structural selection state.
    """
    sections = sorted({segment.section for segment in state.context.segments})
    if not sections:
        return
    available = state.context.limit - state.context.measurer.count(state.output.text)
    share = max(0, available // len(sections))
    costs = {
        index: state.context.measurer.count(state.context.segments[index].text)
        for index in sorted(state.remaining)
    }
    pools = {section: set[int]() for section in sections}
    for index in state.remaining:
        pools[state.context.segments[index].section].add(index)
    for section in sections:
        pool = pools[section]
        spent = 0
        while pool:
            capacity = share - spent
            fitting = {index for index in pool if costs[index] <= capacity}
            if not fitting:
                break
            index = state.context.ranking.next_index(
                fitting,
                state.maximum_similarities,
                state.context.mmr_lambda,
            )
            pool.remove(index)
            if _try_add(state, index):
                spent += costs[index]


def _fill_remaining(
    state: _SelectionState,
    add_candidate: Callable[[_SelectionState, int], bool],
) -> None:
    """Greedily attempt every remaining candidate in live MMR order.

    Args:
        state: Mutable selection state.
        add_candidate: Candidate acceptance behavior for the active strategy.
    """
    while state.remaining:
        index = state.context.ranking.next_index(
            state.remaining,
            state.maximum_similarities,
            state.context.mmr_lambda,
        )
        if add_candidate(state, index):
            continue
        for index in state.context.ranking.ordered_indexes(
            state.remaining,
            state.maximum_similarities,
            state.context.mmr_lambda,
        ):
            if add_candidate(state, index):
                break
        else:
            return


def _try_add(state: _SelectionState, index: int) -> bool:
    """Attempt to add one candidate without heading expansion.

    Args:
        state: Mutable selection state.
        index: Main candidate index.

    Returns:
        Whether the complete candidate fit.
    """
    accepted = _accept_indices(state, index, state.selected | {index})
    if not accepted:
        state.remaining.discard(index)
    return accepted


def _try_add_with_heading(state: _SelectionState, index: int) -> bool:
    """Attempt a query candidate with its nearest heading, then alone.

    Args:
        state: Mutable query-aware selection state.
        index: Main candidate index.

    Returns:
        Whether the candidate fit in either form.
    """
    segment = state.context.segments[index]
    if segment.heading_index is not None and segment.heading_index not in state.selected:
        expanded = state.selected | {segment.heading_index, index}
        if _accept_indices(state, index, expanded):
            return True
    return _try_add(state, index)


def _accept_indices(state: _SelectionState, main_index: int, trial: set[int]) -> bool:
    """Commit a candidate bundle only when its exact composition fits.

    Args:
        state: Mutable selection state.
        main_index: Candidate participating in the MMR sequence.
        trial: Complete set of retained candidate indexes.

    Returns:
        Whether the trial was committed.
    """
    output = _compose(state.context, trial)
    if output is None:
        return False
    added = trial - state.selected
    state.selected = trial
    state.remaining.difference_update(added)
    state.output = output
    state.track_mmr_selection(main_index)
    return True


def _compose(context: _SelectionContext, indexes: set[int]) -> _ComposedOutput | None:
    """Compose source-ordered fragments and add every affordable gap marker.

    Args:
        context: Source and exact budget settings.
        indexes: Candidate indexes to retain.

    Returns:
        Fitting composition, or ``None`` when retained content alone is too large.
    """
    if not indexes:
        return _ComposedOutput("", ())
    segments = [context.segments[index] for index in sorted(indexes)]
    pieces = _output_pieces(context, segments)
    current = [piece.fallback for piece in pieces]
    text = "".join(current)
    if context.measurer.count(text) > context.limit:
        return None
    for index, piece in enumerate(pieces):
        if piece.marked is None:
            continue
        fallback = current[index]
        current[index] = piece.marked
        candidate = "".join(current)
        if context.measurer.count(candidate) <= context.limit:
            text = candidate
        else:
            current[index] = fallback
    return _ComposedOutput(text, _source_spans(context.source, segments))


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
    context: _SelectionContext,
    segments: list[Segment],
) -> list[_OutputPiece]:
    """Describe fixed fragments and optional marker-bearing source gaps.

    Args:
        context: Source and marker settings.
        segments: Retained segments in source order.

    Returns:
        Alternating source and gap pieces.
    """
    pieces: list[_OutputPiece] = []
    first = segments[0]
    if first.start:
        leading_has_content = _NON_WHITESPACE_PATTERN.search(context.source, 0, first.start)
        marked = context.marker + _newlines_before(first.text)
        pieces.append(
            _OutputPiece(
                "" if leading_has_content else context.source[: first.start],
                marked if leading_has_content else None,
            )
        )
    pieces.append(_OutputPiece(first.text))

    for previous, segment in pairwise(segments):
        gap_has_content = _NON_WHITESPACE_PATTERN.search(
            context.source,
            previous.end,
            segment.start,
        )
        if gap_has_content:
            pieces.append(
                _OutputPiece(
                    _plain_separator(previous.text, segment.text),
                    _marked_separator(previous.text, context.marker, segment.text),
                )
            )
        else:
            pieces.append(_OutputPiece(context.source[previous.end : segment.start]))
        pieces.append(_OutputPiece(segment.text))

    last = segments[-1]
    if last.end < len(context.source):
        trailing_has_content = _NON_WHITESPACE_PATTERN.search(context.source, last.end)
        marked = _newlines_after(last.text) + context.marker
        pieces.append(
            _OutputPiece(
                "" if trailing_has_content else context.source[last.end :],
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


def _fallback_output(context: _SelectionContext) -> _ComposedOutput:
    """Retain a measurable prefix of the strongest indivisible candidate.

    Args:
        context: Source, ranking, and budget settings.

    Returns:
        Fitting source-derived fragment, possibly empty.
    """
    if not context.segments:
        return _ComposedOutput("", ())
    index = max(
        range(len(context.segments)),
        key=lambda candidate: (context.ranking.relevance[candidate], -candidate),
    )
    segment = context.segments[index]
    fragment = _fitting_segment(context, segment)
    if not fragment.text:
        return _ComposedOutput("", ())
    return _add_fallback_markers(context, segment, fragment)


def _fitting_segment(context: _SelectionContext, segment: Segment) -> _ComposedOutput:
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
    context: _SelectionContext,
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


def _fitting_plain_prefix(context: _SelectionContext, text: str) -> str:
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
    context: _SelectionContext,
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
    context: _SelectionContext,
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
