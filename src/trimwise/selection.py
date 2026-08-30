"""Select ranked source candidates within one exact output budget."""

from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import cast

from trimwise.composition import (
    _complete_unit_endpoints,
    _compose,
    _ComposedOutput,
    _output_count,
)
from trimwise.measurement import Measurer
from trimwise.models import Strategy
from trimwise.ranking import CandidateRanking
from trimwise.segmentation import Segment, segment_text


@dataclass(frozen=True, slots=True)
class _SelectionContext:
    """Provide immutable source, budget, and ranking data to selection."""

    sources: tuple[str, ...]
    segments: list[Segment]
    source_indexes: tuple[int, ...]
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
    output: tuple[_ComposedOutput, ...] = field(default_factory=tuple)

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
        output=tuple(_ComposedOutput("", ()) for _ in context.sources),
    )


def _prepare_context_candidates(
    sources: tuple[str, ...],
    strategy: Strategy,
) -> tuple[list[Segment], tuple[int, ...]]:
    """Segment sources independently and create globally rankable candidates.

    Args:
        sources: Input source strings in caller order.
        strategy: Resolved operation-wide strategy.

    Returns:
        Flattened candidates and each candidate's input source index.
    """
    segments: list[Segment] = []
    source_indexes: list[int] = []
    section_offset = 0
    for source_index, source in enumerate(sources):
        local_segments = segment_text(source)
        if strategy is Strategy.STRUCTURAL:
            local_segments = _expand_structural_plaintext(local_segments)
        candidate_offset = len(segments)
        for local in local_segments:
            heading_index = (
                candidate_offset + local.heading_index if local.heading_index is not None else None
            )
            segments.append(
                Segment(
                    len(segments),
                    local.start,
                    local.end,
                    local.text,
                    local.kind,
                    section_offset + local.section,
                    heading_index,
                )
            )
            source_indexes.append(source_index)
        if local_segments:
            section_offset += max(segment.section for segment in local_segments) + 1
    return segments, tuple(source_indexes)


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


def _select_structural(
    context: _SelectionContext,
) -> tuple[_ComposedOutput, ...] | None:
    """Select anchors, per-section evidence, then global structural evidence.

    Args:
        context: Queryless selection inputs.

    Returns:
        Fitting composed output, or ``None`` when no complete unit fits.
    """
    state = _new_selection_state(context)
    if len(set(context.source_indexes)) > 1:
        _fill_source_round(state)
        _fill_remaining(state, _try_add)
        return state.output if state.selected else None
    _seed_anchors(state)
    _fill_section_shares(state)
    _fill_remaining(state, _try_add)
    return state.output if state.selected else None


def _select_query_aware(
    context: _SelectionContext,
) -> tuple[_ComposedOutput, ...] | None:
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


def _fill_source_round(state: _SelectionState) -> None:
    """Give each candidate-bearing source one complete structural opportunity.

    Args:
        state: Mutable multi-source structural selection state.
    """
    context = state.context
    for source_index in sorted(set(context.source_indexes)):
        pool = {index for index in state.remaining if context.source_indexes[index] == source_index}
        evidence = {index for index in pool if context.segments[index].kind != "heading"}
        if evidence:
            pool = evidence
        for index in context.ranking.ordered_indexes(
            pool,
            state.maximum_similarities,
            context.mmr_lambda,
        ):
            if _try_add_with_heading(state, index):
                break


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
    available = state.context.limit - _output_count(state.context.measurer, state.output)
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
