"""Verify deterministic lexical, semantic, hybrid, signal, and MMR ranking."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest

from trimwise.measurement import Measurer
from trimwise.models import BudgetUnit
from trimwise.ranking import (
    _RRF_K,
    CandidateRanking,
    _bm25_scores,
    _contextual_ranking_texts,
    _minmax,
    _rank_positions,
    _sparse_cosine,
    rank_hybrid,
    rank_lexical,
    rank_semantic,
    rank_structural,
)
from trimwise.segmentation import Segment
from trimwise.semantic import _SemanticVectors
from trimwise.trimmer import (
    _fill_remaining,
    _fill_section_shares,
    _new_selection_state,
    _SelectionContext,
    _SelectionState,
)


def _segments(*texts: str) -> list[Segment]:
    """Create simple ordered source candidates for ranking tests.

    Args:
        *texts: Candidate texts.

    Returns:
        Source-ordered segments with synthetic spans.
    """
    segments: list[Segment] = []
    cursor = 0
    for index, text in enumerate(texts):
        segments.append(Segment(index, cursor, cursor + len(text), text, "paragraph", 0, None))
        cursor += len(text)
    return segments


def _measurer() -> Measurer:
    """Create a character measurer with the default ranking encoding."""
    return Measurer(BudgetUnit.CHARACTERS, "o200k_base", None)


def _semantic_vectors(*vectors: tuple[float, ...]) -> _SemanticVectors:
    """Create an already normalized semantic matrix for ranking tests.

    Args:
        *vectors: Query row followed by passage rows.

    Returns:
        Float32 semantic vector matrix.
    """
    return _SemanticVectors(np.asarray(vectors, dtype=np.float32))


def _ranking(primary: tuple[float, ...]) -> CandidateRanking:
    """Create a ranking with inert similarity for hybrid-fusion tests.

    Args:
        primary: Raw candidate scores in source order.

    Returns:
        Ranking whose primary and relevance values match the supplied scores.
    """
    return CandidateRanking(primary, primary, lambda _left, _right: 0.0)


def test_bm25_ranks_exact_query_evidence_first() -> None:
    """Give the highest lexical score to the exact matching fact."""
    segments = _segments("cats sleep", "Zephyr launched Tuesday", "unrelated prose")
    ranking = rank_lexical(segments, "Zephyr launch", _measurer())
    assert ranking.primary.index(max(ranking.primary)) == 1


def test_bm25_materializes_query_terms_once() -> None:
    """Reuse one query-term set across all candidate documents."""

    class CountingQuery(list[int]):
        """Count how often BM25 iterates the query tokens."""

        def __init__(self, values: list[int]) -> None:
            """Store query tokens and initialize the iteration count.

            Args:
                values: Query token identifiers.
            """
            super().__init__(values)
            self.iterations = 0

        def __iter__(self) -> Iterator[int]:
            """Count and delegate one iteration over query tokens."""
            self.iterations += 1
            return super().__iter__()

    query = CountingQuery([1, 2, 3])
    _bm25_scores([[1], [2], [3]], query)
    assert query.iterations == 1


def test_contextual_ranking_texts_keep_the_candidate_distinct() -> None:
    """Add local section context while anchoring the current candidate first."""
    segments = [
        Segment(0, 0, 14, "# Installation", "heading", 1, None),
        Segment(1, 14, 31, "Run this command.", "paragraph", 1, 0),
        Segment(2, 31, 59, "It was discontinued in 2024.", "paragraph", 1, 0),
        Segment(3, 59, 80, "Use Orion instead.", "paragraph", 1, 0),
        Segment(4, 80, 89, "# Removal", "heading", 2, None),
        Segment(5, 89, 102, "Delete files.", "paragraph", 2, 4),
    ]

    texts = _contextual_ranking_texts(segments)

    assert texts[2] == (
        "It was discontinued in 2024.\n\n"
        "# Installation\n\nRun this command.\n\n"
        "It was discontinued in 2024.\n\nUse Orion instead."
    )
    assert "# Removal" not in texts[3]
    assert texts[1].count("# Installation") == 1


def test_contextual_bm25_still_prefers_direct_query_evidence() -> None:
    """Keep a matching candidate ahead of neighbors that only inherit its context."""
    segments = _segments("alpha filler", "Zephyr target", "omega filler")
    ranking = rank_lexical(segments, "Zephyr", _measurer())
    assert ranking.primary.index(max(ranking.primary)) == 1


def test_structural_centrality_prefers_repeated_document_vocabulary() -> None:
    """Prefer a candidate central to vocabulary used throughout the document."""
    segments = _segments("alpha beta", "alpha beta gamma", "rare zircon")
    ranking = rank_structural(segments, _measurer())
    assert ranking.primary[1] > ranking.primary[2]


def test_mmr_penalizes_near_duplicate_evidence() -> None:
    """Choose complementary evidence after a highly similar first candidate."""
    similarities = (
        (1.0, 0.99, 0.0),
        (0.99, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    ranking = CandidateRanking(
        (1.0, 0.99, 0.8),
        (1.0, 0.99, 0.8),
        lambda left, right: similarities[left][right],
    )
    index = ranking.next_index({1, 2}, (0.0, 0.99, 0.0), 0.7)
    assert index == 2


def test_mmr_ties_follow_source_order() -> None:
    """Make equal ranking outcomes deterministic."""
    ranking = CandidateRanking((0.0, 0.0), (0.0, 0.0), lambda left, right: left == right)
    assert ranking.next_index({0, 1}, (0.0, 0.0), 0.7) == 0


def test_adaptive_pool_cuts_at_largest_gap_with_recall_buffer() -> None:
    """Keep five candidates beyond the strongest query-score boundary."""
    primary = (1.0, 0.9, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13)
    ranking = CandidateRanking(primary, primary, lambda left, right: left == right)
    assert ranking.adaptive_indexes(set(range(len(primary)))) == set(range(7))


def test_adaptive_pool_is_invariant_to_affine_score_scale() -> None:
    """Derive the cutoff from score shape rather than model-specific values."""
    primary = (0.9, 0.8, 0.7, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04)
    shifted = tuple(score * 0.2 - 0.1 for score in primary)
    original = CandidateRanking(primary, primary, lambda left, right: left == right)
    transformed = CandidateRanking(shifted, shifted, lambda left, right: left == right)
    candidates = set(range(len(primary)))
    assert original.adaptive_indexes(candidates) == transformed.adaptive_indexes(candidates)


def test_adaptive_pool_ignores_extreme_tail_gap() -> None:
    """Do not let a bottom-decile outlier force an almost-full result."""
    primary = (*range(21, 1, -1), -100.0)
    ranking = CandidateRanking(primary, primary, lambda left, right: left == right)
    assert ranking.adaptive_indexes(set(range(len(primary)))) == set(range(6))


def test_adaptive_pool_keeps_a_single_candidate() -> None:
    """Retain the only available query-aware candidate."""
    ranking = CandidateRanking((0.0,), (0.0,), lambda left, right: left == right)
    assert ranking.adaptive_indexes({0}) == {0}


def test_adaptive_pool_breaks_query_score_ties_with_signal() -> None:
    """Let existing fact-like signal choose among equally irrelevant buffer candidates."""
    primary = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    relevance = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4)
    ranking = CandidateRanking(primary, relevance, lambda left, right: left == right)
    assert ranking.adaptive_indexes(set(range(len(primary)))) == {0, 1, 6, 7, 8, 9}


def test_mmr_similarity_is_updated_once_per_remaining_pair() -> None:
    """Compare each selected candidate once with every candidate still eligible."""
    calls: list[tuple[int, int]] = []
    similarities = (
        (1.0, 0.8, 0.1),
        (0.8, 1.0, 0.6),
        (0.1, 0.6, 1.0),
    )

    def similarity(left: int, right: int) -> float:
        """Record and return one pairwise similarity.

        Args:
            left: Remaining candidate index.
            right: Newly selected candidate index.

        Returns:
            Synthetic similarity for the pair.
        """
        calls.append((left, right))
        return similarities[left][right]

    segments = _segments("alpha", "beta", "gamma")
    ranking = CandidateRanking((1.0, 0.9, 0.8), (1.0, 0.9, 0.8), similarity)
    context = _SelectionContext("alphabetagamma", segments, ranking, _measurer(), 20, "...", 0.7)
    state = _new_selection_state(context)
    state.remaining.remove(0)
    state.track_mmr_selection(0)
    state.remaining.remove(1)
    state.track_mmr_selection(1)
    assert (state.maximum_similarities, sorted(calls)) == (
        [0.0, 0.8, 0.6],
        [(1, 0), (2, 0), (2, 1)],
    )


def test_section_shares_skip_ranking_when_no_unit_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave over-share units for global redistribution without ranking them.

    Args:
        monkeypatch: Pytest attribute replacement helper.
    """

    def reject_ranking(*args: object) -> int:
        """Fail if an ineligible provisional candidate reaches MMR ranking.

        Args:
            *args: Ignored ranking call arguments.

        Raises:
            AssertionError: Always, because no candidate fits the share.
        """
        raise AssertionError("section ranking should not run")

    segments = _segments("long", "unit")
    segments[1] = Segment(1, 4, 8, "unit", "paragraph", 1, None)
    ranking = CandidateRanking((1.0, 1.0), (1.0, 1.0), lambda left, right: 0.0)
    context = _SelectionContext("longunit", segments, ranking, _measurer(), 2, "...", 0.7)
    state = _new_selection_state(context)
    monkeypatch.setattr(CandidateRanking, "next_index", reject_ranking)
    _fill_section_shares(state)
    assert state.remaining == {0, 1}


def test_global_fill_uses_linear_best_candidate_scan_when_candidates_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid sorting every remaining candidate after each accepted unit.

    Args:
        monkeypatch: Pytest attribute replacement helper.
    """

    def reject_sorting(*args: object) -> list[int]:
        """Fail if the all-fitting path sorts candidates.

        Args:
            *args: Ignored ranking call arguments.

        Raises:
            AssertionError: Always, because each best candidate is accepted.
        """
        raise AssertionError("all-fitting selection sorted remaining candidates")

    def accept(state: _SelectionState, index: int) -> bool:
        """Remove one accepted candidate from the synthetic selection state.

        Args:
            state: Mutable selection state under test.
            index: Best remaining candidate.

        Returns:
            Always ``True`` because every synthetic candidate fits.
        """
        state.remaining.remove(index)
        return True

    segments = _segments("a", "b", "c")
    ranking = CandidateRanking((1.0, 0.9, 0.8), (1.0, 0.9, 0.8), lambda left, right: 0.0)
    context = _SelectionContext("abc", segments, ranking, _measurer(), 3, "...", 0.7)
    state = _new_selection_state(context)
    monkeypatch.setattr(CandidateRanking, "ordered_indexes", reject_sorting)
    _fill_remaining(state, accept)
    assert not state.remaining


def test_semantic_ranking_uses_query_cosine() -> None:
    """Rank the passage aligned with the query vector first."""
    segments = _segments("aligned", "orthogonal")
    vectors = _semantic_vectors((1.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    ranking = rank_semantic(segments, vectors)
    assert ranking.primary == (1.0, 0.0)


def test_semantic_mmr_updates_similarities_without_scalar_lookups() -> None:
    """Use the semantic matrix bulk updater instead of Python pairwise calls."""

    def reject_scalar_lookup(_: int, __: int) -> float:
        """Fail if semantic MMR falls back to scalar similarity.

        Args:
            _: Ignored first candidate index.
            __: Ignored second candidate index.

        Raises:
            AssertionError: Always, because semantic MMR must update in bulk.
        """
        raise AssertionError("semantic MMR used scalar similarity")

    segments = _segments("selected", "similar", "different")
    vectors = _semantic_vectors(
        (1.0, 0.0),
        (1.0, 0.0),
        (0.8, 0.6),
        (0.0, 1.0),
    )
    ranking = replace(rank_semantic(segments, vectors), similarity=reject_scalar_lookup)
    context = _SelectionContext("", segments, ranking, _measurer(), 20, "...", 0.7)
    state = _new_selection_state(context)
    state.remaining.remove(0)
    state.track_mmr_selection(0)
    np.testing.assert_allclose(state.maximum_similarities, (1.0, 0.8, 0.0))


def test_hybrid_ranking_blends_disagreeing_score_rows() -> None:
    """Give equal blended scores to candidates favored by opposite rankers."""
    segments = _segments("first", "second")
    passages = ((1.0, 0.0), (0.0, 1.0))
    lexical = rank_semantic(segments, _semantic_vectors((1.0, 0.0), *passages))
    semantic = rank_semantic(segments, _semantic_vectors((0.0, 1.0), *passages))
    hybrid = rank_hybrid(segments, lexical, semantic)
    assert hybrid.primary[0] == hybrid.primary[1]


def test_hybrid_ranking_uses_equal_normalized_score_blend() -> None:
    """Preserve score magnitude instead of reducing valid rows to rank positions."""
    hybrid = rank_hybrid(
        _segments("first", "second", "third"),
        _ranking((10.0, 9.0, 0.0)),
        _ranking((0.0, 0.5, 1.0)),
    )
    assert hybrid.primary == pytest.approx((0.5, 0.7, 0.5))


@pytest.mark.parametrize(
    ("lexical_scores", "semantic_scores"),
    [
        ((3.0, 3.0, 3.0), (0.9, 0.5, 0.1)),
        ((0.9, 0.5, 0.1), (float("nan"), 0.5, 0.1)),
    ],
    ids=["flat", "nonfinite"],
)
def test_hybrid_ranking_falls_back_to_rrf_for_unusable_score_rows(
    lexical_scores: tuple[float, ...],
    semantic_scores: tuple[float, ...],
) -> None:
    """Use the previous RRF-60 formula when either score row cannot be blended.

    Args:
        lexical_scores: Raw BM25-like scores for one fallback case.
        semantic_scores: Raw semantic-like scores for one fallback case.
    """
    lexical = _ranking(lexical_scores)
    semantic = _ranking(semantic_scores)
    hybrid = rank_hybrid(_segments("first", "second", "third"), lexical, semantic)
    lexical_positions = _rank_positions(lexical_scores)
    semantic_positions = _rank_positions(semantic_scores)
    expected = tuple(
        1 / (_RRF_K + lexical_positions[index]) + 1 / (_RRF_K + semantic_positions[index])
        for index in range(3)
    )
    assert hybrid.primary == pytest.approx(expected)


def test_signal_boosts_heading_context_urls_numbers_and_identifiers() -> None:
    """Blend the four documented language-neutral indicators."""
    signaled = Segment(0, 0, 30, "api_key https://x.dev 2026-07-14", "paragraph", 1, 3)
    plain = Segment(1, 30, 35, "plain", "paragraph", 0, None)
    ranking = rank_semantic([signaled, plain], _semantic_vectors((1.0,), (1.0,), (1.0,)))
    assert ranking.relevance[0] > ranking.relevance[1]


def test_equal_scores_normalize_to_zero() -> None:
    """Avoid arbitrary relevance when every primary score is equal."""
    assert _minmax([2.0, 2.0]) == [0.0, 0.0]


def test_empty_score_normalization_is_empty() -> None:
    """Handle an empty candidate collection without extrema errors."""
    assert _minmax([]) == []


def test_rank_positions_are_one_indexed_and_stable() -> None:
    """Resolve score ties by source order for RRF."""
    assert _rank_positions((1.0, 1.0, 0.0)) == [1, 2, 3]


def test_sparse_cosine_handles_empty_vectors() -> None:
    """Treat missing lexical evidence as having no similarity."""
    assert _sparse_cosine({}, {1: 1.0}) == 0.0
