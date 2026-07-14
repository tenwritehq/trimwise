"""Rank source segments by salience, relevance, and nonredundancy."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from trimwise.measurement import Measurer
from trimwise.segmentation import Segment
from trimwise.semantic import _SemanticVectors

_BM25_K1 = 1.5
_BM25_B = 0.75
_RRF_K = 60
# ponytail: fixed private calibration; tune only after Trimwise-shaped evidence.
_HYBRID_SEMANTIC_WEIGHT = 0.5
_PRIMARY_WEIGHT = 0.9
_SIGNAL_WEIGHT = 0.1
_ADAPTIVE_RECALL_BUFFER = 5
_ADAPTIVE_SEARCH_SHARE = 0.9
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_NUMBER_OR_DATE_PATTERN = re.compile(r"\b\d+(?:[-/:.]\d+)+\b|\b\d+(?:\.\d+)?\b")
_CODE_IDENTIFIER_PATTERN = re.compile(r"\b(?:[A-Za-z]+_[A-Za-z0-9_]+|[a-z]+[A-Z]\w*)\b")

Similarity = Callable[[int, int], float]
MaximumSimilarityFactory = Callable[[], Sequence[float]]
MaximumSimilarityUpdate = Callable[[Sequence[float], int], None]


@dataclass(frozen=True, slots=True)
class CandidateRanking:
    """Store relevance plus scalar and optional bulk similarity operations."""

    primary: tuple[float, ...]
    relevance: tuple[float, ...]
    similarity: Similarity
    maximum_similarity_factory: MaximumSimilarityFactory | None = None
    maximum_similarity_update: MaximumSimilarityUpdate | None = None

    def adaptive_indexes(self, candidates: set[int]) -> set[int]:
        """Keep candidates through the largest query-score drop plus a recall buffer.

        Args:
            candidates: Candidate indexes eligible for query-aware retrieval.

        Returns:
            Adaptively bounded candidates, independent of absolute score scale.
        """
        ordered = sorted(
            candidates,
            key=lambda index: (-self.primary[index], -self.relevance[index], index),
        )
        if len(ordered) < 2:
            return set(ordered)
        gap_count = len(ordered) - 1
        searchable_gap_count = math.ceil(gap_count * _ADAPTIVE_SEARCH_SHARE)
        boundary = max(
            range(searchable_gap_count),
            key=lambda position: (
                self.primary[ordered[position]] - self.primary[ordered[position + 1]],
                -position,
            ),
        )
        # ponytail: keep this fixed until cross-model quality benchmarks justify a knob.
        retained_count = min(len(ordered), boundary + 1 + _ADAPTIVE_RECALL_BUFFER)
        return set(ordered[:retained_count])

    def new_maximum_similarities(self) -> Sequence[float]:
        """Create strategy-appropriate zeroed MMR state.

        Returns:
            Mutable similarity state indexed by candidate.
        """
        if self.maximum_similarity_factory is not None:
            return self.maximum_similarity_factory()
        return [0.0] * len(self.relevance)

    def next_index(
        self,
        remaining: set[int],
        maximum_similarities: Sequence[float],
        mmr_lambda: float,
    ) -> int:
        """Choose the next MMR candidate with a stable source-order tie break.

        Args:
            remaining: Candidate indexes still eligible for selection.
            maximum_similarities: Greatest redundancy seen for each candidate.
            mmr_lambda: Relevance weight in the MMR objective.

        Returns:
            Index with the greatest relevance-diversity objective.
        """
        return max(
            remaining,
            key=lambda index: self._objective(
                index,
                maximum_similarities[index],
                mmr_lambda,
            ),
        )

    def ordered_indexes(
        self,
        remaining: set[int],
        maximum_similarities: Sequence[float],
        mmr_lambda: float,
    ) -> list[int]:
        """Order eligible candidates by the current MMR objective.

        Args:
            remaining: Candidate indexes still eligible for selection.
            maximum_similarities: Greatest redundancy seen for each candidate.
            mmr_lambda: Relevance weight in the MMR objective.

        Returns:
            Candidate indexes from strongest to weakest with stable ties.
        """
        return sorted(
            remaining,
            key=lambda index: self._objective(
                index,
                maximum_similarities[index],
                mmr_lambda,
            ),
            reverse=True,
        )

    def _objective(
        self,
        index: int,
        maximum_similarity: float,
        mmr_lambda: float,
    ) -> tuple[float, int]:
        """Score one candidate against the already selected evidence.

        Args:
            index: Candidate index to score.
            maximum_similarity: Greatest similarity to retained evidence.
            mmr_lambda: Relevance weight in the MMR objective.

        Returns:
            MMR score and inverse index for deterministic ties.
        """
        score = mmr_lambda * self.relevance[index] - (1 - mmr_lambda) * maximum_similarity
        return score, -index


def rank_structural(segments: list[Segment], measurer: Measurer) -> CandidateRanking:
    """Rank queryless candidates by TF-IDF document centrality.

    Args:
        segments: Ordered source candidates.
        measurer: Provider of consistent subword IDs.

    Returns:
        Centrality ranking with lexical pairwise similarities.
    """
    vectors = _tfidf_vectors(_terms_for_segments(segments, measurer))
    centroid = _centroid(vectors)
    primary = [_sparse_cosine(vector, centroid) for vector in vectors]
    return _build_ranking(segments, primary, _sparse_similarity(vectors))


def rank_lexical(
    segments: list[Segment],
    query: str,
    measurer: Measurer,
) -> CandidateRanking:
    """Rank candidates with BM25 and lexical MMR similarity.

    Args:
        segments: Ordered source candidates.
        query: Nonblank user task or question.
        measurer: Provider of consistent subword IDs.

    Returns:
        BM25 ranking with lexical pairwise similarities.
    """
    documents = _terms_for_segments(segments, measurer)
    query_terms = measurer.token_ids(_normalize_ranking_text(query))
    primary = _bm25_scores(documents, query_terms)
    vectors = _tfidf_vectors(documents)
    return _build_ranking(segments, primary, _sparse_similarity(vectors))


def rank_semantic(
    segments: list[Segment],
    vectors: _SemanticVectors,
) -> CandidateRanking:
    """Rank candidates by semantic cosine similarity.

    Args:
        segments: Ordered source candidates.
        vectors: Normalized query and passage matrix.

    Returns:
        Semantic ranking with dense pairwise similarities.
    """
    ranking = _build_ranking(segments, vectors.query_scores(), vectors.similarity)
    return replace(
        ranking,
        maximum_similarity_factory=vectors.new_maximum_similarities,
        maximum_similarity_update=vectors.update_maximum_similarities,
    )


def rank_hybrid(
    segments: list[Segment],
    lexical: CandidateRanking,
    semantic: CandidateRanking,
) -> CandidateRanking:
    """Fuse valid lexical and semantic score rows equally, with RRF fallback.

    Args:
        segments: Ordered source candidates.
        lexical: BM25 ranking.
        semantic: Embedding ranking.

    Returns:
        Equal score blend or RRF fallback using semantic similarity for MMR.
    """
    scores = (*lexical.primary, *semantic.primary)
    if (
        all(math.isfinite(score) for score in scores)
        and len(set(lexical.primary)) > 1
        and len(set(semantic.primary)) > 1
    ):
        lexical_scores = _minmax(list(lexical.primary))
        semantic_scores = _minmax(list(semantic.primary))
        primary = [
            (1 - _HYBRID_SEMANTIC_WEIGHT) * lexical_score + _HYBRID_SEMANTIC_WEIGHT * semantic_score
            for lexical_score, semantic_score in zip(lexical_scores, semantic_scores, strict=True)
        ]
    else:
        lexical_positions = _rank_positions(lexical.primary)
        semantic_positions = _rank_positions(semantic.primary)
        primary = [
            1 / (_RRF_K + lexical_positions[index]) + 1 / (_RRF_K + semantic_positions[index])
            for index in range(len(segments))
        ]
    ranking = _build_ranking(segments, primary, semantic.similarity)
    return replace(
        ranking,
        maximum_similarity_factory=semantic.maximum_similarity_factory,
        maximum_similarity_update=semantic.maximum_similarity_update,
    )


def _contextual_ranking_texts(segments: list[Segment]) -> list[str]:
    """Give each candidate local context without changing retained source.

    The current candidate is repeated first so direct evidence remains distinguishable from
    evidence inherited through an adjacent candidate's scoring window.

    Args:
        segments: Ordered source candidates with section relationships.

    Returns:
        Candidate-anchored scoring text in source order.
    """
    texts: list[str] = []
    for index, segment in enumerate(segments):
        previous = index - 1 if index and segments[index - 1].section == segment.section else None
        following = (
            index + 1
            if index + 1 < len(segments) and segments[index + 1].section == segment.section
            else None
        )
        context_indexes = dict.fromkeys(
            candidate_index
            for candidate_index in (segment.heading_index, previous, index, following)
            if candidate_index is not None
        )
        if len(context_indexes) == 1:
            texts.append(segment.text)
            continue
        window = "\n\n".join(segments[candidate_index].text for candidate_index in context_indexes)
        texts.append(f"{segment.text}\n\n{window}")
    return texts


def _normalize_ranking_text(text: str) -> str:
    """Normalize text for ranking without changing retained source.

    Args:
        text: Candidate or query text.

    Returns:
        Consistently prefixed Unicode-normalized text.
    """
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return f" {normalized}"


def _terms_for_segments(segments: list[Segment], measurer: Measurer) -> list[list[int]]:
    """Tokenize every source candidate for lexical calculations.

    Args:
        segments: Ordered source candidates.
        measurer: Provider of configured subword IDs.

    Returns:
        Token identifiers for each candidate.
    """
    return [
        measurer.token_ids(_normalize_ranking_text(text))
        for text in _contextual_ranking_texts(segments)
    ]


def _tfidf_vectors(documents: list[list[int]]) -> list[dict[int, float]]:
    """Create normalized sparse TF-IDF vectors for candidate documents.

    Args:
        documents: Token identifiers per candidate.

    Returns:
        L2-normalized sparse vectors.
    """
    document_count = len(documents)
    frequencies = Counter(term for document in documents for term in set(document))
    vectors: list[dict[int, float]] = []
    for document in documents:
        counts = Counter(document)
        length = len(document) or 1
        vector = {
            term: count / length * (math.log((document_count + 1) / (frequencies[term] + 1)) + 1)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({term: value / norm for term, value in vector.items()})
    return vectors


def _centroid(vectors: list[dict[int, float]]) -> dict[int, float]:
    """Average sparse vectors into a document centroid.

    Args:
        vectors: Candidate TF-IDF vectors.

    Returns:
        Sparse mean vector.
    """
    totals: Counter[int] = Counter()
    for vector in vectors:
        totals.update(vector)
    count = len(vectors) or 1
    return {term: value / count for term, value in totals.items()}


def _bm25_scores(documents: list[list[int]], query: list[int]) -> list[float]:
    """Calculate Robertson BM25 scores for all candidates.

    Args:
        documents: Token identifiers per candidate.
        query: Token identifiers from the user query.

    Returns:
        Nonnegative BM25 scores in document order.
    """
    document_count = len(documents)
    average_length = sum(map(len, documents)) / document_count if document_count else 0.0
    document_frequency = Counter(term for document in documents for term in set(document))
    scores: list[float] = []
    for document in documents:
        term_counts = Counter(document)
        length_ratio = len(document) / average_length if average_length else 0.0
        score = 0.0
        for term in set(query):
            frequency = term_counts[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + _BM25_K1 * (1 - _BM25_B + _BM25_B * length_ratio)
            score += inverse_frequency * frequency * (_BM25_K1 + 1) / denominator
        scores.append(score)
    return scores


def _build_ranking(
    segments: list[Segment],
    primary: list[float],
    similarity: Similarity,
) -> CandidateRanking:
    """Normalize primary scores and blend language-neutral signal.

    Args:
        segments: Ordered source candidates.
        primary: Strategy-specific raw relevance.
        similarity: Lazy pairwise redundancy lookup.

    Returns:
        Complete immutable candidate ranking.
    """
    normalized = _minmax(primary)
    relevance = tuple(
        _PRIMARY_WEIGHT * score + _SIGNAL_WEIGHT * _signal_score(segment)
        for score, segment in zip(normalized, segments, strict=True)
    )
    return CandidateRanking(tuple(primary), relevance, similarity)


def _minmax(values: list[float]) -> list[float]:
    """Map scores to the unit interval with deterministic equal-score behavior.

    Args:
        values: Raw strategy scores.

    Returns:
        Values scaled to ``[0, 1]`` or all zeros when constant.
    """
    if not values:
        return []
    minimum = min(values)
    width = max(values) - minimum
    return [(value - minimum) / width for value in values] if width else [0.0] * len(values)


def _signal_score(segment: Segment) -> float:
    """Score language-neutral structural and fact-like signals.

    Args:
        segment: Source candidate to inspect.

    Returns:
        Mean of four binary signal indicators.
    """
    indicators = (
        segment.kind == "heading" or segment.heading_index is not None,
        bool(_URL_PATTERN.search(segment.text)),
        bool(_NUMBER_OR_DATE_PATTERN.search(segment.text)),
        bool(_CODE_IDENTIFIER_PATTERN.search(segment.text)),
    )
    return sum(indicators) / len(indicators)


def _sparse_cosine(left: dict[int, float], right: dict[int, float]) -> float:
    """Compute cosine similarity between sparse numeric vectors.

    Args:
        left: First sparse vector.
        right: Second sparse vector.

    Returns:
        Cosine similarity, or zero for an empty vector.
    """
    if not left or not right:
        return 0.0
    dot_product = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot_product / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _sparse_similarity(
    vectors: list[dict[int, float]],
) -> Similarity:
    """Create a lexical similarity lookup.

    Args:
        vectors: Sparse TF-IDF vectors.

    Returns:
        Pairwise sparse cosine lookup.
    """

    def similarity(left_index: int, right_index: int) -> float:
        """Compare two sparse candidate vectors.

        Args:
            left_index: First candidate index.
            right_index: Second candidate index.

        Returns:
            Sparse cosine similarity.
        """
        return _sparse_cosine(vectors[left_index], vectors[right_index])

    return similarity


def _rank_positions(scores: tuple[float, ...]) -> list[int]:
    """Convert scores into stable one-indexed rank positions.

    Args:
        scores: Raw strategy scores.

    Returns:
        Rank position for each source index.
    """
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    positions = [0] * len(scores)
    for position, index in enumerate(order, start=1):
        positions[index] = position
    return positions
