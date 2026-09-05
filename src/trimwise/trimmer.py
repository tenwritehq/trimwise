"""Orchestrate validation, ranking, selection, composition, and async use."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from trimwise.composition import (
    _ComposedOutput,
    _fallback_candidate_output,
    _fallback_output,
    _text_count,
)
from trimwise.measurement import Measurer, TokenCounter
from trimwise.models import (
    BudgetUnit,
    ContextSourceResult,
    ContextTrimResult,
    SourceSpan,
    Strategy,
    TrimConfig,
    TrimInput,
    TrimResult,
)
from trimwise.ranking import (
    CandidateRanking,
    _contextual_ranking_texts,
    rank_hybrid,
    rank_lexical,
    rank_semantic,
    rank_structural,
)
from trimwise.segmentation import Segment, segment_text
from trimwise.selection import (
    _expand_structural_plaintext,
    _oversized_query_fallback_index,
    _prepare_context_candidates,
    _select_query_aware,
    _select_structural,
    _SelectionContext,
)
from trimwise.semantic import (
    AsyncEmbeddingCallback,
    EmbeddingCallback,
    EmbeddingOutput,
    SemanticEmbedder,
    _PassageBatch,
    _prepare_passage_batch,
    _remap_semantic_vectors,
    _SemanticVectors,
    embed_with_callback,
    invoke_async_embedding_callback,
    normalize_callback_output,
)


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
class _ContextRequest:
    """Collect one validated shared-context call for internal processing."""

    sources: tuple[str, ...]
    limit: int
    unit: BudgetUnit
    strategy: Strategy
    query: str | None
    token_counter: TokenCounter | None
    deduplicate: bool


@dataclass(frozen=True, slots=True)
class _ContextArguments:
    """Collect unvalidated shared-context values before worker processing."""

    sources: tuple[str, ...]
    limit: int
    unit: BudgetUnit | str
    strategy: Strategy | str
    query: str | None
    token_counter: TokenCounter | None
    deduplicate: bool


@dataclass(frozen=True, slots=True)
class _PreparedTrim:
    """Hold measured and segmented input until ranking is available."""

    request: _TrimRequest
    input_count: int
    segments: list[Segment]
    measurer: Measurer


@dataclass(frozen=True, slots=True)
class _PreparedContext:
    """Hold measured multi-source input until shared ranking is available."""

    request: _ContextRequest
    input_counts: tuple[int, ...]
    segments: list[Segment]
    source_indexes: tuple[int, ...]
    measurer: Measurer


@dataclass(frozen=True, slots=True)
class _RankingRequest:
    """Group the inputs needed to rank one segmented document."""

    segments: list[Segment]
    strategy: Strategy
    query: str | None
    measurer: Measurer
    deduplicate: bool = False


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

    def trim_context(
        self,
        sources: Sequence[str],
        limit: int,
        *,
        unit: BudgetUnit | str = BudgetUnit.TOKENS,
        strategy: Strategy | str = Strategy.AUTO,
        query: str | None = None,
        token_counter: Callable[[str], int] | None = None,
        deduplicate: bool = False,
    ) -> ContextTrimResult:
        """Trim many distinct sources under one shared output limit.

        Args:
            sources: Source strings whose excerpts share the requested limit.
            limit: Maximum summed output size in ``unit``.
            unit: Token, whitespace-word, or code-point character budget.
            strategy: Structural, lexical, semantic, hybrid, or automatic ranking.
            query: Task or question required by query-aware strategies.
            token_counter: Optional synchronous token measurement callback.
            deduplicate: Whether identical contextual passages share one embedding.

        Returns:
            Input-aligned excerpts and aggregate measurements.

        Raises:
            TypeError: If an argument is invalid or only an async embedder is available.
            ValueError: If an argument value or strategy/query combination is invalid.
            SemanticBackendError: If an explicitly requested semantic backend fails.
        """
        source_snapshot = _snapshot_sources(sources)
        _validate_deduplicate(deduplicate)
        arguments = _ContextArguments(
            source_snapshot,
            limit,
            unit,
            strategy,
            query,
            token_counter,
            deduplicate,
        )
        return self._trim_context(arguments)

    async def atrim_context(
        self,
        sources: Sequence[str],
        limit: int,
        *,
        unit: BudgetUnit | str = BudgetUnit.TOKENS,
        strategy: Strategy | str = Strategy.AUTO,
        query: str | None = None,
        token_counter: Callable[[str], int] | None = None,
        deduplicate: bool = False,
    ) -> ContextTrimResult:
        """Trim many sources asynchronously under one shared output limit.

        A configured asynchronous embedding callback is awaited on the calling event loop.
        Cancellation propagates to that callback, but cannot stop worker work already running.

        Args:
            sources: Source strings whose excerpts share the requested limit.
            limit: Maximum summed output size in ``unit``.
            unit: Token, whitespace-word, or code-point character budget.
            strategy: Structural, lexical, semantic, hybrid, or automatic ranking.
            query: Task or question required by query-aware strategies.
            token_counter: Optional synchronous token measurement callback.
            deduplicate: Whether identical contextual passages share one embedding.

        Returns:
            Input-aligned excerpts and aggregate measurements.

        Raises:
            TypeError: If an argument has an unsupported type.
            ValueError: If an argument value or strategy/query combination is invalid.
            SemanticBackendError: If an explicitly requested semantic backend fails.
        """
        source_snapshot = _snapshot_sources(sources)
        _validate_deduplicate(deduplicate)
        arguments = _ContextArguments(
            source_snapshot,
            limit,
            unit,
            strategy,
            query,
            token_counter,
            deduplicate,
        )
        callback = self._async_embedding_callback
        if callback is None:
            return await asyncio.to_thread(self._trim_context, arguments)

        prepared = await asyncio.to_thread(self._prepare_context, arguments)
        if isinstance(prepared, ContextTrimResult):
            return prepared
        if prepared.request.strategy not in {Strategy.SEMANTIC, Strategy.HYBRID}:
            return await asyncio.to_thread(self._complete_context, prepared)

        passages = await asyncio.to_thread(_contextual_ranking_texts, prepared.segments)
        batch = await asyncio.to_thread(
            _prepare_passage_batch,
            passages,
            prepared.request.deduplicate,
        )
        output = await invoke_async_embedding_callback(
            callback,
            prepared.request.query or "",
            batch.passages,
        )
        return await asyncio.to_thread(
            self._complete_context_with_embedding_output,
            prepared,
            batch,
            output,
        )

    async def atrim_many(
        self,
        inputs: Sequence[TrimInput],
        *,
        deduplicate: bool = False,
    ) -> list[TrimResult]:
        """Trim independent sources, batching async semantic work by query.

        Inputs are prepared and completed independently. When an asynchronous embedding callback
        is configured, oversized semantic and hybrid inputs with the same normalized query share
        one callback invocation while retaining their own limits and source spans. Exact
        contextual passages can optionally be embedded once per query group.

        Args:
            inputs: Independent trim requests returned in the supplied order.
            deduplicate: Whether to send each exact contextual passage once per query group.

        Returns:
            One measured extractive result per input, in input order.

        Raises:
            TypeError: If an input is not a ``TrimInput`` or an argument has an unsupported type.
            ValueError: If an input has an invalid value or strategy/query combination.
            SemanticBackendError: If an explicitly requested semantic backend fails.
        """
        if not isinstance(deduplicate, bool):
            raise TypeError("deduplicate must be a bool")
        arguments = _batch_arguments(inputs)
        callback = self._async_embedding_callback
        if callback is None:
            return list(
                await asyncio.gather(
                    *(asyncio.to_thread(self._trim, argument) for argument in arguments)
                )
            )

        prepared_inputs = list(
            await asyncio.gather(
                *(asyncio.to_thread(self._prepare, argument) for argument in arguments)
            )
        )
        results: list[TrimResult | None] = [None] * len(prepared_inputs)
        ordinary: list[tuple[int, _PreparedTrim]] = []
        semantic: list[tuple[int, _PreparedTrim]] = []
        for index, prepared in enumerate(prepared_inputs):
            if isinstance(prepared, TrimResult):
                results[index] = prepared
            elif prepared.request.strategy in {Strategy.SEMANTIC, Strategy.HYBRID}:
                semantic.append((index, prepared))
            else:
                ordinary.append((index, prepared))

        ordinary_results = await asyncio.gather(
            *(asyncio.to_thread(self._complete, prepared) for _, prepared in ordinary)
        )
        for (index, _), result in zip(ordinary, ordinary_results, strict=True):
            results[index] = result

        contextual_passages = await asyncio.gather(
            *(
                asyncio.to_thread(_contextual_ranking_texts, prepared.segments)
                for _, prepared in semantic
            )
        )
        groups: dict[str, list[tuple[int, _PreparedTrim, list[str]]]] = {}
        for (index, prepared), passages in zip(semantic, contextual_passages, strict=True):
            groups.setdefault(prepared.request.query or "", []).append((index, prepared, passages))

        for query, group in groups.items():
            passages = [passage for _, _, batch in group for passage in batch]
            batch = _prepare_passage_batch(passages, deduplicate)
            output = await invoke_async_embedding_callback(callback, query, batch.passages)
            vectors = await asyncio.to_thread(
                normalize_callback_output,
                output,
                len(batch.passages),
            )
            offset = 0
            vector_requests: list[tuple[int, _PreparedTrim, _SemanticVectors]] = []
            for index, prepared, request_passages in group:
                end = offset + len(request_passages)
                vector_requests.append(
                    (
                        index,
                        prepared,
                        _remap_semantic_vectors(vectors, batch.rows[offset:end]),
                    )
                )
                offset = end
            completed = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self._complete_with_semantic_vectors,
                        prepared,
                        request_vectors,
                    )
                    for _, prepared, request_vectors in vector_requests
                )
            )
            for (index, _, _), result in zip(vector_requests, completed, strict=True):
                results[index] = result

        return cast(list[TrimResult], results)

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

    def _trim_context(self, arguments: _ContextArguments) -> ContextTrimResult:
        """Run one synchronous shared-context call through every processing stage.

        Args:
            arguments: Unvalidated public shared-context values.

        Returns:
            Input-aligned excerpts and aggregate measurements.
        """
        prepared = self._prepare_context(arguments)
        if isinstance(prepared, ContextTrimResult):
            return prepared
        return self._complete_context(prepared)

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

    def _prepare_context(
        self,
        arguments: _ContextArguments,
    ) -> ContextTrimResult | _PreparedContext:
        """Validate and prepare many sources without invoking semantic inference.

        Args:
            arguments: Unvalidated shared-context values.

        Returns:
            An early result or prepared oversized aggregate input.
        """
        resolved_unit = _parse_unit(arguments.unit)
        resolved_strategy, normalized_query = _resolve_strategy(
            _parse_strategy(arguments.strategy),
            arguments.query,
        )
        request = _ContextRequest(
            arguments.sources,
            arguments.limit,
            resolved_unit,
            resolved_strategy,
            normalized_query,
            arguments.token_counter,
            arguments.deduplicate,
        )
        _validate_context_request(request)
        measurer = Measurer(
            resolved_unit,
            self.config.token_encoding,
            arguments.token_counter,
        )
        input_counts = tuple(measurer.count(source) for source in arguments.sources)
        prepared = _PreparedContext(request, input_counts, [], (), measurer)
        empty_outputs = tuple(_ComposedOutput("", ()) for _ in arguments.sources)
        if arguments.limit == 0:
            return _context_result(prepared, empty_outputs)
        if sum(input_counts) <= arguments.limit:
            outputs = tuple(
                _ComposedOutput(source, (SourceSpan(0, len(source)),) if source else ())
                for source in arguments.sources
            )
            return _context_result(prepared, outputs)

        segments, source_indexes = _prepare_context_candidates(
            arguments.sources,
            resolved_strategy,
        )
        return _PreparedContext(request, input_counts, segments, source_indexes, measurer)

    def _complete(self, prepared: _PreparedTrim) -> TrimResult:
        """Rank and select one prepared input through a synchronous backend.

        Args:
            prepared: Validated, measured, and segmented input.

        Returns:
            Measured extractive trimming result.
        """
        request = _ranking_request(prepared)
        return self._select(prepared, self._rank(request))

    def _complete_context(self, prepared: _PreparedContext) -> ContextTrimResult:
        """Rank and select one prepared shared-context request synchronously.

        Args:
            prepared: Validated and segmented source collection.

        Returns:
            Input-aligned excerpts and aggregate measurements.
        """
        request = _context_ranking_request(prepared)
        if (
            request.strategy in {Strategy.SEMANTIC, Strategy.HYBRID}
            and self._async_embedding_callback is not None
        ):
            raise TypeError(
                "trim_context() cannot use async_embedding_callback; use atrim_context()"
            )
        return self._select_context(prepared, self._rank(request))

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
        return self._complete_with_semantic_vectors(prepared, vectors)

    def _complete_with_semantic_vectors(
        self,
        prepared: _PreparedTrim,
        vectors: _SemanticVectors,
    ) -> TrimResult:
        """Rank and select one prepared input from normalized semantic vectors.

        Args:
            prepared: Validated, measured, and segmented input.
            vectors: Query and candidate vectors already validated by Trimwise.

        Returns:
            Measured extractive trimming result.
        """
        ranking = _rank_with_semantic_vectors(_ranking_request(prepared), vectors)
        return self._select(prepared, ranking)

    def _complete_context_with_embedding_output(
        self,
        prepared: _PreparedContext,
        batch: _PassageBatch,
        output: EmbeddingOutput,
    ) -> ContextTrimResult:
        """Normalize and remap asynchronous context embeddings before selection.

        Args:
            prepared: Validated and segmented source collection.
            batch: Exact backend passages and original occurrence mapping.
            output: Caller-provided query and passage vectors.

        Returns:
            Input-aligned excerpts and aggregate measurements.
        """
        vectors = normalize_callback_output(output, len(batch.passages))
        return self._complete_context_with_semantic_vectors(
            prepared,
            _remap_semantic_vectors(vectors, batch.rows),
        )

    def _complete_context_with_semantic_vectors(
        self,
        prepared: _PreparedContext,
        vectors: _SemanticVectors,
    ) -> ContextTrimResult:
        """Rank and select context candidates from normalized semantic vectors.

        Args:
            prepared: Validated and segmented source collection.
            vectors: Query and candidate vectors in original candidate order.

        Returns:
            Input-aligned excerpts and aggregate measurements.
        """
        ranking = _rank_with_semantic_vectors(_context_ranking_request(prepared), vectors)
        return self._select_context(prepared, ranking)

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
        batch = _prepare_passage_batch(passages, request.deduplicate)
        if self._embedding_callback is not None:
            vectors = embed_with_callback(self._embedding_callback, query, batch.passages)
        else:
            vectors = self._semantic.embed(query, batch.passages)
        vectors = _remap_semantic_vectors(vectors, batch.rows)
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
            (request.text,),
            prepared.segments,
            tuple(0 for _ in prepared.segments),
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
            output[0],
            prepared.input_count,
            request,
            request.strategy,
            prepared.measurer,
        )

    def _select_context(
        self,
        prepared: _PreparedContext,
        ranking: CandidateRanking,
    ) -> ContextTrimResult:
        """Select and measure candidates competing across all input sources.

        Args:
            prepared: Validated and segmented source collection.
            ranking: Operation-wide candidate scores and similarity behavior.

        Returns:
            Input-aligned excerpts and aggregate measurements.
        """
        request = prepared.request
        context = _SelectionContext(
            request.sources,
            prepared.segments,
            prepared.source_indexes,
            ranking,
            prepared.measurer,
            request.limit,
            self.config.omission_marker,
            self.config.mmr_lambda,
        )
        fallback_index = None
        if request.strategy is Strategy.STRUCTURAL:
            outputs = _select_structural(context)
        else:
            fallback_index = _oversized_query_fallback_index(context)
            outputs = None if fallback_index is not None else _select_query_aware(context)
        if outputs is None:
            outputs = (
                _fallback_output(context)
                if fallback_index is None
                else _fallback_candidate_output(context, fallback_index)
            )
        return _context_result(prepared, outputs)


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


def _context_ranking_request(prepared: _PreparedContext) -> _RankingRequest:
    """Build one operation-wide ranking request for prepared sources.

    Args:
        prepared: Validated and segmented source collection.

    Returns:
        Strategy-specific ranking inputs.
    """
    request = prepared.request
    return _RankingRequest(
        prepared.segments,
        request.strategy,
        request.query,
        prepared.measurer,
        request.deduplicate,
    )


def _batch_arguments(inputs: Sequence[TrimInput]) -> list[_TrimArguments]:
    """Convert public batch requests into the ordinary trim call shape.

    Args:
        inputs: User-provided sequence of independent trim requests.

    Returns:
        Unvalidated arguments ready for the established preparation path.

    Raises:
        TypeError: If the input is not a sequence of ``TrimInput`` values.
    """
    if not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence of TrimInput")
    arguments: list[_TrimArguments] = []
    for item in inputs:
        if not isinstance(item, TrimInput):
            raise TypeError("inputs must contain only TrimInput values")
        arguments.append(
            _TrimArguments(
                item.text,
                item.limit,
                item.unit,
                item.strategy,
                item.query,
                item.token_counter,
            )
        )
    return arguments


def _snapshot_sources(sources: Sequence[str]) -> tuple[str, ...]:
    """Validate and snapshot the explicit multi-source collection contract.

    Args:
        sources: Public source collection.

    Returns:
        Stable input-order source tuple.

    Raises:
        TypeError: If the value is not a non-string sequence of strings.
    """
    if isinstance(sources, str) or not isinstance(sources, Sequence):
        raise TypeError("sources must be a sequence of strings")
    snapshot = tuple(sources)
    if any(not isinstance(source, str) for source in snapshot):
        raise TypeError("sources must contain only strings")
    return snapshot


def _validate_deduplicate(deduplicate: bool) -> None:
    """Require an explicit Boolean deduplication choice.

    Args:
        deduplicate: Public exact-passage batching option.

    Raises:
        TypeError: If truthiness could accidentally enable deduplication.
    """
    if not isinstance(deduplicate, bool):
        raise TypeError("deduplicate must be a bool")


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


def _validate_context_request(request: _ContextRequest) -> None:
    """Reuse ordinary budget validation for one shared-context request.

    Args:
        request: Normalized multi-source request.
    """
    _validate_request(
        _TrimRequest(
            "",
            request.limit,
            request.unit,
            request.strategy,
            request.query,
            request.token_counter,
        )
    )
    _validate_deduplicate(request.deduplicate)


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


def _context_result(
    prepared: _PreparedContext,
    outputs: tuple[_ComposedOutput, ...],
) -> ContextTrimResult:
    """Measure input-aligned outputs and enforce their summed hard limit.

    Args:
        prepared: Validated request, source counts, and shared measurer.
        outputs: One composed output for every original source.

    Returns:
        Immutable public shared-context result.

    Raises:
        RuntimeError: If output alignment is broken or the aggregate limit is exceeded.
    """
    request = prepared.request
    if len(outputs) != len(request.sources):
        raise RuntimeError("internal context output alignment failed")
    source_results = tuple(
        ContextSourceResult(
            source_index,
            output.text,
            prepared.input_counts[source_index],
            _text_count(prepared.measurer, output.text),
            output.text != request.sources[source_index],
            output.spans,
        )
        for source_index, output in enumerate(outputs)
    )
    output_count = sum(source.output_count for source in source_results)
    if output_count > request.limit:
        raise RuntimeError("internal composition exceeded the requested limit")
    return ContextTrimResult(
        source_results,
        sum(prepared.input_counts),
        output_count,
        request.limit,
        request.unit,
        request.strategy,
        any(source.trimmed for source in source_results),
    )
