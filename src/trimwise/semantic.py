"""Isolate optional FastEmbed loading, inference, and error translation."""

from __future__ import annotations

import gc
import threading
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

from trimwise.models import SemanticBackendError, TrimConfig

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

EmbeddingOutput: TypeAlias = tuple[object, Iterable[object]]
EmbeddingCallback: TypeAlias = Callable[[str, Sequence[str]], EmbeddingOutput]
AsyncEmbeddingCallback: TypeAlias = Callable[[str, Sequence[str]], Awaitable[EmbeddingOutput]]


class _EmbeddingModel(Protocol):
    """Describe the small FastEmbed surface used by Trimwise."""

    def query_embed(self, query: str, **kwargs: Any) -> Iterable[object]:
        """Generate query vectors.

        Args:
            query: User query text.
            **kwargs: Backend-specific inference options.

        Returns:
            Iterable containing a query vector.
        """
        ...

    def passage_embed(self, passages: list[str], **kwargs: Any) -> Iterable[object]:
        """Generate passage vectors.

        Args:
            passages: Candidate texts.
            **kwargs: Backend-specific inference options.

        Returns:
            Iterable of passage vectors.
        """
        ...


@dataclass(frozen=True, slots=True)
class _SemanticVectors:
    """Hold normalized query and passage vectors in one float32 matrix."""

    matrix: NDArray[np.float32]

    @property
    def passage_count(self) -> int:
        """Return the number of passage rows after the query row."""
        return int(self.matrix.shape[0] - 1)

    def query_scores(self) -> list[float]:
        """Score every passage against the query in one matrix-vector operation.

        Returns:
            Cosine similarities in passage order.
        """
        return cast(list[float], (self.matrix[1:] @ self.matrix[0]).tolist())

    def similarity(self, left_index: int, right_index: int) -> float:
        """Compare two normalized passage rows.

        Args:
            left_index: First passage index.
            right_index: Second passage index.

        Returns:
            Nonnegative cosine similarity.
        """
        passages = self.matrix[1:]
        return max(0.0, float(passages[left_index] @ passages[right_index]))

    def new_maximum_similarities(self) -> Sequence[float]:
        """Create float32 MMR state without importing NumPy on core paths.

        Returns:
            Zeroed similarity state for every passage.
        """
        import numpy as np

        return cast(Sequence[float], np.zeros(self.passage_count, dtype=np.float32))

    def update_maximum_similarities(
        self,
        maximum_similarities: Sequence[float],
        selected_index: int,
    ) -> None:
        """Update exact MMR redundancy with one vectorized maximum.

        Args:
            maximum_similarities: Mutable float32 state created by this vector set.
            selected_index: Newly selected passage index.
        """
        import numpy as np

        passages = self.matrix[1:]
        maximum_array = np.asarray(maximum_similarities, dtype=np.float32)
        np.maximum(
            maximum_array,
            passages @ passages[selected_index],
            out=maximum_array,
        )


def embed_with_callback(
    callback: EmbeddingCallback,
    query: str,
    passages: list[str],
) -> _SemanticVectors:
    """Invoke and validate a synchronous caller-provided embedding backend.

    Args:
        callback: Caller-owned synchronous embedding function.
        query: Nonblank semantic query.
        passages: Contextual candidate texts.

    Returns:
        Normalized query and passage vectors.

    Raises:
        SemanticBackendError: If callback inference or output validation fails.
    """
    try:
        output = callback(query, passages)
    except Exception as error:
        raise _callback_error(
            "inference",
            "check the caller-provided embedding backend",
        ) from error
    return normalize_callback_output(output, len(passages))


async def invoke_async_embedding_callback(
    callback: AsyncEmbeddingCallback,
    query: str,
    passages: list[str],
) -> EmbeddingOutput:
    """Await one caller-provided embedding batch on the active event loop.

    Args:
        callback: Caller-owned asynchronous embedding function.
        query: Nonblank semantic query.
        passages: Contextual candidate texts.

    Returns:
        Unvalidated query and passage vectors.

    Raises:
        SemanticBackendError: If asynchronous callback inference fails.
    """
    try:
        return await callback(query, passages)
    except Exception as error:
        raise _callback_error(
            "inference",
            "check the caller-provided asynchronous embedding backend",
        ) from error


def normalize_callback_output(
    output: EmbeddingOutput,
    expected_passage_count: int,
) -> _SemanticVectors:
    """Validate and normalize one caller-provided embedding result.

    Args:
        output: Query vector paired with passage vectors.
        expected_passage_count: Required number of passage rows.

    Returns:
        Normalized query and passage vectors.

    Raises:
        SemanticBackendError: If vector count, shape, or values are invalid.
    """
    try:
        query_vector, passage_output = output
        passage_vectors = list(passage_output)
        if len(passage_vectors) != expected_passage_count:
            raise ValueError("embedding callback returned an unexpected vector count")
        return _normalize_vectors(query_vector, passage_vectors)
    except Exception as error:
        raise _callback_error(
            "output",
            "return one finite query vector and one equal-dimension vector per passage",
        ) from error


def _callback_error(stage: str, hint: str) -> SemanticBackendError:
    """Create a stable error for a caller-owned semantic backend.

    Args:
        stage: Callback operation that failed.
        hint: Concise remediation guidance.

    Returns:
        Public semantic backend exception.
    """
    return SemanticBackendError(f"Custom embedding callback {stage} failed; {hint}.")


def _normalize_vectors(query: object, passages: list[object]) -> _SemanticVectors:
    """Stack, validate, and normalize backend vectors as float32.

    Args:
        query: Materialized backend query vector.
        passages: Materialized backend passage vectors.

    Returns:
        Validated normalized query and passage matrix.

    Raises:
        ValueError: If vectors are empty, inconsistent, or non-finite.
    """
    import numpy as np

    rows = [np.asarray(vector, dtype=np.float32) for vector in [query, *passages]]
    matrix = np.stack(rows)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("embedding vectors are empty, non-finite, or inconsistent")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("embedding vector norms are non-finite")
    np.divide(matrix, norms, out=matrix, where=norms != 0)
    return _SemanticVectors(matrix)


class SemanticEmbedder:
    """Lazily cache and serialize one FastEmbed model per trimmer."""

    def __init__(self, config: TrimConfig) -> None:
        """Store model configuration and create the per-instance lock.

        Args:
            config: Validated Trimwise configuration.
        """
        self._config = config
        self._model: _EmbeddingModel | None = None
        self._lock = threading.Lock()

    def embed(
        self,
        query: str,
        passages: list[str],
    ) -> _SemanticVectors:
        """Embed a query and passages while serializing backend access.

        Args:
            query: Nonblank query text.
            passages: Candidate source units.

        Returns:
            One normalized float32 matrix containing the query and passages.

        Raises:
            SemanticBackendError: If loading or inference fails.
        """
        with self._lock:
            try:
                model = self._get_model()
                query_vector = self._embed_query(model, query)
                passage_vectors = self._embed_passages(model, passages)
                return self._normalize_vectors(query_vector, passage_vectors)
            finally:
                gc.collect()

    def _get_model(self) -> _EmbeddingModel:
        """Load FastEmbed and initialize the configured model once.

        Returns:
            Cached FastEmbed model.

        Raises:
            SemanticBackendError: If the optional package or model is unavailable.
        """
        if self._model is not None:
            return self._model
        try:
            module = import_module("fastembed")
        except Exception as error:
            raise self._error(
                "import",
                "install trimwise[semantic] or trimwise[semantic-gpu]",
            ) from error
        try:
            model_type = cast(Any, module).TextEmbedding
            self._model = cast(
                _EmbeddingModel,
                model_type(
                    model_name=self._config.embedding_model,
                    **dict(self._config.fastembed_options),
                ),
            )
        except Exception as error:
            raise self._error(
                "initialization",
                "check model, cache, and provider options",
            ) from error
        return self._model

    def _embed_query(self, model: _EmbeddingModel, query: str) -> object:
        """Materialize one query embedding inside the backend lock.

        Args:
            model: Initialized FastEmbed model.
            query: Nonblank user query.

        Returns:
            Backend query vector without boxing its values.

        Raises:
            SemanticBackendError: If query inference returns no usable vector.
        """
        try:
            vectors = list(model.query_embed(query))
            if len(vectors) != 1:
                raise ValueError("query_embed must return exactly one vector")
            return vectors[0]
        except Exception as error:
            raise self._error(
                "query inference",
                "check model and provider compatibility",
            ) from error

    def _embed_passages(
        self,
        model: _EmbeddingModel,
        passages: list[str],
    ) -> list[object]:
        """Materialize all passage embeddings inside the backend lock.

        Args:
            model: Initialized FastEmbed model.
            passages: Candidate source units.

        Returns:
            Backend vectors in passage order without boxed-value conversion.

        Raises:
            SemanticBackendError: If inference fails or returns the wrong count.
        """
        try:
            vectors = list(
                model.passage_embed(
                    passages,
                    batch_size=self._config.embedding_batch_size,
                )
            )
            if len(vectors) != len(passages):
                raise ValueError("passage_embed returned an unexpected vector count")
            return vectors
        except Exception as error:
            raise self._error(
                "passage inference",
                "check model, batch size, and provider",
            ) from error

    def _error(self, stage: str, hint: str) -> SemanticBackendError:
        """Create a consistent staged backend error.

        Args:
            stage: Backend operation that failed.
            hint: Concise remediation guidance.

        Returns:
            Public semantic backend exception.
        """
        return SemanticBackendError(
            f"FastEmbed {stage} failed for {self._config.embedding_model!r}; {hint}."
        )

    def _normalize_vectors(
        self,
        query: object,
        passages: list[object],
    ) -> _SemanticVectors:
        """Stack, validate, and normalize backend vectors as float32.

        Args:
            query: Materialized backend query vector.
            passages: Materialized backend passage vectors.

        Returns:
            Validated normalized query and passage matrix.

        Raises:
            SemanticBackendError: If vectors cannot be compared safely.
        """
        try:
            return _normalize_vectors(query, passages)
        except Exception as error:
            raise self._error(
                "inference output",
                "check model and provider compatibility",
            ) from error
