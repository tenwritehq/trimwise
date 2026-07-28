"""BM25 chunk-ranking baseline with optional source-order reconstruction."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from benchmark.utils.tokens import count_tokens, get_encoding, token_prefix

from .base import CompressionResult, TimedAdapter


@dataclass(frozen=True, slots=True)
class Chunk:
    """Describe a token-window candidate from the input source.

    Attributes:
        text: Decoded window text.
        start: Inclusive token offset in the input.
        end: Exclusive token offset in the input.
    """

    text: str
    start: int
    end: int


def _terms(text: str) -> list[int]:
    """Normalize text into tokenizer IDs used as BM25 terms."""
    return get_encoding().encode(" " + " ".join(text.casefold().split()), disallowed_special=())


def _chunks(text: str, size: int, overlap: int) -> list[Chunk]:
    """Split source text into overlapping tokenizer windows."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must satisfy 0 <= overlap < size")
    enc = get_encoding()
    ids = enc.encode(text, disallowed_special=())
    step = size - overlap
    chunks: list[Chunk] = []
    for start_token in range(0, len(ids), step):
        part_ids = ids[start_token : start_token + size]
        if not part_ids:
            break
        part = enc.decode(part_ids)
        # Source offsets are approximate for this retrieval baseline; output is
        # still selected from decoded source-token windows.
        chunks.append(Chunk(part, start_token, start_token + len(part_ids)))
        if start_token + size >= len(ids):
            break
    return chunks


def _bm25(documents: list[list[int]], query: list[int]) -> list[float]:
    """Score tokenized documents against a tokenized query."""
    count = len(documents)
    if not count:
        return []
    avg_len = sum(map(len, documents)) / count or 1.0
    df = Counter(term for doc in documents for term in set(doc))
    scores: list[float] = []
    for doc in documents:
        tf = Counter(doc)
        norm = 1.5 * (1 - 0.75 + 0.75 * len(doc) / avg_len)
        score = 0.0
        for term in query:
            if not tf[term]:
                continue
            idf = math.log(1 + (count - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * 2.5 / (tf[term] + norm)
        scores.append(score)
    return scores


class BM25Adapter(TimedAdapter):
    """Select fixed-size BM25-ranked token windows."""

    query_aware = True

    def __init__(self, chunk_size: int = 128, overlap: int = 32, source_order: bool = True) -> None:
        """Configure window geometry and emitted ordering.

        Args:
            chunk_size: Token count in each candidate window.
            overlap: Shared token count between consecutive windows.
            source_order: Whether to restore selected windows to input order.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.source_order = source_order
        suffix = "source_order" if source_order else "rank_order"
        self.method_id = f"bm25_{chunk_size}_{overlap}_{suffix}"

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        chunks = _chunks(context, self.chunk_size, self.overlap)
        scores = _bm25([_terms(chunk.text) for chunk in chunks], _terms(query))
        ranked = sorted(range(len(chunks)), key=lambda i: (scores[i], -i), reverse=True)

        # Select by relevance first. Only after selection restore source order.
        selected: list[int] = []
        used = 0
        marker = "\n[…omitted…]\n"
        marker_cost = count_tokens(marker)
        for index in ranked:
            remaining = budget - used - (marker_cost if selected else 0)
            if remaining <= 0:
                break
            piece = chunks[index].text
            if count_tokens(piece) > remaining:
                piece = token_prefix(piece, remaining)
            if not piece:
                continue
            selected.append(index)
            used += count_tokens(piece) + (marker_cost if len(selected) > 1 else 0)
        output_order = sorted(selected) if self.source_order else selected
        output = marker.join(chunks[i].text for i in output_order)
        if count_tokens(output) > budget:
            output = token_prefix(output, budget)
        return CompressionResult(
            self.method_id,
            output,
            metadata={"selected": selected, "scores": scores, "source_order": self.source_order},
        )
