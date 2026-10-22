"""Offer explicit reference checks and provenance-based evidence checks, not accuracy claims."""

from __future__ import annotations

import re

from rag_playground.models import Comparison
from rag_playground.retrieval import Corpus


def _normalized(text: str) -> str:
    """Normalize case, punctuation, and whitespace for a deliberately narrow exact match.

    Args:
        text: Answer or reference text.

    Returns:
        Casefolded word sequence; no semantic matching is attempted.
    """
    return " ".join(re.findall(r"\w+", text.casefold()))


def evaluate(run: Comparison, corpus: Corpus, references: tuple[str, str]) -> dict[str, object]:
    """Separate retrieval coverage, compression retention, and answer string matching.

    Args:
        run: Completed or partially failed comparison.
        corpus: Original indexed chunks for identifying a retrieval miss.
        references: Optional reference answer and one exact supporting quote.

    Returns:
        Explicit checks with their scope; none is a factual-support accuracy score.
    """
    reference, quote = references
    reference, quote = reference.strip(), quote.strip()
    if len(reference) > 4000 or len(quote) > 2000:
        raise ValueError("Reference answers may contain 4,000 characters; supporting quotes 2,000.")
    results: dict[str, object] = {}
    for condition, documents in (("baseline", run.original), ("trimwise", run.compressed)):
        answer = run.generations.get(condition)
        if answer is None:
            continue
        citations = set(re.findall(r"\[(S\d+)\]", answer.text))
        valid = {str(document.metadata["citation"]) for document in documents}
        results[condition] = {
            "normalized_exact_match": _normalized(answer.text) == _normalized(reference)
            if reference
            else None,
            "cited_labels": sorted(citations),
            "unknown_citation_labels": sorted(citations - valid),
        }
    if quote:
        indexed = any(quote in document.page_content for document in corpus.loaded.chunks)
        retrieved = any(quote in document.page_content for document in run.original)
        retained = False
        for document in run.compressed:
            provenance = document.metadata["trimwise"]
            original = run.original[provenance["source_index"]].page_content
            for match in re.finditer(re.escape(quote), original):
                retained |= any(
                    span["start"] <= match.start() and span["end"] >= match.end()
                    for span in provenance["spans"]
                )
        results["supporting_quote"] = {
            "found_in_indexed_chunk": indexed,
            "found_in_retrieved_chunk": retrieved,
            "retained_in_one_source_span": retained,
            "interpretation": (
                (
                    "Quote not found wholly within an indexed chunk; check whitespace or "
                    "chunk boundaries."
                )
                if not indexed
                else "Retrieval missed the supplied quote."
                if not retrieved
                else "Quote survived compression."
                if retained
                else "Compression removed at least part of the supplied quote."
            ),
        }
    results["limitations"] = (
        "Exact answer matching is case/punctuation/whitespace normalized, not semantic grading. "
        "Quote checks require an exact contiguous quote within one chunk "
        "and one retained source span. "
        "Citation labels identify sources; label validity does not prove factual support. "
        "No automatic accuracy score is assigned to your documents."
    )
    return results
