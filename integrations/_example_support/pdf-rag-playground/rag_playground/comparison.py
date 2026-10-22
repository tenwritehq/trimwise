"""Retrieve once, freeze evidence, and compare two conditions without hidden truncation."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from rag_playground.backends import AnswerBackend
from rag_playground.models import Comparison, Evidence, RunSettings
from rag_playground.retrieval import Corpus


def render_context(documents: tuple[Evidence, ...]) -> str:
    """Render exact condition evidence and stable source labels.

    Args:
        documents: Original or compressed evidence with preassigned citation labels.

    Returns:
        Complete context, including source headers and separators.
    """
    return "\n\n".join(
        f"[{document.metadata['citation']}] {document.metadata['filename']!r}, "
        f"page {document.metadata['page']}\n{document.page_content}"
        for document in documents
    )


def compare(corpus: Corpus, backend: AnswerBackend, request: tuple[str, RunSettings]) -> Comparison:
    """Use the same retrieval, model, instructions, and settings for both conditions.

    Args:
        corpus: One completed, session-owned local index.
        backend: Local model or explicitly labelled test backend.
        request: Question and shared comparison settings.

    Returns:
        A run preserving partial failures and exact source references.

    Raises:
        ValueError: If the question is blank or exceeds the playground limit.
    """
    question, settings = request
    question = question.strip()
    if not question or len(question) > 2000:
        raise ValueError("Ask a question containing 1-2,000 characters.")
    started = perf_counter()
    original = corpus.retrieve(question, settings.top_k)
    retrieval_seconds = perf_counter() - started
    for index, document in enumerate(original):
        document.metadata["citation"] = f"S{index + 1}"
    prepared = perf_counter()
    baseline = render_context(original)
    baseline_count = backend.count(baseline)
    baseline_prompt = backend.prompt_count(question, baseline)
    baseline_prepare = perf_counter() - prepared
    compressed_started = perf_counter()
    compressed = corpus.compress(original, (question, settings), backend.count)
    compression_seconds = perf_counter() - compressed_started
    prepared = perf_counter()
    trimmed = render_context(compressed)
    trimmed_count = backend.count(trimmed)
    trimmed_prompt = backend.prompt_count(question, trimmed)
    trimmed_prepare = perf_counter() - prepared
    run = Comparison(
        framework=corpus.framework,
        run_id=uuid4().hex,
        question=question,
        settings=settings,
        corpus_fingerprint=corpus.loaded.fingerprint,
        original=original,
        compressed=compressed,
        contexts={"baseline": baseline, "trimwise": trimmed},
        context_tokens={"baseline": baseline_count, "trimwise": trimmed_count},
        prompt_estimates={"baseline": baseline_prompt, "trimwise": trimmed_prompt},
        retrieval_seconds=retrieval_seconds,
        compression_seconds=compression_seconds,
        model_load_seconds=0,
        model_info=dict(backend.info),
        totals_seconds={
            "baseline": retrieval_seconds + baseline_prepare,
            "trimwise": retrieval_seconds + compression_seconds + trimmed_prepare,
        },
    )
    if not original:
        run.errors = dict.fromkeys(
            ("baseline", "trimwise"),
            "No evidence was retrieved. Re-index documents or change the question.",
        )
        return run
    for condition, count in run.prompt_estimates.items():
        if count + settings.output_tokens + 16 > backend.context_limit:
            message = (
                f"The {condition} prompt needs {count:,} tokens plus {settings.output_tokens} "
                f"answer tokens; the allowance is {backend.context_limit:,}. Reduce top-k or "
                "chunk size and re-index. Neither answer was generated; "
                "the baseline was not truncated."
            )
            run.errors = dict.fromkeys(("baseline", "trimwise"), message)
            return run
    for condition, context in run.contexts.items():
        started = perf_counter()
        try:
            run.generations[condition] = backend.generate(question, context, settings.output_tokens)
        except Exception as error:
            # Runtime errors may contain document text; expose only the exception type.
            run.errors[condition] = (
                f"Generation failed ({type(error).__name__}). Check the selected backend and retry."
            )
        run.totals_seconds[condition] += perf_counter() - started
    if len({generation.model for generation in run.generations.values()}) > 1:
        run.errors["comparison"] = (
            "The runtime returned different model IDs. This pair is not comparable."
        )
    run.model_info.update(backend.info)
    return run
