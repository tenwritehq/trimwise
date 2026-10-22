"""Export explicit comparison evidence without leaking user content by default."""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import asdict
from typing import Any

from rag_playground.backends import INSTRUCTIONS
from rag_playground.models import Comparison


def export_data(run: Comparison, *, include_content: bool = False) -> dict[str, Any]:
    """Build a reproducible run record with an explicit document-content switch.

    Args:
        run: Comparison to export, including errors and optional human feedback.
        include_content: Include question, answers, filenames, and exact condition contexts.

    Returns:
        JSON-compatible, nonsecret run configuration, source references, and measurements.
    """
    sources = []
    for document in run.original:
        metadata = document.metadata
        source = {
            "chunk_id": document.id,
            "document_sha256": metadata["document_sha256"],
            "page": metadata["page"],
            "page_start": metadata["page_start"],
            "page_end": metadata["page_end"],
            "citation": metadata["citation"],
            "text_sha256": hashlib.sha256(document.page_content.encode()).hexdigest(),
        }
        if include_content:
            source.update(filename=metadata["filename"], text=document.page_content)
        sources.append(source)
    generations = {}
    for condition, generation in run.generations.items():
        item = asdict(generation)
        if not include_content:
            item.pop("text")
        generations[condition] = item
    versions = {}
    for name in (
        "trimwise",
        "langchain-trimwise",
        "langchain-core",
        "langchain-chroma",
        "chromadb",
        "llama-index-postprocessor-trimwise",
        "llama-index-core",
        "fastembed",
        "pypdf",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    result: dict[str, Any] = {
        "schema_version": 1,
        "framework": run.framework,
        "run_id": run.run_id,
        "content_included": include_content,
        "corpus_fingerprint": run.corpus_fingerprint,
        "settings": asdict(run.settings),
        "model": run.model_info,
        "sources": sources,
        "retained": [
            {"chunk_id": document.id, **document.metadata["trimwise"]}
            for document in run.compressed
        ],
        "context_tokens": run.context_tokens,
        "prompt_estimates": run.prompt_estimates,
        "timing_seconds": {
            "retrieval": run.retrieval_seconds,
            "compression": run.compression_seconds,
            "model_acquire": run.model_load_seconds,
            "condition_totals": run.totals_seconds,
        },
        "generations": generations,
        "errors": run.errors,
        "feedback": run.feedback,
        "evaluation": run.evaluation,
        "instructions": INSTRUCTIONS,
        "versions": versions,
        "timing_note": (
            "Single-run observations; index and model load are excluded from "
            "condition totals. Baseline generates first; cache/order effects are "
            "possible."
        ),
    }
    if include_content:
        result.update(question=run.question, contexts=run.contexts)
    return result
