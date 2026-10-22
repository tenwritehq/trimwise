"""Record all frozen public-fixture comparisons using real local models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from rag_playground.backends import OllamaBackend
from rag_playground.comparison import compare
from rag_playground.documents import ChunkSettings, load_pdfs
from rag_playground.evaluation import evaluate
from rag_playground.exports import export_data
from rag_playground.models import RunSettings
from rag_playground.retrieval import CorpusBuilder
from rag_playground.session import FIXTURE


def run(builder: CorpusBuilder, framework: str) -> None:
    """Record every public fixture comparison through one native pipeline on CPU.

    Args:
        builder: This example's native index factory.
        framework: Framework name included in the output record.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/local-smoke.json"))
    args = parser.parse_args()
    cases = json.loads(FIXTURE.with_name("questions.json").read_text())
    corpus = builder(load_pdfs([FIXTURE], ChunkSettings()), None)
    started = perf_counter()
    backend = OllamaBackend()
    acquired = perf_counter() - started
    records = []
    try:
        for case in cases:
            run = compare(corpus, backend, (case["question"], RunSettings()))
            run.model_load_seconds = acquired
            support = case["support"]["quote"] if case["support"] else ""
            run.evaluation = evaluate(run, corpus, (case["reference_answer"], support))
            record = export_data(run, include_content=True)
            record["fixture_case"] = case["id"]
            records.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "purpose": "Public synthetic fixture smoke, not an accuracy benchmark.",
                        "framework": framework,
                        "chunk_settings": {"size": 1000, "overlap": 120},
                        "index_seconds": corpus.index_seconds,
                        "embedding_acquire_seconds": corpus.embedding_load_seconds,
                        "runs": records,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "context_tokens": run.context_tokens,
                        "seconds": run.totals_seconds,
                        "errors": run.errors,
                    }
                ),
                flush=True,
            )
    finally:
        backend.close()
        corpus.close()
    if any(record["errors"] for record in records):
        raise SystemExit("Failures retained in the smoke record; inspect before reporting success.")
