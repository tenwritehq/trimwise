"""Verify aggregate output when benchmark dependencies are installed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

aggregate_runner = pytest.importorskip("benchmark.runners.aggregate")


def test_aggregate_reports_thermal_wait_and_peak_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Summarize thermal metadata separately from compression and QA latency."""
    compression_path = tmp_path / "compression.jsonl"
    dataset_path = tmp_path / "dataset.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    summary_path = tmp_path / "summary.csv"
    dataset_path.write_text(
        json.dumps(
            {
                "case_id": "case",
                "context": "alpha beta gamma",
                "gold_evidence": [{"text": "alpha beta"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    compression_path.write_text(
        json.dumps(
            {
                "case_id": "case",
                "method_id": "fixed",
                "query_aware": True,
                "track": "test",
                "budget": 128,
                "status": "success",
                "output": "alpha beta",
                "evidence_recall": 1.0,
                "evidence_f1": 1.0,
                "all_required_evidence_success": True,
                "contains_prohibited_phrase": False,
                "budget_violation": False,
                "compression_ratio": 0.5,
                "token_savings": 10,
                "evidence_recall_per_1k_output_tokens": 1.0,
                "latency_ms": 7.0,
                "actual_tokens": 128,
                "metadata": {
                    "cold_call": True,
                    "thermal_wait_ms": 2_000.0,
                    "thermal_max_temperature_c": 68.0,
                    "thermal_hard_limit_event": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qa_path.write_text(
        json.dumps(
            {
                "case_id": "case",
                "method_id": "fixed",
                "budget": 128,
                "qa_model_id": "gpt_5_4_nano",
                "qa_model": "gpt-5.4-nano-2026-03-17",
                "answer_exact_match": True,
                "answer_token_f1": 1.0,
                "answer_prohibited_phrase": False,
                "qa_latency_ms": 3.0,
                "qa_thermal_wait_ms": 2_000.0,
                "qa_thermal_max_temperature_c": 68.0,
                "qa_thermal_hard_limit_event": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate",
            "--input",
            str(compression_path),
            "--dataset",
            str(dataset_path),
            "--qa-input",
            str(qa_path),
            "--output",
            str(summary_path),
        ],
    )

    aggregate_runner.main()

    summary = summary_path.read_text(encoding="utf-8")
    assert "total_thermal_wait_ms" in summary
    assert "max_qa_thermal_temperature_c" in summary
    assert "normalized_contiguous_case_pass_rate" in summary
    assert "local_ordered_90_case_pass_rate" in summary
