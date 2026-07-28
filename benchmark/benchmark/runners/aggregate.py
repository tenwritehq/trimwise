"""Aggregate JSONL benchmark rows into analysis-ready CSV summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.datasets.loader import iter_cases
from benchmark.metrics.evidence import score_case
from benchmark.metrics.qa import score_answer

ANSWER_TRACKS = frozenset({"adversarial", "evidence_qa", "real_source"})


def _median_or_none(values: pd.Series) -> float | None:
    """Return a median without warning when a metric is unavailable."""
    values = values.dropna()
    return float(values.median()) if not values.empty else None


def _p95_or_none(values: pd.Series) -> float | None:
    """Return p95 without warning when a metric is unavailable."""
    values = values.dropna()
    return float(values.quantile(0.95)) if not values.empty else None


def _maximum_or_none(values: pd.Series) -> float | None:
    """Return a maximum without warning when a metric is unavailable."""
    values = values.dropna()
    return float(values.max()) if not values.empty else None


def _sum_or_zero(values: pd.Series) -> float:
    """Return a numeric sum while treating absent legacy telemetry as zero."""
    values = values.dropna()
    return float(values.sum()) if not values.empty else 0.0


def _case_lookup(dataset: str) -> dict[str, dict[str, Any]]:
    """Load benchmark cases by identifier for result re-scoring.

    Args:
        dataset: JSONL dataset used to produce the supplied benchmark rows.

    Returns:
        Dataset cases keyed by their stable case identifiers.
    """
    return {case["case_id"]: case for case in iter_cases(dataset)}


def _rescore_compression_rows(
    rows: list[dict[str, object]], cases: dict[str, dict[str, Any]]
) -> None:
    """Refresh source-evidence metrics from saved compression outputs.

    Args:
        rows: Compression result rows to update before aggregation.
        cases: Benchmark cases keyed by case identifier.
    """
    for row in rows:
        if row.get("status") != "success":
            continue
        case = cases.get(str(row.get("case_id")))
        if case is None:
            raise ValueError(f"compression result references unknown case: {row.get('case_id')}")
        row.update(score_case(case, str(row.get("output", "")), int(row["budget"])))


def _rescore_qa_rows(rows: list[dict[str, object]], cases: dict[str, dict[str, Any]]) -> None:
    """Refresh saved QA metrics from the final assistant completion.

    Args:
        rows: QA result rows to update before aggregation.
        cases: Benchmark cases keyed by case identifier.
    """
    for row in rows:
        if row.get("qa_status") != "success":
            continue
        case = cases.get(str(row.get("case_id")))
        if case is None:
            raise ValueError(f"QA result references unknown case: {row.get('case_id')}")
        row.update(score_answer(case, str(row.get("qa_output", ""))))


def main() -> None:
    """Write grouped quality, budget, latency, memory, and failure statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/position_controlled_160_results.jsonl")
    parser.add_argument("--dataset", default="data/position_controlled_160.jsonl")
    parser.add_argument("--qa-input", action="append", default=[])
    parser.add_argument("--output", default="results/position_controlled_160_summary.csv")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    cases: dict[str, dict[str, Any]] | None = None
    if any(row.get("status") == "success" and "output" in row for row in rows):
        cases = _case_lookup(args.dataset)
        _rescore_compression_rows(rows, cases)
    frame = pd.DataFrame(rows)
    raw_metadata = frame.pop("metadata") if "metadata" in frame else pd.Series(dtype=object)
    metadata = pd.json_normalize(raw_metadata).add_prefix("metadata_")
    frame = pd.concat([frame.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)
    for column in [
        "metadata_cold_call",
        "metadata_gpu_peak_allocated_mb",
        "metadata_gpu_peak_reserved_mb",
        "metadata_thermal_wait_ms",
        "metadata_thermal_max_temperature_c",
        "metadata_thermal_hard_limit_event",
        "case_pass",
        "required_span_coverage",
        "exact_required_span_coverage",
        "exact_required_evidence_success",
        "ordered_step_coverage",
        "ordered_step_ordered",
    ]:
        if column not in frame:
            frame[column] = pd.NA
    if "evidence_position" not in frame:
        frame["evidence_position"] = "unknown"
    frame["evidence_position"] = frame["evidence_position"].fillna("unknown")
    frame["case_pass"] = frame["status"].eq("success") & frame["case_pass"].eq(True)
    summary = (
        frame.groupby(
            ["method_id", "query_aware", "track", "evidence_position", "budget"],
            dropna=False,
        )
        .agg(
            rows=("case_id", "count"),
            success_rate=("status", lambda values: (values == "success").mean()),
            failures=("status", lambda values: (values != "success").sum()),
            evidence_recall=("evidence_recall", "mean"),
            evidence_f1=("evidence_f1", "mean"),
            all_evidence=("all_required_evidence_success", "mean"),
            case_pass_rate=("case_pass", "mean"),
            required_span_coverage=("required_span_coverage", "mean"),
            exact_required_span_coverage=("exact_required_span_coverage", "mean"),
            exact_required_evidence_success=("exact_required_evidence_success", "mean"),
            ordered_step_coverage=("ordered_step_coverage", "mean"),
            ordered_step_ordered=("ordered_step_ordered", "mean"),
            prohibited_phrase_rate=("contains_prohibited_phrase", "mean"),
            budget_violation_rate=("budget_violation", "mean"),
            median_compression_ratio=("compression_ratio", _median_or_none),
            median_token_savings=("token_savings", _median_or_none),
            evidence_recall_per_1k_output_tokens=("evidence_recall_per_1k_output_tokens", "mean"),
            median_latency_ms=("latency_ms", _median_or_none),
            p95_latency_ms=("latency_ms", _p95_or_none),
            median_output_tokens=("actual_tokens", _median_or_none),
            cold_calls=("metadata_cold_call", "sum"),
            median_gpu_peak_allocated_mb=("metadata_gpu_peak_allocated_mb", _median_or_none),
            median_gpu_peak_reserved_mb=("metadata_gpu_peak_reserved_mb", _median_or_none),
            total_thermal_wait_ms=("metadata_thermal_wait_ms", _sum_or_zero),
            max_thermal_temperature_c=("metadata_thermal_max_temperature_c", _maximum_or_none),
            thermal_hard_limit_events=("metadata_thermal_hard_limit_event", _sum_or_zero),
        )
        .reset_index()
        .sort_values(
            ["track", "evidence_position", "query_aware", "budget", "evidence_f1"],
            ascending=[True, True, False, True, False],
        )
    )
    track_case_pass = frame.groupby(
        ["method_id", "query_aware", "track", "budget"], as_index=False
    ).agg(case_pass_rate=("case_pass", "mean"))
    macro_case_pass = track_case_pass.groupby(
        ["method_id", "query_aware", "budget"], as_index=False
    ).agg(macro_case_pass_rate=("case_pass_rate", "mean"))
    summary = summary.merge(macro_case_pass, on=["method_id", "query_aware", "budget"])
    if args.qa_input:
        qa_rows = []
        for qa_input in args.qa_input:
            qa_path = Path(qa_input)
            with qa_path.open("r", encoding="utf-8") as source:
                qa_rows.extend(json.loads(line) for line in source if line.strip())
        if any(row.get("qa_status") == "success" and "qa_output" in row for row in qa_rows):
            _rescore_qa_rows(qa_rows, cases or _case_lookup(args.dataset))
        qa_frame = pd.DataFrame(qa_rows)
        if "qa_status" in qa_frame:
            qa_frame = qa_frame[qa_frame["qa_status"].eq("success")]
        if "qa_model_id" not in qa_frame:
            qa_frame["qa_model_id"] = "legacy"
        if "qa_model" not in qa_frame:
            qa_frame["qa_model"] = qa_frame["qa_model_id"]
        if "query_aware" not in qa_frame:
            qa_frame["query_aware"] = True
        qa_cases = cases or _case_lookup(args.dataset)
        qa_positions = {
            case_id: case.get("metadata", {}).get("evidence_position", "unknown")
            for case_id, case in qa_cases.items()
        }
        qa_frame["evidence_position"] = qa_frame["case_id"].map(qa_positions).fillna("unknown")
        if "answer_match" not in qa_frame:
            qa_frame["answer_match"] = qa_frame["answer_exact_match"]
        qa_rows_have_track = "track" in qa_frame
        if not qa_rows_have_track:
            qa_frame["track"] = ""
        for column in [
            "qa_thermal_wait_ms",
            "qa_thermal_max_temperature_c",
            "qa_thermal_hard_limit_event",
        ]:
            if column not in qa_frame:
                qa_frame[column] = pd.NA
        if qa_rows_have_track:
            qa_frame = qa_frame[qa_frame["track"].isin(ANSWER_TRACKS)]
        qa_summary = (
            qa_frame.groupby(
                [
                    "qa_model_id",
                    "qa_model",
                    "method_id",
                    "query_aware",
                    "budget",
                    "track",
                    "evidence_position",
                ],
                dropna=False,
            )
            .agg(
                qa_rows=("case_id", "count"),
                qa_answer_match=("answer_match", "mean"),
                qa_token_f1=("answer_token_f1", "mean"),
                qa_prohibited_phrase_rate=("answer_prohibited_phrase", "mean"),
                median_qa_latency_ms=("qa_latency_ms", _median_or_none),
                total_qa_thermal_wait_ms=("qa_thermal_wait_ms", _sum_or_zero),
                max_qa_thermal_temperature_c=("qa_thermal_max_temperature_c", _maximum_or_none),
                qa_thermal_hard_limit_events=("qa_thermal_hard_limit_event", _sum_or_zero),
            )
            .reset_index()
        )
        summary = summary.merge(
            qa_summary[qa_summary["method_id"].ne("full_context")],
            on=["method_id", "query_aware", "budget", "track", "evidence_position"],
            how="left",
        )
        full_prompt = qa_summary[qa_summary["method_id"].eq("full_context")].copy()
        full_prompt = pd.concat(
            [
                full_prompt.assign(query_aware=True),
                full_prompt.assign(query_aware=False),
            ],
            ignore_index=True,
        )
        summary = pd.concat(
            [
                summary,
                full_prompt[
                    [
                        column
                        for column in summary
                        if column in full_prompt and full_prompt[column].notna().any()
                    ]
                ],
            ],
            ignore_index=True,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
