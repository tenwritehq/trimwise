"""Calculate paired bootstrap differences from saved compression outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from benchmark.datasets.loader import iter_cases
from benchmark.metrics.evidence import score_case

COMPARATORS = ("llmlingua", "longllmlingua", "recomp_extractive")
REFERENCE_SETS = {
    "legacy-v1.1": ("trimwise_hybrid",),
    "strict-v1.2": ("trimwise_lexical", "trimwise_hybrid"),
}
METRIC_SETS = {
    "legacy-v1.1": ("case_pass",),
    "strict-v1.2": (
        "normalized_contiguous_case_pass",
        "local_ordered_80_case_pass",
        "local_ordered_90_case_pass",
    ),
}
DEFAULT_OUTPUTS = {
    "legacy-v1.1": "results/position_controlled_160_paired_stats.csv",
    "strict-v1.2": "results/position_controlled_160_evidence_sensitivity_v1_2_paired_stats.csv",
}
BOOTSTRAP_SAMPLES = 10_000
SEED = 20260728


def _results_by_key(
    path: Path,
    cases: dict[str, dict[str, Any]],
    metrics: tuple[str, ...],
    references: tuple[str, ...],
) -> dict[tuple[str, str, int, str], bool]:
    """Load successful query-aware outcomes keyed by case, method, budget, and metric.

    Args:
        path: Compression JSONL produced by the benchmark runner.
        cases: Dataset cases keyed by stable case identifier.
        metrics: Scorer fields to retain for the requested analysis.
        references: Trimwise configurations included in the requested analysis.

    Returns:
        Re-scored binary outcomes for the requested query-aware methods and metrics.

    Raises:
        ValueError: If a row is duplicated, failed, or references an unknown case.
    """
    outcomes: dict[tuple[str, str, int, str], bool] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            method = str(row.get("method_id"))
            if method not in {*COMPARATORS, *references} or not row.get("query_aware", False):
                continue
            if row.get("status") != "success":
                raise ValueError(f"paired analysis requires success: {row['case_id']} {method}")
            case_id = str(row["case_id"])
            case = cases.get(case_id)
            if case is None:
                raise ValueError(f"unknown case: {case_id}")
            budget = int(row["budget"])
            scores = score_case(case, str(row["output"]), budget)
            for metric in metrics:
                key = case_id, method, budget, metric
                if key in outcomes:
                    raise ValueError(f"duplicate compression result: {key}")
                outcomes[key] = bool(scores[metric])
    return outcomes


def _bootstrap_interval(differences: np.ndarray) -> tuple[float, float]:
    """Return a fixed-seed percentile interval for a paired mean difference.

    Args:
        differences: Per-case reference-minus-comparator binary differences.

    Returns:
        Lower and upper 95% percentile-bootstrap interval bounds in percentage points.
    """
    generator = np.random.default_rng(SEED)
    sampled = generator.choice(
        differences, size=(BOOTSTRAP_SAMPLES, len(differences)), replace=True
    )
    interval = np.quantile(sampled.mean(axis=1) * 100, [0.025, 0.975])
    return float(interval[0]), float(interval[1])


def _rows(
    outcomes: dict[tuple[str, str, int, str], bool],
    case_ids: list[str],
    metrics: tuple[str, ...],
    references: tuple[str, ...],
    include_metric: bool,
) -> list[dict[str, object]]:
    """Build paired Trimwise-versus-adapter statistics for each metric and budget.

    Args:
        outcomes: Re-scored main-comparison case outcomes.
        case_ids: Expected evaluation-case identifiers in stable order.
        metrics: Scorer fields represented in the output rows.
        references: Trimwise configurations represented in the output rows.
        include_metric: Whether the output needs a metric identifier column.

    Returns:
        One summary row for each metric, Trimwise reference, comparator, and budget.

    Raises:
        ValueError: If either member of a required pair is absent.
    """
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for budget in (128, 256, 512, 1024):
            for reference_method in references:
                for comparator_method in COMPARATORS:
                    reference = np.array(
                        [
                            outcomes[(case_id, reference_method, budget, metric)]
                            for case_id in case_ids
                        ],
                        dtype=float,
                    )
                    comparator = np.array(
                        [
                            outcomes[(case_id, comparator_method, budget, metric)]
                            for case_id in case_ids
                        ],
                        dtype=float,
                    )
                    lower, upper = _bootstrap_interval(reference - comparator)
                    row: dict[str, object] = {
                        "budget": budget,
                        "reference": reference_method,
                        "comparator": comparator_method,
                        "cases": len(case_ids),
                        "reference_case_pass_rate": round(float(reference.mean()), 6),
                        "comparator_case_pass_rate": round(float(comparator.mean()), 6),
                        "difference_percentage_points": round(
                            float((reference - comparator).mean() * 100), 3
                        ),
                        "ci_95_lower_percentage_points": round(lower, 3),
                        "ci_95_upper_percentage_points": round(upper, 3),
                        "reference_only_passes": int(((reference == 1) & (comparator == 0)).sum()),
                        "comparator_only_passes": int(((reference == 0) & (comparator == 1)).sum()),
                    }
                    if include_metric:
                        row = {"metric": metric, **row}
                    rows.append(row)
    return rows


def main() -> None:
    """Write fixed-seed paired bootstrap statistics for one named metric set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/position_controlled_160_results.jsonl")
    parser.add_argument("--dataset", default="data/position_controlled_160.jsonl")
    parser.add_argument("--metric-set", choices=tuple(METRIC_SETS), default="legacy-v1.1")
    parser.add_argument("--output")
    args = parser.parse_args()
    cases = {case["case_id"]: case for case in iter_cases(args.dataset)}
    metrics = METRIC_SETS[args.metric_set]
    references = REFERENCE_SETS[args.metric_set]
    outcomes = _results_by_key(Path(args.input), cases, metrics, references)
    rows = _rows(
        outcomes,
        sorted(cases),
        metrics,
        references,
        include_metric=args.metric_set == "strict-v1.2",
    )
    destination = Path(args.output or DEFAULT_OUTPUTS[args.metric_set])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
