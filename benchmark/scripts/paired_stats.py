"""Calculate paired case-pass differences from saved compression outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from benchmark.datasets.loader import iter_cases
from benchmark.metrics.evidence import score_case

METHODS = ("llmlingua", "longllmlingua", "recomp_extractive")
REFERENCE = "trimwise_hybrid"
BOOTSTRAP_SAMPLES = 10_000
SEED = 20260728


def _results_by_key(
    path: Path, cases: dict[str, dict[str, Any]]
) -> dict[tuple[str, str, int], bool]:
    """Load successful query-aware case-pass outcomes keyed by case, method, and budget.

    Args:
        path: Compression JSONL produced by the benchmark runner.
        cases: Dataset cases keyed by stable case identifier.

    Returns:
        Re-scored binary case-pass outcomes for the main query-aware methods.

    Raises:
        ValueError: If a row is duplicated, failed, or references an unknown case.
    """
    outcomes: dict[tuple[str, str, int], bool] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            method = str(row.get("method_id"))
            if method not in {*METHODS, REFERENCE} or not row.get("query_aware", False):
                continue
            if row.get("status") != "success":
                raise ValueError(f"paired analysis requires success: {row['case_id']} {method}")
            case_id = str(row["case_id"])
            case = cases.get(case_id)
            if case is None:
                raise ValueError(f"unknown case: {case_id}")
            key = case_id, method, int(row["budget"])
            if key in outcomes:
                raise ValueError(f"duplicate compression result: {key}")
            outcomes[key] = bool(score_case(case, str(row["output"]), key[2])["case_pass"])
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
    outcomes: dict[tuple[str, str, int], bool], case_ids: list[str]
) -> list[dict[str, object]]:
    """Build paired hybrid-versus-baseline statistics for each fixed token budget.

    Args:
        outcomes: Re-scored main-comparison case outcomes.
        case_ids: Expected evaluation-case identifiers in stable order.

    Returns:
        One summary row for each comparator and budget.

    Raises:
        ValueError: If either member of a required pair is absent.
    """
    rows: list[dict[str, object]] = []
    for budget in (128, 256, 512, 1024):
        for method in METHODS:
            reference = np.array(
                [outcomes[(case_id, REFERENCE, budget)] for case_id in case_ids],
                dtype=float,
            )
            comparator = np.array(
                [outcomes[(case_id, method, budget)] for case_id in case_ids],
                dtype=float,
            )
            lower, upper = _bootstrap_interval(reference - comparator)
            rows.append(
                {
                    "budget": budget,
                    "reference": REFERENCE,
                    "comparator": method,
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
            )
    return rows


def main() -> None:
    """Write paired fixed-seed bootstrap statistics for the frozen 160-case evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/position_controlled_160_results.jsonl")
    parser.add_argument("--dataset", default="data/position_controlled_160.jsonl")
    parser.add_argument("--output", default="results/position_controlled_160_paired_stats.csv")
    args = parser.parse_args()
    cases = {case["case_id"]: case for case in iter_cases(args.dataset)}
    outcomes = _results_by_key(Path(args.input), cases)
    rows = _rows(outcomes, sorted(cases))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
