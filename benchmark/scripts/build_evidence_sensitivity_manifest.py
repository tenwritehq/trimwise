"""Freeze v1.2 source-evidence sensitivity inputs before aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from statistics import median
from typing import Any

from benchmark.datasets.loader import iter_cases
from benchmark.metrics.evidence import _normalized_text, _tokens

PROTOCOL_FILES = (
    ".gitignore",
    "benchmark/metrics/evidence.py",
    "benchmark/runners/aggregate.py",
    "tests/test_evidence.py",
    "tests/test_aggregate.py",
    "scripts/build_evidence_sensitivity_manifest.py",
    "README.md",
    "data/manifests/evidence_sensitivity_v1_2_protocol.md",
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File whose bytes should be identified.

    Returns:
        Lowercase SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(repo_root: Path, *arguments: str) -> str:
    """Run one Git command and return its stripped standard output.

    Args:
        repo_root: Repository root containing the frozen benchmark source.
        *arguments: Arguments after the ``git`` executable.

    Returns:
        Standard output without trailing whitespace.

    Raises:
        RuntimeError: If Git cannot complete the requested command.
    """
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _require_committed_protocol(repo_root: Path) -> str:
    """Require the v1.2 implementation and protocol to be cleanly committed.

    Args:
        repo_root: Repository root containing benchmark files.

    Returns:
        Commit SHA that contains the protocol implementation.

    Raises:
        RuntimeError: If a required file is untracked or has unstaged/staged changes.
    """
    repo_paths = [f"benchmark/{path}" for path in PROTOCOL_FILES]
    try:
        for path in repo_paths:
            _git_output(repo_root, "ls-files", "--error-unmatch", path)
    except RuntimeError as exc:
        raise RuntimeError(
            "commit the v1.2 scorer, tests, and protocol before building its manifest"
        ) from exc
    changed = _git_output(repo_root, "status", "--porcelain", "--", *repo_paths)
    if changed:
        raise RuntimeError(
            "commit the v1.2 scorer, tests, and protocol before building its manifest"
        )
    return _git_output(repo_root, "rev-parse", "HEAD")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Load nonblank JSONL records from one frozen artifact.

    Args:
        path: JSONL artifact to read.

    Returns:
        Parsed rows in file order.
    """
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _nearest_rank_quantile(values: list[int], fraction: float) -> int:
    """Return the nearest-rank quantile used by the public diagnostics.

    Args:
        values: Sorted integer values.
        fraction: Quantile fraction in the closed interval from zero to one.

    Returns:
        Element at the nearest-rank quantile index.
    """
    return values[ceil(len(values) * fraction) - 1]


def _overlapping_span_pairs(case: dict[str, Any]) -> int:
    """Count required span pairs overlapping at least half of the shorter span.

    Args:
        case: One benchmark case with required evidence offsets.

    Returns:
        Number of substantially overlapping required-evidence span pairs.
    """
    spans = [item for item in case.get("gold_evidence", []) if item.get("required", True)]
    pairs = 0
    for index, left in enumerate(spans):
        for right in spans[index + 1 :]:
            left_start, left_end = int(left["start"]), int(left["end"])
            right_start, right_end = int(right["start"]), int(right["end"])
            overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
            shorter = min(left_end - left_start, right_end - right_start)
            pairs += bool(shorter and overlap / shorter >= 0.5)
    return pairs


def _dataset_diagnostics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe required evidence without examining any compressed outcome metric.

    Args:
        cases: Frozen benchmark cases.

    Returns:
        Required-span lengths, duplicate-text counts, and overlap counts.

    Raises:
        ValueError: If a required span is blank, tokenless, or absent from its source context.
    """
    token_lengths: list[int] = []
    duplicate_occurrences = 0
    overlap_pairs = 0
    span_count = 0
    spans_per_case: Counter[int] = Counter()
    for case in cases:
        required = [item for item in case.get("gold_evidence", []) if item.get("required", True)]
        spans_per_case[len(required)] += 1
        source = _normalized_text(str(case["context"]))
        for span in required:
            text = str(span["text"])
            normalized = _normalized_text(text)
            tokens = _tokens(text)
            if not normalized:
                raise ValueError(f"blank required evidence span in {case['case_id']}")
            if not tokens:
                raise ValueError(f"tokenless required evidence span in {case['case_id']}")
            occurrences = source.count(normalized)
            if not occurrences:
                raise ValueError(f"required evidence span absent from source in {case['case_id']}")
            span_count += 1
            token_lengths.append(len(tokens))
            duplicate_occurrences += occurrences > 1
        overlap_pairs += _overlapping_span_pairs(case)
    token_lengths.sort()
    return {
        "required_span_count": span_count,
        "required_span_token_length": {
            "minimum": token_lengths[0],
            "median": median(token_lengths),
            "p90": _nearest_rank_quantile(token_lengths, 0.9),
            "maximum": token_lengths[-1],
        },
        "cases_by_required_span_count": dict(sorted(spans_per_case.items())),
        "normalized_required_spans_with_multiple_source_occurrences": duplicate_occurrences,
        "substantially_overlapping_required_span_pairs": overlap_pairs,
    }


def _result_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe stable saved-row identities and status counts.

    Args:
        rows: Frozen compression rows.

    Returns:
        Row-count, status, and unique-identity diagnostics.

    Raises:
        ValueError: If compression result identities are not unique.
    """
    identities = [
        (row["case_id"], row["method_id"], int(row["budget"]), bool(row["query_aware"]))
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("frozen compression rows must have unique result identities")
    return {
        "row_count": len(rows),
        "unique_result_identity_count": len(set(identities)),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in rows).items())),
    }


def main() -> None:
    """Write a committed-protocol manifest before v1.2 aggregation begins."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/position_controlled_160.jsonl")
    parser.add_argument("--input", default="results/position_controlled_160_results.jsonl")
    parser.add_argument(
        "--output",
        default="data/manifests/evidence_sensitivity_v1_2_manifest.json",
    )
    args = parser.parse_args()

    benchmark_root = Path(__file__).resolve().parents[1]
    repo_root = benchmark_root.parent
    dataset = Path(args.dataset)
    input_rows = Path(args.input)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to replace frozen sensitivity manifest: {output}")
    revision = _require_committed_protocol(repo_root)
    cases = list(iter_cases(dataset))
    rows = _read_rows(input_rows)
    manifest = {
        "schema_version": 1,
        "analysis": "post_hoc_source_evidence_sensitivity_v1_2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "implementation_commit": revision,
        "python": sys.version.split()[0],
        "unicode_database": unicodedata.unidata_version,
        "inputs": {
            "dataset": {"path": str(dataset), "sha256": _sha256(dataset), "case_count": len(cases)},
            "compression_rows": {"path": str(input_rows), "sha256": _sha256(input_rows)},
        },
        "scorer_files": {
            path: _sha256(benchmark_root / path)
            for path in ("benchmark/metrics/evidence.py", "benchmark/runners/aggregate.py")
        },
        "dataset_diagnostics": _dataset_diagnostics(cases),
        "compression_row_diagnostics": _result_diagnostics(rows),
        "result_policy": {
            "compression_outputs_regenerated": False,
            "legacy_v1_1_artifacts_modified": False,
            "summary_output": (
                "results/position_controlled_160_evidence_sensitivity_v1_2_summary.csv"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {output}")


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
