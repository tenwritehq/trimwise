"""Source-grounded quality and budget metrics for compression outputs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from benchmark.utils.tokens import count_tokens


def _tokens(text: str) -> list[str]:
    """Return normalized comparison tokens for lightweight evidence scoring."""
    return re.findall(r"[\w./:+-]+", text.casefold())


def overlap(reference: str, candidate: str) -> tuple[float, float, float]:
    """Return token recall, precision, and F1 for two text snippets."""
    gold = Counter(_tokens(reference))
    found = Counter(_tokens(candidate))
    common = sum((gold & found).values())
    recall = common / sum(gold.values()) if gold else 1.0
    precision = common / sum(found.values()) if found else (1.0 if not gold else 0.0)
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    return recall, precision, f1


def _normalized_text(text: str) -> str:
    """Normalize whitespace and casing for source-span comparisons.

    Args:
        text: Source or retained text to compare.

    Returns:
        A case-insensitive, whitespace-normalized representation.
    """
    return " ".join(text.casefold().split())


def _source_hit(reference: str, output: str) -> bool:
    """Return whether an output retains one required source span.

    Args:
        reference: Required source-backed evidence.
        output: Compression output to inspect.

    Returns:
        Whether the span is present exactly or retains at least 80% of its tokens.
    """
    exact = _normalized_text(reference) in _normalized_text(output)
    recall, _, _ = overlap(reference, output)
    return exact or recall >= 0.80


def _coverage(hits: list[bool]) -> float:
    """Return the fraction of retained required items.

    Args:
        hits: Per-item retention results.

    Returns:
        Mean retention, treating an empty requirement set as complete.
    """
    return sum(hits) / len(hits) if hits else 1.0


def _ordered_step_metrics(case: dict[str, Any], output: str) -> tuple[float | None, bool | None]:
    """Measure required procedure-step retention and source order.

    Args:
        case: Benchmark case that may define ordered procedure steps.
        output: Compression output to inspect.

    Returns:
        Coverage and in-order success, or ``None`` values when ordering is not applicable.
    """
    steps = [str(step) for step in case.get("ordered_steps", [])]
    if not steps:
        return None, None
    normalized_output = _normalized_text(output)
    positions = [normalized_output.find(_normalized_text(step)) for step in steps]
    coverage = _coverage([_source_hit(step, output) for step in steps])
    ordered = all(position >= 0 for position in positions) and positions == sorted(positions)
    return coverage, ordered


def score_case(case: dict[str, Any], output: str, budget: int) -> dict[str, Any]:
    """Score evidence retention, task outcomes, safety, and compression.

    Args:
        case: Benchmark case containing required evidence and task metadata.
        output: Compression output to evaluate.
        budget: Requested token limit.

    Returns:
        Source-evidence, task-outcome, safety, and budget metrics for one result.
    """
    spans = [item for item in case.get("gold_evidence", []) if item.get("required", True)]
    gold = "\n".join(str(item["text"]) for item in spans)
    recall, precision, f1 = overlap(gold, output)
    hits = [_source_hit(str(item["text"]), output) for item in spans]
    exact_hits = [str(item["text"]) in output for item in spans]
    ordered_step_coverage, ordered_step_ordered = _ordered_step_metrics(case, output)
    input_tokens = count_tokens(str(case.get("context", output)))
    actual = count_tokens(output)
    prohibited = [str(phrase) for phrase in case.get("prohibited_phrases", [])]
    normalized_prohibited = " ".join(output.casefold().split())
    prohibited_hits = [
        phrase
        for phrase in prohibited
        if " ".join(phrase.casefold().split()) in normalized_prohibited
    ]
    all_required_evidence = all(hits) if hits else True
    case_pass = all_required_evidence and not prohibited_hits and actual <= budget
    return {
        "evidence_recall": recall,
        "evidence_precision": precision,
        "evidence_f1": f1,
        "required_span_hits": hits,
        "required_span_coverage": _coverage(hits),
        "exact_required_span_coverage": _coverage(exact_hits),
        "exact_required_evidence_success": all(exact_hits) if exact_hits else True,
        "ordered_step_coverage": ordered_step_coverage,
        "ordered_step_ordered": ordered_step_ordered,
        "all_required_evidence_success": all_required_evidence,
        "requested_budget": budget,
        "input_tokens": input_tokens,
        "actual_tokens": actual,
        "compression_ratio": input_tokens / actual if actual else None,
        "token_savings": input_tokens - actual,
        "evidence_recall_per_1k_output_tokens": recall * 1000 / actual if actual else 0.0,
        "budget_violation": actual > budget,
        "prohibited_phrase_hits": prohibited_hits,
        "contains_prohibited_phrase": bool(prohibited_hits),
        "case_pass": case_pass,
    }
