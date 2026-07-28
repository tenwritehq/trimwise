"""Source-grounded quality and budget metrics for compression outputs."""

from __future__ import annotations

import re
from collections import Counter
from math import ceil
from typing import Any

from benchmark.utils.tokens import count_tokens

_COMPARISON_TOKEN = re.compile(r"[\w./:+-]+")


def _tokens(text: str) -> list[str]:
    """Return case-folded comparison tokens for source-evidence scoring.

    The token pattern retains Unicode word characters, ``.``, ``/``, ``:``, ``+``, and ``-``.
    It consequently keeps identifiers, paths, URLs, and hyphenated terms intact where possible.

    Args:
        text: Source or retained text to tokenize.

    Returns:
        Case-folded comparison tokens in source order.
    """
    return _COMPARISON_TOKEN.findall(text.casefold())


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


def _legacy_source_hit(reference: str, output: str) -> bool:
    """Return the historical whole-output bag-of-token retention decision.

    Args:
        reference: Required source-backed evidence.
        output: Compression output to inspect.

    Returns:
        Whether the span is present exactly or retains at least 80% of its tokens anywhere in the
        complete output.
    """
    exact = _normalized_text(reference) in _normalized_text(output)
    recall, _, _ = overlap(reference, output)
    return exact or recall >= 0.80


def _normalized_exact_hit(reference: str, output: str) -> bool:
    """Return whether a required span occurs after case and whitespace normalization.

    Args:
        reference: Required source-backed evidence.
        output: Compression output to inspect.

    Returns:
        Whether the complete normalized reference occurs in the normalized output.
    """
    return _normalized_text(reference) in _normalized_text(output)


def _longest_common_subsequence_length(reference: list[str], window: list[str]) -> int:
    """Return the longest common subsequence length for two token sequences.

    Args:
        reference: Normalized tokens from one required source span.
        window: One contiguous output-token window.

    Returns:
        The maximum number of reference tokens occurring in order in the window.
    """
    previous = [0] * (len(window) + 1)
    for reference_token in reference:
        current = [0]
        for index, output_token in enumerate(window, start=1):
            if reference_token == output_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _local_ordered_match_count(reference: list[str], output: list[str]) -> int:
    """Return maximum local ordered source-token retention.

    Each candidate output window has exactly ``len(reference)`` tokens unless the remaining output
    is shorter. It is sufficient to start at output tokens present in the reference: shifting an
    arbitrary window to its first matched token retains every match and stays within the same
    maximum window length.

    Args:
        reference: Case-folded comparison tokens from one required source span.
        output: Case-folded comparison tokens from the compression output.

    Returns:
        The largest longest-common-subsequence count across bounded output windows.
    """
    if not reference or not output:
        return 0
    reference_terms = set(reference)
    best = 0
    for start, output_token in enumerate(output):
        if output_token not in reference_terms:
            continue
        best = max(
            best,
            _longest_common_subsequence_length(reference, output[start : start + len(reference)]),
        )
        if best == len(reference):
            return best
    return best


def _passes_local_ordered_retention(matches: int, reference_length: int, threshold: float) -> bool:
    """Apply a local ordered-retention threshold using integer arithmetic.

    Args:
        matches: Longest common subsequence count in the best bounded output window.
        reference_length: Number of comparison tokens in the required source span.
        threshold: Required fraction of source tokens that must survive.

    Returns:
        Whether the retained count reaches ``ceil(threshold * reference_length)``.
    """
    return matches >= ceil(threshold * reference_length)


def _required_span_texts(case: dict[str, Any]) -> list[str]:
    """Return validated required source spans from one benchmark case.

    Args:
        case: Benchmark case containing annotated source evidence.

    Returns:
        Required evidence texts in their dataset order.

    Raises:
        ValueError: If a required span is blank or has no comparison tokens.
    """
    texts = [
        str(item["text"]) for item in case.get("gold_evidence", []) if item.get("required", True)
    ]
    for text in texts:
        if not _normalized_text(text):
            raise ValueError("required evidence spans must not be blank")
        if not _tokens(text):
            raise ValueError("required evidence spans must contain comparison tokens")
    return texts


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
    coverage = _coverage([_legacy_source_hit(step, output) for step in steps])
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
    spans = _required_span_texts(case)
    gold = "\n".join(spans)
    recall, precision, f1 = overlap(gold, output)
    legacy_hits = [_legacy_source_hit(span, output) for span in spans]
    exact_hits = [span in output for span in spans]
    normalized_contiguous_hits = [_normalized_exact_hit(span, output) for span in spans]
    output_tokens = _tokens(output)
    span_tokens = [_tokens(span) for span in spans]
    local_ordered_matches = [
        _local_ordered_match_count(tokens, output_tokens) for tokens in span_tokens
    ]
    local_ordered_recalls = [
        matches / len(tokens)
        for matches, tokens in zip(local_ordered_matches, span_tokens, strict=True)
    ]
    local_ordered_80_hits = [
        _passes_local_ordered_retention(matches, len(tokens), 0.80)
        for matches, tokens in zip(local_ordered_matches, span_tokens, strict=True)
    ]
    local_ordered_90_hits = [
        _passes_local_ordered_retention(matches, len(tokens), 0.90)
        for matches, tokens in zip(local_ordered_matches, span_tokens, strict=True)
    ]
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
    all_required_evidence = all(legacy_hits) if legacy_hits else True
    normalized_contiguous_evidence = (
        all(normalized_contiguous_hits) if normalized_contiguous_hits else True
    )
    local_ordered_80_evidence = all(local_ordered_80_hits) if local_ordered_80_hits else True
    local_ordered_90_evidence = all(local_ordered_90_hits) if local_ordered_90_hits else True
    case_constraints_pass = not prohibited_hits and actual <= budget
    case_pass = all_required_evidence and case_constraints_pass
    normalized_contiguous_case_pass = normalized_contiguous_evidence and case_constraints_pass
    local_ordered_80_case_pass = local_ordered_80_evidence and case_constraints_pass
    local_ordered_90_case_pass = local_ordered_90_evidence and case_constraints_pass
    return {
        "evidence_recall": recall,
        "evidence_precision": precision,
        "evidence_f1": f1,
        "required_span_hits": legacy_hits,
        "required_span_coverage": _coverage(legacy_hits),
        "exact_required_span_coverage": _coverage(exact_hits),
        "exact_required_evidence_success": all(exact_hits) if exact_hits else True,
        "normalized_contiguous_required_span_coverage": _coverage(normalized_contiguous_hits),
        "normalized_contiguous_required_evidence_success": normalized_contiguous_evidence,
        "local_ordered_required_span_recall": _coverage(local_ordered_recalls),
        "local_ordered_80_required_span_coverage": _coverage(local_ordered_80_hits),
        "local_ordered_80_required_evidence_success": local_ordered_80_evidence,
        "local_ordered_90_required_span_coverage": _coverage(local_ordered_90_hits),
        "local_ordered_90_required_evidence_success": local_ordered_90_evidence,
        "ordered_step_coverage": ordered_step_coverage,
        "ordered_step_ordered": ordered_step_ordered,
        "all_required_evidence_success": all_required_evidence,
        "legacy_case_pass": case_pass,
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
        "normalized_contiguous_case_pass": normalized_contiguous_case_pass,
        "local_ordered_80_case_pass": local_ordered_80_case_pass,
        "local_ordered_90_case_pass": local_ordered_90_case_pass,
    }
