"""Verify strict source-evidence retention sensitivity metrics."""

from __future__ import annotations

import pytest
from benchmark.metrics.evidence import score_case


def test_legacy_bag_recall_accepts_scattered_tokens() -> None:
    """Show the historical metric can accept tokens separated across an output."""
    result = score_case(
        {"context": "", "gold_evidence": [{"text": "alpha beta gamma delta epsilon"}]},
        "alpha unrelated beta unrelated gamma unrelated delta unrelated epsilon",
        128,
    )

    assert result["case_pass"] is True


def test_local_ordered_80_rejects_scattered_tokens() -> None:
    """Require 80 percent retention to occur in one bounded ordered window."""
    result = score_case(
        {"context": "", "gold_evidence": [{"text": "alpha beta gamma delta epsilon"}]},
        "alpha unrelated beta unrelated gamma unrelated delta unrelated epsilon",
        128,
    )

    assert result["local_ordered_80_case_pass"] is False
    assert result["local_ordered_required_span_recall"] == 0.6


def test_local_ordered_90_rejects_scattered_tokens() -> None:
    """Require 90 percent retention to occur in one bounded ordered window."""
    result = score_case(
        {"context": "", "gold_evidence": [{"text": "alpha beta gamma delta epsilon"}]},
        "alpha unrelated beta unrelated gamma unrelated delta unrelated epsilon",
        128,
    )

    assert result["local_ordered_90_case_pass"] is False


def test_normalized_exact_retention_ignores_case_and_whitespace() -> None:
    """Recognize a complete required span after the declared normalization."""
    result = score_case(
        {"context": "", "gold_evidence": [{"text": "Rule 7:\nReturn JSON"}]},
        "Before\n\nrule 7:    return json\n\nAfter",
        128,
    )

    assert result["normalized_contiguous_case_pass"] is True


def test_local_ordered_80_accepts_one_partially_retained_span() -> None:
    """Accept a single ordered window that retains exactly 80 percent of a span."""
    result = score_case(
        {
            "context": "",
            "gold_evidence": [{"text": "one two three four five six seven eight nine ten"}],
        },
        "one two three four five six seven eight",
        128,
    )

    assert result["local_ordered_80_case_pass"] is True
    assert result["local_ordered_required_span_recall"] == 0.8


def test_local_ordered_90_rejects_80_percent_retention() -> None:
    """Keep the 90 percent sensitivity threshold distinct from the 80 percent one."""
    result = score_case(
        {
            "context": "",
            "gold_evidence": [{"text": "one two three four five six seven eight nine ten"}],
        },
        "one two three four five six seven eight",
        128,
    )

    assert result["local_ordered_90_case_pass"] is False


def test_local_ordered_retention_rejects_reversed_tokens() -> None:
    """Reject reference tokens appearing locally but in the opposite source order."""
    result = score_case(
        {"context": "", "gold_evidence": [{"text": "alpha beta gamma delta epsilon"}]},
        "epsilon delta gamma beta alpha",
        128,
    )

    assert result["local_ordered_required_span_recall"] == 0.2
    assert result["local_ordered_80_case_pass"] is False


def test_local_ordered_retention_does_not_reuse_repeated_tokens() -> None:
    """Count repeated tokens no more often than they occur in one output window."""
    result = score_case(
        {"context": "", "gold_evidence": [{"text": "alpha alpha beta"}]},
        "alpha beta",
        128,
    )

    assert result["local_ordered_required_span_recall"] == pytest.approx(2 / 3)
    assert result["local_ordered_80_case_pass"] is False


def test_local_ordered_retention_scores_a_shorter_output_window() -> None:
    """Score the complete output when it is shorter than the required span."""
    result = score_case(
        {"context": "", "gold_evidence": [{"text": "one two three four five"}]},
        "one two three four",
        128,
    )

    assert result["local_ordered_required_span_recall"] == 0.8
    assert result["local_ordered_80_case_pass"] is True
    assert result["local_ordered_90_case_pass"] is False


def test_case_constraints_fail_an_otherwise_retained_span() -> None:
    """Keep prohibited-content and budget checks conjunctive with new span metrics."""
    prohibited = score_case(
        {
            "context": "",
            "gold_evidence": [{"text": "alpha beta"}],
            "prohibited_phrases": ["forbidden"],
        },
        "alpha beta forbidden",
        128,
    )
    over_budget = score_case(
        {"context": "", "gold_evidence": [{"text": "alpha beta"}]},
        "alpha beta",
        0,
    )

    assert prohibited["normalized_contiguous_required_evidence_success"] is True
    assert prohibited["normalized_contiguous_case_pass"] is False
    assert over_budget["local_ordered_90_required_evidence_success"] is True
    assert over_budget["local_ordered_90_case_pass"] is False


def test_case_fails_when_one_of_multiple_required_spans_is_missing() -> None:
    """Require every independently annotated span to meet each retention metric."""
    result = score_case(
        {
            "context": "",
            "gold_evidence": [{"text": "alpha beta"}, {"text": "gamma delta"}],
        },
        "alpha beta",
        128,
    )

    assert result["normalized_contiguous_required_span_coverage"] == 0.5
    assert result["normalized_contiguous_case_pass"] is False
    assert result["local_ordered_80_case_pass"] is False


def test_blank_required_span_is_invalid_data() -> None:
    """Reject a blank required span instead of counting it as automatically retained."""
    with pytest.raises(ValueError, match="must not be blank"):
        score_case(
            {"context": "", "gold_evidence": [{"text": " \n\t "}]},
            "anything",
            128,
        )
