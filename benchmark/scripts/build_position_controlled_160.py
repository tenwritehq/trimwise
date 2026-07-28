"""Build and verify the separate 160-case position-controlled benchmark set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

Case = dict[str, Any]

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "benchmark_cases.jsonl"
OUTPUT_PATH = ROOT / "data" / "position_controlled_160.jsonl"
REVIEW_PATH = ROOT / "data" / "position_controlled_160_review.md"
POSITION_TARGET = 40
NATURAL_END_COUNT = 15
POSITION_ORDER = ("beginning", "middle", "end", "multiple")
REVIEW_TEXT_LIMIT = 120
VARIANT_TRACK_QUOTAS = {
    "real_source": 8,
    "evidence_qa": 5,
    "instruction": 4,
    "procedure": 4,
    "structured": 2,
    "adversarial": 2,
}
VARIANT_SOURCE_CASE_IDS = {
    "adversarial": ("adversarial-01-q2", "adversarial-02-q2"),
    "real_source": (
        "real-public-013",
        "real-public-015",
        "real-public-032",
        "real-public-041",
        "real-public-055",
        "real-public-066",
        "real-public-074",
        "real-public-081",
    ),
    "evidence_qa": (
        "synthetic-qa-01-q1",
        "synthetic-qa-02-q2",
        "synthetic-qa-03-q1",
        "synthetic-qa-04-q2",
        "synthetic-qa-05-q1",
    ),
}


def _load_cases(path: Path) -> list[Case]:
    """Load nonblank JSONL benchmark rows.

    Args:
        path: JSONL dataset to read.

    Returns:
        Parsed benchmark cases in file order.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _required_spans(case: Case) -> list[dict[str, Any]]:
    """Return the source spans required for one benchmark case.

    Args:
        case: Benchmark case to inspect.

    Returns:
        Required evidence spans in source order.
    """
    return [span for span in case["gold_evidence"] if span.get("required", True)]


def _required_span_index(case: Case) -> int:
    """Return the list index of the sole required evidence span.

    Args:
        case: Single-required-span case to inspect.

    Returns:
        Index of the required span in ``gold_evidence``.

    Raises:
        ValueError: If the case does not have exactly one required span.
    """
    indices = [
        index for index, span in enumerate(case["gold_evidence"]) if span.get("required", True)
    ]
    if len(indices) != 1:
        raise ValueError(f"{case['case_id']} does not have exactly one required span")
    return indices[0]


def _position(case: Case) -> str:
    """Classify a case by required-evidence placement in its supplied context.

    Args:
        case: Benchmark case with at least one required evidence span.

    Returns:
        Beginning, middle, end, or multiple.
    """
    spans = _required_spans(case)
    if len(spans) > 1:
        return "multiple"
    midpoint = (spans[0]["start"] + spans[0]["end"]) / 2
    ratio = midpoint / len(case["context"])
    if ratio < 1 / 3:
        return "beginning"
    if ratio > 2 / 3:
        return "end"
    return "middle"


def _grouped_cases(cases: list[Case]) -> dict[tuple[str, str], deque[Case]]:
    """Group sorted cases by task track and source type for round-robin selection.

    Args:
        cases: Candidate cases for one evidence-position stratum.

    Returns:
        Sorted deques keyed by task track and source type.
    """
    groups: dict[tuple[str, str], deque[Case]] = defaultdict(deque)
    for case in sorted(cases, key=lambda row: row["case_id"]):
        key = case["track"], case["metadata"]["source_type"]
        groups[key].append(case)
    return dict(groups)


def _select_diverse(cases: list[Case], count: int) -> list[Case]:
    """Select a deterministic, track/source-type-diverse subset.

    Args:
        cases: Candidate cases from one position stratum.
        count: Number of rows required.

    Returns:
        Selected cases in deterministic round-robin order.

    Raises:
        ValueError: If the requested count is unavailable.
    """
    groups = _grouped_cases(cases)
    selected: list[Case] = []
    while len(selected) < count:
        wrote_row = False
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].popleft())
                wrote_row = True
        if not wrote_row:
            raise ValueError(f"needed {count} cases but found only {len(selected)}")
    return selected


def _variant_candidates(cases: list[Case], track: str) -> list[Case]:
    """Return safe source rows that can become controlled end variants.

    Args:
        cases: Natural benchmark rows to consider.
        track: Task track whose rows should be retained.

    Returns:
        Single-span, non-end cases whose evidence text occurs exactly once.
    """
    candidates = []
    for case in cases:
        spans = _required_spans(case)
        if case["track"] != track or len(spans) != 1 or _position(case) == "end":
            continue
        span = spans[0]
        context = case["context"]
        if (
            context[span["start"] : span["end"]] == span["text"]
            and context.count(span["text"]) == 1
        ):
            candidates.append(case)
    return candidates


def _reviewed_variant_sources(cases: list[Case], track: str) -> list[Case]:
    """Return manually reviewed, safe sources for a track when configured.

    Args:
        cases: Complete natural benchmark corpus.
        track: Task track whose optional reviewed sources should be loaded.

    Returns:
        Reviewed source cases, or an empty list for generic track selection.

    Raises:
        ValueError: If a configured source is unavailable, unsafe, or repeats a document.
    """
    case_ids = VARIANT_SOURCE_CASE_IDS.get(track)
    if case_ids is None:
        return []
    candidates = {case["case_id"]: case for case in _variant_candidates(cases, track)}
    selected = [candidates[case_id] for case_id in case_ids if case_id in candidates]
    if len(selected) != len(case_ids):
        raise ValueError(f"{track} has an unavailable reviewed variant source")
    if len({case["document_id"] for case in selected}) != len(selected):
        raise ValueError(f"{track} reviewed variant sources repeat a document")
    return selected


def _select_variant_sources(cases: list[Case]) -> list[Case]:
    """Select the 25 distinct-source rows used to create end variants.

    Args:
        cases: Complete natural benchmark corpus.

    Returns:
        Source cases in the configured track mix.

    Raises:
        ValueError: If a track lacks enough distinct-source candidates.
    """
    selected: list[Case] = []
    for track, quota in VARIANT_TRACK_QUOTAS.items():
        reviewed = _reviewed_variant_sources(cases, track)
        if reviewed:
            if len(reviewed) != quota:
                raise ValueError(f"{track} reviewed variant sources do not meet its quota")
            selected.extend(reviewed)
            continue
        candidates = _variant_candidates(cases, track)
        groups = _grouped_cases(candidates)
        seen_documents: set[str] = set()
        track_selected: list[Case] = []
        while len(track_selected) < quota:
            wrote_row = False
            for key in sorted(groups):
                bucket = groups[key]
                while bucket and str(bucket[0].get("document_id")) in seen_documents:
                    bucket.popleft()
                if bucket and len(track_selected) < quota:
                    case = bucket.popleft()
                    track_selected.append(case)
                    seen_documents.add(str(case.get("document_id")))
                    wrote_row = True
            if not wrote_row:
                raise ValueError(f"{track} lacks {quota} distinct-source variant candidates")
        selected.extend(track_selected)
    return selected


def _natural_case(case: Case) -> Case:
    """Copy one natural row with explicit position-control metadata.

    Args:
        case: Unchanged source row from the natural corpus.

    Returns:
        Copied case annotated as naturally positioned.
    """
    natural = copy.deepcopy(case)
    natural["metadata"]["evidence_position"] = _position(case)
    natural["metadata"]["position_origin"] = "natural"
    return natural


def _relocated_context(case: Case) -> tuple[str, int]:
    """Return the context formed by moving a unique required span to the end.

    Args:
        case: Safe, single-span source case selected for controlled relocation.

    Returns:
        Relocated context and the new inclusive character offset of its evidence span.
    """
    span = _required_spans(case)[0]
    context = case["context"]
    remainder = context[: span["start"]] + context[span["end"] :]
    separator = "\n" if remainder.endswith("\n") else "\n\n"
    return remainder + separator + span["text"], len(remainder) + len(separator)


def _end_variant(case: Case) -> Case:
    """Move one unique required span to the end while preserving its exact text.

    Args:
        case: Safe, single-span source case selected for controlled relocation.

    Returns:
        A provenance-linked end-position variant.
    """
    source = copy.deepcopy(case)
    span_index = _required_span_index(source)
    span = source["gold_evidence"][span_index]
    variant_context, variant_start = _relocated_context(source)
    source["case_id"] = f"{case['case_id']}-position-end"
    source["context"] = variant_context
    source["gold_evidence"][span_index] = {
        **span,
        "start": variant_start,
        "end": variant_start + len(span["text"]),
    }
    source["metadata"]["evidence_position"] = "end"
    source["metadata"]["position_origin"] = "controlled_relocation"
    source["metadata"]["derived_from_case_id"] = case["case_id"]
    source["metadata"]["derived_from_span_start"] = span["start"]
    source["metadata"]["derived_from_span_end"] = span["end"]
    return source


def _build_cases(cases: list[Case]) -> list[Case]:
    """Build the 160-row natural-plus-controlled positional evaluation set.

    Args:
        cases: Complete untouched natural benchmark corpus.

    Returns:
        Forty rows per evidence-position stratum.
    """
    variants = _select_variant_sources(cases)
    variant_ids = {case["case_id"] for case in variants}
    natural = {
        position: _select_diverse(
            [
                case
                for case in cases
                if _position(case) == position and case["case_id"] not in variant_ids
            ],
            POSITION_TARGET,
        )
        for position in ("beginning", "middle", "multiple")
    }
    natural["end"] = [case for case in cases if _position(case) == "end"]
    if len(natural["end"]) != NATURAL_END_COUNT:
        raise ValueError(f"expected {NATURAL_END_COUNT} natural end cases")
    rows = [*(_natural_case(case) for case in natural["beginning"])]
    rows.extend(_natural_case(case) for case in natural["middle"])
    rows.extend(_natural_case(case) for case in natural["end"])
    rows.extend(_end_variant(case) for case in variants)
    rows.extend(_natural_case(case) for case in natural["multiple"])
    return sorted(rows, key=lambda case: (POSITION_ORDER.index(_position(case)), case["case_id"]))


def _validate(cases: list[Case], source_cases: list[Case]) -> None:
    """Assert source integrity, lineage, uniqueness, and exact position balance.

    Args:
        cases: Built position-controlled rows to validate.
        source_cases: Untouched natural source corpus used for derivations.

    Raises:
        ValueError: If a row breaks a positional, provenance, or span invariant.
    """
    source_by_id = {case["case_id"]: case for case in source_cases}
    case_ids = [case["case_id"] for case in cases]
    if len(cases) != POSITION_TARGET * len(POSITION_ORDER) or len(case_ids) != len(set(case_ids)):
        raise ValueError("position-controlled dataset must contain 160 unique cases")
    expected_positions = Counter(dict.fromkeys(POSITION_ORDER, POSITION_TARGET))
    if Counter(_position(case) for case in cases) != expected_positions:
        raise ValueError("position-controlled dataset is not exactly balanced")
    natural_ids = {
        case["case_id"] for case in cases if case["metadata"]["position_origin"] == "natural"
    }
    variants = [
        case for case in cases if case["metadata"]["position_origin"] == "controlled_relocation"
    ]
    if len(variants) != POSITION_TARGET - NATURAL_END_COUNT:
        raise ValueError("position-controlled dataset must contain 25 end variants")
    source_ids = set()
    for case in cases:
        spans = _required_spans(case)
        for span in spans:
            if case["context"][span["start"] : span["end"]] != span["text"]:
                raise ValueError(f"invalid required span in {case['case_id']}")
        if case["metadata"]["evidence_position"] != _position(case):
            raise ValueError(f"stale evidence position in {case['case_id']}")
    for variant in variants:
        source_id = variant["metadata"]["derived_from_case_id"]
        source = source_by_id[source_id]
        source_ids.add(source_id)
        if source_id in natural_ids or _position(variant) != "end":
            raise ValueError(f"invalid variant lineage for {variant['case_id']}")
        span = _required_spans(variant)[0]
        if variant["context"].count(span["text"]) != 1:
            raise ValueError(f"duplicate variant evidence in {variant['case_id']}")
        source_span = _required_spans(source)[0]
        if span["text"] != source_span["text"]:
            raise ValueError(f"altered variant evidence in {variant['case_id']}")
        expected_context, expected_start = _relocated_context(source)
        if variant["context"] != expected_context or span["start"] != expected_start:
            raise ValueError(f"invalid relocation in {variant['case_id']}")
    if len(source_ids) != len(variants):
        raise ValueError("controlled variants must use distinct source cases")


def _render_review(cases: list[Case], source_digest: str) -> str:
    """Render the one-file reviewer manifest for the generated evaluation set.

    Args:
        cases: Validated position-controlled rows.
        source_digest: SHA-256 digest of the untouched source JSONL.

    Returns:
        Markdown review manifest with rows, lineage, and distributions.
    """
    positions = Counter(_position(case) for case in cases)
    tracks = Counter(case["track"] for case in cases)
    lines = [
        "# Position-controlled 160-case review",
        "",
        (
            "The original 250-case corpus remains unchanged; the separate 160-case "
            "evaluation contains 135 natural cases and 25 controlled relocations."
        ),
        "",
        f"Source dataset SHA-256: `{source_digest}`",
        "",
        "## Required manual check",
        "",
        (
            "Review the 25 rows marked `controlled_relocation`: their required text is moved "
            "unchanged to the end of the context; the source case is excluded from this set."
        ),
        "",
        "## Distribution",
        "",
        "| Position | Cases |",
        "| --- | ---: |",
        *(f"| {position} | {positions[position]} |" for position in POSITION_ORDER),
        "",
        "| Track | Cases |",
        "| --- | ---: |",
        *(f"| {track} | {count} |" for track, count in sorted(tracks.items())),
        "",
        "## Cases",
        "",
        "| Case | Position | Origin | Track | Source type | Derived from |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        metadata = case["metadata"]
        lines.append(
            "| {case_id} | {position} | {origin} | {track} | {source_type} | {derived} |".format(
                case_id=case["case_id"],
                position=_position(case),
                origin=metadata["position_origin"],
                track=case["track"],
                source_type=metadata["source_type"],
                derived=metadata.get("derived_from_case_id", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Controlled end variants",
            "",
            "| Variant | Source | Original span | New span | Required text preview |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in cases:
        metadata = case["metadata"]
        if metadata["position_origin"] != "controlled_relocation":
            continue
        span = _required_spans(case)[0]
        preview = " ".join(span["text"].split())[:REVIEW_TEXT_LIMIT].replace("|", "\\|")
        lines.append(
            (
                "| {variant} | {source} | {old_start}:{old_end} | {new_start}:{new_end} "
                "| {preview} |"
            ).format(
                variant=case["case_id"],
                source=metadata["derived_from_case_id"],
                old_start=metadata["derived_from_span_start"],
                old_end=metadata["derived_from_span_end"],
                new_start=span["start"],
                new_end=span["end"],
                preview=preview,
            )
        )
    return "\n".join(lines) + "\n"


def _write_outputs(cases: list[Case], source_digest: str) -> None:
    """Write the generated JSONL dataset and reviewer manifest.

    Args:
        cases: Validated position-controlled cases to persist.
        source_digest: SHA-256 digest of the exact source JSONL bytes.
    """
    OUTPUT_PATH.write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    REVIEW_PATH.write_text(
        _render_review(cases, source_digest),
        encoding="utf-8",
    )


def main() -> None:
    """Build the controlled dataset or verify the committed generated artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source_digest = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    source_cases = _load_cases(SOURCE_PATH)
    cases = _build_cases(source_cases)
    _validate(cases, source_cases)
    expected_jsonl = "".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases
    )
    expected_review = _render_review(cases, source_digest)
    if args.check:
        if (
            OUTPUT_PATH.read_text(encoding="utf-8") != expected_jsonl
            or REVIEW_PATH.read_text(encoding="utf-8") != expected_review
        ):
            raise ValueError("generated position-controlled artifacts are stale")
        print("position-controlled dataset is current and valid")
        return
    _write_outputs(cases, source_digest)
    print(f"wrote {len(cases)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
