"""Load benchmark cases and reproducible case selections from JSONL."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_cases(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated JSON records from one benchmark dataset.

    Args:
        path: JSONL dataset path.

    Yields:
        Parsed benchmark case records.

    Raises:
        ValueError: If a nonblank JSONL row is invalid.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc


def selected_cases(path: str | Path, case_ids: Iterable[str] | None) -> Iterator[dict[str, Any]]:
    """Yield all cases or only a configured, reproducible case selection.

    Args:
        path: JSONL dataset path.
        case_ids: Case identifiers to retain, or ``None`` for every case.

    Yields:
        Dataset records selected for the benchmark run.
    """
    selected = set(case_ids or ())
    for case in iter_cases(path):
        if not selected or case["case_id"] in selected:
            yield case
