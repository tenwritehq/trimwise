"""Run configured context compressors and optional downstream QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from benchmark.adapters import build_adapter
from benchmark.datasets.loader import selected_cases
from benchmark.metrics.evidence import score_case
from benchmark.thermal import ThermalGate, thermal_metadata


def stable_seed(base: int, *parts: object) -> int:
    """Derive a stable integer seed from benchmark identity fields.

    Args:
        base: Configured seed shared by the run.
        *parts: Case, method, and budget identity fields.

    Returns:
        A deterministic unsigned integer seed.
    """
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def existing_keys(path: Path) -> set[tuple[str, str, int, bool]]:
    """Read successful case, method, budget, and query-access keys."""
    keys: set[tuple[str, str, int, bool]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "success":
                keys.add(
                    (
                        row["case_id"],
                        row["method_id"],
                        int(row["budget"]),
                        bool(row.get("query_aware", True)),
                    )
                )
    return keys


def configure_runtime(config: dict[str, Any]) -> str:
    """Configure persistent model caches and enforce an optional CUDA requirement."""
    cache_dir = Path(config.get("cache_dir", "cache/huggingface")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "transformers"))
    os.environ.pop("TRANSFORMERS_CACHE", None)
    if config.get("require_cuda", False) or config.get("use_gpu", False):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("GPU benchmark requires the torch package") from exc
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU benchmark requested, but CUDA is unavailable")
    return str(cache_dir)


def main() -> None:
    """Run configured compressors and append auditable JSONL result rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/position_controlled_160.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    thermal_gate = ThermalGate.from_config(config.get("thermal"))
    thermal_gate.preflight()
    cache_dir = configure_runtime(config)
    dataset = Path(config["dataset"])
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(output)
    method_specs = [
        {**spec, "cache_dir": cache_dir, "use_gpu": bool(config.get("use_gpu", False))}
        for spec in config["methods"]
        if spec.get("enabled", True)
    ]
    methods = [build_adapter(spec) for spec in method_specs]
    cases = list(selected_cases(dataset, config.get("case_ids")))
    if args.limit is not None:
        cases = cases[: args.limit]
    budgets = list(map(int, config["budgets"]))
    total_work = len(cases) * len(methods) * len(budgets)
    written_rows = 0
    skipped_rows = 0
    started = time.perf_counter()
    print(
        f"[benchmark] start cases={len(cases)} methods={len(methods)} budgets={budgets} "
        f"work={total_work} cache={cache_dir}",
        flush=True,
    )

    with output.open("a", encoding="utf-8") as sink:
        for adapter in methods:
            method_started = time.perf_counter()
            method_written = 0
            method_failures = 0
            print(
                f"[benchmark] method={adapter.method_id} start "
                f"model_backed={getattr(adapter, 'model_backed', False)}",
                flush=True,
            )
            try:
                for case_index, case in enumerate(cases, 1):
                    context = str(case["context"])
                    query = str(case.get("query", "")) if adapter.query_aware else ""
                    for budget in budgets:
                        key = (case["case_id"], adapter.method_id, budget, adapter.query_aware)
                        if key in completed:
                            skipped_rows += 1
                            continue
                        seed = stable_seed(int(config.get("seed", 242)), *key)
                        thermal_label = (
                            f"compression {adapter.method_id} {case['case_id']} {budget}"
                        )
                        thermal_before = thermal_gate.before_work(thermal_label)
                        result = adapter.compress(context, query, budget, seed)
                        thermal_after = thermal_gate.after_work(thermal_label)
                        result.metadata.update(thermal_metadata(thermal_before, thermal_after))
                        row: dict[str, Any] = {
                            "case_id": case["case_id"],
                            "document_id": case.get("document_id"),
                            "track": case.get("track"),
                            "source_type": case.get("metadata", {}).get("source_type"),
                            "evidence_position": case.get("metadata", {}).get("evidence_position"),
                            "method_id": adapter.method_id,
                            "query_aware": bool(adapter.query_aware),
                            "budget": budget,
                            "status": result.status,
                            "latency_ms": result.latency_ms,
                            "output": result.output,
                            "error_type": result.error_type,
                            "error_message": result.error_message,
                            "traceback": result.traceback,
                            "metadata": result.metadata,
                        }
                        if result.status == "success":
                            row.update(score_case(case, result.output, budget))
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                        sink.flush()
                        completed.add(key)
                        written_rows += 1
                        method_written += 1
                        method_failures += result.status != "success"
                        if method_written == 1:
                            print(
                                f"[benchmark] method={adapter.method_id} first_call "
                                f"status={result.status} cold={result.metadata.get('cold_call')} "
                                f"model_load_ms={result.metadata.get('model_load_ms')} "
                                f"gpu={result.metadata.get('gpu_available')}",
                                flush=True,
                            )
                    if case_index == 1 or case_index % 10 == 0 or case_index == len(cases):
                        print(
                            f"[benchmark] method={adapter.method_id} "
                            f"cases={case_index}/{len(cases)} rows={method_written} "
                            f"elapsed_s={time.perf_counter() - method_started:.1f}",
                            flush=True,
                        )
            finally:
                adapter.close()
                print(
                    f"[benchmark] method={adapter.method_id} done rows={method_written} "
                    f"failures={method_failures} "
                    f"elapsed_s={time.perf_counter() - method_started:.1f}",
                    flush=True,
                )

    print(
        f"[benchmark] compression_done written={written_rows} skipped={skipped_rows} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
