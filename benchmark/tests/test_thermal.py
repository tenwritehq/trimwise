"""Verify read-only thermal safety behavior without invoking NVIDIA tooling."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import benchmark.runners.run_compression as compression_runner
import benchmark.thermal as thermal
import pytest
import yaml
from benchmark.adapters.base import CompressionResult


def _desktop_processes() -> str:
    """Return process monitoring output containing desktop-only GPU clients.

    Returns:
        A minimal ``nvidia-smi pmon`` response with no pure-compute process.
    """
    return "# gpu pid type sm mem enc dec jpg ofa command\n0 42 C+G - - - - - - kwin_wayland\n"


def _telemetry_responses(temperatures: list[str], processes: str | None = None) -> Any:
    """Build a fake NVIDIA command runner with deterministic temperatures.

    Args:
        temperatures: Temperature values returned in query order.
        processes: Optional process-monitoring output.

    Returns:
        Callable replacement for the NVIDIA command helper.
    """
    readings = iter(temperatures)
    process_output = processes or _desktop_processes()

    def run(command: list[str]) -> str:
        """Return the matching simulated NVIDIA command response.

        Args:
            command: NVIDIA command selected by the thermal gate.

        Returns:
            Simulated command output.
        """
        return process_output if command[1] == "pmon" else f"{next(readings)}\n"

    return run


def test_gate_allows_desktop_gpu_clients_and_records_no_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept C+G desktop clients while no pure CUDA worker is present."""
    monkeypatch.setattr(thermal, "_run_nvidia_smi", _telemetry_responses(["50"]))

    checkpoint = thermal.ThermalGate.from_config({"enabled": True}).before_work("test")

    assert checkpoint == thermal.ThermalCheckpoint(50.0, 50.0, 0.0, False, 50.0)


def test_gate_rejects_foreign_pure_compute_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a C-only worker before it can compete with benchmark models."""
    processes = "# header\n0 123 C - - - - - - semantic-server\n"
    monkeypatch.setattr(thermal, "_run_nvidia_smi", _telemetry_responses([], processes))

    with pytest.raises(thermal.ThermalSafetyError, match="pid=123"):
        thermal.ThermalGate.from_config({"enabled": True}).preflight()


def test_gate_waits_to_resume_temperature_and_records_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait through a 75 C event until the configured 60 C resume threshold."""
    monkeypatch.setattr(thermal, "_run_nvidia_smi", _telemetry_responses(["75", "68", "60"]))
    monkeypatch.setattr(thermal.time, "sleep", lambda _: None)

    checkpoint = thermal.ThermalGate.from_config({"enabled": True}).after_work("test")

    assert checkpoint.temperature_before_c == 75.0
    assert checkpoint.temperature_after_c == 60.0
    assert checkpoint.hard_limit_event is True


def test_gate_stops_when_cooldown_timeout_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise instead of waiting indefinitely when cooling cannot recover."""
    monotonic_values = iter([0.0, 1_801.0])
    monkeypatch.setattr(thermal, "_run_nvidia_smi", _telemetry_responses(["68"]))
    monkeypatch.setattr(thermal.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(thermal.ThermalSafetyError, match="did not cool"):
        thermal.ThermalGate.from_config({"enabled": True}).before_work("test")


def test_gate_stops_when_temperature_telemetry_is_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed telemetry rather than running without a temperature signal."""
    monkeypatch.setattr(thermal, "_run_nvidia_smi", _telemetry_responses(["N/A"]))

    with pytest.raises(thermal.ThermalSafetyError, match="invalid NVIDIA temperature"):
        thermal.ThermalGate.from_config({"enabled": True}).before_work("test")


def test_gate_stops_when_temperature_telemetry_is_not_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject NaN telemetry rather than allowing it to bypass the temperature threshold."""
    monkeypatch.setattr(thermal, "_run_nvidia_smi", _telemetry_responses(["nan"]))

    with pytest.raises(thermal.ThermalSafetyError, match="invalid NVIDIA temperature"):
        thermal.ThermalGate.from_config({"enabled": True}).before_work("test")


def test_thermal_metadata_keeps_wait_separate_from_model_latency() -> None:
    """Expose temperatures and wait time without creating a latency field."""
    before = thermal.ThermalCheckpoint(68.0, 60.0, 2_000.0, False)
    after = thermal.ThermalCheckpoint(70.0, 60.0, 4_000.0, True, 75.0)

    assert thermal.thermal_metadata(before, after) == {
        "thermal_temperature_before_c": 60.0,
        "thermal_temperature_after_c": 70.0,
        "thermal_max_temperature_c": 75.0,
        "thermal_wait_ms": 6_000.0,
        "thermal_hard_limit_event": True,
    }


class RecordingThermalGate:
    """Record compression boundaries without querying a real GPU."""

    def __init__(self) -> None:
        """Initialize the recorded boundary labels."""
        self.preflight_calls = 0
        self.before_labels: list[str] = []
        self.after_labels: list[str] = []

    def preflight(self) -> None:
        """Record thermal startup validation."""
        self.preflight_calls += 1

    def before_work(self, label: str) -> thermal.ThermalCheckpoint:
        """Record the boundary immediately before compression.

        Args:
            label: Compression work identifier.

        Returns:
            A no-wait thermal checkpoint.
        """
        self.before_labels.append(label)
        return thermal.ThermalCheckpoint(60.0, 60.0, 0.0, False)

    def after_work(self, label: str) -> thermal.ThermalCheckpoint:
        """Record the boundary immediately after compression.

        Args:
            label: Compression work identifier.

        Returns:
            A thermal checkpoint with separate cooldown time.
        """
        self.after_labels.append(label)
        return thermal.ThermalCheckpoint(68.0, 60.0, 2_000.0, False)


class FixedAdapter:
    """Return one fixed compression result for runner integration testing."""

    method_id = "fixed"
    query_aware = True
    model_backed = False

    def compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        """Return a fixed result while preserving the runner's supplied arguments.

        Args:
            context: Context selected for compression.
            query: Query supplied to the adapter.
            budget: Token budget supplied by the runner.
            seed: Deterministic seed supplied by the runner.

        Returns:
            A successful result with a known compute-only latency.
        """
        del context, query, budget, seed
        return CompressionResult("fixed", "evidence", latency_ms=7.0)

    def close(self) -> None:
        """Satisfy the adapter lifecycle used by the runner."""


def test_compression_runner_writes_thermal_metadata_without_changing_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply shared thermal boundaries around one compression result."""
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "results.jsonl"
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps({"case_id": "case", "context": "source", "query": "question"}) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": str(dataset_path),
                "output": str(output_path),
                "budgets": [128],
                "methods": [{"name": "fixed"}],
                "qa": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    gate = RecordingThermalGate()
    monkeypatch.setattr(compression_runner.ThermalGate, "from_config", lambda _: gate)
    monkeypatch.setattr(compression_runner, "build_adapter", lambda _: FixedAdapter())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_compression", "--config", str(config_path)],
    )

    compression_runner.main()

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["latency_ms"] == 7.0
    assert row["metadata"]["thermal_wait_ms"] == 2_000.0
    assert gate.preflight_calls == 1
    assert gate.before_labels == gate.after_labels
    assert "compression fixed case 128" in gate.before_labels
