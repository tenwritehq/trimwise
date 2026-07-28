"""Read NVIDIA telemetry and pause benchmark work before unsafe heat accumulates."""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

COMMAND_TIMEOUT_SECONDS = 5
COOLDOWN_POLL_SECONDS = 2
DEFAULT_PAUSE_TEMPERATURE_C = 68.0
DEFAULT_RESUME_TEMPERATURE_C = 60.0
DEFAULT_HARD_TEMPERATURE_C = 75.0
DEFAULT_COOLDOWN_TIMEOUT_SECONDS = 1_800.0


class ThermalSafetyError(RuntimeError):
    """Report a condition that prevents a thermal benchmark from continuing safely."""


@dataclass(frozen=True, slots=True)
class ThermalSettings:
    """Store the temperature thresholds used to gate benchmark work."""

    enabled: bool
    pause_temperature_c: float = DEFAULT_PAUSE_TEMPERATURE_C
    resume_temperature_c: float = DEFAULT_RESUME_TEMPERATURE_C
    hard_temperature_c: float = DEFAULT_HARD_TEMPERATURE_C
    cooldown_timeout_seconds: float = DEFAULT_COOLDOWN_TIMEOUT_SECONDS

    @classmethod
    def from_config(cls, config: object) -> ThermalSettings:
        """Build validated settings from one optional benchmark configuration section.

        Args:
            config: Mapping supplied under the benchmark's ``thermal`` key.

        Returns:
            Thermal settings, disabled when no configuration is supplied.

        Raises:
            ValueError: If the supplied settings are malformed or unordered.
        """
        if config is None:
            return cls(enabled=False)
        if not isinstance(config, Mapping):
            raise ValueError("thermal must be a mapping")
        enabled = config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("thermal.enabled must be a boolean")
        settings = cls(
            enabled=enabled,
            pause_temperature_c=_temperature_setting(
                config, "pause_temperature_c", DEFAULT_PAUSE_TEMPERATURE_C
            ),
            resume_temperature_c=_temperature_setting(
                config, "resume_temperature_c", DEFAULT_RESUME_TEMPERATURE_C
            ),
            hard_temperature_c=_temperature_setting(
                config, "hard_temperature_c", DEFAULT_HARD_TEMPERATURE_C
            ),
            cooldown_timeout_seconds=_positive_setting(
                config, "cooldown_timeout_seconds", DEFAULT_COOLDOWN_TIMEOUT_SECONDS
            ),
        )
        temperatures_are_ordered = (
            settings.resume_temperature_c
            < settings.pause_temperature_c
            < settings.hard_temperature_c
        )
        if not temperatures_are_ordered:
            raise ValueError(
                "thermal temperatures must satisfy resume_temperature_c < "
                "pause_temperature_c < hard_temperature_c"
            )
        return settings


@dataclass(frozen=True, slots=True)
class ThermalCheckpoint:
    """Capture one temperature check and any cooldown it required."""

    temperature_before_c: float
    temperature_after_c: float
    wait_ms: float
    hard_limit_event: bool
    peak_temperature_c: float | None = None


class ThermalGate:
    """Pause only at benchmark work boundaries using read-only NVIDIA telemetry."""

    def __init__(self, settings: ThermalSettings) -> None:
        """Initialize a gate for the current benchmark process.

        Args:
            settings: Validated thermal settings for this run.
        """
        self._settings = settings
        self._process_id = os.getpid()

    @classmethod
    def from_config(cls, config: object) -> ThermalGate:
        """Create a gate from an optional benchmark configuration section.

        Args:
            config: Mapping supplied under the benchmark's ``thermal`` key.

        Returns:
            A configured thermal gate.
        """
        return cls(ThermalSettings.from_config(config))

    @property
    def enabled(self) -> bool:
        """Return whether this gate actively reads NVIDIA telemetry."""
        return self._settings.enabled

    def preflight(self) -> None:
        """Reject GPU contention and cool the card before CUDA initialization."""
        if not self.enabled:
            return
        self._reject_foreign_compute_processes()
        self._wait_until_ready("startup")

    def before_work(self, label: str) -> ThermalCheckpoint | None:
        """Cool the GPU before starting one compression call or QA batch.

        Args:
            label: Human-readable work identifier for progress output.

        Returns:
            The checkpoint observed before the work, or ``None`` when disabled.
        """
        if not self.enabled:
            return None
        self._reject_foreign_compute_processes()
        return self._wait_until_ready(label)

    def after_work(self, label: str) -> ThermalCheckpoint | None:
        """Cool the GPU after one compression call or QA batch.

        Args:
            label: Human-readable work identifier for progress output.

        Returns:
            The checkpoint observed after the work, or ``None`` when disabled.
        """
        if not self.enabled:
            return None
        self._reject_foreign_compute_processes()
        return self._wait_until_ready(label)

    def _wait_until_ready(self, label: str) -> ThermalCheckpoint:
        """Wait until a hot GPU reaches the configured resume temperature.

        Args:
            label: Human-readable work identifier for progress output.

        Returns:
            The observed temperatures, wait time, and hard-limit state.

        Raises:
            ThermalSafetyError: If telemetry fails or cooling exceeds the timeout.
        """
        temperature_before_c = self._temperature_c()
        peak_temperature_c = temperature_before_c
        hard_limit_event = temperature_before_c >= self._settings.hard_temperature_c
        if temperature_before_c < self._settings.pause_temperature_c:
            return ThermalCheckpoint(
                temperature_before_c,
                temperature_before_c,
                0.0,
                hard_limit_event,
                peak_temperature_c,
            )
        started = time.monotonic()
        print(
            f"[thermal] {label} temperature_c={temperature_before_c:.1f} "
            f"waiting_for_c={self._settings.resume_temperature_c:.1f}",
            flush=True,
        )
        while True:
            elapsed_seconds = time.monotonic() - started
            if elapsed_seconds >= self._settings.cooldown_timeout_seconds:
                raise ThermalSafetyError(
                    f"GPU did not cool to {self._settings.resume_temperature_c:.1f} C within "
                    f"{self._settings.cooldown_timeout_seconds:.0f} seconds"
                )
            time.sleep(COOLDOWN_POLL_SECONDS)
            self._reject_foreign_compute_processes()
            temperature_after_c = self._temperature_c()
            peak_temperature_c = max(peak_temperature_c, temperature_after_c)
            hard_limit_event = (
                hard_limit_event or temperature_after_c >= self._settings.hard_temperature_c
            )
            if temperature_after_c <= self._settings.resume_temperature_c:
                wait_ms = (time.monotonic() - started) * 1_000
                print(
                    f"[thermal] {label} resumed temperature_c={temperature_after_c:.1f} "
                    f"wait_ms={wait_ms:.0f}",
                    flush=True,
                )
                return ThermalCheckpoint(
                    temperature_before_c,
                    temperature_after_c,
                    wait_ms,
                    hard_limit_event,
                    peak_temperature_c,
                )

    def _temperature_c(self) -> float:
        """Read the single visible GPU's core temperature from ``nvidia-smi``.

        Returns:
            The core temperature in degrees Celsius.

        Raises:
            ThermalSafetyError: If NVIDIA telemetry is unavailable or malformed.
        """
        output = _run_nvidia_smi(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        )
        temperatures = [line.strip() for line in output.splitlines() if line.strip()]
        if len(temperatures) != 1:
            raise ThermalSafetyError("thermal mode requires exactly one visible NVIDIA GPU")
        try:
            temperature_c = float(temperatures[0])
        except ValueError as exc:
            raise ThermalSafetyError(
                f"invalid NVIDIA temperature telemetry: {temperatures[0]!r}"
            ) from exc
        if not math.isfinite(temperature_c):
            raise ThermalSafetyError(f"invalid NVIDIA temperature telemetry: {temperatures[0]!r}")
        return temperature_c

    def _reject_foreign_compute_processes(self) -> None:
        """Reject pure-compute GPU clients other than this benchmark process.

        Raises:
            ThermalSafetyError: If a foreign CUDA compute process is active.
        """
        output = _run_nvidia_smi(["nvidia-smi", "pmon", "-c", "1"])
        for line in output.splitlines():
            fields = line.split(maxsplit=8)
            if len(fields) < 4 or not fields[0].isdigit() or not fields[1].isdigit():
                continue
            process_id = int(fields[1])
            process_type = fields[2]
            process_name = fields[-1]
            if process_type == "C" and process_id != self._process_id:
                raise ThermalSafetyError(
                    f"foreign CUDA compute process detected: pid={process_id} name={process_name}"
                )


def thermal_metadata(
    before: ThermalCheckpoint | None, after: ThermalCheckpoint | None
) -> dict[str, float | bool]:
    """Return JSON-safe work metadata while leaving compute latency untouched.

    Args:
        before: Checkpoint taken immediately before model work.
        after: Checkpoint taken immediately after model work.

    Returns:
        Empty metadata when disabled, otherwise temperatures and cooldown time.
    """
    if before is None or after is None:
        return {}
    return {
        "thermal_temperature_before_c": before.temperature_after_c,
        "thermal_temperature_after_c": after.temperature_before_c,
        "thermal_max_temperature_c": max(
            before.temperature_before_c,
            after.temperature_before_c,
            before.peak_temperature_c or before.temperature_before_c,
            after.peak_temperature_c or after.temperature_before_c,
        ),
        "thermal_wait_ms": before.wait_ms + after.wait_ms,
        "thermal_hard_limit_event": before.hard_limit_event or after.hard_limit_event,
    }


def _temperature_setting(config: Mapping[str, Any], name: str, default: float) -> float:
    """Read one finite temperature setting from configuration.

    Args:
        config: Thermal configuration mapping.
        name: Configuration key to read.
        default: Value used when the key is absent.

    Returns:
        The configured temperature.

    Raises:
        ValueError: If the value is not numeric.
    """
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"thermal.{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"thermal.{name} must be finite")
    return value


def _positive_setting(config: Mapping[str, Any], name: str, default: float) -> float:
    """Read one positive duration setting from configuration.

    Args:
        config: Thermal configuration mapping.
        name: Configuration key to read.
        default: Value used when the key is absent.

    Returns:
        The configured duration in seconds.

    Raises:
        ValueError: If the duration is not positive.
    """
    value = _temperature_setting(config, name, default)
    if value <= 0:
        raise ValueError(f"thermal.{name} must be positive")
    return value


def _run_nvidia_smi(command: list[str]) -> str:
    """Run one bounded read-only NVIDIA telemetry command.

    Args:
        command: ``nvidia-smi`` command and arguments to execute.

    Returns:
        Standard output from the completed command.

    Raises:
        ThermalSafetyError: If the command is unavailable, slow, or unsuccessful.
    """
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ThermalSafetyError("unable to read NVIDIA telemetry") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown nvidia-smi error"
        raise ThermalSafetyError(f"unable to read NVIDIA telemetry: {message}")
    return completed.stdout
