"""Keep the public benchmark protocol explicit and reproducible."""

from __future__ import annotations

from pathlib import Path

import yaml


def _canonical_config() -> dict[str, object]:
    """Load the sole public benchmark configuration.

    Returns:
        Parsed canonical benchmark settings.
    """
    config_path = Path(__file__).parents[1] / "configs" / "position_controlled_160.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_canonical_config_pins_the_openai_evaluators() -> None:
    """Keep the public evaluator snapshots and deterministic settings explicit."""
    qa = _canonical_config()["openai_qa"]

    assert (
        qa["reasoning_effort"],
        qa["max_output_tokens"],
        [(model["id"], model["model"]) for model in qa["models"]],
    ) == (
        "none",
        128,
        [
            ("gpt_5_4_nano", "gpt-5.4-nano-2026-03-17"),
            ("gpt_5_4_mini", "gpt-5.4-mini-2026-03-17"),
            ("gpt_5_6_luna", "gpt-5.6-luna"),
        ],
    )


def test_canonical_config_enables_the_documented_thermal_profile() -> None:
    """Keep the shipped thermal thresholds aligned with the benchmark protocol."""
    thermal = _canonical_config()["thermal"]

    assert thermal == {
        "enabled": True,
        "pause_temperature_c": 70,
        "resume_temperature_c": 60,
        "hard_temperature_c": 75,
        "cooldown_timeout_seconds": 1800,
    }


def test_pyproject_pins_the_llmlingua_compatible_transformers_release() -> None:
    """Prevent an incompatible Transformers cache API upgrade."""
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"

    assert "transformers==4.43.1" in pyproject_path.read_text(encoding="utf-8")
