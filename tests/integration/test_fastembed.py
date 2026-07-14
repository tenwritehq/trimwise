"""Exercise the real optional multilingual FastEmbed model."""

from __future__ import annotations

import pytest

from trimwise import Trimmer

pytestmark = pytest.mark.integration


def test_multilingual_model_selects_paraphrased_evidence() -> None:
    """Retrieve relevant multilingual evidence and honor the final budget."""
    pytest.importorskip("fastembed", reason="install trimwise[semantic] for integration tests")
    source = (
        "The cafeteria menu changes each Monday.\n\n"
        "परियोजना सूर्य 14 जुलाई 2026 को शुरू हुई।\n\n"
        "Routine operational notes follow this paragraph.\n"
    )
    result = Trimmer().trim(
        source,
        55,
        unit="characters",
        strategy="semantic",
        query="परियोजना सूर्य कब आरंभ हुई?",
    )
    assert "14 जुलाई 2026" in result.text
    assert result.output_count <= 55
