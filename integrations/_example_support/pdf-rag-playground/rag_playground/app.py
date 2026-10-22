"""Launch the standalone PDF RAG playground on localhost without public sharing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_playground.retrieval import CorpusBuilder

# Disable tracing/analytics before importing libraries that inspect their environment.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["GGML_VK_VISIBLE_DEVICES"] = "-1"


def launch(builder: CorpusBuilder, framework: str, port: int) -> None:
    """Serve a native example with CPU-only generation and private temporary storage.

    Args:
        builder: Native framework index factory.
        framework: Fixed framework label for this example.
        port: Loopback port, allowing both examples to run simultaneously.
    """
    with tempfile.TemporaryDirectory(prefix="trimwise-rag-") as directory:
        os.environ["GRADIO_TEMP_DIR"] = str(Path(directory) / "gradio")
        import gradio as gr

        from rag_playground.ui import ROOT, build_app

        app = build_app(Path(directory), builder, framework)
        theme = gr.themes.Base(
            primary_hue="teal",
            radius_size=gr.themes.sizes.radius_none,
            neutral_hue="slate",
            font=["system-ui", "sans-serif"],
            font_mono=["ui-monospace", "monospace"],
        ).set(
            checkbox_background_color_selected="#08675e",
            checkbox_border_color="#6c858c",
            checkbox_border_color_selected="#08675e",
            checkbox_label_background_fill="#ffffff",
            checkbox_label_background_fill_hover="#f0f6f3",
            checkbox_label_background_fill_selected="#e1f3ee",
            checkbox_label_text_color="#17282e",
            checkbox_label_text_color_selected="#08675e",
            checkbox_label_border_color="#b5c5c8",
            checkbox_label_border_color_selected="#08675e",
        )
        # Native controls must share the workspace palette when the OS prefers dark mode.
        for name in vars(theme):
            if name.endswith("_dark"):
                setattr(theme, name, getattr(theme, name.removesuffix("_dark")))
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            show_error=False,
            max_file_size="10mb",
            footer_links=[],
            enable_monitoring=False,
            run_history=False,
            css=(ROOT / "styles.css").read_text(encoding="utf-8"),
            theme=theme,
            state_session_capacity=20,
        )
