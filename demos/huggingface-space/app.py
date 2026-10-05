"""Run a CPU Trimwise workbench with model-free defaults and optional embeddings."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from html import escape
from pathlib import Path
from time import perf_counter

import gradio as gr

from trimwise import (
    BudgetUnit,
    ContextSource,
    ContextSourceResult,
    SemanticBackendError,
    Strategy,
    TrimConfig,
    Trimmer,
)
from trimwise.measurement import Measurer

CONFIG = TrimConfig(fastembed_options={"threads": 2}, embedding_batch_size=16)
MEASURER = Measurer(BudgetUnit.TOKENS, CONFIG.token_encoding, None)
TRIMMER = Trimmer(CONFIG)
METHODS = ("lexical", "structural", "semantic", "hybrid", "auto")
SEPARATOR = "\n\n"
MAX_SOURCE_CHARS = 12_000
MAX_TOTAL_CHARS = 18_000
MAX_INPUT_TOKENS = 4_096
MAX_INPUT_LINES = 300
MAX_QUERY_CHARS = 800
MAX_BUDGET = 2_048
QUEUE_SIZE = 8
CONCURRENCY = 2

SINGLE_PRESET = (
    (
        "# Release notebook\n\n"
        "The team opened the planning meeting on Monday. The agenda covered staffing, "
        "documentation, and a review of the previous release.\n\n"
        "## Design review\n\n"
        "The dashboard uses compact tables. The navigation labels were reviewed. "
        "Several screenshots were collected for the design archive.\n\n"
        "## Routine checks\n\n"
        "The documentation build passed. The search page loaded. The support rota "
        "was updated. The staging homepage used the expected title.\n\n"
        "## Earlier discussion\n\n"
        "The planning notes proposed updating the onboarding checklist. The team "
        "scheduled a separate meeting for translation work and accessibility reviews.\n\n"
        "## Deployment decision\n\n"
        "The production rollout is blocked by an expired signing certificate. "
        "Mira owns certificate renewal, due Friday at 16:00 UTC.\n\n"
        "## Archive\n\n"
        "The meeting ended with a reminder to file the routine notes.",
    ),
    "What blocks the production rollout, who owns renewal, and when is it due?",
    64,
)
MULTIPLE_PRESET = (
    (
        "# Incident log\n\n"
        "The morning shift checked dashboards, network links, and deployment status. "
        "The network was healthy. Logs were copied into the incident workspace.\n\n"
        "Routine checks found no change to the frontend or the scheduled maintenance window. "
        "The team reviewed old incident tickets while collecting metrics.\n\n"
        "The outage was caused by the invoice worker retry loop ignoring backoff settings, "
        "which exhausted the database connection pool.",
        "# Operations notes\n\n"
        "The support rota lists the daytime and evening shifts. The documentation "
        "review is scheduled for next week. Dashboard screenshots remain in the archive.\n\n"
        "Network checks passed. The homepage title and help-page links were correct. "
        "The team discussed future changes to the onboarding checklist.",
        "# Recovery log\n\n"
        "The afternoon team checked routine alerts and recorded the shift handover. "
        "The planned design review continued in a separate meeting.\n\n"
        "The status page received regular updates while the team compared worker metrics.\n\n"
        "The invoice worker retry loop was disabled. Service recovered at 14:32 UTC. "
        "A regression test now verifies that retries respect backoff settings.",
    ),
    "Which retry loop caused the outage, and when did service recover?",
    96,
)

STYLESHEET = Path(__file__).with_name("styles.css")
BRAND_MARK = """<svg viewBox="0 0 512 512" aria-hidden="true" focusable="false">
<rect width="512" height="512" rx="104" fill="#11161a"/>
<path d="M178 106h-72v301h72v-25h-48V131h48zm156 0v25h48v251h-48v25h72V106z" fill="#f3f0e8"/>
<path d="M184 178h154v13H184zm0 72h154v13H184zm0 72h154v13H184z" fill="#3f4a50"/>
<path d="M184 170h103v28H184zm56 72h98v28h-98zm-40 72h112v28H200zm153-155h12v194h-12z"
fill="#39e0d0"/>
<path d="M341 181h25v9h-25zm0 71h25v9h-25zm0 71h25v9h-25z" fill="#39e0d0"/>
</svg>"""
METRIC_LABELS = (
    "Original context tokens",
    "Retained context tokens",
    "Token reduction",
    "Trim API wall time",
)


@dataclass(frozen=True)
class DemoResult:
    """Hold one request's measured outputs without storing them between calls.

    Attributes:
        originals: Input strings in their original positions.
        original_context: Complete untrimmed assembly used by the baseline.
        context: Trimwise's final budgeted text.
        baseline: Source-preserving prefix of the same original assembly.
        sources: Input-aligned evidence rows from the public result.
        original_tokens: Measured original assembly, including source labels.
        retained_tokens: Public result's final output count.
        evidence_tokens: Public result's evidence-only input count.
        baseline_tokens: Measured prefix output count.
        trim_ms: Wall time awaiting the public trim API, excluding baseline and UI work.
        strategy: Concrete ranking strategy used for this request.
    """

    originals: tuple[str, ...]
    original_context: str
    context: str
    baseline: str
    sources: tuple[ContextSourceResult, ...]
    original_tokens: int
    retained_tokens: int
    evidence_tokens: int
    baseline_tokens: int
    trim_ms: float
    strategy: Strategy

    @property
    def reduction_percent(self) -> float:
        """Return reduction relative to the measured original assembly; empty means zero."""
        if not self.original_tokens:
            return 0.0
        return 100 * (1 - self.retained_tokens / self.original_tokens)


def validate_inputs(
    sources: tuple[object, ...], query: object, budget: object
) -> tuple[tuple[str, ...], str | None, int]:
    """Validate CPU-hosted demo limits before segmentation or ranking.

    Args:
        sources: One or three unmodified source strings from the UI or API.
        query: Optional question text.
        budget: Integer-valued token ceiling, including zero.

    Returns:
        Source strings, a stripped query or None, and an integer budget.

    Raises:
        ValueError: If types or hosting limits are invalid.
    """
    if len(sources) not in (1, 3) or any(not isinstance(text, str) for text in sources):
        raise ValueError("Supply one or three text sources.")
    texts = tuple(str(text) for text in sources)
    if any(len(text) > MAX_SOURCE_CHARS for text in texts):
        raise ValueError(f"Each source is limited to {MAX_SOURCE_CHARS:,} characters.")
    if sum(map(len, texts)) > MAX_TOTAL_CHARS:
        raise ValueError(f"Sources together are limited to {MAX_TOTAL_CHARS:,} characters.")
    if sum(len(text.splitlines()) for text in texts) > MAX_INPUT_LINES:
        raise ValueError(f"Sources together are limited to {MAX_INPUT_LINES} lines.")
    if not isinstance(query, str) or len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"The query must be text of at most {MAX_QUERY_CHARS} characters.")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not 0 <= budget <= MAX_BUDGET
        or int(budget) != budget
    ):
        raise ValueError(f"The budget must be a whole number from 0 to {MAX_BUDGET:,}.")
    if sum(MEASURER.count(text) for text in texts) > MAX_INPUT_TOKENS:
        raise ValueError(f"Source evidence is limited to {MAX_INPUT_TOKENS:,} tokens in total.")
    return texts, query.strip() or None, int(budget)


def assemble_sources(sources: tuple[str, ...]) -> str:
    """Assemble untrimmed evidence exactly as the multi-source API renders full rows.

    Args:
        sources: Original strings in input order, including empty positions.

    Returns:
        A single raw source or labeled nonempty sources joined by the fixed separator.
    """
    if len(sources) == 1:
        return sources[0]
    return SEPARATOR.join(f"[Source {index}]\n{text}" for index, text in enumerate(sources) if text)


async def run_demo(
    sources: tuple[object, ...], query: object, budget: object, method: object = "lexical"
) -> DemoResult:
    """Await the public single or shared-budget API and measure a fair prefix baseline.

    Args:
        sources: One or three source inputs; no content is cached or persisted.
        query: Optional question; Lexical and Auto use Structural when blank.
        budget: Final context token ceiling.
        method: Ranking method; lexical is the model-free default.

    Returns:
        Measured results and source provenance for this call only.

    Raises:
        ValueError: If an input exceeds the demo's limits.
        SemanticBackendError: If the managed embedding backend fails.
    """
    texts, question, limit = await asyncio.to_thread(validate_inputs, sources, query, budget)
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("Choose Lexical, Structural, Semantic, Hybrid, or Auto.")
    strategy = Strategy(method)
    if strategy is Strategy.LEXICAL and not question:
        strategy = Strategy.STRUCTURAL
    if strategy in (Strategy.SEMANTIC, Strategy.HYBRID) and not question:
        raise ValueError("Semantic and Hybrid require a nonblank query.")
    original_context = assemble_sources(texts)
    start = perf_counter()
    if len(texts) == 1:
        single = await TRIMMER.atrim(texts[0], limit, strategy=strategy, query=question)
        context, input_count, output_count = single.text, single.input_count, single.output_count
        rows = (
            ContextSourceResult(
                0,
                single.text,
                single.input_count,
                single.output_count,
                single.trimmed,
                single.spans,
            ),
        )
        strategy = single.strategy
    else:
        multiple = await TRIMMER.atrim_context(
            [ContextSource(text, prefix=f"[Source {index}]\n") for index, text in enumerate(texts)],
            limit,
            strategy=strategy,
            query=question,
            separator=SEPARATOR,
        )
        assert multiple.text is not None
        context, input_count, output_count = (
            multiple.text,
            multiple.input_count,
            multiple.output_count,
        )
        rows = multiple.sources
        strategy = multiple.strategy
    trim_ms = (perf_counter() - start) * 1000
    baseline = await asyncio.to_thread(MEASURER.fitting_prefix, original_context, limit)
    return DemoResult(
        texts,
        original_context,
        context,
        baseline,
        rows,
        MEASURER.count(original_context),
        output_count,
        input_count,
        MEASURER.count(baseline),
        trim_ms,
        strategy,
    )


def render_sources(result: DemoResult) -> str:
    """Escape excerpts and show source-local, half-open Python character ranges.

    Args:
        result: Current request's output and original source strings.

    Returns:
        Safe HTML with an input-aligned entry even for a noncontributing source.
    """
    parts = []
    for row in result.sources:
        ranges = "".join(
            f"<li><code>[{span.start}, {span.end})</code>"
            f"<pre>{escape(result.originals[row.source_index][span.start : span.end])}</pre></li>"
            for span in row.spans
        )
        excerpt = f"<pre>{escape(row.text)}</pre>" if row.text else "<p>No retained evidence.</p>"
        mapping = (
            '<details class="span-map">'
            f"<summary>{len(row.spans)} original character "
            f"{'range' if len(row.spans) == 1 else 'ranges'}</summary>"
            f"<ol>{ranges}</ol></details>"
            if ranges
            else ""
        )
        parts.append(
            '<details class="source-result" open>'
            f'<summary><span class="source-index">{row.source_index:02d}</span>'
            f'<span>Source {row.source_index}</span><span class="source-count">'
            f"{row.input_count} &rarr; {row.output_count} evidence tokens</span></summary>"
            f"{excerpt}{mapping}</details>"
        )
    return "".join(parts)


def render_statistics(result: DemoResult | None = None) -> str:
    """Render measured values, or explicit placeholders before a request completes.

    Args:
        result: Completed request, or None for an empty result view.

    Returns:
        Accessible statistics HTML shared by initial, reset, and completed states.
    """
    values = (
        (
            f"{result.original_tokens:,}",
            f"{result.retained_tokens:,}",
            f"{result.reduction_percent:.1f}%",
            f"{result.trim_ms:.1f} ms",
        )
        if result
        else ("&ndash;",) * len(METRIC_LABELS)
    )
    statistics = (
        '<div class="metrics" role="status" aria-live="polite" aria-atomic="true">'
        + "".join(
            f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>'
            for value, label in zip(values, METRIC_LABELS, strict=True)
        )
        + "</div>"
    )
    note = (
        f"{result.strategy.value.capitalize()} &middot; "
        f"{result.evidence_tokens:,} original evidence tokens &middot; "
        f"Prefix baseline: {result.baseline_tokens:,} tokens"
        if result
        else "Awaiting trim"
    )
    return statistics + f'<p class="result-note">{note}</p>'


def display_result(result: DemoResult) -> tuple[str, str, str, str]:
    """Format real measurements, plain-text outputs, and safe provenance for both tabs.

    Args:
        result: Completed isolated request.

    Returns:
        Statistics HTML, final context, baseline text, and source mapping HTML.
    """
    return render_statistics(result), result.context, result.baseline, render_sources(result)


def method_notice(method: str) -> str:
    """Disclose the selected backend without importing or loading any model.

    Args:
        method: Selected dropdown value.

    Returns:
        Static, safe Markdown describing model choice and cold-load behavior.
    """
    if method in ("semantic", "hybrid"):
        model = CONFIG.embedding_model
        return (
            f"**Embedding model:** [{model}](https://huggingface.co/{model}) "
            "&middot; FastEmbed / ONNX CPU. Downloads and loads only when vectors are needed; "
            "the first load can take longer. No LLM or API key."
        )
    if method == "structural":
        return "Structural &middot; No embedding model."
    return "Lexical with a query; structural when blank. No embedding model."


def build_tab(preset: tuple[tuple[str, ...], str, int]) -> None:
    """Create shared controls, async events, and outputs inside the active tab.

    Args:
        preset: Synthetic sources, optional question, and default budget for this mode.
    """
    texts, question, limit = preset
    with gr.Row(elem_classes="workspace"):
        with gr.Column(scale=5, min_width=300, elem_classes="input-pane"):
            gr.HTML('<h2 class="pane-title">Source material</h2>')
            sources = [
                gr.Textbox(
                    value=text,
                    label=f"Source {index}",
                    lines=12 if len(texts) == 1 else 4,
                    max_lines=14 if len(texts) == 1 else 6,
                    max_length=MAX_SOURCE_CHARS,
                    elem_classes=["field", "source-field"],
                )
                for index, text in enumerate(texts)
            ]
            with gr.Row(elem_classes="request-controls"):
                query = gr.Textbox(
                    value=question,
                    label="Query (optional)",
                    lines=2,
                    max_length=MAX_QUERY_CHARS,
                    scale=3,
                    min_width=180,
                    elem_classes="field",
                )
                budget = gr.Number(
                    value=limit,
                    label="Shared token budget" if len(texts) > 1 else "Token budget",
                    minimum=0,
                    maximum=MAX_BUDGET,
                    step=1,
                    scale=1,
                    min_width=140,
                    elem_classes=["field", "budget-field"],
                )
            method = gr.Dropdown(
                choices=[(value.capitalize(), value) for value in METHODS],
                value="lexical",
                label="Method",
                elem_classes=["field", "method-field"],
            )
            backend_notice = gr.Markdown(method_notice("lexical"), elem_classes="backend-notice")
            with gr.Row(elem_classes="action-bar"):
                trim = gr.Button("Trim", variant="primary", elem_classes="trim-button")
                reset = gr.Button("Reset", elem_classes="reset-button")
                example = gr.Button("Load example", elem_classes="example-button")
        with gr.Column(scale=7, min_width=300, elem_classes="output-pane"):
            gr.HTML('<h2 class="pane-title">Retained evidence</h2>')
            statistics = gr.HTML(render_statistics(), elem_classes="statistics")
            context = gr.Textbox(
                label="Retained context",
                placeholder="No retained context yet.",
                interactive=False,
                lines=8,
                max_lines=14,
                buttons=["copy"],
                elem_classes=["field", "output-field"],
            )
            with gr.Accordion(
                "Prefix truncation (illustrative baseline)",
                open=False,
                elem_classes="result-section",
            ):
                baseline = gr.Textbox(
                    label="Prefix of the same original context",
                    interactive=False,
                    lines=6,
                    buttons=["copy"],
                    elem_classes=["field", "output-field", "baseline-field"],
                )
            with gr.Accordion(
                "Source excerpts and character spans", open=True, elem_classes="result-section"
            ):
                excerpts = gr.HTML()

    outputs = [statistics, context, baseline, excerpts]
    inputs = [*sources, query, budget, method]

    async def handle_trim(*values: object) -> tuple[str, str, str, str]:
        """Return this event's result or a bounded error without exposing input text.

        Args:
            values: Source strings followed by query, budget, and method.

        Returns:
            Shared display values for the active tab.

        Raises:
            gr.Error: If validation fails or trimming cannot complete.
        """
        try:
            result = await run_demo(tuple(values[:-3]), values[-3], values[-2], values[-1])
            return display_result(result)
        except ValueError as error:
            raise gr.Error(str(error)) from None
        except SemanticBackendError:
            raise gr.Error(
                "The embedding model could not load or run. Check the Space's model download "
                "access and resources, or select Lexical."
            ) from None
        except Exception:
            raise gr.Error("Trimming failed. Try a smaller input.") from None

    async def clear() -> tuple[object, ...]:
        """Clear inputs and results without retaining server-side session content.

        Returns:
            Empty source/query/output values and the mode's default budget.
        """
        return (
            *("" for _ in texts),
            "",
            limit,
            "lexical",
            render_statistics(),
            "",
            "",
            "",
            method_notice("lexical"),
        )

    async def load_example() -> tuple[object, ...]:
        """Restore this mode's synthetic preset and clear earlier output.

        Returns:
            Preset inputs and empty result values.
        """
        return (
            *texts,
            question,
            limit,
            "lexical",
            render_statistics(),
            "",
            "",
            "",
            method_notice("lexical"),
        )

    async def disclose_backend(value: str) -> str:
        """Update the model disclosure without starting inference or a download.

        Args:
            value: Selected method.

        Returns:
            Backend description for the selected mode.
        """
        return method_notice(value)

    method.change(disclose_backend, method, backend_notice, queue=False, api_visibility="private")

    event = trim.click(
        handle_trim,
        inputs=inputs,
        outputs=outputs,
        api_name="trim_single" if len(texts) == 1 else "trim_multiple",
        concurrency_id="trimwise_cpu",
        concurrency_limit=CONCURRENCY,
        trigger_mode="once",
    )
    for button, handler in ((reset, clear), (example, load_example)):
        button.click(
            handler,
            outputs=[*inputs, *outputs, backend_notice],
            queue=False,
            cancels=[event],
            api_visibility="private",
        )


def build_app() -> gr.Blocks:
    """Build the two-tab CPU demo with one bounded queue shared by both modes.

    Returns:
        Launchable Gradio application with analytics disabled.
    """
    MEASURER.count("")
    with gr.Blocks(title="Trimwise | Context workbench", analytics_enabled=False) as demo:
        if os.environ.get("SPACES_ZERO_GPU") == "1":
            import spaces

            @spaces.GPU(duration=1)
            def zero_gpu_registration() -> None:
                """Provide ZeroGPU's required handler without wrapping CPU trimming."""

            # ZeroGPU scans registered handlers; normal requests never invoke this one.
            gr.Button(visible=False).click(zero_gpu_registration, api_visibility="private")
        gr.HTML(
            '<header class="workbench-heading"><div class="brand">'
            f"{BRAND_MARK}<h1>trim<span>wise</span></h1>"
            '<span class="workbench-label">Context workbench</span></div>'
            '<nav aria-label="Resources">'
            '<a href="https://trimwise.readthedocs.io/en/latest/" target="_blank" '
            'rel="noopener noreferrer">Docs</a>'
            '<a href="https://github.com/tenwritehq/trimwise" target="_blank" '
            'rel="noopener noreferrer">GitHub</a>'
            '<a class="api-link" href="https://trimwise.aatbit.com" target="_blank" '
            'rel="noopener noreferrer" title="Free API with in-memory text processing; '
            'see setup and privacy details">Free &amp; private API <span aria-hidden="true">'
            "&#8599;</span></a></nav></header>"
        )
        gr.HTML(
            '<div class="runtime-bar">'
            '<span class="runtime-status">CPU &middot; Local library</span>'
            f"<span>tiktoken <code>{CONFIG.token_encoding}</code></span>"
            '<span class="ranking-label">Model-free by default &middot; No LLM or API key</span>'
            "</div>"
        )
        gr.Markdown(
            "Please do not submit sensitive data. This demo does not save your text or results; "
            "Hugging Face may log activity on its infrastructure.",
            elem_classes="privacy-notice",
        )
        with gr.Tabs(elem_id="modes"):
            with gr.Tab("Single source"):
                build_tab(SINGLE_PRESET)
            with gr.Tab("Multiple sources"):
                build_tab(MULTIPLE_PRESET)
        gr.Markdown(
            "The budget covers retained context, including emitted source labels and separators. "
            "Reserve instructions, the query, and eventual LLM output separately.",
            elem_classes="budget-notice",
        )
        with gr.Accordion("Measurement & limits", open=False, elem_classes="measurement-notes"):
            gr.Markdown(
                "Reduction = 100 &times; (1 &minus; retained / original assembled tokens); "
                "empty input = 0%. Time measures the awaited trim API, excluding queue wait, "
                "validation, baseline, and rendering. Spans are `[start, end)` Python character "
                "offsets in each original source; labels and omission markers have no spans. "
                "Per-source token counts measure evidence and omission markers only.\n\n"
                f"**CPU limits:** {MAX_SOURCE_CHARS:,} characters per source, "
                f"{MAX_TOTAL_CHARS:,} characters / {MAX_INPUT_TOKENS:,} evidence tokens / "
                f"{MAX_INPUT_LINES} lines total; query {MAX_QUERY_CHARS} characters; "
                f"budget 0&ndash;{MAX_BUDGET:,} tokens. A tight budget may omit whole sources. "
                "The prefix baseline can end inside a source label or passage; neither method "
                "guarantees enough evidence to answer a question."
            )
        gr.HTML(
            '<footer class="workbench-footer"><span>Trimwise &middot; Open source</span>'
            '<span>Prefer HTTP? <a href="https://trimwise.aatbit.com" target="_blank" '
            'rel="noopener noreferrer">Free API setup &amp; privacy details '
            '<span aria-hidden="true">&#8599;</span></a></span></footer>'
        )
    return demo.queue(max_size=QUEUE_SIZE, default_concurrency_limit=CONCURRENCY, api_open=False)


def workbench_theme() -> gr.themes.Base:
    """Use local system fonts and the public Trimwise site's teal and neutral palette.

    Returns:
        Native Gradio theme tokens for light and dark component states.
    """
    return gr.themes.Default(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="zinc",
        radius_size="sm",
        font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#111315",
        body_text_color="#171c20",
        body_text_color_dark="#f0f3f4",
        body_text_color_subdued="#58636b",
        body_text_color_subdued_dark="#a6b2b9",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#111315",
        background_fill_secondary="#f4f7f8",
        background_fill_secondary_dark="#1b1f22",
        block_background_fill="#ffffff",
        block_background_fill_dark="#111315",
        block_border_width="0px",
        block_shadow="none",
        block_shadow_dark="none",
        border_color_primary="#dce3e7",
        border_color_primary_dark="#353e44",
        input_background_fill="#f7f9fa",
        input_background_fill_dark="#191d20",
        input_border_color="#c2cdd3",
        input_border_color_dark="#59646b",
        input_border_width="1px",
        input_shadow="none",
        input_shadow_focus="none",
        button_primary_background_fill="#087e76",
        button_primary_background_fill_dark="#39e0d0",
        button_primary_background_fill_hover="#06645e",
        button_primary_background_fill_hover_dark="#7be9de",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#11161a",
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
    )


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1",
        share=False,
        show_error=False,
        max_threads=4,
        run_history=False,
        enable_monitoring=False,
        blocked_paths=[str(Path.cwd().resolve())],
        footer_links=[],
        theme=workbench_theme(),
        css_paths=[STYLESHEET],
    )
