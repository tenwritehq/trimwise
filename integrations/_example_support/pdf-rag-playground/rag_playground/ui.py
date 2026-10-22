"""Build a local document workspace with a transparent two-condition comparison."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr
from gradio.blocks import Block

from rag_playground.documents import ChunkSettings
from rag_playground.evaluation import evaluate
from rag_playground.exports import export_data
from rag_playground.models import RunSettings
from rag_playground.presentation import (
    EMPTY_ANSWERS,
    EMPTY_EVIDENCE,
    EMPTY_METRICS,
    answers,
    evidence,
    library,
    metrics,
)
from rag_playground.retrieval import CorpusBuilder
from rag_playground.session import FIXTURE, Session

ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTION = "What caused the Atlas outage, and when did service recover?"


def _close_session(session: Session) -> None:
    """Release per-browser data after Gradio expires its state.

    Args:
        session: Server-side browser state.
    """
    session.close()


def build_app(runtime_root: Path, builder: CorpusBuilder, framework: str) -> gr.Blocks:
    """Create an accessible Gradio workbench with per-session collections without starting a server.

    Args:
        runtime_root: Example-owned temporary directory for uploads and exports.
        builder: This example's native framework index factory.
        framework: Fixed framework label for the current example.

    Returns:
        A Blocks app ready for a localhost-only launch.
    """

    def new_session() -> Session:
        """Allocate a fresh corpus owner for each browser tab.

        Returns:
            Independent session state.
        """
        return Session(runtime_root, builder)

    app = gr.Blocks(
        title=f"Trimwise / {framework} RAG Lab", analytics_enabled=False, delete_cache=(300, 3600)
    )
    with app:
        state = gr.State(new_session, time_to_live=3600, delete_callback=_close_session)
        gr.HTML(f"""<a class="skip-link" href="#question-workspace">Skip to question</a>
<header class="app-header"><div class="wordmark"><span class="brand-brackets">[tw]</span>
<span>trimwise <span class="wordmark-divider">/</span> <strong>{framework} RAG</strong></span></div>
<span class="local-badge">LOCAL WORKSPACE</span></header>""")
        with gr.Row(elem_id="workspace", equal_height=False):
            with gr.Column(scale=1, min_width=270, elem_id="document-sidebar"):
                gr.HTML(
                    '<div class="sidebar-heading"><span class="eyebrow">YOUR KNOWLEDGE '
                    "BASE</span><h2>Documents</h2><p>Bring the evidence.<br>See what makes "
                    "the cut.</p></div>"
                )
                uploads = gr.File(
                    label="Upload text PDFs",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                    elem_id="pdf-upload",
                )
                gr.Markdown(
                    "Up to 5 PDFs · 10 MiB each · selectable text only", elem_classes="small-note"
                )
                index_button = gr.Button("Index PDFs", variant="primary", elem_id="index-button")
                example_button = gr.Button("Load the example", elem_id="example-button")
                library_output = gr.HTML(
                    '<div class="library-empty">No PDFs indexed yet.<br>Upload your files '
                    "or start with the example.</div>",
                    elem_id="library-status",
                )
                with gr.Accordion("Chunking settings", open=False):
                    chunk_size = gr.Slider(
                        200, 2000, value=1000, step=100, label="Chunk size · characters"
                    )
                    overlap = gr.Slider(0, 300, value=120, step=20, label="Overlap · characters")
                    gr.Markdown(
                        "Changing files or chunk settings requires indexing again. Each index "
                        "replaces this session's full document set."
                    )
                reset_button = gr.Button(
                    "Reset session", variant="secondary", elem_id="reset-button"
                )
                gr.HTML(
                    '<div class="privacy-note"><strong>Your documents stay '
                    "here.</strong><p>Parsing, embeddings, and indexing run locally. Answer "
                    "generation uses your local Ollama server. No document text is sent to "
                    "a cloud API.</p></div>"
                )
            with gr.Column(scale=4, min_width=320, elem_id="question-workspace"):
                gr.HTML(
                    '<div class="workspace-heading"><div class="eyebrow">LESS CONTEXT. '
                    "INSPECT THE DIFFERENCE.</div><h1>Ask your documents.</h1><p>One "
                    f"{framework} retrieval. Two answers. Every retained passage traceable to its "
                    "source.</p></div>"
                )
                question = gr.Textbox(
                    label="Your question",
                    placeholder="Ask something your PDFs can answer…",
                    lines=2,
                    max_lines=5,
                    elem_id="question",
                )
                with gr.Row(elem_id="run-controls"):
                    gr.Markdown("Qwen3 4B · **CPU only**")
                    top_k = gr.Slider(
                        1, 12, value=6, step=1, label="Retrieved chunks · top-k", scale=1
                    )
                    budget = gr.Slider(
                        16,
                        2048,
                        value=256,
                        step=16,
                        label="Shared evidence budget · tokens",
                        scale=2,
                    )
                with gr.Accordion("Answer settings", open=False):
                    with gr.Row():
                        answer_tokens = gr.Slider(
                            32,
                            512,
                            value=160,
                            step=32,
                            label="Answer allowance · tokens per condition",
                        )
                        strategy = gr.Dropdown(
                            ["lexical", "structural", "auto"],
                            value="lexical",
                            label="Trimwise strategy",
                        )
                    gr.Markdown(
                        "Qwen3-4B-Instruct Q4 uses temperature zero and a 4,096-token context "
                        "window. Both conditions use the same instructions and output "
                        "allowance. Labels and prompt instructions sit outside the evidence "
                        "budget."
                    )
                compare_button = gr.Button(
                    "Compare answers", variant="primary", size="lg", elem_id="compare-button"
                )
                status = gr.Textbox(
                    "Ready when you are. Upload PDFs, or load the example to get started.",
                    label="Run status",
                    interactive=False,
                    lines=2,
                    elem_id="run-status",
                )
                metrics_output = gr.HTML(EMPTY_METRICS, elem_id="context-metrics")
                with gr.Tabs(elem_id="result-tabs"):
                    with gr.Tab("Answer comparison"):
                        answers_output = gr.HTML(EMPTY_ANSWERS, elem_id="answer-results")
                    with gr.Tab("Evidence inspector"):
                        evidence_output = gr.HTML(EMPTY_EVIDENCE, elem_id="evidence-results")
                    with gr.Tab("Run details"):
                        gr.Markdown(
                            "Timings are single-run observations. Model/index load "
                            "is separate. "
                            "Baseline generation runs first; cache and order effects "
                            "can affect "
                            "latency."
                        )
                        details_output = gr.JSON(
                            value={}, label="Configuration, counts, source references, and timings"
                        )
                    with gr.Tab("Evaluate & export"):
                        feedback = gr.Radio(
                            [
                                "Unrated",
                                "Baseline better",
                                "Trimwise better",
                                "Tie",
                                "Both incorrect",
                            ],
                            value="Unrated",
                            label="Your assessment",
                        )
                        reference = gr.Textbox(
                            label="Reference answer · optional",
                            lines=2,
                            placeholder="For a narrow normalized exact-match check",
                        )
                        support = gr.Textbox(
                            label="Supporting quote · optional",
                            lines=2,
                            placeholder="Paste an exact quote contained within one indexed chunk",
                        )
                        evaluate_button = gr.Button("Save assessment & check evidence")
                        evaluation_output = gr.JSON(
                            value={}, label="Explicit checks · not an accuracy score"
                        )
                        include_content = gr.Checkbox(
                            value=False,
                            label=(
                                "Include question, answers, filenames, and document text in the "
                                "exported JSON"
                            ),
                        )
                        export_button = gr.Button("Export this comparison")
                        download = gr.File(label="Comparison export", interactive=False)
                gr.HTML(
                    f'<footer class="workspace-footer"><span>Built with {framework} + '
                    "Trimwise</span>"
                    "<span>Source spans show provenance. They do not prove "
                    "an answer is correct.</span></footer>"
                )

        outputs = [
            library_output,
            status,
            metrics_output,
            answers_output,
            evidence_output,
            details_output,
            evaluation_output,
            feedback,
            download,
        ]
        controls: set[Block] = {
            state,
            uploads,
            question,
            top_k,
            budget,
            chunk_size,
            overlap,
            answer_tokens,
            strategy,
        }

        def cleared() -> dict[Any, Any]:
            """Clear prior answers and exports before a new operation starts.

            Returns:
                Updates for derived comparison outputs.
            """
            return {
                metrics_output: EMPTY_METRICS,
                answers_output: EMPTY_ANSWERS,
                evidence_output: EMPTY_EVIDENCE,
                details_output: {},
                evaluation_output: {},
                feedback: "Unrated",
                download: None,
            }

        def index_selected(values: dict[Any, Any]) -> dict[Any, Any]:
            """Index the full current upload selection without mixing stale documents.

            Args:
                values: Server-validated Gradio component values.

            Returns:
                Updated library/status and cleared old results.
            """
            session = values[state]
            result = cleared()
            try:
                session.index(
                    [Path(path) for path in values[uploads] or []],
                    ChunkSettings(int(values[chunk_size]), int(values[overlap])),
                )
                result[status] = "Indexed locally. Ask a question to compare both conditions."
            except Exception as error:
                result[status] = (
                    "Indexing stopped: "
                    f"{str(error) if isinstance(error, ValueError) else type(error).__name__}"
                )
            result[library_output] = library(session)
            return result

        def load_example(values: dict[Any, Any]) -> dict[Any, Any]:
            """Use the bundled public PDF through the same real indexing path.

            Args:
                values: Current session and indexing controls.

            Returns:
                Updated upload, question, library, and result controls.
            """
            values[uploads] = [str(FIXTURE)]
            result = index_selected(values)
            result.update({uploads: [str(FIXTURE)], question: DEFAULT_QUESTION})
            return result

        def run_compare(values: dict[Any, Any]) -> Iterator[dict[Any, Any]]:
            """Display progress, then perform one frozen-retrieval comparison.

            Args:
                values: Current session, controls, and local runtime settings.

            Yields:
                Cleared busy state followed by completed results or a safe error.
            """
            session = values[state]
            result = cleared()
            session.run = None
            result[status] = (
                "Comparing… First use may download the tokenizer and load the local model. "
                "Both answers "
                "use the same retrieved evidence."
            )
            yield result
            try:
                settings = RunSettings(
                    int(values[top_k]),
                    int(values[budget]),
                    int(values[answer_tokens]),
                    values[strategy],
                )
                run = session.compare_selected(
                    (values[question], settings),
                    (
                        [Path(path) for path in values[uploads] or []],
                        ChunkSettings(int(values[chunk_size]), int(values[overlap])),
                    ),
                )
                result.update(
                    {
                        metrics_output: metrics(run),
                        answers_output: answers(run),
                        evidence_output: evidence(run),
                        details_output: export_data(run),
                        status: "Comparison recorded. Inspect the answers and evidence below."
                        if not run.errors
                        else (
                            "Comparison needs attention. See the condition messages; no baseline "
                            "truncation was applied."
                        ),
                    }
                )
            except Exception as error:
                message = (
                    str(error)
                    if isinstance(error, ValueError)
                    else f"{type(error).__name__}. Check the selected backend and retry."
                )
                result[status] = f"Comparison stopped: {message}"
            yield result

        def reset(session: Session) -> dict[Any, Any]:
            """Reset only the current session's collection, results, and generated exports.

            Args:
                session: Current browser state.

            Returns:
                Fresh empty controls, with the upload cache policy made explicit.
            """
            session.reset()
            return {
                **cleared(),
                uploads: None,
                library_output: library(session),
                status: (
                    "Session index and exports cleared. Temporary upload/download cache "
                    "expires within one hour or when the app exits."
                ),
            }

        def save_evaluation(
            session: Session, rating: str, reference_text: str, quote: str
        ) -> dict[Any, Any]:
            """Save explicit human feedback and optional narrow reference checks.

            Args:
                session: Current run owner.
                rating: Human assessment.
                reference_text: Optional reference answer.
                quote: Optional exact source quote.

            Returns:
                Evaluation data or a clear action message.
            """
            with session.lock:
                if session.run is None or session.corpus is None:
                    return {status: "Run a comparison before evaluating it."}
                session.run.feedback = rating
                try:
                    session.run.evaluation = evaluate(
                        session.run, session.corpus, (reference_text, quote)
                    )
                except ValueError as error:
                    return {status: str(error)}
                return {
                    evaluation_output: session.run.evaluation,
                    details_output: export_data(session.run),
                    status: "Assessment saved locally for this run.",
                }

        def export(session: Session, content: bool) -> dict[Any, Any]:
            """Create only an explicitly requested downloadable run record.

            Args:
                session: Current run owner.
                content: Permission to include user document content.

            Returns:
                Downloadable file or a prerequisite message.
            """
            try:
                return {
                    download: str(session.export(content)),
                    status: "Export ready. Content inclusion follows the checkbox above.",
                }
            except ValueError as error:
                return {status: str(error), download: None}

        events: dict[str, Any] = {
            "api_name": False,
            "concurrency_id": "playground",
            "concurrency_limit": 1,
        }
        index_button.click(index_selected, inputs=controls, outputs=outputs, **events)
        example_button.click(
            load_example, inputs=controls, outputs=[*outputs, uploads, question], **events
        )
        compare_button.click(
            run_compare, inputs=controls, outputs=outputs, show_progress="minimal", **events
        )
        reset_button.click(reset, inputs=state, outputs=[*outputs, uploads], **events)
        evaluate_button.click(
            save_evaluation,
            inputs=[state, feedback, reference, support],
            outputs=[status, evaluation_output, details_output],
            **events,
        )
        export_button.click(
            export, inputs=[state, include_content], outputs=[download, status], **events
        )
    app.queue(max_size=8, default_concurrency_limit=1, api_open=False)
    return app
