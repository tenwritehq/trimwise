"""Render escaped answers and exact source-backed retention highlights."""

from __future__ import annotations

import re
from html import escape

from rag_playground.models import Comparison
from rag_playground.session import Session

EMPTY_ANSWERS = """<div class="answer-grid">
<section class="answer-card baseline"><div class="eyebrow">01 / WITHOUT TRIMWISE</div>
<h3>The full retrieved context</h3><p class="muted">Your first answer will appear here.
Every retrieved passage goes to the answer model.</p>
<div class="waiting">Awaiting a question</div></section>
<section class="answer-card trimmed"><div class="eyebrow">02 / WITH TRIMWISE</div>
<h3>Evidence within your budget</h3><p class="muted">The same passages, question, and model.
Trimwise selects evidence under one shared allowance.</p>
<div class="waiting">Awaiting a question</div></section></div>"""
EMPTY_EVIDENCE = (
    '<div class="empty-panel"><h3>Follow the evidence.</h3><p>After a '
    "comparison, inspect the exact retrieved text and see what Trimwise "
    "kept or removed.</p></div>"
)
EMPTY_METRICS = (
    '<div class="metric-strip"><div><span>RETRIEVED '
    "CONTEXT</span><strong>—</strong></div><div><span>WITH "
    "TRIMWISE</span><strong>—</strong></div><div><span>CONTEXT "
    "REDUCTION</span><strong>—</strong></div></div>"
)


def library(session: Session) -> str:
    """Show the active indexed corpus, independently of pending upload selections.

    Args:
        session: Current browser session.

    Returns:
        Escaped filenames, counts, extraction warnings, and index status.
    """
    if session.corpus is None:
        return (
            '<div class="library-empty">No PDFs indexed yet.<br>Upload your files '
            "or start with the example.</div>"
        )
    corpus = session.corpus
    rows = []
    for record in corpus.loaded.files:
        warning = (
            f" · skipped textless pages {escape(str(record['skipped_pages']))}"
            if record["skipped_pages"]
            else ""
        )
        rows.append(
            f'<div class="file-row"><span '
            f'class="file-type">PDF</span><div><strong>{escape(str(record["name"]))}</strong>'
            f"<small>{record['text_pages']} text pages{warning}</small></div></div>"
        )
    return (
        "".join(rows)
        + (
            f'<div class="library-summary">{len(corpus.loaded.chunks)} chunks · '
            f"{corpus.framework} index<br>"
        )
        + (
            f"Index {corpus.index_seconds:.2f}s · embedding acquire "
            f"{corpus.embedding_load_seconds:.2f}s</div>"
        )
    )


def metrics(run: Comparison) -> str:
    """Compare fully rendered contexts, including source labels and separators.

    Args:
        run: Current comparison.

    Returns:
        Three labelled context metrics without unsupported quality or speed claims.
    """
    before, after = run.context_tokens["baseline"], run.context_tokens["trimwise"]
    reduction = 100 * (before - after) / before if before else 0.0
    return f"""<div class="metric-strip">
<div><span>RETRIEVED CONTEXT</span><strong>{before:,}<small> tokens</small></strong></div>
<div><span>WITH TRIMWISE</span><strong>{after:,}<small> tokens</small></strong></div>
<div><span>CONTEXT REDUCTION</span>
<strong>{reduction:.1f}<small>% this run</small></strong></div></div>
<p class="metric-note">Includes source labels and separators.
Shared evidence allowance:
{run.settings.budget:,} tokens.
Instructions and answer space are additional.</p>"""


def _answer(run: Comparison, condition: str) -> str:
    """Link only valid source labels after escaping all model-produced text.

    Args:
        run: Current comparison.
        condition: Baseline or trimmed condition.

    Returns:
        Safe answer markup and explicit usage/timing or error information.
    """
    if condition in run.errors:
        return f'<div class="run-error" role="status">{escape(run.errors[condition])}</div>'
    generation = run.generations.get(condition)
    if generation is None:
        return '<p class="muted">No answer was generated.</p>'
    documents = run.original if condition == "baseline" else run.compressed
    labels = {document.metadata["citation"] for document in documents}
    text = escape(generation.text or "The model returned no visible answer text.")

    def citation(match: re.Match[str]) -> str:
        """Turn a validated generated label into a local excerpt link.

        Args:
            match: A bracketed source label from escaped text.

        Returns:
            Local link, or unchanged label when the source was not supplied.
        """
        label = match.group(1)
        if label not in labels:
            return (
                f'<span class="invalid-citation" title="This label was not supplied to '
                f'this condition">[{label}]</span>'
            )
        return f'<a class="citation" href="#excerpt-{label}">[{label}]</a>'

    text = re.sub(r"\[(S\d+)\]", citation, text)
    return (
        f'<div class="answer-text">{text}</div>'
        f'<div class="answer-foot"><span>{generation.prompt_tokens:,} prompt · '
        f"{generation.output_tokens:,} output tokens</span>"
        f"<span>{generation.seconds:.2f}s generation · "
        f"{run.totals_seconds[condition]:.2f}s total</span></div>"
        f'<div class="completion-note">{escape(generation.status)} · '
        f"{escape(generation.usage_kind)}</div>"
    )


def answers(run: Comparison) -> str:
    """Show both answers and locally linked exact extracted source excerpts.

    Args:
        run: Current frozen retrieval comparison.

    Returns:
        Safe side-by-side answer panels followed by citation targets.
    """
    sources = []
    for document in run.original:
        label = document.metadata["citation"]
        sources.append(
            f'<details class="excerpt" id="excerpt-{label}"><summary>'
            f'<span class="source-label">[{label}]</span> '
            f"{escape(document.metadata['filename'])} · page {document.metadata['page']}</summary>"
            f"<pre>{escape(document.page_content)}</pre></details>"
        )
    warning = (
        f'<p class="run-error">{escape(run.errors["comparison"])}</p>'
        if "comparison" in run.errors
        else ""
    )
    return (
        warning + '<div class="answer-grid"><section class="answer-card baseline">'
        '<div class="eyebrow">01 / WITHOUT TRIMWISE</div><h3>Full retrieved context</h3>'
        + _answer(run, "baseline")
        + '</section><section class="answer-card trimmed">'
        '<div class="eyebrow">02 / WITH TRIMWISE</div><h3>Shared evidence budget</h3>'
        + _answer(run, "trimwise")
        + "</section></div>"
        '<div class="source-excerpts"><h3>Sources supplied to this run</h3>'
        '<p class="muted">Citations link to extracted excerpts, not a correctness verdict. '
        "Open a source to inspect it.</p>" + "".join(sources) + "</div>"
    )


def evidence(run: Comparison) -> str:
    """Highlight retained spans in the original evidence, leaving removed text visible.

    Args:
        run: Current run with original evidence and adapter span provenance.

    Returns:
        Exact escaped source text with explicit retained/removed styling and context details.
    """
    retained = {
        document.metadata["trimwise"]["source_index"]: document for document in run.compressed
    }
    rows = []
    for index, original in enumerate(run.original):
        document = retained.get(index)
        spans = document.metadata["trimwise"]["spans"] if document else []
        parts = []
        cursor = 0
        for span in spans:
            parts.append(f"<del>{escape(original.page_content[cursor : span['start']])}</del>")
            parts.append(
                f"<mark>{escape(original.page_content[span['start'] : span['end']])}</mark>"
            )
            cursor = span["end"]
        parts.append(f"<del>{escape(original.page_content[cursor:])}</del>")
        state = "Contributes evidence" if document else "No evidence retained"
        rows.append(
            f'<article class="evidence-card"><header><div><span class="source-label">'
            f"[{original.metadata['citation']}]</span> "
            f"<strong>{escape(original.metadata['filename'])}</strong>"
            f"<small>Page {original.metadata['page']} · "
            f"characters {original.metadata['page_start']}-"
            f"{original.metadata['page_end']}</small></div>"
            f'<span class="source-state">{state}</span></header>'
            f'<pre class="evidence-text">{"".join(parts)}</pre></article>'
        )
    contexts = "".join(
        (
            f'<details class="excerpt"><summary>Exact rendered {name} '
            f"context</summary><pre>{escape(text)}</pre></details>"
        )
        for name, text in run.contexts.items()
    )
    return (
        '<div class="evidence-legend"><span><mark>Retained source text</mark></span>'
        "<span><del>Removed source text</del></span></div>"
        '<p class="muted">Highlights refer to exact extracted page text. '
        "Omission markers are added separately and have no source span.</p>"
        + "".join(rows)
        + contexts
    )
