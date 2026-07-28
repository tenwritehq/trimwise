# ruff: noqa: E501
"""Render separate query-aware and queryless benchmark reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

FULL_PROMPT_METHOD = "full_context"
LEGACY_CASE_PASS_LABEL = "Legacy v1.1 case-pass rate"
POSITION_ORDER = ("beginning", "middle", "end", "multiple")
COMPRESSION_GROUP_COLUMNS = (
    "method_id",
    "query_aware",
    "track",
    "evidence_position",
    "budget",
)
EVALUATOR_LABELS = {
    "gpt_5_4_nano": "GPT-5.4 Nano",
    "gpt_5_4_mini": "GPT-5.4 Mini",
    "gpt_5_6_luna": "GPT-5.6 Luna",
}

METHOD_LABELS = {
    "naive_first_n": "Prefix baseline",
    "head_tail": "Head + tail baseline",
    "trimwise_structural": "Trimwise Structural",
    "trimwise_lexical": "Trimwise Lexical",
    "trimwise_hybrid": "Trimwise Hybrid",
    "llmlingua": "LLMLingua GPT-2 token-pruning adapter",
    "llmlingua_queryless": "LLMLingua GPT-2 token-pruning adapter (no question)",
    "longllmlingua": "LongLLMLingua GPT-2 single-context adapter",
    "llmlingua2": "LLMLingua2",
    "recomp_extractive": "RECOMP NQ extractive sentence adapter",
}

METHOD_COLORS = {
    "naive_first_n": "#475569",
    "head_tail": "#64748b",
    "trimwise_structural": "#0f766e",
    "trimwise_lexical": "#155eef",
    "trimwise_hybrid": "#0f766e",
    "llmlingua": "#cc79a7",
    "llmlingua_queryless": "#cc79a7",
    "longllmlingua": "#e69f00",
    "llmlingua2": "#7b61ff",
    "recomp_extractive": "#222222",
}

TASK_GROUPS = {
    "Question answering": ("adversarial", "evidence_qa", "real_source"),
    "Following instructions": ("instruction",),
    "Procedures": ("procedure",),
    "Structured text": ("structured",),
}


@dataclass(frozen=True)
class ReportSpec:
    """Describe one fair comparison report.

    Attributes:
        query_aware: Whether every compressor receives the case question.
        directory: Directory name below the requested output directory.
        title: Report heading.
        lede: One-sentence statement of the comparison condition.
        methods: Method identifiers included in the report.
    """

    query_aware: bool
    directory: str
    title: str
    lede: str
    methods: tuple[str, ...]


REPORTS = (
    ReportSpec(
        query_aware=True,
        directory="query-aware",
        title="Historical v1.1 context selection with a question",
        lede="Every method receives the same source and question. This archived report uses the permissive v1.1 bag-of-token case-pass metric; see the v1.2 sensitivity report for the strict source-span result.",
        methods=(
            "trimwise_lexical",
            "trimwise_hybrid",
            "llmlingua",
            "longllmlingua",
            "recomp_extractive",
        ),
    ),
    ReportSpec(
        query_aware=False,
        directory="queryless",
        title="Historical v1.1 context selection without a question",
        lede="Every method receives only the source. The question is used later, after the text has been shortened. This archived report uses the v1.1 metric.",
        methods=(
            "naive_first_n",
            "head_tail",
            "trimwise_structural",
            "llmlingua_queryless",
            "llmlingua2",
        ),
    ),
)


def plot_summary(
    input_path: Path,
    output_dir: Path,
    natural_input_path: Path | None = None,
    relocated_end_input_path: Path | None = None,
) -> None:
    """Create one static report for each valid question-access condition.

    Args:
        input_path: Aggregated CSV emitted by the benchmark aggregate runner.
        output_dir: Root directory receiving an index and two report directories.
        natural_input_path: Optional aggregate containing only naturally placed cases.
        relocated_end_input_path: Optional aggregate containing only controlled relocations.

    Raises:
        ValueError: If no configured fair track has compression rows.
    """
    import matplotlib.pyplot as plt

    if (natural_input_path is None) != (relocated_end_input_path is None):
        raise ValueError("natural and relocated-end inputs must be supplied together")
    frame = pd.read_csv(input_path)
    natural_frame = pd.read_csv(natural_input_path) if natural_input_path else None
    relocated_end_frame = (
        pd.read_csv(relocated_end_input_path) if relocated_end_input_path else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    available_reports = []
    for report in REPORTS:
        rows = _compression_rows(frame, report)
        if rows.empty:
            continue
        report_dir = output_dir / report.directory
        report_dir.mkdir(parents=True, exist_ok=True)
        figures = _render_figures(
            plt,
            frame,
            rows,
            report,
            report_dir,
            natural_frame,
            relocated_end_frame,
        )
        _write_guide(report, report_dir, natural_frame is not None)
        _write_report(report, frame, rows, figures, report_dir, natural_frame is not None)
        available_reports.append(report)
    if not available_reports:
        raise ValueError("summary does not contain configured query-aware or queryless rows")
    _write_index(available_reports, output_dir)


def _compression_rows(frame: pd.DataFrame, report: ReportSpec) -> pd.DataFrame:
    """Return one main-result row per configured method and budget.

    Args:
        frame: Aggregated benchmark rows.
        report: Fairness-track definition.

    Returns:
        Main comparison rows without QA evaluator duplication.
    """
    rows = _compression_summary_rows(frame)
    rows = rows[
        rows["query_aware"].eq(report.query_aware) & rows["method_id"].isin(report.methods)
    ].copy()
    if rows.empty:
        return rows
    return (
        rows.groupby(["method_id", "budget"], as_index=False)
        .agg(
            macro_case_pass_rate=("macro_case_pass_rate", "first"),
            median_latency_ms=("median_latency_ms", "median"),
            median_output_tokens=("median_output_tokens", "median"),
            success_rate=("success_rate", "mean"),
            budget_violation_rate=("budget_violation_rate", "mean"),
        )
        .sort_values(["budget", "method_id"])
    )


def _compression_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove QA-evaluator copies before aggregating compressor measurements.

    Args:
        frame: Combined aggregate table, including optional QA evaluator columns.

    Returns:
        One row for each recorded compressor, track, position, and budget group.
    """
    available_columns = [column for column in COMPRESSION_GROUP_COLUMNS if column in frame]
    return frame.drop_duplicates(subset=available_columns).copy()


def _weighted_rows(
    rows: pd.DataFrame, group_columns: list[str], value_columns: list[str], weight_column: str
) -> pd.DataFrame:
    """Average rate columns using each aggregate row's original case count.

    Args:
        rows: Aggregated rows containing numeric values and their case counts.
        group_columns: Columns defining the requested output rows.
        value_columns: Rate columns to average.
        weight_column: Column holding the number of contributing cases.

    Returns:
        Weighted values keyed by ``group_columns``.
    """
    if rows.empty:
        return pd.DataFrame(columns=[*group_columns, *value_columns])
    work = rows.copy()
    weights = pd.to_numeric(work[weight_column], errors="coerce").fillna(0.0)
    work["_weight"] = weights
    totals = work.groupby(group_columns, as_index=False, dropna=False)["_weight"].sum()
    for column in value_columns:
        work[f"_{column}"] = pd.to_numeric(work[column], errors="coerce") * weights
        numerators = work.groupby(group_columns, as_index=False, dropna=False)[f"_{column}"].sum()
        totals = totals.merge(numerators, on=group_columns, how="left")
        totals[column] = totals[f"_{column}"].div(totals["_weight"]).where(totals["_weight"].gt(0))
        totals = totals.drop(columns=f"_{column}")
    return totals.drop(columns="_weight")


def _position_rows(frame: pd.DataFrame, report: ReportSpec) -> pd.DataFrame:
    """Return observed case-pass rates for each evidence position.

    Args:
        frame: Aggregated benchmark rows.
        report: Fairness-track definition.

    Returns:
        One weighted case-pass rate per method, budget, and position.
    """
    if "evidence_position" not in frame:
        return pd.DataFrame()
    rows = _compression_summary_rows(frame)
    rows = rows[
        rows["query_aware"].eq(report.query_aware)
        & rows["method_id"].isin(report.methods)
        & rows["evidence_position"].isin(POSITION_ORDER)
    ]
    return _weighted_rows(
        rows,
        ["method_id", "budget", "evidence_position"],
        ["case_pass_rate"],
        "rows",
    )


def _overall_case_pass_rows(frame: pd.DataFrame, report: ReportSpec) -> pd.DataFrame:
    """Return all-case pass rates for one report condition.

    Args:
        frame: Aggregate table for one complete or filtered case cohort.
        report: Fairness-track definition.

    Returns:
        One case-pass rate per method and budget, weighted by contributing cases.
    """
    rows = _compression_summary_rows(frame)
    rows = rows[rows["query_aware"].eq(report.query_aware) & rows["method_id"].isin(report.methods)]
    return _weighted_rows(rows, ["method_id", "budget"], ["case_pass_rate"], "rows")


def _cohort_size(frame: pd.DataFrame, report: ReportSpec, position: str | None = None) -> int:
    """Return the stable case count represented by one comparison cohort.

    Args:
        frame: Aggregate table for one complete or filtered case cohort.
        report: Fairness-track definition.
        position: Optional evidence-position value to retain.

    Returns:
        Number of source cases in the selected cohort, or zero when absent.
    """
    rows = _compression_summary_rows(frame)
    rows = rows[rows["query_aware"].eq(report.query_aware) & rows["method_id"].isin(report.methods)]
    if position is not None:
        rows = rows[rows["evidence_position"].eq(position)]
    if rows.empty:
        return 0
    counts = rows.groupby(["method_id", "budget"], dropna=False)["rows"].sum()
    return int(counts.iloc[0])


def _end_origin_rows(
    frame: pd.DataFrame,
    natural_frame: pd.DataFrame,
    relocated_end_frame: pd.DataFrame,
    report: ReportSpec,
) -> pd.DataFrame:
    """Return natural, relocated, and combined end-position case-pass rates.

    Args:
        frame: Full 160-case aggregate table.
        natural_frame: Aggregate table filtered to naturally placed cases.
        relocated_end_frame: Aggregate table filtered to controlled relocations.
        report: Fairness-track definition.

    Returns:
        One rate per method, budget, and labeled end-position cohort.
    """
    cohorts = (
        ("Natural", natural_frame),
        ("Relocated", relocated_end_frame),
        ("Combined", frame),
    )
    parts = []
    for label, cohort_frame in cohorts:
        values = _position_rows(cohort_frame, report)
        values = values[values["evidence_position"].eq("end")].copy()
        if values.empty:
            continue
        size = _cohort_size(cohort_frame, report, "end")
        values["end_cohort"] = f"{label}\nn={size}"
        parts.append(values)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _qa_rows(frame: pd.DataFrame, report: ReportSpec) -> pd.DataFrame:
    """Return answer-model outcomes without mixing evaluator identities.

    Args:
        frame: Aggregated benchmark rows with optional QA metrics.
        report: Fairness-track definition.

    Returns:
        One weighted answer-match rate per evaluator, method, and budget.
    """
    required = {"qa_model_id", "qa_answer_match", "qa_rows"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows = frame[frame["query_aware"].eq(report.query_aware)].dropna(
        subset=["qa_model_id", "qa_answer_match"]
    )
    return _weighted_rows(
        rows,
        ["qa_model_id", "method_id", "budget"],
        ["qa_answer_match"],
        "qa_rows",
    )


def _task_rows(frame: pd.DataFrame, report: ReportSpec) -> pd.DataFrame:
    """Return case-pass rates grouped into the benchmark's four task families.

    Args:
        frame: Aggregated benchmark rows.
        report: Fairness-track definition.

    Returns:
        One weighted case-pass rate per method, budget, and task family.
    """
    rows = _compression_summary_rows(frame)
    rows = rows[
        rows["query_aware"].eq(report.query_aware) & rows["method_id"].isin(report.methods)
    ].copy()
    rows["task_family"] = ""
    for family, tracks in TASK_GROUPS.items():
        rows.loc[rows["track"].isin(tracks), "task_family"] = family
    rows = rows[rows["task_family"].ne("")]
    return _weighted_rows(
        rows,
        ["method_id", "budget", "task_family"],
        ["case_pass_rate"],
        "rows",
    )


def _render_figures(
    plt: object,
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    report: ReportSpec,
    report_dir: Path,
    natural_frame: pd.DataFrame | None,
    relocated_end_frame: pd.DataFrame | None,
) -> list[tuple[str, str, str]]:
    """Render the compact figure set for one fairness track.

    Args:
        plt: Matplotlib pyplot module.
        frame: Full aggregate table.
        rows: Main compression rows for this report.
        report: Fairness-track definition.
        report_dir: Directory receiving the report's figure files.
        natural_frame: Optional aggregate restricted to naturally placed cases.
        relocated_end_frame: Optional aggregate restricted to controlled relocations.

    Returns:
        Filename, title, and caption entries for the report HTML.
    """
    position_rows = _position_rows(frame, report)
    end_origin_rows = (
        _end_origin_rows(frame, natural_frame, relocated_end_frame, report)
        if natural_frame is not None and relocated_end_frame is not None
        else pd.DataFrame()
    )
    if report.query_aware:
        figures = [
            _render_utility_chart(plt, rows, report, report_dir),
            _render_natural_utility_chart(
                plt,
                _overall_case_pass_rows(natural_frame, report)
                if natural_frame is not None
                else pd.DataFrame(),
                _cohort_size(natural_frame, report) if natural_frame is not None else 0,
                report,
                report_dir,
            ),
            _render_position_chart(plt, position_rows, report, report_dir),
            _render_end_origin_chart(plt, end_origin_rows, report, report_dir),
            _render_tradeoff_chart(plt, rows, report, report_dir),
            _render_task_chart(plt, _task_rows(frame, report), report, report_dir),
        ]
    else:
        figures = [
            _render_position_utility_chart(plt, position_rows, report, report_dir),
            _render_end_origin_chart(plt, end_origin_rows, report, report_dir),
        ]
    if report.query_aware:
        qa_figure = _render_qa_chart(plt, _qa_rows(frame, report), report, report_dir)
        if qa_figure is not None:
            figures.insert(2, qa_figure)
    return [figure for figure in figures if figure is not None]


def _render_utility_chart(
    plt: object, rows: pd.DataFrame, report: ReportSpec, report_dir: Path
) -> tuple[str, str, str]:
    """Render the primary case-pass-versus-budget chart.

    Args:
        plt: Matplotlib pyplot module.
        rows: Main compression rows.
        report: Fairness-track definition.
        report_dir: Directory receiving the figure.

    Returns:
        Figure metadata for the report.
    """
    figure, axis = plt.subplots(figsize=(15.5, 6.0))
    handles: dict[str, object] = {}
    for method_id in report.methods:
        values = rows[rows["method_id"].eq(method_id)].sort_values("budget")
        if values.empty:
            continue
        (line,) = axis.plot(
            values["budget"],
            values["macro_case_pass_rate"],
            color=_method_color(method_id),
            label=_method_label(method_id),
            linewidth=2.6 if method_id.startswith("trimwise_") else 1.8,
            marker="o",
            markersize=6,
        )
        handles[_method_label(method_id)] = line
    axis.set_ylim(0, 1)
    axis.set_xlabel("Maximum kept text (tokens)")
    axis.set_ylabel(LEGACY_CASE_PASS_LABEL)
    axis.set_xticks(sorted(rows["budget"].dropna().unique()))
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, report_dir / "utility_vs_budget.png", handles)
    return (
        "utility_vs_budget.png",
        "Historical v1.1 case pass at each context size",
        "Higher is better under the archived bag-of-token scorer. Question answering, instructions, procedures, and structured-source tasks have equal weight.",
    )


def _render_natural_utility_chart(
    plt: object,
    rows: pd.DataFrame,
    case_count: int,
    report: ReportSpec,
    report_dir: Path,
) -> tuple[str, str, str] | None:
    """Render the natural-placement sensitivity analysis for query-aware methods.

    Args:
        plt: Matplotlib pyplot module.
        rows: All-case pass rates from the natural-only aggregate.
        case_count: Number of naturally placed cases represented by ``rows``.
        report: Fairness-track definition.
        report_dir: Directory receiving the figure.

    Returns:
        Figure metadata when natural-only rows exist, otherwise ``None``.
    """
    if rows.empty:
        return None
    figure, axis = plt.subplots(figsize=(15.5, 6.0))
    handles: dict[str, object] = {}
    for method_id in report.methods:
        values = rows[rows["method_id"].eq(method_id)].sort_values("budget")
        if values.empty:
            continue
        (line,) = axis.plot(
            values["budget"],
            values["case_pass_rate"],
            color=_method_color(method_id),
            label=_method_label(method_id),
            linewidth=2.6 if method_id.startswith("trimwise_") else 1.8,
            marker="o",
            markersize=6,
        )
        handles[_method_label(method_id)] = line
    axis.set_ylim(0, 1)
    axis.set_xlabel("Maximum kept text (tokens)")
    axis.set_ylabel(LEGACY_CASE_PASS_LABEL)
    axis.set_title(f"Natural-placement sensitivity (n={case_count}; positions remain unbalanced)")
    axis.set_xticks(sorted(rows["budget"].dropna().unique()))
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, report_dir / "natural_only_vs_budget.png", handles)
    return (
        "natural_only_vs_budget.png",
        "Historical v1.1 natural-placement check",
        f"The 25 controlled relocations are excluded. This uses {case_count} saved natural cases; it is not position-balanced because only 15 natural cases have end-position evidence.",
    )


def _render_position_utility_chart(
    plt: object, rows: pd.DataFrame, report: ReportSpec, report_dir: Path
) -> tuple[str, str, str] | None:
    """Render separate queryless utility curves for every evidence position.

    Args:
        plt: Matplotlib pyplot module.
        rows: Position-level case-pass rows.
        report: Queryless report definition.
        report_dir: Directory receiving the figure.

    Returns:
        Figure metadata when position data exists, otherwise ``None``.
    """
    if rows.empty:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 11.5), sharex=True, sharey=True)
    handles: dict[str, object] = {}
    for axis, position in zip(axes.flat, POSITION_ORDER, strict=True):
        panel = rows[rows["evidence_position"].eq(position)]
        for method_id in report.methods:
            values = panel[panel["method_id"].eq(method_id)].sort_values("budget")
            if values.empty:
                continue
            (line,) = axis.plot(
                values["budget"],
                values["case_pass_rate"],
                color=_method_color(method_id),
                label=_method_label(method_id),
                linewidth=2.6 if method_id.startswith("trimwise_") else 1.8,
                marker="o",
                markersize=6,
            )
            handles[_method_label(method_id)] = line
        axis.set_title(position.capitalize())
        axis.set_ylim(0, 1)
        axis.set_xticks(sorted(panel["budget"].dropna().unique()))
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Maximum kept text (tokens)")
    for axis in axes[:, 0]:
        axis.set_ylabel(LEGACY_CASE_PASS_LABEL)
    _save_figure(figure, report_dir / "utility_by_position.png", handles)
    return (
        "utility_by_position.png",
        "Historical v1.1 case pass by source location",
        "Each panel shows where the needed source text appears under the archived scorer. The 40-case end panel combines 15 natural end cases with 25 controlled relocations, which the next figure separates.",
    )


def _render_position_chart(
    plt: object, rows: pd.DataFrame, report: ReportSpec, report_dir: Path
) -> tuple[str, str, str] | None:
    """Render a position-stratified case-pass heatmap.

    Args:
        plt: Matplotlib pyplot module.
        rows: Position-level case-pass rows.
        report: Fairness-track definition.
        report_dir: Directory receiving the figure.

    Returns:
        Figure metadata when position data exists, otherwise ``None``.
    """
    if rows.empty:
        return None
    budgets = sorted(rows["budget"].dropna().unique())
    figure, axes = plt.subplots(1, len(budgets), figsize=(4.1 * len(budgets), 4.8), squeeze=False)
    image = None
    for index, budget in enumerate(budgets):
        axis = axes[0][index]
        panel = rows[rows["budget"].eq(budget)]
        values = np.full((len(report.methods), len(POSITION_ORDER)), np.nan)
        for row_index, method_id in enumerate(report.methods):
            for column_index, position in enumerate(POSITION_ORDER):
                match = panel[
                    panel["method_id"].eq(method_id) & panel["evidence_position"].eq(position)
                ]
                if not match.empty:
                    values[row_index, column_index] = match.iloc[0]["case_pass_rate"]
        image = axis.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_title(f"{int(budget)} tokens")
        axis.set_xticks(range(len(POSITION_ORDER)), ("Start", "Middle", "End", "Multiple"))
        axis.set_yticks(
            range(len(report.methods)),
            [_method_label(value) for value in report.methods] if index == 0 else (),
        )
        for row_index, column_index in np.ndindex(values.shape):
            value = values[row_index, column_index]
            if not np.isnan(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="#ffffff" if value >= 0.55 else "#0f172a",
                    fontsize=9,
                    fontweight="bold",
                )
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label=LEGACY_CASE_PASS_LABEL,
            orientation="horizontal",
            fraction=0.06,
            pad=0.17,
        )
    figure.subplots_adjust(bottom=0.28, wspace=0.5)
    _save_figure(figure, report_dir / "position_robustness.png")
    return (
        "position_robustness.png",
        "Does source location matter?",
        "The full position-controlled suite has 40 cases at each location. The combined end column contains 15 natural end cases and 25 controlled relocations; the next figure separates them.",
    )


def _render_end_origin_chart(
    plt: object, rows: pd.DataFrame, report: ReportSpec, report_dir: Path
) -> tuple[str, str, str] | None:
    """Render the natural-versus-relocated decomposition of end-position rows.

    Args:
        plt: Matplotlib pyplot module.
        rows: Case-pass rates for natural, relocated, and combined end cohorts.
        report: Fairness-track definition.
        report_dir: Directory receiving the figure.

    Returns:
        Figure metadata when all end-origin cohorts are present, otherwise ``None``.
    """
    if rows.empty:
        return None
    budgets = sorted(rows["budget"].dropna().unique())
    cohorts = tuple(rows["end_cohort"].drop_duplicates())
    column_count = min(2, len(budgets))
    row_count = (len(budgets) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(10.8, 4.2 * row_count),
        squeeze=False,
        sharey=True,
    )
    for index, budget in enumerate(budgets):
        axis = axes.flat[index]
        panel = rows[rows["budget"].eq(budget)]
        values = np.full((len(report.methods), len(cohorts)), np.nan)
        for row_index, method_id in enumerate(report.methods):
            for column_index, cohort in enumerate(cohorts):
                match = panel[panel["method_id"].eq(method_id) & panel["end_cohort"].eq(cohort)]
                if not match.empty:
                    values[row_index, column_index] = match.iloc[0]["case_pass_rate"]
        axis.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_title(f"{int(budget)} tokens")
        axis.set_xticks(range(len(cohorts)), cohorts, fontsize=9)
        axis.set_yticks(range(len(report.methods)))
        if index % column_count == 0:
            axis.set_yticklabels([_method_label(value) for value in report.methods])
        else:
            axis.tick_params(labelleft=False)
        for row_index, column_index in np.ndindex(values.shape):
            value = values[row_index, column_index]
            if not np.isnan(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="#ffffff" if value >= 0.55 else "#0f172a",
                    fontsize=9,
                    fontweight="bold",
                )
    for axis in axes.flat[len(budgets) :]:
        axis.axis("off")
    figure.subplots_adjust(bottom=0.14, hspace=0.42, wspace=0.24)
    _save_figure(figure, report_dir / "end_origin_breakdown.png")
    return (
        "end_origin_breakdown.png",
        "End-position construction check",
        "Natural end cases (n=15), controlled end relocations (n=25), and their combined 40-case controlled end stratum are reported separately. The combined column is a weighted view of the first two, not an independent sample.",
    )


def _render_qa_chart(
    plt: object, rows: pd.DataFrame, report: ReportSpec, report_dir: Path
) -> tuple[str, str, str] | None:
    """Render answer pass by evaluator and compressed-context budget.

    Args:
        plt: Matplotlib pyplot module.
        rows: Weighted answer-match rows.
        report: Fairness-track definition.
        report_dir: Directory receiving the figure.

    Returns:
        Figure metadata when evaluator rows exist, otherwise ``None``.
    """
    if rows.empty:
        return None
    models = sorted(rows["qa_model_id"].dropna().unique())
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.8), squeeze=False, sharey=True)
    handles: dict[str, object] = {}
    for index, model_id in enumerate(models):
        axis = axes.flat[index]
        panel = rows[rows["qa_model_id"].eq(model_id)]
        for method_id in report.methods:
            values = panel[panel["method_id"].eq(method_id) & panel["budget"].notna()].sort_values(
                "budget"
            )
            if values.empty:
                continue
            (line,) = axis.plot(
                values["budget"],
                values["qa_answer_match"],
                color=_method_color(method_id),
                label=_method_label(method_id),
                linewidth=2.4 if method_id.startswith("trimwise_") else 1.6,
                marker="o",
            )
            handles[_method_label(method_id)] = line
        full_prompt = panel[panel["method_id"].eq(FULL_PROMPT_METHOD)]["qa_answer_match"].dropna()
        if not full_prompt.empty:
            handles["Uncompressed source"] = axis.axhline(
                full_prompt.iloc[0],
                color="#334155",
                linestyle="--",
                linewidth=1.5,
                label="Uncompressed source",
            )
        axis.set_ylim(0, 1)
        axis.set_title(EVALUATOR_LABELS.get(str(model_id), str(model_id)))
        axis.set_xlabel("Context budget (tokens)")
        axis.grid(axis="y", alpha=0.25)
        if index % 2 == 0:
            axis.set_ylabel("Answer-match rate")
    legend_axis = axes.flat[len(models)]
    legend_axis.axis("off")
    legend_axis.legend(
        handles.values(),
        handles.keys(),
        frameon=False,
        loc="center",
        ncols=1,
        fontsize="small",
    )
    for axis in axes.flat[len(models) + 1 :]:
        axis.axis("off")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.92, hspace=0.35, wspace=0.24)
    _save_figure(figure, report_dir / "answer_pass.png")
    return (
        "answer_pass.png",
        "Downstream answer match",
        "Each panel uses a different downstream evaluator. The dashed reference uses the complete source with the same prompt and has no compression budget.",
    )


def _render_tradeoff_chart(
    plt: object, rows: pd.DataFrame, report: ReportSpec, report_dir: Path
) -> tuple[str, str, str]:
    """Render the sole quality-versus-latency trade-off chart.

    Args:
        plt: Matplotlib pyplot module.
        rows: Main compression rows.
        report: Fairness-track definition.
        report_dir: Directory receiving the figure.

    Returns:
        Figure metadata for the report.
    """
    budgets = sorted(rows["budget"].dropna().unique())
    figure, axes = plt.subplots(
        1, len(budgets), figsize=(4.0 * len(budgets), 4.7), squeeze=False, sharey=True
    )
    handles: dict[str, object] = {}
    for index, budget in enumerate(budgets):
        axis = axes[0][index]
        panel = rows[rows["budget"].eq(budget)]
        for method_id in report.methods:
            values = panel[panel["method_id"].eq(method_id)]
            if values.empty:
                continue
            value = values.iloc[0]
            handles[_method_label(method_id)] = axis.scatter(
                value["median_latency_ms"],
                value["macro_case_pass_rate"],
                color=_method_color(method_id),
                s=62,
                label=_method_label(method_id),
            )
        axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_title(f"{int(budget)} tokens")
        axis.set_xlabel("Time to trim (ms, log scale)")
        axis.grid(alpha=0.25)
        if index == 0:
            axis.set_ylabel(LEGACY_CASE_PASS_LABEL)
    _save_figure(figure, report_dir / "utility_vs_latency.png", handles)
    return (
        "utility_vs_latency.png",
        "Historical v1.1 case pass and trimming time",
        "Higher and farther left is preferable. Timing comes from the recorded hardware and run conditions.",
    )


def _render_task_chart(
    plt: object, rows: pd.DataFrame, report: ReportSpec, report_dir: Path
) -> tuple[str, str, str] | None:
    """Render task-family case-pass rates without duplicating latency charts.

    Args:
        plt: Matplotlib pyplot module.
        rows: Task-family case-pass rows.
        report: Fairness-track definition.
        report_dir: Directory receiving the figure.

    Returns:
        Figure metadata when task-family rows exist, otherwise ``None``.
    """
    if rows.empty:
        return None
    budgets = sorted(rows["budget"].dropna().unique())
    families = tuple(TASK_GROUPS)
    figure, axes = plt.subplots(1, len(budgets), figsize=(4.1 * len(budgets), 4.8), squeeze=False)
    image = None
    for index, budget in enumerate(budgets):
        axis = axes[0][index]
        panel = rows[rows["budget"].eq(budget)]
        values = np.full((len(report.methods), len(families)), np.nan)
        for row_index, method_id in enumerate(report.methods):
            for column_index, family in enumerate(families):
                match = panel[panel["method_id"].eq(method_id) & panel["task_family"].eq(family)]
                if not match.empty:
                    values[row_index, column_index] = match.iloc[0]["case_pass_rate"]
        image = axis.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_title(f"{int(budget)} tokens")
        axis.set_xticks(range(len(families)), families, rotation=32, ha="right")
        axis.set_yticks(
            range(len(report.methods)),
            [_method_label(value) for value in report.methods] if index == 0 else (),
        )
        for row_index, column_index in np.ndindex(values.shape):
            value = values[row_index, column_index]
            if not np.isnan(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="#ffffff" if value >= 0.55 else "#0f172a",
                    fontsize=9,
                    fontweight="bold",
                )
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label=LEGACY_CASE_PASS_LABEL,
            orientation="horizontal",
            fraction=0.06,
            pad=0.17,
        )
    figure.subplots_adjust(bottom=0.31, wspace=0.5)
    _save_figure(figure, report_dir / "task_breakdown.png")
    return (
        "task_breakdown.png",
        "Historical v1.1 results by task type",
        "Question answering, following instructions, ordered procedures, and structured source text stay separate rather than being hidden inside one archived score.",
    )


def _save_figure(figure: object, path: Path, handles: dict[str, object] | None = None) -> None:
    """Save a chart with an external legend that never covers observations.

    Args:
        figure: Matplotlib figure to finalize.
        path: Destination PNG path.
        handles: Optional legend entries keyed by display label.
    """
    if handles:
        figure.legend(
            handles.values(),
            handles.keys(),
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            loc="lower center",
            ncols=min(3, len(handles)),
            fontsize="small",
        )
    if handles:
        figure.subplots_adjust(bottom=0.22)
    figure.subplots_adjust(top=0.92)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def _method_label(method_id: str) -> str:
    """Return a stable human-readable benchmark method label.

    Args:
        method_id: Internal method identifier.

    Returns:
        Display label for reports and figures.
    """
    return METHOD_LABELS.get(method_id, method_id)


def _method_color(method_id: str) -> str:
    """Return a consistent chart color for one method.

    Args:
        method_id: Internal method identifier.

    Returns:
        Matplotlib color value.
    """
    return METHOD_COLORS.get(method_id, "#334155")


def _figure_markup(figures: list[tuple[str, str, str]]) -> str:
    """Create accessible figure cards from generated chart metadata.

    Args:
        figures: Filename, title, and caption entries.

    Returns:
        HTML figure cards.
    """
    return "\n".join(
        f"""<article class="figure-card"><figure>
<img src="{escape(filename)}" alt="{escape(title)}" width="1600" height="960" loading="lazy">
<figcaption><strong>{escape(title)}</strong> — {escape(caption)}</figcaption>
</figure></article>"""
        for filename, title, caption in figures
    )


def _scope_markup(rows: pd.DataFrame, report: ReportSpec) -> str:
    """Describe the recorded comparison condition without inventing metadata.

    Args:
        rows: Main compression rows.
        report: Fairness-track definition.

    Returns:
        HTML facts list for the report header.
    """
    budgets = ", ".join(str(int(value)) for value in sorted(rows["budget"].dropna().unique()))
    methods = ", ".join(
        _method_label(method) for method in report.methods if method in set(rows["method_id"])
    )
    access = "Source and question" if report.query_aware else "Source only"
    return f"""<dl class="facts">
<div><dt>Information available</dt><dd>{escape(access)}</dd></div>
<div><dt>Maximum kept text</dt><dd>{escape(budgets)} tokens</dd></div>
<div><dt>Methods shown</dt><dd>{escape(methods)}</dd></div>
<div><dt>Where needed text appears</dt><dd>Beginning · middle · end · several places</dd></div>
</dl>"""


def _reliability_markup(frame: pd.DataFrame, report: ReportSpec) -> str:
    """Create a compact run-reliability table for one report.

    Args:
        frame: Full aggregate table.
        report: Fairness-track definition.

    Returns:
        HTML table of success, budget, latency, memory, and thermal metrics.
    """
    rows = _compression_summary_rows(frame)
    rows = rows[rows["query_aware"].eq(report.query_aware) & rows["method_id"].isin(report.methods)]
    if rows.empty:
        return ""
    summary = rows.groupby("method_id", as_index=False).agg(
        success_rate=("success_rate", "mean"),
        budget_violation_rate=("budget_violation_rate", "mean"),
        p95_latency_ms=("p95_latency_ms", "median"),
        gpu_memory_mb=("median_gpu_peak_reserved_mb", "median"),
        max_temperature_c=("max_thermal_temperature_c", "max"),
        thermal_wait_ms=("total_thermal_wait_ms", "sum"),
    )
    summary = summary.set_index("method_id")
    table_rows = []
    for method_id in report.methods:
        if method_id not in summary.index:
            continue
        row = summary.loc[method_id]
        temperature = (
            "Not recorded" if pd.isna(row.max_temperature_c) else f"{row.max_temperature_c:.0f} °C"
        )
        table_rows.append(
            "<tr>"
            f"<td>{escape(_method_label(method_id))}</td>"
            f"<td>{row.success_rate:.1%}</td>"
            f"<td>{row.budget_violation_rate:.1%}</td>"
            f"<td>{row.p95_latency_ms:.1f} ms</td>"
            f"<td>{row.gpu_memory_mb:.0f} MB</td>"
            f"<td>{temperature}</td>"
            f"<td>{row.thermal_wait_ms / 1_000:.1f} s</td>"
            "</tr>"
        )
    return (
        """<div class="table-wrap"><table><thead><tr>
<th>Method</th><th>Completed</th><th>Over the limit</th><th>95th-percentile time</th><th>Typical GPU memory</th><th>Highest GPU temperature</th><th>Cooling pause</th>
</tr></thead><tbody>"""
        + "\n".join(table_rows)
        + "</tbody></table></div>"
    )


def _write_guide(report: ReportSpec, report_dir: Path, has_origin_sensitivity: bool) -> None:
    """Write concise metric definitions beside a report.

    Args:
        report: Fairness-track definition.
        report_dir: Directory receiving the Markdown guide.
        has_origin_sensitivity: Whether natural and relocation aggregates were supplied.
    """
    access = "the source and question" if report.query_aware else "only the source"
    primary_metric = (
        "- **Legacy v1.1 overall case pass**: question answering, instructions, procedures, and structured-source tasks count equally. A result succeeds when needed source text survives under the archived bag-of-token scorer, prohibited text is absent, and the output stays within its limit."
        if report.query_aware
        else "- **Legacy v1.1 case pass by source location**: each curve uses only one 40-case source-location group. A result succeeds when needed source text survives under the archived scorer, prohibited text is absent, and the output stays within its limit."
    )
    metric_definitions = (
        (
            primary_metric,
            "- **Source location**: needed text appears at the beginning, middle, end, or several places in four equally sized groups.",
            "- **Answer match**: generated answers contain the accepted answer facts. It checks this benchmark's answers; it is not a general measure of reasoning quality.",
            "- **Time to trim**: recorded on this hardware. Cooling pauses are reported separately and are not included in the trimming time.",
        )
        if report.query_aware
        else (
            primary_metric,
            "- **Why there is no single score**: the report does not average away differences caused by where needed text appears.",
        )
    )
    if has_origin_sensitivity:
        metric_definitions += (
            "- **Position construction**: the original 250-case corpus remains unchanged. The separate 160-case evaluation contains 135 natural cases and 25 controlled relocations. The end stratum is shown as natural (n=15), relocated (n=25), and combined (n=40).",
        )
    report_dir.joinpath("plot_guide.md").write_text(
        "\n".join(
            (
                "# How to read this report",
                "",
                f"Every method on this page receives {access}. Do not compare its scores with the other report.",
                "",
                *metric_definitions,
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_report(
    report: ReportSpec,
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    figures: list[tuple[str, str, str]],
    report_dir: Path,
    has_origin_sensitivity: bool,
) -> None:
    """Write one self-contained publication-style HTML benchmark report.

    Args:
        report: Fairness-track definition.
        frame: Full aggregate table.
        rows: Main compression rows.
        figures: Generated figure metadata.
        report_dir: Directory receiving the report files.
        has_origin_sensitivity: Whether natural and relocation aggregates were supplied.
    """
    figure_blocks = _figure_markup(figures)
    position_notice = (
        "The original 250-case corpus remains unchanged; the separate 160-case evaluation contains 135 natural cases and 25 controlled relocations. The end-position figures report natural end (n=15), relocated end (n=25), and the combined controlled end stratum (n=40)."
        if has_origin_sensitivity
        else "This position-controlled suite is a diagnostic of source location, not a naturally representative corpus."
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trimwise benchmark — {escape(report.title)}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#526074; --rule:#d8dee8; --paper:#fff; --surface:#f7f9fc; --accent:#155eef; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
a {{ color:var(--accent); }} .shell {{ max-width:1680px; margin:auto; padding:0 32px; }} header {{ border-bottom:1px solid var(--rule); padding:22px 0; }} .brand {{ font-weight:750; letter-spacing:-.02em; }} main {{ padding:64px 0 76px; }} h1,h2 {{ letter-spacing:-.035em; line-height:1.08; }} h1 {{ font-size:clamp(2.3rem,5vw,4.6rem); max-width:15ch; margin:0; }} h2 {{ font-size:1.7rem; margin:0 0 14px; }} .lede {{ color:var(--muted); font-size:1.18rem; max-width:65ch; }} .notice {{ border-left:4px solid var(--accent); color:var(--muted); margin:32px 0; max-width:76ch; padding:10px 16px; }}
.facts {{ display:grid; gap:1px; grid-template-columns:repeat(4,minmax(0,1fr)); background:var(--rule); margin:42px 0 0; }} .facts div {{ background:var(--surface); min-width:0; padding:18px; }} dt {{ color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }} dd {{ margin:7px 0 0; overflow-wrap:anywhere; }} section {{ border-top:1px solid var(--rule); padding:50px 0; }} .figure-grid {{ display:grid; gap:28px; grid-template-columns:minmax(0,1fr); }} .figure-card {{ background:var(--surface); border:1px solid var(--rule); margin:0; padding:20px; }} figure {{ margin:0; }} img {{ display:block; height:auto; width:100%; }} figcaption {{ color:var(--muted); font-size:.9rem; margin-top:12px; }} .table-wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; min-width:860px; width:100%; }} th,td {{ border-bottom:1px solid var(--rule); padding:12px; text-align:left; vertical-align:top; }} th {{ color:var(--muted); font-size:.76rem; letter-spacing:.05em; text-transform:uppercase; }} footer {{ border-top:1px solid var(--rule); color:var(--muted); font-size:.9rem; padding:26px 0; }}
@media (max-width:760px) {{ .shell {{ padding:0 18px; }} main {{ padding-top:42px; }} .facts {{ grid-template-columns:minmax(0,1fr); }} }}
</style>
</head>
<body>
<header><div class="shell"><a class="brand" href="../index.html">Trimwise benchmark</a></div></header>
<main class="shell">
<p>POSITION-CONTROLLED EVALUATION SUITE</p>
<h1>{escape(report.title)}</h1>
<p class="lede">{escape(report.lede)}</p>
<p class="notice">{escape(position_notice)}</p>
{_scope_markup(rows, report)}
<section><h2>Results</h2><div class="figure-grid">{figure_blocks}</div></section>
<section><h2>Completion and resource use</h2><p class="lede">A useful method must finish and stay within the requested limit.</p>{_reliability_markup(frame, report)}</section>
<section><h2>How scores are calculated</h2><p>See <a href="plot_guide.md">metric definitions</a>. The underlying summary data is available in the supplied CSV file.</p></section>
</main>
<footer><div class="shell">Benchmark summary. Results are separated by whether each method receives the question.</div></footer>
</body></html>"""
    report_dir.joinpath("index.html").write_text(page, encoding="utf-8")


def _write_index(reports: list[ReportSpec], output_dir: Path) -> None:
    """Write the small root page linking available benchmark reports.

    Args:
        reports: Fairness tracks that produced report directories.
        output_dir: Root report directory.
    """
    links = "\n".join(
        f'<li><a href="{escape(report.directory)}/index.html">{escape(report.title)}</a></li>'
        for report in reports
    )
    follow_up_links = "".join(
        f'<li><a href="{directory}/index.html">{label}</a></li>\n'
        for directory, label in (
            ("direct-retrieval-v1", "Direct fixed-window retrieval follow-up"),
            ("ablation-v1", "Exploratory Hybrid component study"),
        )
        if output_dir.joinpath(directory, "index.html").is_file()
    )
    links = (
        '<li><a href="evidence-sensitivity-v1-2/index.html">Strict source-span survival</a></li>\n'
        + follow_up_links
        + links
    )
    output_dir.joinpath("index.html").write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Trimwise benchmark reports</title>
<body style="font:18px/1.5 system-ui,sans-serif;max-width:720px;margin:10vh auto;padding:24px"><h1>Trimwise benchmark reports</h1><p>The strict source-span report is the primary result. The direct-retrieval and component studies are separate follow-ups over the same suite. The two legacy reports are preserved only as historical bag-of-token diagnostics; their scores cannot be compared across question access conditions. Each historical report separates the 135 natural cases from the 25 controlled end relocations.</p><ul>{links}</ul></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    """Parse report paths and render both fair comparison reports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/position_controlled_160_summary.csv")
    parser.add_argument("--natural-input")
    parser.add_argument("--relocated-end-input")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    plot_summary(
        Path(args.input),
        Path(args.output_dir),
        Path(args.natural_input) if args.natural_input else None,
        Path(args.relocated_end_input) if args.relocated_end_input else None,
    )


if __name__ == "__main__":
    main()
