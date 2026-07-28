# ruff: noqa: E501
"""Render strict direct-retrieval baseline figures from a combined summary."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BUDGETS = (128, 256, 512, 1024)
METRIC = "normalized_contiguous_case_pass_rate"
METHODS = (
    ("trimwise_lexical", "Trimwise Lexical", "#155eef"),
    ("trimwise_hybrid", "Trimwise Hybrid", "#0f766e"),
    ("bm25_128_32_source_order", "BM25 fixed-window", "#7e22ce"),
    ("embedding_topk_source_order", "Embedding top-k", "#0891b2"),
    ("embedding_mmr_source_order", "Embedding + MMR", "#be123c"),
)
POSITIONS = ("beginning", "middle", "end", "multiple")


def _parse_arguments() -> argparse.Namespace:
    """Return command-line arguments for direct-retrieval report generation.

    Returns:
        Paths for the combined summary and report directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/position_controlled_160_direct_retrieval_summary.csv",
        type=Path,
    )
    parser.add_argument("--output-dir", default="reports/direct-retrieval-v1", type=Path)
    parser.add_argument(
        "--docs-output",
        default=Path("../docs/assets/benchmark/direct_retrieval_vs_budget.svg"),
        type=Path,
    )
    parser.add_argument(
        "--docs-position-output",
        default=Path("../docs/assets/benchmark/direct_retrieval_by_position_256.svg"),
        type=Path,
    )
    parser.add_argument(
        "--docs-report-output",
        default=Path("../docs/benchmark-report/direct-retrieval.html"),
        type=Path,
    )
    return parser.parse_args()


def _weighted_rates(summary: pd.DataFrame, groups: list[str], metric: str = METRIC) -> pd.DataFrame:
    """Calculate all-case or stratum-weighted strict pass rates.

    Args:
        summary: Grouped aggregate CSV containing per-track rows.
        groups: Additional columns that define the requested result strata.
        metric: Aggregate rate column to weight by case count.

    Returns:
        One weighted strict rate for each method, budget, and requested stratum.

    Raises:
        ValueError: If a direct baseline is absent from the aggregate input.
    """
    method_ids = [method_id for method_id, _, _ in METHODS]
    rows = summary.loc[
        summary["query_aware"].eq(True) & summary["method_id"].isin(method_ids),
        ["method_id", "budget", *groups, "rows", metric],
    ].copy()
    if rows.empty:
        raise ValueError("summary contains no direct-retrieval comparison rows")
    rows["weighted_passes"] = rows["rows"] * rows[metric]
    grouped = rows.groupby(["method_id", "budget", *groups], as_index=False).agg(
        rows=("rows", "sum"), weighted_passes=("weighted_passes", "sum")
    )
    grouped[metric] = grouped["weighted_passes"].div(grouped["rows"])
    expected = {(method_id, budget) for method_id, _, _ in METHODS for budget in BUDGETS}
    actual = set(zip(grouped["method_id"], grouped["budget"], strict=True))
    missing = expected - actual
    if missing:
        raise ValueError(f"summary is missing direct-retrieval rates: {sorted(missing)}")
    return grouped


def _render_budget_figure(rates: pd.DataFrame, output_dir: Path, docs_output: Path) -> None:
    """Render all-budget strict case-pass comparison.

    Args:
        rates: All-case weighted strict pass rates.
        output_dir: Directory receiving PNG and SVG figures.
        docs_output: MkDocs-local SVG receiving the same comparison figure.
    """
    plt.rcParams["svg.fonttype"] = "none"
    figure, axis = plt.subplots(figsize=(12, 7.5))
    for method_id, label, color in METHODS:
        method_rates = rates.loc[rates["method_id"].eq(method_id)].set_index("budget")
        axis.plot(
            BUDGETS,
            [method_rates.loc[budget, METRIC] * 100 for budget in BUDGETS],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(BUDGETS, [f"{budget:,}" for budget in BUDGETS])
    axis.set_ylim(0, 100)
    axis.set_xlabel("Output-token budget")
    axis.set_ylabel("Feasible strict case pass (%)")
    axis.grid(axis="y", color="#cbd5e1", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Trimwise versus direct fixed-window retrieval", fontsize=17, fontweight="bold")
    axis.set_title(
        "Normalized contiguous required-span containment · 160 position-controlled cases",
        fontsize=10.5,
        color="#475569",
        pad=12,
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9)
    figure.subplots_adjust(bottom=0.28, top=0.86, left=0.1, right=0.98)
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"direct_retrieval_vs_budget.{suffix}", dpi=180)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(docs_output)
    plt.close(figure)


def _render_position_figure(rates: pd.DataFrame, output_dir: Path, docs_output: Path) -> None:
    """Render the strict 256-token result separately for every evidence position.

    Args:
        rates: Position-stratified weighted strict pass rates.
        output_dir: Directory receiving PNG and SVG figures.
        docs_output: MkDocs-local SVG receiving the same position figure.
    """
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, position in zip(axes.flat, POSITIONS, strict=True):
        values = rates.loc[
            rates["evidence_position"].eq(position) & rates["budget"].eq(256)
        ].set_index("method_id")
        for index, (method_id, _label, color) in enumerate(METHODS):
            axis.bar(index, values.loc[method_id, METRIC] * 100, color=color, width=0.72)
        axis.set_title(position.capitalize())
        axis.set_ylim(0, 100)
        axis.set_xticks(
            range(len(METHODS)), [label for _, label, _ in METHODS], rotation=24, ha="right"
        )
        axis.grid(axis="y", color="#cbd5e1", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Direct retrieval comparison by evidence position at 256 tokens",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(0.02, 0.5, "Feasible strict case pass (%)", va="center", rotation="vertical")
    figure.tight_layout(rect=(0.04, 0.02, 1, 0.94))
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"direct_retrieval_by_position_256.{suffix}", dpi=180)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(docs_output)
    plt.close(figure)


def _rate_table(rates: pd.DataFrame) -> str:
    """Build the all-budget strict-pass HTML table.

    Args:
        rates: All-case weighted strict pass rates.

    Returns:
        Escaped HTML table with one row per evaluated method.
    """
    header = "".join(f"<th>{budget:,}</th>" for budget in BUDGETS)
    body = []
    for method_id, label, _ in METHODS:
        method_rates = rates.loc[rates["method_id"].eq(method_id)].set_index("budget")
        cells = "".join(
            f"<td>{method_rates.loc[budget, METRIC] * 100:.1f}%</td>" for budget in BUDGETS
        )
        body.append(f"<tr><th>{escape(label)}</th>{cells}</tr>")
    return (
        '<table style="border-collapse:collapse;width:100%;margin:20px 0">'
        f"<thead><tr><th>Method</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _feasibility_table(summary: pd.DataFrame) -> str:
    """Build the 256-token evidence and feasibility diagnostics table.

    Args:
        summary: Combined aggregate CSV containing direct-retrieval methods.

    Returns:
        Escaped HTML table separating span survival, final pass, and prohibited text.
    """
    evidence = _weighted_rates(summary, [], "normalized_contiguous_required_evidence_success")
    feasible = _weighted_rates(summary, [], METRIC)
    prohibited = _weighted_rates(summary, [], "prohibited_phrase_rate")
    body = []
    for method_id, label, _ in METHODS:
        evidence_rate = evidence.loc[
            (evidence["method_id"].eq(method_id)) & evidence["budget"].eq(256),
            "normalized_contiguous_required_evidence_success",
        ].item()
        feasible_rate = feasible.loc[
            (feasible["method_id"].eq(method_id)) & feasible["budget"].eq(256), METRIC
        ].item()
        prohibited_rate = prohibited.loc[
            (prohibited["method_id"].eq(method_id)) & prohibited["budget"].eq(256),
            "prohibited_phrase_rate",
        ].item()
        body.append(
            f"<tr><th>{escape(label)}</th><td>{evidence_rate * 100:.1f}%</td>"
            f"<td>{feasible_rate * 100:.1f}%</td><td>{prohibited_rate * 100:.1f}%</td></tr>"
        )
    return (
        '<table style="border-collapse:collapse;width:100%;margin:20px 0">'
        "<thead><tr><th>Method</th><th>Span survival</th><th>Feasible pass</th>"
        "<th>Prohibited text</th></tr></thead><tbody>"
        f"{''.join(body)}</tbody></table>"
    )


def _report_page(table: str, feasibility: str, figure_path: str, position_figure_path: str) -> str:
    """Build the shared standalone direct-retrieval report page.

    Args:
        table: Rendered all-budget strict-pass table.
        feasibility: Rendered evidence and feasibility diagnostics table.
        figure_path: Relative SVG path for the all-budget figure.
        position_figure_path: Relative SVG path for the position figure.

    Returns:
        Complete report HTML.
    """
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trimwise direct retrieval baselines</title>
<body style="font:18px/1.5 system-ui,sans-serif;max-width:1180px;margin:6vh auto;padding:24px;color:#172033">
<p style="color:#526177;font-weight:700">QUERY-AWARE DIRECT BASELINES</p>
<h1>Does Trimwise add more than ordinary retrieval?</h1>
<p>Every method receives the same source, question, budgets, and strict source-span scorer.
The three direct baselines use 128-token fixed windows with 32-token overlap and reconstruct their
selected source windows in source order. No downstream LLM calls are included.</p>
<img src="{figure_path}" alt="Strict case pass versus budget for Trimwise and direct retrieval baselines." style="width:100%;height:auto">
{table}
<h2>Evidence survival versus feasible pass at 256 tokens</h2>
<p>Span survival ignores prohibited content. Feasible pass also requires prohibited text to be absent
and the output to fit its budget, so it reveals when retrieving more evidence also retrieves content
the case explicitly forbids.</p>
{feasibility}
<img src="{position_figure_path}" alt="Strict case pass by evidence position at 256 tokens." style="width:100%;height:auto">
</body></html>"""


def _write_report(
    summary: pd.DataFrame, rates: pd.DataFrame, output_dir: Path, docs_report_output: Path
) -> None:
    """Write a compact standalone direct-retrieval report.

    Args:
        summary: Combined grouped aggregate CSV containing direct-retrieval methods.
        rates: All-case weighted strict pass rates.
        output_dir: Directory receiving the report HTML.
        docs_report_output: MkDocs-local version of the standalone report.
    """
    table = _rate_table(rates)
    feasibility = _feasibility_table(summary)
    output_dir.joinpath("index.html").write_text(
        _report_page(
            table,
            feasibility,
            "direct_retrieval_vs_budget.svg",
            "direct_retrieval_by_position_256.svg",
        ),
        encoding="utf-8",
    )
    docs_report_output.parent.mkdir(parents=True, exist_ok=True)
    docs_report_output.write_text(
        _report_page(
            table,
            feasibility,
            "../assets/benchmark/direct_retrieval_vs_budget.svg",
            "../assets/benchmark/direct_retrieval_by_position_256.svg",
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Render the direct-retrieval figures and standalone HTML report."""
    arguments = _parse_arguments()
    summary = pd.read_csv(arguments.input)
    all_case_rates = _weighted_rates(summary, [])
    position_rates = _weighted_rates(summary, ["evidence_position"])
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _render_budget_figure(all_case_rates, arguments.output_dir, arguments.docs_output)
    _render_position_figure(position_rates, arguments.output_dir, arguments.docs_position_output)
    _write_report(summary, all_case_rates, arguments.output_dir, arguments.docs_report_output)


if __name__ == "__main__":
    main()
