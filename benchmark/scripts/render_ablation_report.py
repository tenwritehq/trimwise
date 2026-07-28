# ruff: noqa: E501
"""Render strict Trimwise Hybrid component-ablation figures and an HTML report."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BUDGETS = (128, 256, 512, 1024)
METRIC = "normalized_contiguous_case_pass_rate"
METHODS = (
    ("trimwise_hybrid", "Trimwise Hybrid", "#0f766e"),
    ("trimwise_hybrid_no_mmr", "Hybrid, relevance-only", "#7b61ff"),
    ("trimwise_hybrid_no_evidence_cutoff", "Hybrid, no evidence cutoff", "#e69f00"),
    ("trimwise_hybrid_fixed_windows", "Hybrid, fixed windows", "#cc79a7"),
)


def _parse_arguments() -> argparse.Namespace:
    """Return command-line paths for the ablation report.

    Returns:
        Parsed report input and output locations.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/position_controlled_160_ablation_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--paired-input",
        default="results/position_controlled_160_ablation_paired_stats.csv",
        type=Path,
    )
    parser.add_argument("--output-dir", default="reports/ablation-v1", type=Path)
    parser.add_argument(
        "--docs-output",
        default=Path("../docs/assets/benchmark/ablation_vs_hybrid.svg"),
        type=Path,
    )
    parser.add_argument(
        "--docs-report-output",
        default=Path("../docs/benchmark-report/ablation.html"),
        type=Path,
    )
    return parser.parse_args()


def _all_case_rates(summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate row-weighted strict all-case rates for the four configurations.

    Args:
        summary: Track- and position-grouped aggregate CSV.

    Returns:
        One strict pass rate per configuration and output budget.

    Raises:
        ValueError: If an expected configuration or budget is absent.
    """
    method_ids = [method_id for method_id, _, _ in METHODS]
    rates = summary.loc[
        summary["query_aware"].eq(True) & summary["method_id"].isin(method_ids),
        ["method_id", "budget", "rows", METRIC],
    ].copy()
    rates["weighted_passes"] = rates["rows"] * rates[METRIC]
    rates = rates.groupby(["method_id", "budget"], as_index=False).agg(
        rows=("rows", "sum"), weighted_passes=("weighted_passes", "sum")
    )
    rates[METRIC] = rates["weighted_passes"].div(rates["rows"])
    expected = {(method_id, budget) for method_id, _, _ in METHODS for budget in BUDGETS}
    actual = set(zip(rates["method_id"], rates["budget"], strict=True))
    if missing := expected - actual:
        raise ValueError(f"summary is missing ablation rates: {sorted(missing)}")
    return rates


def _render_figure(rates: pd.DataFrame, output_dir: Path, docs_output: Path) -> None:
    """Render the all-budget strict case-pass ablation figure.

    Args:
        rates: One all-case rate per method and budget.
        output_dir: Directory that receives PNG and SVG artifacts.
        docs_output: MkDocs-local SVG receiving the same figure.
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
    figure.suptitle(
        "What does each Trimwise Hybrid component contribute?", fontsize=17, fontweight="bold"
    )
    axis.set_title(
        "Normalized contiguous required-span containment · 160 position-controlled cases",
        fontsize=10.5,
        color="#475569",
        pad=12,
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    figure.subplots_adjust(bottom=0.26, top=0.86, left=0.1, right=0.98)
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"ablation_vs_hybrid.{suffix}", dpi=180)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(docs_output)
    plt.close(figure)


def _rate_table(rates: pd.DataFrame) -> str:
    """Build the strict all-budget HTML rate table.

    Args:
        rates: All-case strict pass rates.

    Returns:
        Escaped HTML table for each configuration and budget.
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
        f"<thead><tr><th>Configuration</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _interval_table(paired: pd.DataFrame) -> str:
    """Build the primary-metric paired difference table.

    Args:
        paired: Paired-bootstrap CSV for Hybrid against each ablation.

    Returns:
        Escaped HTML table of full-Hybrid minus ablation differences.
    """
    primary = paired.loc[paired["metric"].eq("normalized_contiguous_case_pass")]
    labels = {method_id: label for method_id, label, _ in METHODS}
    body = []
    for comparator in labels:
        if comparator == "trimwise_hybrid":
            continue
        rows = primary.loc[primary["comparator"].eq(comparator)].set_index("budget")
        cells = "".join(
            "<td>"
            f"{rows.loc[budget, 'difference_percentage_points']:+.1f} "
            f"[{rows.loc[budget, 'ci_95_lower_percentage_points']:+.1f}, "
            f"{rows.loc[budget, 'ci_95_upper_percentage_points']:+.1f}]"
            "</td>"
            for budget in BUDGETS
        )
        body.append(f"<tr><th>Hybrid - {escape(labels[comparator])}</th>{cells}</tr>")
    header = "".join(f"<th>{budget:,}</th>" for budget in BUDGETS)
    return (
        '<table style="border-collapse:collapse;width:100%;margin:20px 0">'
        f"<thead><tr><th>Paired difference (percentage points)</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _report_page(rates: pd.DataFrame, paired: pd.DataFrame, figure_path: str) -> str:
    """Build the shared standalone component-study report page.

    Args:
        rates: Strict all-case rates.
        paired: Paired-bootstrap results.
        figure_path: Relative SVG path for the component-study figure.

    Returns:
        Complete report HTML.
    """
    rate_table = _rate_table(rates)
    interval_table = _interval_table(paired)
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trimwise Hybrid component study</title>
<body style="font:18px/1.5 system-ui,sans-serif;max-width:1180px;margin:6vh auto;padding:24px;color:#172033">
<p style="color:#526177;font-weight:700">QUERY-AWARE COMPONENT STUDY</p>
<h1>What does each Hybrid component contribute?</h1>
<p>Each variant uses the same 160 sources, questions, budgets, encoder, lexical--semantic fusion,
source-order reconstruction, strict source-span measure, and thermal-gated runtime. It changes exactly one mechanism:
MMR becomes relevance-only, the adaptive evidence cutoff is removed, or Markdown-aware source
segments become exact non-overlapping 128-token windows. No downstream LLM calls are included.</p>
<img src="{figure_path}" alt="Strict case pass versus budget for Trimwise Hybrid and three component variants." style="width:100%;height:auto">
{rate_table}
<h2>Paired strict differences</h2>
<p>Values are complete Hybrid minus the variant in percentage points; brackets are fixed-suite
95% paired bootstrap intervals. The 128-token results support the adaptive cutoff and structural
segments on this suite. MMR has no consistent observed effect, and the higher-budget intervals do
not distinguish the cutoff or fixed-window variants. This exploratory study does not establish
universal component effects.</p>
{interval_table}
</body></html>"""


def _write_report(
    rates: pd.DataFrame, paired: pd.DataFrame, output_dir: Path, docs_report_output: Path
) -> None:
    """Write standalone and documentation-hosted component-study reports.

    Args:
        rates: Strict all-case rates.
        paired: Paired-bootstrap results.
        output_dir: Directory that receives the standalone HTML report.
        docs_report_output: MkDocs-local version of the standalone report.
    """
    output_dir.joinpath("index.html").write_text(
        _report_page(rates, paired, "ablation_vs_hybrid.svg"),
        encoding="utf-8",
    )
    docs_report_output.parent.mkdir(parents=True, exist_ok=True)
    docs_report_output.write_text(
        _report_page(rates, paired, "../assets/benchmark/ablation_vs_hybrid.svg"),
        encoding="utf-8",
    )


def main() -> None:
    """Render strict ablation figures and a standalone HTML report."""
    arguments = _parse_arguments()
    summary = pd.read_csv(arguments.input)
    paired = pd.read_csv(arguments.paired_input)
    rates = _all_case_rates(summary)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _render_figure(rates, arguments.output_dir, arguments.docs_output)
    _write_report(rates, paired, arguments.output_dir, arguments.docs_report_output)


if __name__ == "__main__":
    main()
