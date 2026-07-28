"""Render the v1.2 strict source-span-survival figure from frozen results."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METHODS = (
    ("trimwise_lexical", "Trimwise Lexical", "#155eef"),
    ("trimwise_hybrid", "Trimwise Hybrid", "#0f766e"),
    ("recomp_extractive", "RECOMP NQ extractive\nsentence adapter", "#9a3412"),
    ("llmlingua", "LLMLingua GPT-2\ntoken-pruning adapter", "#b45309"),
    ("longllmlingua", "LongLLMLingua GPT-2\nsingle-context adapter", "#c2410c"),
)
METRIC = "normalized_contiguous_case_pass_rate"


def _parse_arguments() -> argparse.Namespace:
    """Return command-line arguments for the static figure renderer.

    Returns:
        Parsed paths for the frozen summary and rendered figure outputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/position_controlled_160_evidence_sensitivity_v1_2_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/evidence-sensitivity-v1-2"),
    )
    parser.add_argument(
        "--readme-output",
        type=Path,
        default=Path("../assets/readme/query-aware-benchmark.svg"),
    )
    parser.add_argument(
        "--docs-output",
        type=Path,
        default=Path("../docs/assets/benchmark/normalized_contiguous_vs_budget.png"),
    )
    parser.add_argument(
        "--docs-report-output",
        type=Path,
        default=Path("../docs/benchmark-report/index.html"),
    )
    return parser.parse_args()


def _all_case_rates(summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate all-case strict pass rates for each query-aware method.

    Args:
        summary: The frozen aggregate CSV, grouped by track and evidence position.

    Returns:
        One row per evaluated method and token budget with a weighted strict rate.
    """
    method_ids = [method_id for method_id, _, _ in METHODS]
    rows = summary.loc[
        summary["query_aware"].eq(True) & summary["method_id"].isin(method_ids),
        ["method_id", "budget", "rows", METRIC],
    ].copy()
    rows["weighted_rate"] = rows["rows"] * rows[METRIC]
    grouped = rows.groupby(["method_id", "budget"], as_index=False).agg(
        rows=("rows", "sum"),
        weighted_rate=("weighted_rate", "sum"),
    )
    grouped[METRIC] = grouped["weighted_rate"].div(grouped["rows"])
    return grouped


def render_figure(
    summary_path: Path,
    output_dir: Path,
    readme_output: Path,
    docs_output: Path,
    docs_report_output: Path,
) -> None:
    """Render the primary strict metric as PNG and SVG files.

    Args:
        summary_path: Frozen v1.2 aggregate CSV.
        output_dir: Directory receiving the report PNG and SVG.
        readme_output: Repository README SVG path.
        docs_output: MkDocs-local PNG path.
        docs_report_output: MkDocs-local detailed report path.

    Raises:
        ValueError: If an evaluated method is missing a tested budget.
    """
    rates = _all_case_rates(pd.read_csv(summary_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    docs_report_output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    fig, axis = plt.subplots(figsize=(12, 7.5))
    budgets = (128, 256, 512, 1024)
    for method_id, label, color in METHODS:
        method_rates = rates.loc[rates["method_id"].eq(method_id)].set_index("budget")
        if set(method_rates.index) != set(budgets):
            raise ValueError(f"missing v1.2 rates for {method_id}")
        axis.plot(
            budgets,
            [method_rates.loc[budget, METRIC] * 100 for budget in budgets],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, [f"{budget:,}" for budget in budgets])
    axis.set_ylim(0, 80)
    axis.set_xlabel("Output-token budget")
    axis.set_ylabel("Cases passing (%)")
    axis.grid(axis="y", color="#cbd5e1", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Complete source-span survival under a fixed budget",
        fontsize=17,
        fontweight="bold",
    )
    axis.set_title(
        "Normalized contiguous required-span containment · 160 position-controlled cases",
        fontsize=10.5,
        color="#475569",
        pad=12,
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.5,
    )
    fig.subplots_adjust(bottom=0.28, top=0.86, left=0.1, right=0.98)
    docs_svg_output = docs_output.with_suffix(".svg")
    for output_path in (
        output_dir / "normalized_contiguous_vs_budget.png",
        output_dir / "normalized_contiguous_vs_budget.svg",
        readme_output,
        docs_output,
        docs_svg_output,
    ):
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        if output_path.suffix == ".svg":
            _strip_svg_trailing_whitespace(output_path)
    plt.close(fig)
    _write_report(rates, output_dir, docs_report_output)


def _strip_svg_trailing_whitespace(path: Path) -> None:
    """Remove renderer whitespace that would otherwise create noisy diffs.

    Args:
        path: SVG generated by Matplotlib.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _write_report(
    rates: pd.DataFrame,
    output_dir: Path,
    docs_report_output: Path,
) -> None:
    """Write matching standalone and documentation-hosted strict reports.

    Args:
        rates: All-case strict rates keyed by method and budget.
        output_dir: Directory receiving the static report.
        docs_report_output: MkDocs-local detailed report path.
    """
    budgets = (128, 256, 512, 1024)
    rows = []
    for method_id, label, _ in METHODS:
        method_rates = rates.loc[rates["method_id"].eq(method_id)].set_index("budget")
        cells = "".join(
            f'<td style="padding:10px 12px;border-bottom:1px solid #d8dee9">'
            f"{method_rates.loc[budget, METRIC] * 100:.1f}%</td>"
            for budget in budgets
        )
        rows.append(
            '<tr><th style="padding:10px 12px;border-bottom:1px solid #d8dee9;text-align:left">'
            f"{escape(label.replace(chr(10), ' '))}</th>{cells}</tr>"
        )
    report_references = (
        'See the <a href="../../data/manifests/evidence_sensitivity_v1_2_protocol.md">'
        "protocol</a>,\n"
        '<a href="../../data/manifests/evidence_sensitivity_v1_2_manifest.json">'
        "frozen manifest</a>, and\n"
        '<a href="../../results/position_controlled_160_evidence_sensitivity_v1_2_summary.csv">\n'
        "complete summary</a>."
    )
    docs_references = (
        'See the <a href="../benchmark/">benchmark documentation</a> for the protocol,\n'
        "frozen manifest, complete summary, and reproduction commands."
    )
    output_dir.joinpath("index.html").write_text(
        _report_page(rows, "normalized_contiguous_vs_budget.svg", report_references),
        encoding="utf-8",
    )
    docs_report_output.write_text(
        _report_page(
            rows, "../assets/benchmark/normalized_contiguous_vs_budget.svg", docs_references
        ),
        encoding="utf-8",
    )


def _report_page(rows: list[str], image_path: str, references: str) -> str:
    """Return the shared strict-report HTML document.

    Args:
        rows: Pre-rendered method result rows.
        image_path: Relative SVG path for the report location.
        references: Footer links appropriate to the report location.

    Returns:
        Complete standalone HTML document.
    """
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trimwise benchmark — v1.2 strict source-span survival</title>
<body style="font:18px/1.5 system-ui,sans-serif;max-width:1100px;margin:6vh auto;
padding:24px;color:#172033">
<p style="color:#526177;font-weight:700">POSITION-CONTROLLED EVALUATION SUITE</p>
<h1>v1.2 strict source-span survival</h1>
<p style="max-width:75ch;color:#526177">
Normalized contiguous required-span containment is the post-hoc primary metric over frozen outputs.
Every annotated source span must survive as one contiguous normalized passage; prohibited content
and budget violations still fail the case.
</p>
<img src="{image_path}"
     alt="Normalized contiguous required-span containment by output-token budget."
     style="width:100%;height:auto">
<table style="border-collapse:collapse;width:100%;margin:28px 0">
<thead><tr><th style="padding:10px 12px;text-align:left">Evaluated method or adapter</th>
<th style="padding:10px 12px">128</th><th style="padding:10px 12px">256</th>
<th style="padding:10px 12px">512</th><th style="padding:10px 12px">1,024</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>
The local ordered 80% and 90% sensitivity checks preserve the same ordering at every budget.
{references}
</p>
</body>
</html>"""


def main() -> None:
    """Run the figure renderer from the benchmark directory."""
    arguments = _parse_arguments()
    render_figure(
        arguments.input,
        arguments.output_dir,
        arguments.readme_output,
        arguments.docs_output,
        arguments.docs_report_output,
    )


if __name__ == "__main__":
    main()
