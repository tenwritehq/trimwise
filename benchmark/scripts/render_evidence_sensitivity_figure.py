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
BUDGETS = (128, 256, 512, 1024)


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
        "--natural-input",
        type=Path,
        default=Path(
            "results/position_controlled_160_evidence_sensitivity_v1_2_natural_only_summary.csv"
        ),
    )
    parser.add_argument(
        "--relocated-input",
        type=Path,
        default=Path(
            "results/position_controlled_160_evidence_sensitivity_v1_2_relocated_only_summary.csv"
        ),
    )
    parser.add_argument(
        "--without-self-input",
        type=Path,
        default=Path(
            "results/position_controlled_160_evidence_sensitivity_v1_2_without_self_sources_summary.csv"
        ),
    )
    parser.add_argument(
        "--paired-input",
        type=Path,
        default=Path("results/position_controlled_160_evidence_sensitivity_v1_2_paired_stats.csv"),
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


def _all_case_rates(summary: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    """Calculate weighted query-aware rates for each evaluated method.

    Args:
        summary: The frozen aggregate CSV, grouped by track and evidence position.
        metric: Aggregate metric column to weight by the number of cases.

    Returns:
        One row per evaluated method and token budget with a weighted strict rate.
    """
    method_ids = [method_id for method_id, _, _ in METHODS]
    rows = summary.loc[
        summary["query_aware"].eq(True) & summary["method_id"].isin(method_ids),
        ["method_id", "budget", "rows", metric],
    ].copy()
    rows["weighted_rate"] = rows["rows"] * rows[metric]
    grouped = rows.groupby(["method_id", "budget"], as_index=False).agg(
        rows=("rows", "sum"),
        weighted_rate=("weighted_rate", "sum"),
    )
    grouped[metric] = grouped["weighted_rate"].div(grouped["rows"])
    return grouped


def render_figure(
    summary_path: Path,
    natural_path: Path,
    relocated_path: Path,
    without_self_path: Path,
    paired_path: Path,
    output_dir: Path,
    readme_output: Path,
    docs_output: Path,
    docs_report_output: Path,
) -> None:
    """Render the primary strict metric as PNG and SVG files.

    Args:
        summary_path: Frozen v1.2 aggregate CSV.
        natural_path: Strict v1.2 aggregate limited to naturally placed cases.
        relocated_path: Strict v1.2 aggregate limited to controlled relocations.
        without_self_path: Strict v1.2 aggregate excluding Trimwise-source cases.
        paired_path: Strict v1.2 paired-bootstrap summary.
        output_dir: Directory receiving the report PNG and SVG.
        readme_output: Repository README SVG path.
        docs_output: MkDocs-local PNG path.
        docs_report_output: MkDocs-local detailed report path.

    Raises:
        ValueError: If an evaluated method is missing a tested budget.
    """
    summary = pd.read_csv(summary_path)
    natural_summary = pd.read_csv(natural_path)
    relocated_summary = pd.read_csv(relocated_path)
    without_self_summary = pd.read_csv(without_self_path)
    paired = pd.read_csv(paired_path)
    rates = _all_case_rates(summary)
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
    _render_cohort_figure(
        natural_summary,
        relocated_summary,
        output_dir,
        docs_output.with_name("natural_vs_relocated.png"),
    )
    _write_report(
        summary,
        rates,
        natural_summary,
        relocated_summary,
        without_self_summary,
        paired,
        output_dir,
        docs_report_output,
    )


def _strip_svg_trailing_whitespace(path: Path) -> None:
    """Remove renderer whitespace that would otherwise create noisy diffs.

    Args:
        path: SVG generated by Matplotlib.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _render_cohort_figure(
    natural_summary: pd.DataFrame,
    relocated_summary: pd.DataFrame,
    output_dir: Path,
    docs_output: Path,
) -> None:
    """Render the natural-placement and relocation cohorts without pooling them.

    Args:
        natural_summary: Strict aggregate for the 135 naturally placed cases.
        relocated_summary: Strict aggregate for the 25 controlled relocations.
        output_dir: Directory receiving standalone report figures.
        docs_output: MkDocs-local PNG destination.

    Raises:
        ValueError: If a cohort lacks one evaluated method at a tested budget.
    """
    cohorts = (
        (natural_summary, "Natural placement\n135 cases; end position is unbalanced"),
        (relocated_summary, "Controlled relocation\n25 end-position construction cases"),
    )
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(2, 1, figsize=(11, 9.5), sharex=True, sharey=True)
    for axis, (summary, title) in zip(axes, cohorts, strict=True):
        rates = _all_case_rates(summary)
        for method_id, label, color in METHODS:
            method_rates = rates.loc[rates["method_id"].eq(method_id)].set_index("budget")
            if set(method_rates.index) != set(BUDGETS):
                raise ValueError(f"missing v1.2 cohort rates for {method_id}")
            axis.plot(
                BUDGETS,
                [method_rates.loc[budget, METRIC] * 100 for budget in BUDGETS],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=color,
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, [f"{budget:,}" for budget in BUDGETS])
        axis.set_ylim(0, 100)
        axis.set_title(title, fontsize=11, pad=12)
        axis.grid(axis="y", color="#cbd5e1", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Output-token budget")
    axes[0].set_ylabel("Feasible strict case pass (%)")
    figure.suptitle(
        "Natural placement and controlled relocation are reported separately",
        fontsize=16,
        fontweight="bold",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    figure.subplots_adjust(bottom=0.2, top=0.88, left=0.1, right=0.98, hspace=0.42)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    for destination in (
        output_dir / "natural_vs_relocated.png",
        output_dir / "natural_vs_relocated.svg",
        docs_output,
        docs_output.with_suffix(".svg"),
    ):
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        if destination.suffix == ".svg":
            _strip_svg_trailing_whitespace(destination)
    plt.close(figure)


def _group_rates(summary: pd.DataFrame, group: str, metric: str = METRIC) -> pd.DataFrame:
    """Calculate weighted strict rates for one benchmark grouping.

    Args:
        summary: Aggregate CSV grouped by track and evidence position.
        group: Existing aggregate column used to split the rates.
        metric: Aggregate metric column to weight by the number of cases.

    Returns:
        One weighted row per group, method, and tested budget.
    """
    method_ids = [method_id for method_id, _, _ in METHODS]
    rows = summary.loc[
        summary["query_aware"].eq(True) & summary["method_id"].isin(method_ids),
        ["method_id", "budget", group, "rows", metric],
    ].copy()
    rows["weighted_rate"] = rows["rows"] * rows[metric]
    grouped = rows.groupby([group, "method_id", "budget"], as_index=False).agg(
        rows=("rows", "sum"),
        weighted_rate=("weighted_rate", "sum"),
    )
    grouped[metric] = grouped["weighted_rate"].div(grouped["rows"])
    return grouped


def _method_label(method_id: str) -> str:
    """Return the report label for one evaluated method identifier.

    Args:
        method_id: Stable benchmark method identifier.

    Returns:
        Human-readable method or adapter label.
    """
    return next(label.replace("\n", " ") for key, label, _ in METHODS if key == method_id)


def _value(rates: pd.DataFrame, method_id: str, budget: int, metric: str = METRIC) -> float:
    """Return one weighted result value after checking its expected identity.

    Args:
        rates: Weighted rates indexed by method and budget columns.
        method_id: Evaluated method identifier.
        budget: Token budget to select.
        metric: Result column to retrieve.

    Returns:
        The selected aggregate rate.

    Raises:
        ValueError: If the requested rate is missing or ambiguous.
    """
    selected = rates.loc[rates["method_id"].eq(method_id) & rates["budget"].eq(budget), metric]
    if len(selected) != 1:
        raise ValueError(f"missing or ambiguous rate for {method_id} at {budget}")
    return float(selected.iloc[0])


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return one compact escaped report table.

    Args:
        headers: Table column labels.
        rows: Plain-text cell values in display order.

    Returns:
        HTML table markup using the report's shared inline styling.
    """
    header_cells = "".join(
        f'<th style="padding:10px 12px;text-align:left">{escape(header)}</th>' for header in headers
    )
    body_rows = "".join(
        "<tr>"
        + "".join(
            f'<td style="padding:10px 12px;border-bottom:1px solid #d8dee9">{escape(cell)}</td>'
            for cell in row
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<table style="border-collapse:collapse;width:100%;margin:20px 0;font-size:0.92em">'
        f"<thead><tr>{header_cells}</tr></thead><tbody>{body_rows}</tbody></table>"
    )


def _rate_rows(rates: pd.DataFrame) -> list[list[str]]:
    """Build all-budget primary-rate table rows.

    Args:
        rates: All-case primary-metric rates.

    Returns:
        Method-labelled percentage rows in stable display order.
    """
    return [
        [
            _method_label(method_id),
            *[f"{_value(rates, method_id, budget) * 100:.1f}%" for budget in BUDGETS],
        ]
        for method_id, _, _ in METHODS
    ]


def _single_budget_rows(
    rates: pd.DataFrame, budget: int, group: str, values: tuple[str, ...]
) -> list[list[str]]:
    """Build a stable method-by-group percentage table at one token budget.

    Args:
        rates: Grouped rates from ``_group_rates``.
        budget: Token budget represented by the table.
        group: Grouping column present in ``rates``.
        values: Group values in presentation order.

    Returns:
        One table row per evaluated method.
    """
    rows: list[list[str]] = []
    for method_id, _, _ in METHODS:
        percentages = []
        for value in values:
            selected = rates.loc[
                rates["method_id"].eq(method_id)
                & rates["budget"].eq(budget)
                & rates[group].eq(value),
                METRIC,
            ]
            percentages.append("—" if selected.empty else f"{selected.iloc[0] * 100:.1f}%")
        rows.append([_method_label(method_id), *percentages])
    return rows


def _write_report(
    summary: pd.DataFrame,
    rates: pd.DataFrame,
    natural_summary: pd.DataFrame,
    relocated_summary: pd.DataFrame,
    without_self_summary: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
    docs_report_output: Path,
) -> None:
    """Write matching standalone and documentation-hosted strict reports.

    Args:
        summary: All-case strict aggregate CSV.
        rates: Weighted all-case primary rates.
        natural_summary: Strict aggregate for naturally placed cases.
        relocated_summary: Strict aggregate for controlled relocations.
        without_self_summary: Strict aggregate excluding Trimwise-source cases.
        paired: Strict v1.2 paired-bootstrap summary.
        output_dir: Directory receiving the static report.
        docs_report_output: MkDocs-local detailed report path.
    """
    evidence_metric = "normalized_contiguous_required_evidence_success"
    violation_metric = "budget_violation_rate"
    local_80_metric = "local_ordered_80_case_pass_rate"
    local_90_metric = "local_ordered_90_case_pass_rate"
    evidence_rates = _all_case_rates(summary, evidence_metric)
    violation_rates = _all_case_rates(summary, violation_metric)
    local_80_rates = _all_case_rates(summary, local_80_metric)
    local_90_rates = _all_case_rates(summary, local_90_metric)
    natural_rates = _all_case_rates(natural_summary)
    relocated_rates = _all_case_rates(relocated_summary)
    without_self_rates = _all_case_rates(without_self_summary)
    feasibility_rows = [
        [
            _method_label(method_id),
            f"{_value(evidence_rates, method_id, 256, evidence_metric) * 100:.1f}%",
            f"{_value(rates, method_id, 256) * 100:.1f}%",
            f"{_value(violation_rates, method_id, 256, violation_metric) * 100:.1f}%",
        ]
        for method_id, _, _ in METHODS
    ]
    local_rows = [
        [
            _method_label(method_id),
            f"{_value(rates, method_id, 256) * 100:.1f}%",
            f"{_value(local_80_rates, method_id, 256, local_80_metric) * 100:.1f}%",
            f"{_value(local_90_rates, method_id, 256, local_90_metric) * 100:.1f}%",
        ]
        for method_id, _, _ in METHODS
    ]
    cohort_rows = [
        [
            _method_label(method_id),
            f"{_value(natural_rates, method_id, 256) * 100:.1f}%",
            f"{_value(relocated_rates, method_id, 256) * 100:.1f}%",
            f"{_value(without_self_rates, method_id, 256) * 100:.1f}%",
        ]
        for method_id, _, _ in METHODS
    ]
    position_rows = _single_budget_rows(
        _group_rates(summary, "evidence_position"),
        256,
        "evidence_position",
        ("beginning", "middle", "end", "multiple"),
    )
    task_rows = _single_budget_rows(
        _group_rates(summary, "track"),
        256,
        "track",
        ("adversarial", "evidence_qa", "instruction", "procedure", "real_source", "structured"),
    )
    paired_rows = []
    paired_subset = paired.loc[
        paired["metric"].eq("normalized_contiguous_case_pass")
        & paired["reference"].eq("trimwise_hybrid")
        & paired["comparator"].eq("recomp_extractive")
    ].sort_values("budget")
    for row in paired_subset.itertuples(index=False):
        paired_rows.append(
            [
                f"{row.budget:,}",
                f"{row.difference_percentage_points:+.1f} pp",
                (
                    f"[{row.ci_95_lower_percentage_points:+.1f}, "
                    f"{row.ci_95_upper_percentage_points:+.1f}] pp"
                ),
            ]
        )
    report_content = {
        "primary_rows": _html_table(
            ["Evaluated method or adapter", *[str(budget) for budget in BUDGETS]],
            _rate_rows(rates),
        ),
        "feasibility_rows": _html_table(
            [
                "Evaluated method or adapter",
                "Span survival",
                "Feasible case pass",
                "Budget violation",
            ],
            feasibility_rows,
        ),
        "local_rows": _html_table(
            ["Evaluated method or adapter", "Contiguous", "Local ordered 80%", "Local ordered 90%"],
            local_rows,
        ),
        "cohort_rows": _html_table(
            [
                "Evaluated method or adapter",
                "Natural n=135",
                "Relocated n=25",
                "Without self-source n=150",
            ],
            cohort_rows,
        ),
        "position_rows": _html_table(
            ["Evaluated method or adapter", "Beginning", "Middle", "End", "Multiple"],
            position_rows,
        ),
        "task_rows": _html_table(
            [
                "Evaluated method or adapter",
                "Adversarial",
                "Evidence QA",
                "Instruction",
                "Procedure",
                "Real source",
                "Structured",
            ],
            task_rows,
        ),
        "paired_rows": _html_table(
            ["Budget", "Hybrid - RECOMP", "Fixed-seed 95% paired bootstrap interval"],
            paired_rows,
        ),
    }
    report_references = (
        'See the <a href="../../data/manifests/'
        'evidence_sensitivity_v1_2_protocol.md">protocol</a>, '
        '<a href="../../data/manifests/'
        'evidence_sensitivity_v1_2_manifest.json">frozen manifest</a>, '
        'and the <a href="../../results/'
        'position_controlled_160_evidence_sensitivity_v1_2_summary.csv">all-case summary</a>. '
        'The <a href="../../results/'
        "position_controlled_160_evidence_sensitivity_v1_2_natural_only_"
        'summary.csv">natural-only</a>, '
        '<a href="../../results/'
        "position_controlled_160_evidence_sensitivity_v1_2_relocated_only_"
        'summary.csv">relocated-only</a>, '
        '<a href="../../results/'
        "position_controlled_160_evidence_sensitivity_v1_2_without_self_sources_"
        'summary.csv">self-source exclusion</a>, '
        'and <a href="../../results/'
        'position_controlled_160_evidence_sensitivity_v1_2_paired_stats.csv">'
        "paired bootstrap</a> CSVs "
        "are released beside it."
    )
    docs_references = (
        'See the <a href="../benchmark/">benchmark documentation</a> for the protocol, '
        "frozen manifest, complete CSVs, and reproduction commands."
    )
    output_dir.joinpath("index.html").write_text(
        _report_page(
            report_content,
            "normalized_contiguous_vs_budget.svg",
            "natural_vs_relocated.svg",
            report_references,
        ),
        encoding="utf-8",
    )
    docs_report_output.write_text(
        _report_page(
            report_content,
            "../assets/benchmark/normalized_contiguous_vs_budget.svg",
            "../assets/benchmark/natural_vs_relocated.svg",
            docs_references,
        ),
        encoding="utf-8",
    )


def _report_page(
    content: dict[str, str], primary_image_path: str, cohort_image_path: str, references: str
) -> str:
    """Return the shared strict-report HTML document.

    Args:
        content: Pre-rendered tables keyed by report section.
        primary_image_path: Relative SVG path for the primary all-case figure.
        cohort_image_path: Relative SVG path for the cohort-comparison figure.
        references: Footer links appropriate to the report location.

    Returns:
        Complete standalone HTML document.
    """
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trimwise benchmark — v1.2 strict source-span survival</title>
<body style="font:18px/1.5 system-ui,sans-serif;max-width:1180px;
margin:6vh auto;padding:24px;color:#172033">
<p style="color:#526177;font-weight:700">POSITION-CONTROLLED EVALUATION SUITE</p>
<h1>v1.2 strict source-span survival</h1>
<p style="max-width:78ch;color:#526177">
Normalized contiguous required-span containment is the post-hoc primary metric over frozen outputs.
Every annotated source span must survive as one contiguous normalized passage; prohibited content
and budget violations still fail the case. No compressor or QA calls were rerun.</p>
<img src="{primary_image_path}"
alt="Normalized contiguous required-span containment by output-token budget."
style="width:100%;height:auto">
{content["primary_rows"]}
<h2>Evidence survival and operational feasibility</h2>
<p>At 256 tokens, span survival ignores prohibited text and budget compliance; feasible case
pass requires all three. This separates retrieval failure from an output that violates the shared
evaluator budget.</p>
{content["feasibility_rows"]}
<h2>Local ordered-retention sensitivities</h2>
<p>These 256-token values are feasible case pass under the predeclared post-hoc metrics. The 80%
and 90% tests require ordered matching within one reference-length output window; complete
contiguous containment remains primary.</p>
{content["local_rows"]}
<h2>Natural, relocated, and self-source sensitivity checks</h2>
<p>The natural-only subset is unbalanced because it contains 15 natural end cases. The 25 relocated
rows are a construction diagnostic, not a causal substitute for natural placement. The self-source
check removes every <code>real-trimwise-*</code> case.</p>
<img src="{cohort_image_path}"
alt="Strict case pass separately for natural placement and controlled relocation cohorts."
style="width:100%;height:auto">
{content["cohort_rows"]}
<h2>Where the strict result succeeds and fails</h2>
<p>All values below are 256-token feasible strict case pass. Evidence positions are balanced in the
160-case primary suite; task counts are not equal.</p>
{content["position_rows"]}
{content["task_rows"]}
<h2>Paired uncertainty: Hybrid versus RECOMP adapter</h2>
<p>Fixed-seed percentile bootstrap intervals over paired case-level differences in the 160-case
frozen suite. They are descriptive uncertainty intervals for the post-hoc v1.2 analysis, not
preregistered hypothesis tests.</p>
{content["paired_rows"]}
<p>The complete local-80%, local-90%, and both Trimwise configuration intervals are in the paired-
bootstrap CSV. {references}</p>
</body>
</html>"""


def main() -> None:
    """Run the figure renderer from the benchmark directory."""
    arguments = _parse_arguments()
    render_figure(
        arguments.input,
        arguments.natural_input,
        arguments.relocated_input,
        arguments.without_self_input,
        arguments.paired_input,
        arguments.output_dir,
        arguments.readme_output,
        arguments.docs_output,
        arguments.docs_report_output,
    )


if __name__ == "__main__":
    main()
