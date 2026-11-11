"""Write a standalone HW4 HTML report from the saved analysis results."""

from numbers import Integral, Real
from pathlib import Path

import mistune
import pandas as pd
from mistune.toc import add_toc_hook, render_toc_ul


PERIODS = ("1994–2022", "1994–2003", "2004–2013", "2014–2022")
OVERALL_INDEX = "Overall EW"
TITLE = "HW4 — TASS hedge funds and Fung–Hsieh benchmarks"

REPORT_CSS = """
:root { color-scheme: light; --ink: #1b2530; --navy: #173553; --rule: #dfe6ec; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: #f7f9fb; font: 16px/1.55 system-ui, sans-serif; }
main { max-width: 1240px; margin: auto; padding: 36px 28px 60px; }
h1, h2 { color: var(--navy); line-height: 1.22; }
h1 { font: bold clamp(1.9rem, 4vw, 2.5rem)/1.18 Georgia, serif; margin: 0 0 1.2rem; }
h2 { margin: 2.25rem 0 0.9rem; padding-top: 1rem; border-top: 1px solid #becbd7; }
p, main > ul { max-width: 95ch; }
li + li { margin-top: 0.35rem; }
a { color: #145987; text-underline-offset: 0.15em; }
nav { border-block: 1px solid var(--rule); padding-block: 0.75rem; }
nav ul { display: flex; flex-wrap: wrap; gap: 0.45rem 1.2rem; margin: 0; padding: 0; list-style: none; }
nav li + li { margin: 0; }
nav a { font-size: 0.88rem; font-weight: 600; }
code { font-size: 0.9em; background: #e8f0f6; padding: 0.08em 0.25em; }
.table-scroll { overflow-x: auto; margin: 1rem 0 1.35rem; background: #fff; }
.table-scroll:focus-visible { outline: 2px solid #145987; outline-offset: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
th, td { padding: 0.5rem 0.65rem; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
th { color: var(--navy); background: #e8f0f6; }
td:first-child { font-weight: 500; }
tbody tr:nth-child(even) { background: #f4f7fa; }
.footnotes { font-size: 0.88rem; }
@media (max-width: 650px) {
    main { padding: 24px 16px 40px; }
    table { min-width: max-content; }
    td { max-width: 18rem; }
}
@media print {
    body { background: #fff; font-size: 10pt; }
    main { padding: 0; }
    nav { display: none; }
    h2 { break-after: avoid; }
    tr { break-inside: avoid; }
    .table-scroll { overflow: visible; }
    table { min-width: 0; font-size: 8pt; }
    th, td { padding: 0.25rem 0.3rem; }
}
"""

TABLES = (
    ("tass_tables", "TASS tables and joins"),
    ("sample_selection", "Sample selection"),
    ("strategy_summary", "Fund characteristics by strategy"),
    ("fund_characteristics", "Fund characteristics across strategies"),
    ("liquidity_terms", "Reported liquidity frequencies"),
    ("index_summary", "Monthly return indexes"),
)

COLUMN_LABELS = {
    "alpha_annualized": "Annualized alpha",
    "alpha_change": "Late minus early (pp/month)",
    "alpha_monthly": "Monthly alpha",
    "baseline_alpha": "Baseline alpha",
    "baseline_p": "Baseline p",
    "change_ci": "95% CI (pp/month)",
    "contents": "Contents",
    "fund_months": "Product-months",
    "hac_p": "HAC p",
    "hac_t": "HAC t",
    "index": "Index",
    "link": "Linking fields",
    "mean_incentive_fee_pct": "Mean incentive fee (%)",
    "mean_management_fee_pct": "Mean management fee (%)",
    "mean_products": "Mean products per month",
    "mean_return_pct": "Mean return (%)",
    "median_assets_usd_millions": "Median assets (USD millions)",
    "median_lockup_months": "Median lockup (months)",
    "median_notice_days": "Median notice (days)",
    "monthly_ci": "95% CI (monthly)",
    "p25": "25th percentile",
    "p75": "75th percentile",
    "r_squared": "R²",
    "revised_alpha": "Revised alpha",
    "revised_p": "Revised p",
    "share_pct": "Share (%)",
    "std": "Std. dev.",
    "volatility_pct": "Return std. dev. (%)",
}


def _markdown_cell(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, Integral):
        return f"{value:,}"
    if isinstance(value, Real):
        return f"{value:,.2f}"
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def _markdown_table(frame):
    headings = [
        COLUMN_LABELS.get(column, column.replace("_", " ").capitalize())
        for column in frame.columns
    ]
    rows = [headings, ["---"] * len(headings)]
    rows.extend(frame.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(map(_markdown_cell, row)) + " |" for row in rows)


def write_html_report(path: Path, markdown_text: str) -> None:
    """Render report tables, links, and footnotes with embedded styling."""
    renderer = mistune.create_markdown(escape=True, plugins=["table", "footnotes"])
    add_toc_hook(renderer, min_level=2, max_level=2)
    content, state = renderer.parse(markdown_text)
    toc = render_toc_ul(state.env["toc_items"])
    if toc:
        nav = f'<nav aria-label="Report sections">\n{toc}</nav>'
        content = content.replace("</h1>", f"</h1>\n{nav}", 1)
    content = content.replace("<table>", '<div class="table-scroll" tabindex="0"><table>')
    content = content.replace("</table>", "</table></div>")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<main>
{content}</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _p_value(value):
    if pd.isna(value):
        return "—"
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _alpha_pct(value):
    return "—" if pd.isna(value) else f"{value:+.3%}"


def _interval(lower, upper, *, percentage_points=False):
    if pd.isna(lower) or pd.isna(upper):
        return "—"
    if percentage_points:
        return f"[{100 * lower:+.3f}, {100 * upper:+.3f}]"
    return f"[{_alpha_pct(lower)}, {_alpha_pct(upper)}]"


def _alpha_section(regressions, factor_spec, index_summary):
    lines = [
        "## Fung–Hsieh seven-factor model",
        "",
        "For index *i* and month *t*, the model is",
        "",
        "`Rᵢ,ₜ = αᵢ + β₁ SPₜ + β₂ SIZEₜ + β₃ ΔY10ₜ + β₄ Δ(Baa − Y10)ₜ "
        "+ β₅ PTFSBDₜ + β₆ PTFSFXₜ + β₇ PTFSCOMₜ + εᵢ,ₜ`",
        "",
        "The dependent variable is the raw monthly hedge fund index return. Returns and yield changes enter as "
        "decimals. OLS re-estimates the intercept and seven factor loadings in each window using the months "
        "available for that index. Two-sided t-tests of zero alpha and 95% intervals use Newey–West HAC "
        "standard errors with three observation lags and a small-sample correction.",
        "",
    ]
    if factor_spec:
        lines.extend([factor_spec, ""])
        if "updated specification" in factor_spec or "Russell 2000 minus S&P 500" in factor_spec:
            lines.extend([
                "The original Table I uses the Wilshire 1750 Small Cap minus Wilshire 750 Large Cap "
                "total-return spread.",
                "",
            ])

    by_index_and_period = {(row.index, row.sample): row for row in regressions.itertuples(index=False)}
    overall_rows = []
    for period in PERIODS:
        estimate = by_index_and_period.get((OVERALL_INDEX, period))
        if estimate is None:
            continue
        overall_rows.append([
            period,
            _alpha_pct(estimate.alpha),
            _alpha_pct(estimate.alpha_annualized),
            estimate.t_stat,
            _p_value(estimate.p_value),
            _interval(estimate.ci_lower, estimate.ci_upper),
            estimate.nobs,
            "—" if pd.isna(estimate.r_squared) else f"{estimate.r_squared:.3f}",
        ])
    lines.extend([
        "## Overall equal-weighted alpha",
        "",
        "Annualized alpha is 12 times monthly alpha. The confidence interval is for monthly alpha.",
        "",
        _markdown_table(pd.DataFrame(overall_rows, columns=[
            "sample", "alpha_monthly", "alpha_annualized", "hac_t", "hac_p", "monthly_ci", "months", "r_squared",
        ])),
        "",
    ])

    first, middle, last = [by_index_and_period.get((OVERALL_INDEX, period)) for period in PERIODS[1:]]
    if all(row is not None and pd.notna(row.alpha) for row in (first, middle, last)):
        change_pp = 100 * (last.alpha - first.alpha)
        if first.alpha <= middle.alpha <= last.alpha and change_pp > 0:
            movement = "rose across the subperiods"
        elif first.alpha >= middle.alpha >= last.alpha and change_pp < 0:
            movement = "fell across the subperiods"
        elif first.alpha == middle.alpha == last.alpha:
            movement = "was the same in each subperiod"
        else:
            movement = "reversed direction across the subperiods"
        significant = [
            row.sample for row in (first, middle, last)
            if pd.notna(row.p_value) and row.p_value < 0.05
        ]
        inference = ""
        if significant:
            inference = f" Overall alpha was significant at the 5% level in {', '.join(significant)}."
        elif all(pd.notna(row.p_value) for row in (first, middle, last)):
            inference = " Each overall subperiod p-value was at least 0.05."
        lines.extend([
            f"Overall equal-weighted alpha {movement}. The final-minus-initial change was "
            f"{change_pp:+.2f} percentage points per month.{inference}",
            "",
        ])

    indexes = regressions.loc[regressions["index"].ne(OVERALL_INDEX), "index"].drop_duplicates().tolist()
    strategy_rows = []
    changes = []
    for index in indexes:
        cells = [index]
        for period in PERIODS:
            estimate = by_index_and_period.get((index, period))
            if estimate is None:
                cells.append("—")
            elif pd.isna(estimate.alpha):
                cells.append(f"— (n={estimate.nobs:,})")
            else:
                p = _p_value(estimate.p_value)
                marker = "†" if pd.notna(estimate.p_value) and estimate.p_value < 0.05 else ""
                if p == "—":
                    p_label = "p unavailable"
                else:
                    p_label = f"p{p}" if p.startswith("<") else f"p={p}"
                cells.append(f"{_alpha_pct(estimate.alpha)}{marker} ({p_label}, n={estimate.nobs:,})")
        strategy_rows.append(cells)
        early = by_index_and_period.get((index, PERIODS[1]))
        late = by_index_and_period.get((index, PERIODS[3]))
        if early is not None and late is not None and pd.notna(early.alpha) and pd.notna(late.alpha):
            changes.append((index, late.alpha - early.alpha))
    lines.extend([
        "## Strategy alphas across periods",
        "",
        "Cells show monthly alpha, its two-sided p-value, and the number of return months. † marks "
        "p < 0.05 using the unrounded p-value. The [detailed estimates](alpha_estimates.csv) include "
        "factor loadings, t-statistics, and confidence intervals.",
        "",
        _markdown_table(pd.DataFrame(strategy_rows, columns=["strategy", *PERIODS])),
        "",
    ])
    if changes:
        declines = sum(change < 0 for _, change in changes)
        increases = sum(change > 0 for _, change in changes)
        largest = min(changes, key=lambda item: item[1])
        largest_decline = (
            f" The largest decline was {abs(100 * largest[1]):.3f} percentage points per month for {largest[0]}."
            if largest[1] < 0 else ""
        )
        lines.extend([
            f"Across the {len(changes)} strategies with estimates in both the first and last subperiods, "
            f"{declines} had lower alpha and {increases} had higher alpha in the last subperiod.{largest_decline} "
            "These comparisons use point estimates. The strategy p-values test each period's alpha against zero.",
            "",
        ])
    significant = regressions.loc[regressions["index"].ne(OVERALL_INDEX) & regressions["p_value"].lt(0.05)]
    significant_counts = significant["sample"].value_counts().reindex(PERIODS[1:], fill_value=0)
    significance_summary = ", ".join(f"{count} in {period}" for period, count in significant_counts.items())
    lines.extend([f"The number of strategies with p < 0.05 was {significance_summary}.", ""])

    strategy_subperiods = regressions.loc[
        regressions["index"].ne(OVERALL_INDEX) & regressions["sample"].isin(PERIODS[1:])
    ]
    strategy_products = index_summary.loc[index_summary["index"].ne(OVERALL_INDEX)].dropna(subset=["mean_products"])
    if not strategy_subperiods.empty and not strategy_products.empty:
        fewest_months = strategy_subperiods.nsmallest(1, "nobs").iloc[0]
        fewest_products = strategy_products.nsmallest(1, "mean_products").iloc[0]
        lines.extend([
            f"Thin coverage limits precision for some strategies. {fewest_months['index']} contributes "
            f"{fewest_months['nobs']:,} months in {fewest_months['sample']}. "
            f"{fewest_products['index']} averages {fewest_products['mean_products']:.1f} eligible products "
            "per contributing month over the full sample.",
            "",
        ])
    return lines


def _sensitivity_section(regressions, sensitivity, details):
    lines = ["## Sensitivity to a single product-month", ""]
    if details.get("excluded_count") == 1:
        month = pd.Period(details["month"], freq="M").strftime("%B %Y")
        lines.extend([
            f"The sensitivity check omits the {details['excluded_return_pct']:+,.2f}% reported return for TASS "
            f"product {details['productreference']} in {month}. The baseline retains this return. "
            f"The revised indexes retain {month} and recompute its equal-weighted returns from the remaining products.",
            "",
        ])
        for index in details["affected_indexes"]:
            original = details["baseline_returns_pct"].get(index)
            revised = details["sensitivity_returns_pct"].get(index)
            if pd.notna(original) and pd.notna(revised):
                remaining = details["remaining_products"][index]
                lines.extend([
                    f"The {index} return for {month} changes from {original:+.3f}% to {revised:+.3f}% "
                    f"with {remaining:,} products in the revised index.",
                    "",
                ])

    baseline = regressions.set_index(["index", "sample"])
    revised = sensitivity.set_index(["index", "sample"])
    rows = []
    for (index, period), estimate in revised.iterrows():
        original = baseline.loc[(index, period)]
        rows.append([
            index,
            period,
            _alpha_pct(original["alpha"]),
            _p_value(original["p_value"]),
            _alpha_pct(estimate["alpha"]),
            _p_value(estimate["p_value"]),
        ])
    lines.extend([
        "Alphas are monthly. The same seven factors and HAC inference apply to both sets of estimates. "
        "The [sensitivity estimates](alpha_sensitivity.csv) include the re-estimated factor loadings and intervals.",
        "",
        _markdown_table(pd.DataFrame(rows, columns=[
            "index", "sample", "baseline_alpha", "baseline_p", "revised_alpha", "revised_p",
        ])),
        "",
    ])

    for index in sensitivity["index"].drop_duplicates():
        early, late = (index, PERIODS[1]), (index, PERIODS[3])
        if early not in revised.index or late not in revised.index:
            continue
        original_change = baseline.loc[late, "alpha"] - baseline.loc[early, "alpha"]
        revised_change = revised.loc[late, "alpha"] - revised.loc[early, "alpha"]
        if pd.notna(original_change) and pd.notna(revised_change):
            lines.extend([
                f"For {index}, the late-minus-early alpha change is {100 * original_change:+.3f} percentage "
                f"points per month in the baseline and {100 * revised_change:+.3f} after the omission.",
                "",
            ])
    return lines


def _alpha_change_section(alpha_changes):
    rows = [
        [
            row.specification,
            "—" if pd.isna(row.alpha_change) else f"{100 * row.alpha_change:+.3f}",
            row.t_stat,
            _p_value(row.p_value),
            _interval(row.ci_lower, row.ci_upper, percentage_points=True),
            row.nobs,
        ]
        for row in alpha_changes.itertuples(index=False)
    ]
    lines = [
        "## Test of the overall alpha change",
        "",
        "The pooled model lets the intercept and all seven slopes vary by subperiod. A HAC contrast "
        "tests equality of the 2014–2022 and 1994–2003 intercepts directly. The contrast subtracts the "
        "early alpha from the late alpha and is reported in percentage points per month. The HAC settings "
        "match the individual regressions.",
        "",
        _markdown_table(pd.DataFrame(rows, columns=[
            "specification", "alpha_change", "hac_t", "hac_p", "change_ci", "months",
        ])),
        "",
        "The [contrast estimates](alpha_change_tests.csv) give the unrounded results.",
        "",
    ]
    available = alpha_changes.dropna(subset=["p_value"])
    for row in available.itertuples(index=False):
        label = "For the baseline" if row.specification == "Baseline" else "After the omission"
        conclusion = "excludes" if row.p_value < 0.05 else "includes"
        lines.extend([f"{label}, the 95% confidence interval {conclusion} zero.", ""])
    if len(available) == 2:
        baseline, revised = available.itertuples(index=False)
        if (
            abs(revised.alpha_change) < abs(baseline.alpha_change)
            and revised.ci_upper - revised.ci_lower < baseline.ci_upper - baseline.ci_lower
        ):
            lines.extend([
                "Omitting the product-month reduces the magnitude of the estimated change and narrows its confidence interval.",
                "",
            ])
    return lines


def write_report(
    path: Path,
    tables: dict[str, pd.DataFrame],
    sample: dict,
    regressions: pd.DataFrame | None = None,
    factor_spec: str | None = None,
    *,
    sensitivity: pd.DataFrame | None = None,
    alpha_changes: pd.DataFrame | None = None,
) -> None:
    """Write an HTML report, with an optional factor regression section."""
    lines = [
        f"# {TITLE}",
        "",
        "The sample window is January 1994 through December 2022. Product tables link through "
        "`productreference`. Supplied benchmark series link separately through `indexreference`. "
        "The overall and strategy equal-weighted indexes are formed from product returns. "
        "[WRDS TASS data dictionary](https://wrds-www.wharton.upenn.edu/data-dictionary/tr_tass/)",
        "",
    ]
    for name, title in TABLES:
        lines.extend([f"## {title}", "", _markdown_table(tables[name]), ""])
        if name == "sample_selection" and sample.get("notes"):
            lines.extend([*(f"- {note}" for note in sample["notes"]), ""])

    if sample.get("findings"):
        lines.extend(["## Descriptive findings", "", *(f"- {finding}" for finding in sample["findings"]), ""])
    if regressions is not None:
        lines.extend(_alpha_section(regressions, factor_spec, tables["index_summary"]))
        if sensitivity is not None and not sensitivity.empty:
            lines.extend(_sensitivity_section(regressions, sensitivity, sample.get("sensitivity", {})))
        if alpha_changes is not None and not alpha_changes.empty:
            lines.extend(_alpha_change_section(alpha_changes))
        lines.extend([
            "The seven-factor model is from Fung and Hsieh (2004), *Hedge Fund Benchmarks: A Risk-Based Approach*. "
            "The three trend factors come from Fung and Hsieh (2001), *The Risk in Hedge Fund Strategies: "
            "Theory and Evidence from Trend Followers*, Review of Financial Studies 14, 313–341.[^trend] "
            "[Hsieh's factor library](https://people.duke.edu/~dah7/HFRFData.htm)",
            "",
            "[^trend]: [Trend-following factor workbook](https://people.duke.edu/~dah7/DataLibrary/TF-Fac.xls)",
            "",
        ])

    write_html_report(path, "\n".join(lines))
