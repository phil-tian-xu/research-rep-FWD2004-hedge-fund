"""Analyze locally downloaded TASS data for HW4."""

import argparse

import pandas as pd

from core.factors import load_factors
from core.regressions import estimate_alphas, test_overall_alpha_change
from core.report import write_report
from core.tass import analyze_tass
from utils.path_info import data_path, results_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptive-only", action="store_true",
        help="Produce TASS sample summaries and indexes before factor inputs are available.",
    )
    args = parser.parse_args()
    raw_dir = data_path / "raw"

    factors = None
    if not args.descriptive_only:
        print("Reading local factor inputs ...", flush=True)
        factors = load_factors(raw_dir)
        print(f"Equity inputs: {factors.attrs['equity_source']}", flush=True)
    print("Preparing the TASS sample and monthly equal-weighted indexes ...", flush=True)
    index_returns, tables, sample, sensitivity_returns = analyze_tass(raw_dir)

    alphas = None
    if factors is not None:
        print("Estimating full-sample and subperiod alphas ...", flush=True)
        alphas = estimate_alphas(index_returns, factors)
        changes = [test_overall_alpha_change(index_returns["Overall EW"], factors, "Baseline")]
        sensitivity = None
        if not sensitivity_returns.empty:
            sensitivity = estimate_alphas(sensitivity_returns, factors)
            changes.append(test_overall_alpha_change(
                sensitivity_returns["Overall EW"], factors, "April 1999 observation excluded",
            ))
        alpha_changes = pd.DataFrame(changes)

    results_path.mkdir(parents=True, exist_ok=True)
    index_returns.to_csv(results_path / "strategy_indexes_monthly.csv", index_label="month")
    for name, table in tables.items():
        table.to_csv(results_path / f"{name}.csv", index=False)

    if alphas is None:
        report_path = results_path / "descriptive_report.html"
        write_report(report_path, tables, sample, None, None)
    else:
        factors.to_csv(results_path / "factors_monthly.csv", index_label="month")
        alphas.to_csv(results_path / "alpha_estimates.csv", index=False)
        alpha_changes.to_csv(results_path / "alpha_change_tests.csv", index=False)
        if sensitivity is not None:
            sensitivity_returns.to_csv(results_path / "strategy_indexes_sensitivity.csv", index_label="month")
            sensitivity.to_csv(results_path / "alpha_sensitivity.csv", index=False)
        report_path = results_path / "report.html"
        write_report(
            report_path, tables, sample, alphas, factors.attrs["factor_spec"],
            sensitivity=sensitivity, alpha_changes=alpha_changes,
        )
    print(f"Saved {report_path} and CSV tables in {results_path}", flush=True)


if __name__ == "__main__":
    main()
