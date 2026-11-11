# HW4 · Fung–Hsieh hedge fund benchmarks

The assignment is in [docs/requirements.md](docs/requirements.md). `main.py` reads the local raw extracts and writes the TASS summaries, monthly equal-weighted indexes, seven-factor estimates, and a brief report under `results/`.

## Raw inputs

- `data/raw/wrds/tass_catalog.csv.gz` and `productdetails.csv.gz`, `productdetailsadddate.csv.gz`, and the completed `productperformance.csv.gz` under `data/raw/wrds/tr_tass/`.
- `data/raw/factors/TF-Fac.xls` and `data/raw/factors/fred_dgs10_dbaa_daily.csv` from the download step.
- `indexdetails.csv.gz` and `indexperformance.csv.gz` under `data/raw/wrds/tr_tass/` supply S&P 500 and Russell 2000 total-return levels for the default run.

To use an equity export, place `datastream_equity_monthly.csv` in `data/raw/factors/`. The script gives that file priority. Use USD month-end total-return index **levels** from December 1993 through December 2022 and ISO dates such as `1993-12-31`.

| Specification | Equity CSV header |
| --- | --- |
| Original Wilshire size spread | `date,sp500_tr,wilshire_1750_tr,wilshire_750_tr` |
| Updated size spread | `date,sp500_tr,russell_2000_tr` |

The default run uses Hsieh’s updated Russell 2000 minus S&P 500 spread. Supply the two Wilshire columns for the original Table I spread. The generated report identifies the source and definition used. December 1993 levels supply the baseline for January 1994. [Hsieh’s factor library](https://people.duke.edu/~dah7/HFRFData.htm)

For the original spread, the Bloomberg total-return series are `W5KSCT Index` and `W5KLCT Index`. Request monthly levels. The provider lists daily history starting in December 1998 and monthly history starting in June 1978. [Wilshire small-cap factsheet](https://assets-global.website-files.com/63e3e50fdce0bcaff7861530/662013b3643978b000e63793_wilshire-us-small-cap-fact-sheet.pdf) · [Wilshire large-cap factsheet](https://assets-global.website-files.com/63e3e50fdce0bcaff7861530/662012cc7b28c29c9f005370_wilshire-us-large-cap-fact-sheet.pdf)

## Run the analysis

From the repository root in PowerShell, run the full assignment with the local inputs above.

```powershell
uv run --no-project --python 3.13 --no-build --with pandas==2.2.3 --with statsmodels==0.14.5 --with xlrd==2.0.2 --with mistune==3.4.0 python main.py
```

Append `--descriptive-only` to generate the TASS tables and `results/descriptive_report.html` while the factor inputs are pending. The analysis uses local data and requires no WRDS session.

The full run writes the standalone HTML report `results/report.html`, `results/alpha_estimates.csv`, `results/factors_monthly.csv`, `results/strategy_indexes_monthly.csv`, and the descriptive CSV tables. Alphas are monthly decimals in the estimates CSV. The report presents them as percentages and the estimates CSV includes annualized alpha computed as 12 times monthly alpha. Each index and period gets its own OLS fit with Newey–West standard errors using three observation lags.

The baseline uses all eligible TASS product-months. The April 1999 sensitivity omits one extreme TASS product return from that month's overall and strategy averages. `results/strategy_indexes_sensitivity.csv` and `results/alpha_sensitivity.csv` record the affected indexes and their re-estimated alphas. `results/alpha_change_tests.csv` tests the overall late-minus-early alpha change with a pooled regression that allows the intercept and all factor loadings to vary by subperiod.

Monthly indexes give each eligible TASS product one equal weight. The sample uses USD net-of-fee monthly returns from both live and graveyard products and excludes months before a known TASS entry date. The report shows the sample counts and how missing entry dates and strategy labels are handled.
