"""Fung–Hsieh seven-factor regressions on monthly TASS index returns."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


FACTOR_COLUMNS = [
    "sp500", "size_spread", "treasury_change", "credit_spread_change",
    "ptfs_bond", "ptfs_currency", "ptfs_commodity",
]
SAMPLES = (
    ("1994–2022", "1994-01", "2022-12"),
    ("1994–2003", "1994-01", "2003-12"),
    ("2004–2013", "2004-01", "2013-12"),
    ("2014–2022", "2014-01", "2022-12"),
)
HAC_LAGS = 3


def estimate_alphas(index_returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Fit separate monthly OLS models with an intercept and HAC inference.

    Both the dependent variable and the S&P 500 factor are total returns in
    decimals. Each fit uses the months in which its index return is available.
    HAC lags count successive observations within that fit.
    """
    if not index_returns.index.is_unique or not factors.index.is_unique:
        raise ValueError("Index returns and factors must each have one row per month.")

    records = []
    for index_name, returns in index_returns.items():
        panel = factors[FACTOR_COLUMNS].join(returns.rename("index_return")).sort_index()
        for label, start, end in SAMPLES:
            sample = panel.loc[start:end].dropna().astype(float)
            nobs = len(sample)
            row = {
                "index": index_name,
                "sample": label,
                "nobs": nobs,
                "first_month": str(sample.index[0]) if nobs else None,
                "last_month": str(sample.index[-1]) if nobs else None,
                "alpha": np.nan,
                "alpha_annualized": np.nan,
                "alpha_se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "r_squared": np.nan,
                "status": "estimated",
                **{f"beta_{name}": np.nan for name in FACTOR_COLUMNS},
            }
            records.append(row)
            if nobs <= len(FACTOR_COLUMNS) + 1:
                row["status"] = "too few observations"
                continue

            design = sm.add_constant(sample[FACTOR_COLUMNS], has_constant="add")
            if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
                row["status"] = "rank-deficient factor matrix"
                continue

            fit = sm.OLS(sample["index_return"], design).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
                use_t=True,
            )
            alpha = fit.params["const"]
            row.update(alpha=alpha, alpha_annualized=12 * alpha, r_squared=fit.rsquared)
            row.update({f"beta_{name}": fit.params[name] for name in FACTOR_COLUMNS})
            se = fit.bse["const"]
            if not np.isfinite(se) or se <= 0:
                row["status"] = "HAC standard error unavailable"
                continue

            lower, upper = fit.conf_int().loc["const"]
            row.update(
                alpha_se=se, t_stat=fit.tvalues["const"], p_value=fit.pvalues["const"],
                ci_lower=lower, ci_upper=upper,
            )
    return pd.DataFrame.from_records(records)


def test_overall_alpha_change(
    returns: pd.Series, factors: pd.DataFrame, specification: str,
) -> dict:
    """Test late-minus-early alpha with a pooled regression and period-specific slopes."""
    panel = factors[FACTOR_COLUMNS].join(returns.rename("index_return")).sort_index()
    sample = panel.loc[SAMPLES[0][1]:SAMPLES[0][2]].dropna().astype(float)
    base = sm.add_constant(sample[FACTOR_COLUMNS], has_constant="add")
    blocks = []
    for label, start, end in SAMPLES[1:]:
        period = pd.Series(
            (sample.index >= pd.Period(start, freq="M")) & (sample.index <= pd.Period(end, freq="M")),
            index=sample.index,
        )
        blocks.append(base.mul(period, axis=0).add_prefix(f"{label} "))
    design = pd.concat(blocks, axis=1)
    row = {
        "specification": specification,
        "nobs": len(sample),
        "alpha_change": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
    }
    if len(sample) <= design.shape[1] or np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        return row

    fit = sm.OLS(sample["index_return"], design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
        use_t=True,
    )
    contrast = np.zeros(design.shape[1])
    contrast[design.columns.get_loc(f"{SAMPLES[-1][0]} const")] = 1
    contrast[design.columns.get_loc(f"{SAMPLES[1][0]} const")] = -1
    test = fit.t_test(contrast)
    lower, upper = test.conf_int()[0]
    row.update(
        alpha_change=np.asarray(test.effect).item(),
        t_stat=np.asarray(test.tvalue).item(),
        p_value=np.asarray(test.pvalue).item(),
        ci_lower=lower,
        ci_upper=upper,
    )
    return row
