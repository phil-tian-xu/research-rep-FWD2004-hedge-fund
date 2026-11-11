"""Construct the monthly Fung–Hsieh factors from local source files."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd


SAMPLE_MONTHS = pd.period_range("1994-01", "2022-12", freq="M", name="month")
LEVEL_MONTHS = pd.period_range("1993-12", "2022-12", freq="M", name="month")
EQUITY_FILE = "datastream_equity_monthly.csv"
YIELD_FILE = "fred_dgs10_dbaa_daily.csv"
TREND_FILE = "TF-Fac.xls"
TREND_COLUMNS = {"PTFSBD": "ptfs_bond", "PTFSFX": "ptfs_currency", "PTFSCOM": "ptfs_commodity"}
WILSHIRE_COLUMNS = ["wilshire_1750_tr", "wilshire_750_tr"]
TASS_EQUITY_NAMES = {"S&P 500 TR": "sp500_tr", "Russell 2000 TR": "russell_2000_tr"}
RUSSELL_SPREAD = "Russell 2000 minus S&P 500 total return (Hsieh’s updated specification)"


def _require_months(frame: pd.DataFrame, months: pd.PeriodIndex, path: Path) -> pd.DataFrame:
    frame = frame.loc[frame.index.isin(months)]
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique()
        raise ValueError(f"{path.name} has duplicate months: {', '.join(map(str, duplicates[:8]))}.")

    frame = frame.reindex(months)
    invalid = ~np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)
    if invalid.any():
        missing = months[invalid]
        examples = ", ".join(map(str, missing[:8]))
        raise ValueError(f"{path.name} has missing or invalid data in {len(missing)} required months ({examples}).")
    return frame


def _equity_returns(levels: pd.DataFrame, small: str, large: str, path: Path) -> pd.DataFrame:
    levels = _require_months(levels, LEVEL_MONTHS, path)
    if (levels <= 0).any().any():
        raise ValueError(f"{path.name} requires strictly positive total-return index levels.")

    returns = levels.pct_change(fill_method=None).loc[SAMPLE_MONTHS]
    return pd.DataFrame({"sp500": returns["sp500_tr"], "size_spread": returns[small] - returns[large]})


def _load_equity_export(path: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path, dtype={"date": "string"})
    if not {"date", "sp500_tr"}.issubset(frame):
        raise ValueError(f"{path.name} requires date and sp500_tr columns.")

    if set(WILSHIRE_COLUMNS).issubset(frame):
        small, large = WILSHIRE_COLUMNS
        size_definition = "Wilshire 1750 Small Cap minus Wilshire 750 Large Cap total return (Table I)"
    elif set(WILSHIRE_COLUMNS).intersection(frame):
        raise ValueError(f"{path.name} must contain both Wilshire total-return level columns or neither.")
    elif "russell_2000_tr" in frame:
        small, large = "russell_2000_tr", "sp500_tr"
        size_definition = RUSSELL_SPREAD
    else:
        raise ValueError(f"{path.name} requires both Wilshire level columns or russell_2000_tr.")

    dates = frame["date"].str.strip()
    if not dates.str.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", na=False).all():
        raise ValueError(f"{path.name} requires dates in YYYY-MM or YYYY-MM-DD format.")
    dates = pd.to_datetime(dates.mask(dates.str.len().eq(7), dates + "-01"), format="%Y-%m-%d")
    columns = list(dict.fromkeys(["sp500_tr", small, large]))
    levels = frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    levels.index = pd.PeriodIndex(dates, freq="M", name="month")
    return _equity_returns(levels, small, large, path), size_definition


def _load_tass_equity(tass_dir: Path) -> pd.DataFrame:
    details_path = tass_dir / "indexdetails.csv.gz"
    performance_path = tass_dir / "indexperformance.csv.gz"
    if not details_path.is_file() or not performance_path.is_file():
        raise FileNotFoundError(
            f"Equity factors require either {EQUITY_FILE} or the two TASS benchmark extracts "
            "indexdetails.csv.gz and indexperformance.csv.gz. See the README."
        )

    details = pd.read_csv(details_path, usecols=["indexreference", "indexname"])
    details["indexname"] = details["indexname"].str.strip()
    details = details.loc[details["indexname"].isin(TASS_EQUITY_NAMES)].drop_duplicates()
    if (
        set(details["indexname"]) != set(TASS_EQUITY_NAMES)
        or details["indexname"].duplicated().any()
        or details["indexreference"].isna().any()
        or details["indexreference"].duplicated().any()
    ):
        raise ValueError(f"{details_path.name} must identify one S&P 500 TR and one Russell 2000 TR series.")
    columns = details.set_index("indexreference")["indexname"].map(TASS_EQUITY_NAMES).to_dict()

    performance = pd.read_csv(performance_path, usecols=["indexreference", "monthenddate", "nav"])
    performance = performance.loc[performance["indexreference"].isin(columns)].copy()
    performance["month"] = pd.to_datetime(performance["monthenddate"], format="%Y-%m-%d").dt.to_period("M")
    performance = performance.loc[performance["month"].isin(LEVEL_MONTHS)].drop_duplicates()
    if performance.duplicated(["month", "indexreference"]).any():
        raise ValueError(f"{performance_path.name} has conflicting NAV rows for a required equity index-month.")
    performance["nav"] = pd.to_numeric(performance["nav"], errors="coerce").astype(float)
    levels = performance.pivot(index="month", columns="indexreference", values="nav")
    levels = levels.rename(columns=columns).reindex(columns=list(TASS_EQUITY_NAMES.values()))
    return _equity_returns(levels, "russell_2000_tr", "sp500_tr", performance_path)


def _load_yields(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path, na_values=["."])
    daily.columns = daily.columns.str.strip().str.lower()
    date_column = "observation_date" if "observation_date" in daily else "date"
    if not {date_column, "dgs10", "dbaa"}.issubset(daily):
        raise ValueError(f"{path.name} requires a FRED date column and DGS10 and DBAA yields.")

    dates = pd.to_datetime(daily[date_column], format="%Y-%m-%d")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"{path.name} has missing or duplicate daily dates.")
    daily = daily[["dgs10", "dbaa"]].apply(pd.to_numeric, errors="coerce").astype(float)
    daily.index = pd.DatetimeIndex(dates)
    daily = daily.loc[np.isfinite(daily).all(axis=1)].sort_index()
    # Use the last day when both yields are observed so the spread uses the same date.
    month_end = daily.groupby(daily.index.to_period("M")).last()
    month_end = _require_months(month_end, LEVEL_MONTHS, path) / 100
    levels = pd.DataFrame({
        "treasury_change": month_end["dgs10"],
        "credit_spread_change": month_end["dbaa"] - month_end["dgs10"],
    })
    return levels.diff().loc[SAMPLE_MONTHS]


def _load_trend(path: Path) -> pd.DataFrame:
    workbook = xlrd.open_workbook(str(path))
    sheet = workbook.sheet_by_name("TF-Fac") if "TF-Fac" in workbook.sheet_names() else workbook.sheet_by_index(0)
    for header_row in range(sheet.nrows):
        headers = [re.sub(r"[^A-Z0-9]", "", str(value).upper()) for value in sheet.row_values(header_row)]
        date_label = "YYYYMM" if "YYYYMM" in headers else "DATE"
        if {date_label, *TREND_COLUMNS}.issubset(headers):
            break
    else:
        raise ValueError(f"Could not find the month, PTFSBD, PTFSFX, and PTFSCOM header in {path.name}.")

    date_column = headers.index(date_label)
    return_columns = [headers.index(name) for name in TREND_COLUMNS]
    rows, months = [], []
    for row in range(header_row + 1, sheet.nrows):
        cell = sheet.cell(row, date_column)
        if cell.ctype == xlrd.XL_CELL_DATE:
            month = pd.Period(xlrd.xldate_as_datetime(cell.value, workbook.datemode), freq="M")
        else:
            label = str(cell.value).strip().removesuffix(".0")
            if re.fullmatch(r"\d{4}(?:0[1-9]|1[0-2])", label):
                month = pd.Period(f"{label[:4]}-{label[4:]}", freq="M")
            elif re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", label):
                month = pd.Period(pd.Timestamp(label), freq="M")
            else:
                continue
        if month < SAMPLE_MONTHS[0] or month > SAMPLE_MONTHS[-1]:
            continue

        values = []
        for column in return_columns:
            # Excel percentage cells store numeric fractions. Text with a % sign is in percent units.
            value = str(sheet.cell_value(row, column)).strip()
            scale = 100 if value.endswith("%") else 1
            values.append(pd.to_numeric(value.removesuffix("%"), errors="coerce") / scale)
        rows.append(values)
        months.append(month)

    frame = pd.DataFrame(rows, columns=list(TREND_COLUMNS.values()), index=pd.PeriodIndex(months, freq="M"))
    return _require_months(frame, SAMPLE_MONTHS, path)


def load_factors(raw_dir: Path) -> pd.DataFrame:
    """Load all seven factors in decimal units for January 1994–December 2022."""
    factor_dir = raw_dir / "factors"
    missing = [name for name in (YIELD_FILE, TREND_FILE) if not (factor_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing factor input files in {factor_dir}: {', '.join(missing)}. See the README.")

    equity_path = factor_dir / EQUITY_FILE
    if equity_path.is_file():
        equity, size_definition = _load_equity_export(equity_path)
        equity_source = EQUITY_FILE
    else:
        equity = _load_tass_equity(raw_dir / "wrds" / "tr_tass")
        size_definition = RUSSELL_SPREAD
        equity_source = "LSEG TASS published S&P 500 TR and Russell 2000 TR NAVs"
    yields = _load_yields(factor_dir / YIELD_FILE)
    trend = _load_trend(factor_dir / TREND_FILE)
    factors = pd.concat([equity, yields, trend], axis=1)
    factors.attrs["size_spread_definition"] = size_definition
    factors.attrs["equity_source"] = equity_source
    factors.attrs["factor_spec"] = (
        f"S&P 500 total return. Size spread is {size_definition}. "
        f"Equity index levels come from {equity_source}. "
        "Bond factors are changes in the Treasury yield and Baa-minus-Treasury yield spread "
        "using each month's last common observation in FRED. Trend factors are the bond, currency, "
        "and commodity lookback-straddle returns from the Fung–Hsieh workbook."
    )
    return factors
