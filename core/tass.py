"""Build HW4's hedge fund indexes from the local Lipper TASS extract."""

from pathlib import Path

import numpy as np
import pandas as pd


START_MONTH = "1994-01"
END_MONTH = "2022-12"
PRODUCT = "productreference"
UNCLASSIFIED = "Unclassified"
OVERALL = "Overall EW"

PROFILE_COLUMNS = [
    PRODUCT, "primarycategory", "currencycode", "grossnett", "trackingfrequency",
    "live_graveyard", "managementfee", "incentivefee", "minimuminvestment",
    "subscriptionfrequency", "redemptionfrequency", "redemptionnoticeperiod",
    "lockupperiod", "payoutperiod",
]
PERFORMANCE_COLUMNS = [PRODUCT, "date", "rateofreturn", "estimatedassets", "estimatedactual"]

TABLE_CONTENTS = {
    "assetsinstrumentsdetails": ("Assets and instruments used by a product", PRODUCT),
    "companydetails": ("Companies associated with each product", PRODUCT),
    "focusdetails": ("Sector, investment, and geographic focus and investment approach flags", PRODUCT),
    "indexdetails": ("Names, abbreviations, and coverage of published indexes", "indexreference to indexperformance"),
    "indexperformance": ("Published monthly index levels and returns", "indexreference to indexdetails"),
    "peopledetails": ("People associated with each product", PRODUCT),
    "productdetails": ("Product profile, strategy, fees, liquidity, and current live/graveyard status", PRODUCT),
    "productdetailsadddate": ("Dates products were added to and removed from TASS", PRODUCT),
    "productdropdetails": ("Drop dates, reasons, and last performance dates", PRODUCT),
    "productperformance": ("Dated returns, NAV, assets, and estimate flags", PRODUCT),
    "serviceproviderdetails": ("Service provider companies associated with each product", PRODUCT),
    "stockexchanges": ("Stock exchange lookup", "stockexchangeid to stockexchangexref"),
    "stockexchangexref": ("Product-to-stock-exchange links", "productreference and stockexchangeid"),
    "tradingexchange": ("Trading exchange lookup", "exchangeid to tradingexchangexref"),
    "tradingexchangexref": ("Product-to-trading-exchange links", "productreference and exchangeid"),
    "wrds_productdetails": ("WRDS product profile table", PRODUCT),
}


def _inventory(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path, usecols=["table_name", "column_name"])
    records = []
    for table, columns in catalog.groupby("table_name", sort=True)["column_name"]:
        contents, link = TABLE_CONTENTS.get(table, ("See WRDS catalog", ""))
        records.append((table, len(columns), contents, link))
    return pd.DataFrame(records, columns=["table", "columns", "contents", "link"])


def _load_sample(performance_path: Path, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stream the raw history, then reconcile fund-month duplicates across chunks."""
    steps = [
        "Dated performance records in the sample window",
        "Matched to a TASS product profile",
        "USD-denominated products",
        "Net-of-fees returns",
        "Monthly tracking frequency",
        "Nonmissing finite returns of at least -100%",
        "Remove months before known TASS entry",
        "Final unique product-months",
    ]
    counts = [0] * len(steps)
    products = [set() for _ in steps]

    def count(step: int, frame: pd.DataFrame) -> None:
        counts[step] += len(frame)
        products[step].update(frame[PRODUCT].dropna().unique())

    usd = profiles.index[profiles["currencycode"].eq("USD")]
    net = profiles.index[profiles["grossnett"].eq("N")]
    monthly = profiles.index[profiles["trackingfrequency"].eq("MONTHLY")]
    pieces = []
    with pd.read_csv(
        performance_path, usecols=PERFORMANCE_COLUMNS, chunksize=250_000,
        dtype={PRODUCT: "float64"},
    ) as reader:
        for chunk in reader:
            chunk["date"] = pd.to_datetime(chunk["date"], format="%Y-%m-%d", errors="coerce")
            chunk = chunk.loc[chunk["date"].between("1994-01-01", "2022-12-31")]
            count(0, chunk)
            chunk = chunk.loc[chunk[PRODUCT].isin(profiles.index)]
            count(1, chunk)
            chunk = chunk.loc[chunk[PRODUCT].isin(usd)]
            count(2, chunk)
            chunk = chunk.loc[chunk[PRODUCT].isin(net)]
            count(3, chunk)
            chunk = chunk.loc[chunk[PRODUCT].isin(monthly)].copy()
            count(4, chunk)
            chunk["rateofreturn"] = pd.to_numeric(chunk["rateofreturn"], errors="coerce")
            valid = np.isfinite(chunk["rateofreturn"]) & chunk["rateofreturn"].ge(-100)
            chunk = chunk.loc[valid].copy()
            count(5, chunk)
            chunk["month"] = chunk["date"].dt.to_period("M")
            added = chunk[PRODUCT].map(profiles["entry_month"])
            chunk = chunk.loc[added.isna() | chunk["month"].ge(added)].copy()
            count(6, chunk)
            chunk["estimatedassets"] = pd.to_numeric(chunk["estimatedassets"], errors="coerce")
            valid_assets = np.isfinite(chunk["estimatedassets"]) & chunk["estimatedassets"].gt(0)
            chunk["estimatedassets"] = chunk["estimatedassets"].where(valid_assets)
            if not chunk.empty:
                pieces.append(chunk)

    if not pieces:
        raise ValueError("TASS has no eligible USD net-of-fees monthly returns for 1994–2022.")
    panel = pd.concat(pieces, ignore_index=True)
    keys = [PRODUCT, "month"]
    duplicated = panel.duplicated(keys, keep=False)
    conflicts = 0
    if duplicated.any():
        candidates = panel.loc[duplicated].copy()
        latest_date = candidates.groupby(keys)["date"].transform("max")
        candidates = candidates.loc[candidates["date"].eq(latest_date)].copy()
        candidates["actual"] = candidates["estimatedactual"].eq("A")
        prefer_actual = candidates.groupby(keys)["actual"].transform("max")
        candidates = candidates.loc[candidates["actual"].eq(prefer_actual)].copy()
        unique_return = candidates.groupby(keys)["rateofreturn"].transform("nunique").eq(1)
        conflicts = len(candidates.loc[~unique_return, keys].drop_duplicates())
        candidates = candidates.loc[unique_return].copy()
        # Agreeing nonmissing asset values can supplement a duplicate row with missing assets.
        asset_group = candidates.groupby(keys)["estimatedassets"]
        candidates["estimatedassets"] = asset_group.transform("max").where(asset_group.transform("nunique").eq(1))
        candidates = candidates.drop_duplicates(keys).drop(columns="actual")
        panel = pd.concat([panel.loc[~duplicated], candidates], ignore_index=True)
    if panel.empty:
        raise ValueError("No eligible TASS fund-month returns remain after reconciling duplicates.")
    count(7, panel)
    selection = pd.DataFrame({"step": steps, "products": list(map(len, products)), "fund_months": counts})
    return panel, selection, conflicts


def april_1999_sensitivity(panel: pd.DataFrame, index_returns: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rebuild the affected index returns without the identified April 1999 observation.

    The panel contains validated product-months, strategy labels and TASS returns
    in percentage points. The input can contain just April 1999.
    """
    month = pd.Period("1999-04", freq="M")
    product = 8
    april = panel.loc[panel["month"].eq(month)]
    # Match the reported +31,208% print without excluding an ordinary revised return.
    excluded = april[PRODUCT].eq(product) & april["rateofreturn"].gt(30_000)
    info = {
        "productreference": product,
        "month": str(month),
        "excluded_return_pct": None,
        "excluded_count": int(excluded.sum()),
        "affected_indexes": [],
        "remaining_products": {},
        "baseline_returns_pct": {},
        "sensitivity_returns_pct": {},
    }
    if info["excluded_count"] == 0:
        return index_returns.iloc[:, :0].copy(), info
    if info["excluded_count"] != 1:
        raise ValueError("Expected one reconciled April 1999 observation for TASS productreference 8.")

    observation = april.loc[excluded].iloc[0]
    strategy = observation["strategy"]
    remaining = april.loc[~excluded]
    groups = {OVERALL: remaining["rateofreturn"]}
    if strategy != UNCLASSIFIED:
        groups[strategy] = remaining.loc[remaining["strategy"].eq(strategy), "rateofreturn"]

    revised = index_returns.loc[:, list(groups)].copy()
    info["excluded_return_pct"] = float(observation["rateofreturn"])
    info["affected_indexes"] = list(groups)
    for name, values in groups.items():
        before = index_returns.at[month, name]
        after = float(values.mean()) if len(values) else np.nan
        revised.at[month, name] = after / 100
        info["remaining_products"][name] = len(values)
        info["baseline_returns_pct"][name] = float(before * 100) if pd.notna(before) else None
        info["sensitivity_returns_pct"][name] = after if pd.notna(after) else None
    return revised, info


def analyze_tass(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict, pd.DataFrame]:
    """Return baseline decimal indexes, descriptives, sample notes, and sensitivity indexes."""
    wrds_dir = Path(raw_dir) / "wrds"
    tables_dir = wrds_dir / "tr_tass"
    performance_path = tables_dir / "productperformance.csv.gz"
    profile_path = tables_dir / "productdetails.csv.gz"
    entry_path = tables_dir / "productdetailsadddate.csv.gz"
    catalog_path = wrds_dir / "tass_catalog.csv.gz"
    for path in (performance_path, profile_path, entry_path, catalog_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required TASS file missing: {path}. Finish the download before running HW4.")

    profiles = pd.read_csv(profile_path, usecols=PROFILE_COLUMNS, dtype={PRODUCT: "float64"})
    profiles = profiles.dropna(subset=[PRODUCT]).drop_duplicates().set_index(PRODUCT, verify_integrity=True)
    entries = pd.read_csv(entry_path, usecols=[PRODUCT, "dateaddedtotass"], dtype={PRODUCT: "float64"})
    entries = entries.dropna(subset=[PRODUCT]).drop_duplicates().set_index(PRODUCT, verify_integrity=True)
    entry_dates = pd.to_datetime(entries["dateaddedtotass"], format="%Y-%m-%d", errors="coerce")
    profiles["entry_month"] = entry_dates.dt.to_period("M")
    for column in ("currencycode", "grossnett", "trackingfrequency", "live_graveyard"):
        profiles[column] = profiles[column].astype("string").str.strip().str.upper()
    strategy = profiles["primarycategory"].astype("string").str.strip()
    unnamed = strategy.str.lower().isin(["", "undefined", "not defined", "unknown"])
    profiles["strategy"] = strategy.mask(unnamed).fillna(UNCLASSIFIED)
    for column in ("managementfee", "incentivefee", "minimuminvestment", "redemptionnoticeperiod", "lockupperiod", "payoutperiod"):
        values = pd.to_numeric(profiles[column], errors="coerce")
        profiles[column] = values.where(np.isfinite(values) & values.ge(0))

    panel, selection, conflicts = _load_sample(performance_path, profiles)
    panel["strategy"] = panel[PRODUCT].map(profiles["strategy"])
    # The source's reported returns match 100 * (NAV_t / NAV_{t-1} - 1) for adjacent observations.
    panel["return"] = panel["rateofreturn"] / 100
    months = pd.period_range(START_MONTH, END_MONTH, freq="M", name="month")
    named = panel.loc[panel["strategy"].ne(UNCLASSIFIED)]
    strategy_returns = named.groupby(["month", "strategy"])["return"].mean().unstack("strategy")
    overall_returns = panel.groupby("month")["return"].mean().rename(OVERALL)
    returns = pd.concat([overall_returns, strategy_returns], axis=1).reindex(months)
    sensitivity_returns, sensitivity_info = april_1999_sensitivity(panel, returns)
    strategy_sizes = named.groupby(["month", "strategy"])["return"].size().unstack("strategy")
    overall_sizes = panel.groupby("month")["return"].size().rename(OVERALL)
    sizes = pd.concat([overall_sizes, strategy_sizes], axis=1).reindex(months)

    funds = profiles.loc[panel[PRODUCT].unique()].copy()
    positive_assets = panel.loc[panel["estimatedassets"].notna(), [PRODUCT, "date", "estimatedassets"]]
    latest_assets = positive_assets.sort_values("date").drop_duplicates(PRODUCT, keep="last").set_index(PRODUCT)
    funds["assets_usd_millions"] = latest_assets["estimatedassets"] / 1e6
    fund_groups = funds.groupby("strategy", sort=True)
    strategy_summary = fund_groups.agg(
        products=("strategy", "size"),
        median_assets_usd_millions=("assets_usd_millions", "median"),
        mean_management_fee_pct=("managementfee", "mean"),
        mean_incentive_fee_pct=("incentivefee", "mean"),
        median_notice_days=("redemptionnoticeperiod", "median"),
        median_lockup_months=("lockupperiod", "median"),
    ).join(panel.groupby("strategy")["rateofreturn"].agg(
        fund_months="size", mean_return_pct="mean", volatility_pct="std",
    )).reset_index()

    characteristics = {
        "Latest positive reported assets (USD millions)": funds["assets_usd_millions"],
        "Management fee (%)": funds["managementfee"],
        "Incentive fee (%)": funds["incentivefee"],
        "Minimum investment (USD)": funds["minimuminvestment"],
        "Redemption notice (days)": funds["redemptionnoticeperiod"],
        "Lockup (months)": funds["lockupperiod"],
        "Payout period (days)": funds["payoutperiod"],
        "Monthly return (%; pooled fund-months)": panel["rateofreturn"],
    }
    fund_characteristics = pd.DataFrame([
        (label, int(values.count()), values.mean(), values.std(), values.quantile(.25),
         values.median(), values.quantile(.75))
        for label, values in characteristics.items()
    ], columns=["characteristic", "n", "mean", "std", "p25", "median", "p75"])

    liquidity = []
    for term, column in [("Subscription", "subscriptionfrequency"), ("Redemption", "redemptionfrequency")]:
        frequency = funds[column].astype("string").str.strip().replace({"": "Missing", "Not Defined": "Missing"}).fillna("Missing")
        for label, count in frequency.value_counts().items():
            liquidity.append((term, label, count, 100 * count / len(funds)))
    liquidity_terms = pd.DataFrame(liquidity, columns=["term", "frequency", "products", "share_pct"])

    index_summary = pd.DataFrame({
        "months": returns.count(),
        "mean_return_pct": 100 * returns.mean(),
        "volatility_pct": 100 * returns.std(),
        "mean_products": sizes.mean(),
    }).rename_axis("index").reset_index()

    metadata = {
        "products": len(funds),
        "fund_months": len(panel),
        "sensitivity": sensitivity_info,
        "notes": [
            "Use USD products explicitly marked net of fees and monthly, with finite monthly returns of at least −100%. "
            "TASS rateofreturn is in percentage points. No return winsorization is applied.",
            "Remove returns dated before the calendar month of known TASS entry. Keep products with missing entry "
            "dates, whose histories may still contain backfill. Early coverage can be thin after the entry screen.",
            "Use each available product-month once. For duplicates, prefer the latest dated row, then an actual "
            "observation on that date. Drop unresolved return conflicts and set conflicting asset values to missing.",
            "Include live and graveyard products, including funds of funds. Overall EW includes unclassified products. "
            "Named strategy indexes use the supplied primary category and omit unclassified products.",
            "Each index averages available product returns equally within a calendar month. Product IDs can represent "
            "share classes. Product characteristics and strategy labels come from the available snapshot.",
            "Size is the latest positive reported assets per product within the eligible sample. Numeric summaries use "
            "nonmissing values, and each liquidity frequency share uses all eligible products as the denominator.",
            "The sample-selection table counts source records before product-month reconciliation. Published vendor "
            "indexes are listed in the table inventory. The homework indexes are built from product returns.",
        ],
        "findings": [],
    }
    findings = metadata["findings"]
    leaders = strategy_summary.loc[strategy_summary["strategy"].ne(UNCLASSIFIED)].nlargest(1, "products")
    if not leaders.empty:
        leader = leaders.iloc[0]
        findings.append(f"The largest named strategy by product count was {leader['strategy']} "
                        f"with {leader['products']:,} products ({100 * leader['products'] / len(funds):.1f}% of the sample).")
    graveyard = int(funds["live_graveyard"].eq("GRAVEYARD").sum())
    missing_entry = int(funds["entry_month"].isna().sum())
    unclassified = int(funds["strategy"].eq(UNCLASSIFIED).sum())
    removed = int(selection.loc[5, "fund_months"] - selection.loc[6, "fund_months"])
    extreme = int(panel["rateofreturn"].ge(100).sum())
    findings.append(f"The final sample has {len(funds):,} products and {len(panel):,} fund-months. "
                    f"The extract classifies {graveyard:,} sampled products as graveyard and {unclassified:,} "
                    f"as unclassified by strategy. {missing_entry:,} sampled products lack a TASS entry date.")
    findings.append(f"The entry screen removes {removed:,} otherwise eligible records. "
                    f"Duplicate reconciliation drops {conflicts:,} product-months with unresolved return conflicts. "
                    f"The final sample includes {extreme:,} reported monthly returns of at least 100%.")
    early = panel.loc[panel["month"].dt.year.eq(1994), PRODUCT]
    if not early.empty:
        early_missing = int(early.map(profiles["entry_month"]).isna().sum())
        findings.append(f"The 1994 sample contributes {len(early):,} fund-months, including {early_missing:,} "
                        f"({100 * early_missing / len(early):.1f}%) with no recorded entry date.")
    for label, column, suffix in [
        ("Reported assets", "assets_usd_millions", " million USD"),
        ("Management fee", "managementfee", "%"),
        ("Incentive fee", "incentivefee", "%"),
        ("Redemption notice", "redemptionnoticeperiod", " days"),
        ("Lockup", "lockupperiod", " months"),
    ]:
        median = funds[column].median()
        if pd.notna(median):
            findings.append(f"{label} has a sample median of {median:,.2f}{suffix} among reporting products.")
    redemption = liquidity_terms.loc[liquidity_terms["term"].eq("Redemption") & liquidity_terms["frequency"].ne("Missing")]
    if not redemption.empty:
        mode = redemption.nlargest(1, "products").iloc[0]
        findings.append(f"{mode['frequency']} is the most common known redemption schedule "
                        f"({mode['products']:,} products, {mode['share_pct']:.1f}% of the sample).")
    summary = index_summary.set_index("index").loc[OVERALL]
    findings.append(f"The overall EW index has {int(summary['months']):,} months with data. Its average monthly "
                    f"return is {summary['mean_return_pct']:.2f}% and its monthly volatility is {summary['volatility_pct']:.2f}%.")
    return returns, {
        "tass_tables": _inventory(catalog_path),
        "sample_selection": selection,
        "strategy_summary": strategy_summary,
        "fund_characteristics": fund_characteristics,
        "liquidity_terms": liquidity_terms,
        "index_summary": index_summary,
    }, metadata, sensitivity_returns
