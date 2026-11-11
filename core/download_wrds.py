"""Download the raw TASS and factor inputs for HW4.

TASS is archived as delivered by WRDS. Keeping the full tables preserves fund
IDs, active and graveyard histories, and any dates used to identify backfill.
The analysis sample is January 1994 through December 2022.
Completed productperformance batches are reused if a run is interrupted.

The equity inputs require a separate LSEG Datastream export. Save month-end
S&PCOMP(RI) and FRUSS2L(RI) levels for December 1993 through December 2022 under
data/raw/factors/. These series follow Hsieh's updated size-spread definition
at https://people.duke.edu/~dah7/HFRFData.htm.
"""

import argparse
import csv
import gzip
import os
import re
from contextlib import closing
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from utils.path_info import data_path


START = "1994-01-01"
STOP = "2023-01-01"  # Exclusive.
WRDS_DIR = data_path / "raw" / "wrds"
TASS_SCHEMA = "tr_tass"
TASS_DIR = WRDS_DIR / TASS_SCHEMA
CATALOG_FILE = WRDS_DIR / "tass_catalog.csv.gz"
RF_FILE = WRDS_DIR / "risk_free_monthly.csv.gz"
PUBLIC_DIR = data_path / "raw" / "factors"
PUBLIC_URLS = {
    "TF-Fac.xls": "https://people.duke.edu/~dah7/DataLibrary/TF-Fac.xls",
    "fred_dgs10_dbaa_daily.csv": (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        # December 1993 supplies the baseline for the January 1994 changes.
        "?id=DGS10,DBAA&cosd=1993-12-01&coed=2022-12-31"
    ),
}

CATALOG_SQL = """
    SELECT n.nspname AS table_schema, c.relname AS table_name, c.relkind,
           obj_description(c.oid, 'pg_class') AS table_description,
           a.attnum AS ordinal_position, a.attname AS column_name,
           format_type(a.atttypid, a.atttypmod) AS data_type,
           col_description(c.oid, a.attnum) AS column_description
    FROM pg_catalog.pg_namespace AS n
    JOIN pg_catalog.pg_class AS c ON c.relnamespace = n.oid
    JOIN pg_catalog.pg_attribute AS a ON a.attrelid = c.oid
    WHERE n.nspname = %(schema)s
      AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
      AND NOT c.relispartition
      AND a.attnum > 0 AND NOT a.attisdropped
      AND has_schema_privilege(n.oid, 'USAGE')
      AND has_table_privilege(c.oid, 'SELECT')
    ORDER BY c.relname, a.attnum
"""

RF_SQL = """
    SELECT date, rf
    FROM ff.factors_monthly
    WHERE date >= %(start)s AND date < %(stop)s
    ORDER BY date
"""


def write_query(db, sql, target, *, params=None, allow_empty=False):
    """Write a WRDS result in batches and publish it once complete."""
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {target.name} ...", flush=True)
    with TemporaryDirectory(dir=target.parent) as temp_dir:
        temporary = Path(temp_dir) / target.name
        rows = 0
        header = True
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
            for chunk in db.raw_sql(
                sql,
                params=params,
                coerce_float=False,
                dtype_backend="numpy_nullable",
                chunksize=100_000,
                return_iter=True,
            ):
                chunk.to_csv(output, index=False, header=header)
                header = False
                rows += len(chunk)
        if header or (not rows and not allow_empty):
            raise RuntimeError(f"WRDS returned no data for {target.name}.")
        temporary.replace(target)
    print(f"Saved {rows:,} rows to {target}", flush=True)


def download_product_performance(db, sql, target, *, force):
    """Query disjoint date ranges and assemble the original raw CSV."""
    batch_dir = target.parent / ".productperformance_batches"
    batches = [(f"before_{START}", '"date" < %(stop)s', {"stop": START})]
    for year in range(int(START[:4]), int(STOP[:4])):
        start, stop = f"{year}-01-01", f"{year + 1}-01-01"
        batches.append((str(year), '"date" >= %(start)s AND "date" < %(stop)s',
                        {"start": start, "stop": stop}))
    batches.extend([
        (f"from_{STOP}", '"date" >= %(start)s', {"start": STOP}),
        ("undated", '"date" IS NULL', None),
    ])

    # Each batch is an atomic, self-contained gzip CSV with a header.
    parts = [batch_dir / f"{label}.csv.gz" for label, _, _ in batches]
    if force:
        for part in parts:
            part.unlink(missing_ok=True)

    for number, ((label, predicate, params), part) in enumerate(zip(batches, parts), start=1):
        progress = f"[productperformance {number}/{len(batches)}] {label}"
        if part.is_file():
            print(f"{progress}: using completed batch", flush=True)
            continue
        print(f"{progress}: querying WRDS", flush=True)
        write_query(db, f"{sql} WHERE {predicate}", part, params=params, allow_empty=True)

    print(f"Assembling {target.name} ...", flush=True)
    with TemporaryDirectory(dir=target.parent) as temp_dir:
        temporary = Path(temp_dir) / target.name
        header = None
        with gzip.open(temporary, "wb") as output:
            for part in parts:
                with gzip.open(part, "rb") as source:
                    batch_header = source.readline()
                    if not batch_header or (header is not None and batch_header != header):
                        raise RuntimeError(f"Unexpected CSV header in {part}. Use --force to refresh.")
                    if header is None:
                        header = batch_header
                        output.write(header)
                    copyfileobj(source, output)
        temporary.replace(target)
    print(f"Saved {target}", flush=True)


def tass_downloads(selected_tables, force):
    with gzip.open(CATALOG_FILE, "rt", encoding="utf-8", newline="") as source:
        tables = dict.fromkeys(row["table_name"] for row in csv.DictReader(source))
    unknown = set(selected_tables or ()) - tables.keys()
    if unknown:
        raise ValueError(f"Tables absent from {TASS_SCHEMA}: {', '.join(sorted(unknown))}.")

    pending = []
    for name in dict.fromkeys(selected_tables or tables):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError(f"Unexpected WRDS table name: {name!r}")
        target = TASS_DIR / f"{name}.csv.gz"
        if force or not target.is_file():
            pending.append((name, target))
    return pending


def download_public_inputs(force):
    for filename, url in PUBLIC_URLS.items():
        target = PUBLIC_DIR / filename
        if target.is_file() and not force:
            print(f"Skipping {target.name} (already downloaded)")
            continue
        print(f"Downloading {target.name} ...", flush=True)
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if filename.endswith(".xls"):
            valid = content.startswith(bytes.fromhex("D0 CF 11 E0 A1 B1 1A E1"))
        else:
            header = content.partition(b"\n")[0].decode("utf-8-sig")
            valid = {"DGS10", "DBAA"} <= set(next(csv.reader([header])))
        if not valid:
            raise RuntimeError(f"Unexpected file format returned by {url}")

        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent) as temp_dir:
            temporary = Path(temp_dir) / target.name
            temporary.write_bytes(content)
            temporary.replace(target)
        print(f"Saved {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", help="WRDS username (defaults to WRDS_USERNAME or a prompt)")
    parser.add_argument("--only", nargs="+", choices=("tass", "rf", "public"), help="Download selected inputs")
    parser.add_argument("--tass-table", nargs="+", help="Download only these TASS tables")
    parser.add_argument("--force", action="store_true", help="Refresh the requested inputs")
    args = parser.parse_args()

    selected = args.only or ("tass", "rf", "public")
    if args.tass_table and "tass" not in selected:
        parser.error("--tass-table requires the tass input.")

    refresh_catalog = "tass" in selected and (args.force or not CATALOG_FILE.is_file())
    pending_tables = []
    if "tass" in selected and not refresh_catalog:
        pending_tables = tass_downloads(args.tass_table, args.force)
    download_rf = "rf" in selected and (args.force or not RF_FILE.is_file())

    if refresh_catalog or pending_tables or download_rf:
        import wrds

        username = (args.username or os.environ.get("WRDS_USERNAME", "")).strip()
        print("Connecting to WRDS. Approve the Duo Mobile push if prompted.", flush=True)
        with closing(wrds.Connection(wrds_username=username)) as db:
            if refresh_catalog:
                write_query(db, CATALOG_SQL, CATALOG_FILE, params={"schema": TASS_SCHEMA})
                pending_tables = tass_downloads(args.tass_table, args.force)
            for name, target in pending_tables:
                sql = f'SELECT * FROM "{TASS_SCHEMA}"."{name}"'
                if name == "productperformance":
                    download_product_performance(db, sql, target, force=args.force)
                else:
                    write_query(db, sql, target, allow_empty=True)
            if download_rf:
                # Available for an excess-return specification.
                write_query(db, RF_SQL, RF_FILE, params={"start": START, "stop": STOP})
    elif "tass" in selected or "rf" in selected:
        print("Requested WRDS files are already downloaded.")

    if "public" in selected:
        download_public_inputs(args.force)


if __name__ == "__main__":
    main()
