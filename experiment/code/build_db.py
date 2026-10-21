#!/usr/bin/env python3
"""Build a sampled NYC yellow-taxi SQLite database for text-to-SQL evaluation.

Downloads one month of TLC yellow-taxi trip records plus the taxi zone lookup,
cleans obviously broken rows, samples N trips uniformly at random, and writes:

    trips          sampled trip records
    zones          TLC taxi zone lookup (location_id -> borough / zone)
    payment_types  payment_type code -> name

Usage:
    python build_db.py --month 2024-01 --rows 200000 --out data/taxi.db
"""
import argparse
import csv
import datetime as dt
import random
import sqlite3
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# Source column (lowercased; TLC capitalisation varies by year) -> our column.
REQUIRED = {
    "vendorid": "vendor_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
    "ratecodeid": "rate_code_id",
    "pulocationid": "pu_location_id",
    "dolocationid": "do_location_id",
    "payment_type": "payment_type",
    "fare_amount": "fare_amount",
    "tip_amount": "tip_amount",
    "tolls_amount": "tolls_amount",
    "total_amount": "total_amount",
}
OPTIONAL = {
    "congestion_surcharge": "congestion_surcharge",
    "airport_fee": "airport_fee",
}

PAYMENT_TYPES = [
    (1, "Credit card"),
    (2, "Cash"),
    (3, "No charge"),
    (4, "Dispute"),
    (5, "Unknown"),
    (6, "Voided trip"),
]

# Comments are kept in sqlite_master, so schema.sql doubles as prompt context.
DDL = """
CREATE TABLE zones (
    location_id   INTEGER PRIMARY KEY,  -- TLC taxi zone id
    borough       TEXT,                 -- 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR'
    zone          TEXT,                 -- zone name, e.g. 'JFK Airport', 'Upper East Side North'
    service_zone  TEXT                  -- 'Yellow Zone', 'Boro Zone', 'Airports', 'EWR'
);

CREATE TABLE payment_types (
    payment_type_id  INTEGER PRIMARY KEY,
    name             TEXT               -- 'Credit card', 'Cash', 'No charge', 'Dispute', 'Unknown', 'Voided trip'
);

CREATE TABLE trips (
    trip_id               INTEGER PRIMARY KEY,
    vendor_id             INTEGER,      -- TPEP provider code
    pickup_datetime       TEXT,         -- 'YYYY-MM-DD HH:MM:SS', local time
    dropoff_datetime      TEXT,         -- 'YYYY-MM-DD HH:MM:SS', local time
    passenger_count       INTEGER,
    trip_distance         REAL,         -- miles
    rate_code_id          INTEGER,      -- 1 standard, 2 JFK, 3 Newark, 4 Nassau/Westchester, 5 negotiated, 6 group ride, 99 unknown
    pu_location_id        INTEGER REFERENCES zones(location_id),  -- pickup zone
    do_location_id        INTEGER REFERENCES zones(location_id),  -- dropoff zone
    payment_type          INTEGER REFERENCES payment_types(payment_type_id),
    fare_amount           REAL,         -- metered fare, USD
    tip_amount            REAL,         -- USD; cash tips are not recorded
    tolls_amount          REAL,         -- USD
    total_amount          REAL,         -- USD, total charged (excludes cash tips)
    congestion_surcharge  REAL,         -- USD
    airport_fee           REAL          -- USD
);

CREATE INDEX idx_trips_pu ON trips(pu_location_id);
CREATE INDEX idx_trips_do ON trips(do_location_id);
CREATE INDEX idx_trips_pickup ON trips(pickup_datetime);
"""

TRIP_COLS = [
    "trip_id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
    "trip_distance", "rate_code_id", "pu_location_id", "do_location_id", "payment_type",
    "fare_amount", "tip_amount", "tolls_amount", "total_amount", "congestion_surcharge",
    "airport_fee",
]
INT_COLS = {"vendor_id", "passenger_count", "rate_code_id", "pu_location_id",
            "do_location_id", "payment_type"}
MONEY_COLS = {"fare_amount", "tip_amount", "tolls_amount", "total_amount",
              "congestion_surcharge", "airport_fee"}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"using cached {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "taxi-sql-bench/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def load_trips(path: Path) -> pa.Table:
    by_lower = {name.lower(): name for name in pq.read_schema(path).names}
    missing = [c for c in REQUIRED if c not in by_lower]
    if missing:
        sys.exit(f"parquet is missing expected columns: {missing}")
    wanted = {**REQUIRED, **{k: v for k, v in OPTIONAL.items() if k in by_lower}}
    table = pq.read_table(path, columns=[by_lower[c] for c in wanted])
    table = table.rename_columns([wanted[n.lower()] for n in table.column_names])
    for col in OPTIONAL.values():
        if col not in table.column_names:
            table = table.append_column(col, pa.nulls(table.num_rows, pa.float64()))
    return table


def clean(table: pa.Table, year: int, month: int) -> pa.Table:
    pu, do = table["pickup_datetime"], table["dropoff_datetime"]
    start = dt.datetime(year, month, 1)
    end = dt.datetime(year + month // 12, month % 12 + 1, 1)
    ts = pu.type
    dur_s = pc.divide(
        pc.cast(pc.subtract(do, pu), pa.int64()),
        {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[ts.unit],
    )
    dist = table["trip_distance"]

    def between(col, lo, hi):
        return pc.and_(pc.greater_equal(table[col], lo), pc.less_equal(table[col], hi))

    conds = [
        pc.greater_equal(pu, pa.scalar(start, ts)),
        pc.less(pu, pa.scalar(end, ts)),
        pc.greater_equal(dur_s, 60),            # at least a minute
        pc.less_equal(dur_s, 6 * 3600),         # at most six hours
        pc.greater(dist, 0),
        pc.less(dist, 200),
        pc.less_equal(pc.multiply(dist, 3600), pc.multiply(dur_s, 100)),  # <= 100 mph
        pc.greater_equal(table["fare_amount"], 0),
        pc.greater_equal(table["tip_amount"], 0),
        pc.greater_equal(table["total_amount"], 0),
        between("pu_location_id", 1, 263),      # 264/265 are unknown / outside NYC
        between("do_location_id", 1, 263),
        between("payment_type", 1, 6),
        pc.is_valid(table["passenger_count"]),
    ]
    mask = conds[0]
    for c in conds[1:]:
        mask = pc.and_(mask, c)
    return table.filter(pc.fill_null(mask, False))


def to_row(i: int, rec: dict) -> tuple:
    out = []
    for col in TRIP_COLS:
        if col == "trip_id":
            out.append(i)
            continue
        v = rec[col]
        if v is None:
            out.append(None)
        elif col in ("pickup_datetime", "dropoff_datetime"):
            out.append(v.strftime("%Y-%m-%d %H:%M:%S"))
        elif col in INT_COLS:
            out.append(int(v))
        elif col in MONEY_COLS:
            out.append(round(float(v), 2))
        else:
            out.append(float(v))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", default="2024-01", help="YYYY-MM of the TLC file (default 2024-01)")
    ap.add_argument("--rows", type=int, default=200_000, help="number of trips to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/taxi.db"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    year, month = map(int, args.month.split("-"))
    trip_file = download(TRIP_URL.format(month=args.month), args.raw_dir / f"yellow_tripdata_{args.month}.parquet")
    zone_file = download(ZONE_URL, args.raw_dir / "taxi_zone_lookup.csv")

    raw = load_trips(trip_file)
    table = clean(raw, year, month)
    print(f"{raw.num_rows:,} raw rows -> {table.num_rows:,} after cleaning")
    if table.num_rows < args.rows:
        sys.exit(f"only {table.num_rows:,} clean rows, fewer than --rows {args.rows:,}")

    idx = sorted(random.Random(args.seed).sample(range(table.num_rows), args.rows))
    sample = table.take(pa.array(idx))
    sample = sample.take(pc.sort_indices(sample, sort_keys=[("pickup_datetime", "ascending")]))
    rows = [to_row(i, rec) for i, rec in enumerate(sample.to_pylist(), start=1)]

    with open(zone_file, newline="", encoding="utf-8") as f:
        zones = [(int(r["LocationID"]), r["Borough"], r["Zone"], r["service_zone"])
                 for r in csv.DictReader(f)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    conn.executescript(DDL)
    conn.executemany("INSERT INTO zones VALUES (?, ?, ?, ?)", zones)
    conn.executemany("INSERT INTO payment_types VALUES (?, ?)", PAYMENT_TYPES)
    conn.executemany(f"INSERT INTO trips VALUES ({', '.join('?' * len(TRIP_COLS))})", rows)
    conn.commit()
    conn.execute("ANALYZE")
    schema = [r[0] for r in conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
    conn.close()
    tmp.replace(args.out)

    schema_path = args.out.with_name("schema.sql")
    schema_path.write_text(";\n\n".join(schema) + ";\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(rows):,} trips, {len(zones)} zones) and {schema_path}")


if __name__ == "__main__":
    main()
