"""Aggregate citation outcome CSV files into a single wide-format parquet.

Reads all *_*.csv.gz files from a citation_outcomes directory, joins them
on record_id, and writes a single parquet file with one row per document
and a column for each outcome metric.

This is the step that produces the table `accelscan.denominator` reads
(`config.OUTCOMES_PATH`): `outcomes.py` writes one CSV per metric and horizon, and
nothing downstream knows how to join those.

Usage:
    python -m accelscan.semanticscholar.aggregate \
        --outcomes_dir data/semanticscholar/2026-02-10/processed/citation_outcomes \
        --out data/semanticscholar/2026-02-10/processed/citation_outcomes/semanticscholar_outcomes.parquet.gz
"""

import argparse
import os
import logging

import polars as pl

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate citation outcome CSVs into a single wide parquet."
    )
    parser.add_argument(
        "--outcomes_dir",
        type=str,
        required=True,
        help="Directory containing *_*.csv.gz outcome files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for the aggregated parquet file.",
    )
    args = parser.parse_args()

    outcomes_dir = args.outcomes_dir
    if not os.path.isdir(outcomes_dir):
        logging.error(f"Directory {outcomes_dir} does not exist.")
        return

    files = sorted(
        f for f in os.listdir(outcomes_dir)
        if f.endswith(".csv.gz")
    )
    if not files:
        logging.error(f"No .csv.gz files found in {outcomes_dir}.")
        return

    logging.info(f"Found {len(files)} outcome files in {outcomes_dir}")

    result = None
    for fname in files:
        fpath = os.path.join(outcomes_dir, fname)
        logging.info(f"  Loading {fname}...")
        df = pl.read_csv(fpath, schema_overrides={"record_id": pl.Utf8})

        # Ensure record_id is String for consistent joining
        if "record_id" not in df.columns:
            logging.warning(f"  {fname} has no 'record_id' column, skipping.")
            continue
        if df["record_id"].dtype != pl.Utf8:
            df = df.with_columns(pl.col("record_id").cast(pl.Utf8))

        logging.info(
            f"    {df.height} rows, columns: {[c for c in df.columns if c != 'record_id']}"
        )

        if result is None:
            result = df
        else:
            result = result.join(df, on="record_id", how="full", coalesce=True)

    if result is None:
        logging.error("No valid data loaded. Exiting.")
        return

    # Sort for deterministic output
    result = result.sort("record_id")

    logging.info(
        f"Final table: {result.height} rows, {len(result.columns)} columns"
    )
    logging.info(f"Columns: {result.columns}")

    # Write output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result.write_parquet(args.out, compression="gzip")
    logging.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
