"""Vectorized `i_k` (forward citations) and `bcites_k` over a huge edge list.

`outcomes.py` is the lab's general CD-index implementation: it loads the whole edge
list into pandas, builds `defaultdict(set)` indexes, and walks years in Python with
a multiprocessing pool. That is fine at 10^8 edges and hopeless at the 5.7x10^9 of
the 2026-07-28 Semantic Scholar release -- `drop_duplicates` alone never finished,
and polars cannot even hold the filtered 4.9x10^9 rows in one frame (its row index
is 32-bit).

For `i` and `bcites` none of that machinery is needed, because both reduce to a
group-by once the year window is expressed as an arithmetic condition:

    i_k(f)      = |{p : p cites f, citing_year(p) <= pubyear(f) + k}|
    bcites_k(f) = |{r : f cites r}|

`read_citations_from_file` already guarantees `citing_year >= cited_year` for every
edge, and for an edge pointing at f, `cited_year == pubyear(f)`. So the window
condition is exactly `citing_year - cited_year <= k`, and `i_k` is one filtered
count per cited id. `bcites_k` is an out-degree: every edge out of f has
`citing_year == pubyear(f)`, so the horizon never binds and the value is the same
for all k (which is why the lab's `bcites_300` is horizon-agnostic in practice).

Scale is handled by hash-partitioning on the *cited* id. All copies of an edge
pointing at f land in the same bucket, so dedup and counting are exact within a
bucket and never need a global frame. Peak memory is one bucket.

`cd_k` genuinely needs set intersections over a 2-hop neighbourhood and is NOT
implemented here -- use `outcomes.py` for it, on a focal-scoped subset.

    python -m accelscan.semanticscholar.fast_outcomes
    python -m accelscan.semanticscholar.fast_outcomes --time_horizons 2 3 5 10 --cd_types i bcites
"""

import argparse
import glob
import os
import shutil
import sys

import polars as pl
from tqdm import tqdm

from accelscan.semanticscholar.outcomes import (CITATION_LOC, CITED_VAR,
                                                CITING_VAR, DATE_VAR_CITED,
                                                DATE_VAR_CITING, END_YEAR,
                                                SAVE_LOC)

# 4.9e9 filtered edges / 16 ~= 3e8 rows (~7 GB) per bucket. Fewer, larger buckets
# beat many small ones: pass 1 writes one file per (input part, bucket), so 64
# buckets would mean 25,792 files on a shared filesystem.
N_BUCKETS = 16
HORIZONS = (2, 3, 5)


def _filtered(path: str, end_year: int) -> pl.DataFrame:
    """One input part, with exactly the filters read_citations_from_file applies.

    Those are: both years known, no self-citation, no citation from the past, and
    citing year within the window. Kept here rather than downstream so the volume
    drops before anything is written.
    """
    return (pl.read_parquet(path,
                            columns=[CITING_VAR, CITED_VAR,
                                     DATE_VAR_CITING, DATE_VAR_CITED])
            .drop_nulls([DATE_VAR_CITING, DATE_VAR_CITED])
            .filter((pl.col(CITING_VAR) != pl.col(CITED_VAR))
                    & (pl.col(DATE_VAR_CITING) >= pl.col(DATE_VAR_CITED))
                    & (pl.col(DATE_VAR_CITING) <= end_year))
            .with_columns(pl.col(DATE_VAR_CITING).cast(pl.Int32),
                          pl.col(DATE_VAR_CITED).cast(pl.Int32)))


def bucket_edges(citation_loc: str, work_dir: str, end_year: int = END_YEAR,
                 n_buckets: int = N_BUCKETS) -> tuple[int, int]:
    """Pass 1: filter every part and split it by hash(cited id) into buckets.

    Bucketing on the cited id is what makes the later dedup and count exact
    without a global frame: every edge pointing at a given paper is in one bucket.
    """
    paths = ([citation_loc] if citation_loc.endswith('.parquet')
             else sorted(glob.glob(os.path.join(citation_loc, '*.parquet'))))
    if not paths:
        raise SystemExit(f'no parquet parts under {citation_loc}')
    for b in range(n_buckets):
        os.makedirs(os.path.join(work_dir, f'b{b:03d}'), exist_ok=True)

    seen = kept = 0
    for path in tqdm(paths, desc='bucketing'):
        d = pl.read_parquet(path, columns=[CITING_VAR]).height
        seen += d
        f = _filtered(path, end_year)
        kept += f.height
        f = f.with_columns(_b=pl.col(CITED_VAR).hash(seed=0) % n_buckets)
        for (b,), part in f.group_by('_b'):
            part.drop('_b').write_parquet(
                os.path.join(work_dir, f'b{int(b):03d}',
                             os.path.basename(path)))
        del f
    print(f'{seen:,} edges -> {kept:,} after filters ({100 * kept / seen:.1f}%), '
          f'in {n_buckets} buckets', file=sys.stderr)
    return seen, kept


def count_outcomes(work_dir: str, horizons=HORIZONS,
                   n_buckets: int = N_BUCKETS) -> dict[str, pl.DataFrame]:
    """Pass 2: per bucket, dedup and count. Returns {column_name: (id, value)}.

    Also collects each paper's publication year from both endpoint roles, which is
    what defines the universe the lab's loop iterates (`fids` = papers that cite in
    a year, `bids` = papers cited that year which never cite).
    """
    i_parts = {f'i_{k}': [] for k in horizons}
    b_parts, years_parts = [], []
    for b in tqdm(range(n_buckets), desc='counting'):
        files = glob.glob(os.path.join(work_dir, f'b{b:03d}', '*.parquet'))
        if not files:
            continue
        e = pl.concat([pl.read_parquet(f) for f in files]).unique(
            subset=[CITING_VAR, CITED_VAR])
        lag = (pl.col(DATE_VAR_CITING) - pl.col(DATE_VAR_CITED))
        for k in horizons:
            i_parts[f'i_{k}'].append(
                e.filter(lag <= k).group_by(CITED_VAR).len()
                .rename({CITED_VAR: 'record_id', 'len': f'i_{k}'}))
        b_parts.append(e.group_by(CITING_VAR).len()
                       .rename({CITING_VAR: 'record_id', 'len': 'bcites'}))
        years_parts.append(pl.concat([
            e.select(record_id=pl.col(CITING_VAR), year=pl.col(DATE_VAR_CITING)),
            e.select(record_id=pl.col(CITED_VAR), year=pl.col(DATE_VAR_CITED)),
        ]).unique())
        del e

    out = {}
    for col, parts in i_parts.items():
        # buckets partition on cited id, so a paper appears in exactly one part
        out[col] = pl.concat(parts) if parts else pl.DataFrame(
            schema={'record_id': pl.Int64, col: pl.UInt32})
    # citing ids are NOT bucket-local (a paper cites many papers), so sum
    out['bcites'] = (pl.concat(b_parts).group_by('record_id')
                     .agg(bcites=pl.col('bcites').sum())
                     if b_parts else
                     pl.DataFrame(schema={'record_id': pl.Int64,
                                          'bcites': pl.UInt32}))
    out['_universe'] = (pl.concat(years_parts).group_by('record_id')
                        .agg(year=pl.col('year').min())
                        if years_parts else
                        pl.DataFrame(schema={'record_id': pl.Int64,
                                             'year': pl.Int32}))
    return out


def write_csvs(res: dict[str, pl.DataFrame], save_loc: str,
               horizons=HORIZONS, cd_types=('i',)) -> list[str]:
    """Write one CSV per metric, in the schema `aggregate.py` expects.

    A paper in the universe with no qualifying citation gets 0, not a missing row:
    the lab's loop calls `cited_dict.get(fix, set())` for every focal id, so an
    uncited paper is a genuine zero.
    """
    os.makedirs(save_loc, exist_ok=True)
    universe = res['_universe'].select('record_id')
    written = []
    cols = ([f'i_{k}' for k in horizons] if 'i' in cd_types else []) \
        + (['bcites'] if 'bcites' in cd_types else [])
    for col in cols:
        df = (universe.join(res[col], on='record_id', how='left')
              .with_columns(pl.col(col).fill_null(0).cast(pl.Int64))
              .sort('record_id'))
        path = os.path.join(save_loc, f'{col}.csv.gz')
        # aggregate.py globs *.csv.gz and polars' read_csv decompresses it
        df.write_csv(path, compression='gzip')
        print(f'wrote {path} ({df.height:,} papers, '
              f'{df[col].sum():,} total)', file=sys.stderr)
        written.append(path)
    return written


def write_parquet(res: dict[str, pl.DataFrame], out_path: str,
                  horizons=HORIZONS, cd_types=('i',)) -> str:
    """Write ONE table with every metric as a column, joined on record_id.

    This is what `aggregate.py` produces from the per-metric CSVs, so writing it
    here skips that round trip -- which is not a micro-optimization at this scale:
    the universe is ~2.4e8 papers, so the CSV path means writing ~5 x 600 MB of
    gzip and reading it all back to do joins we can do in memory.
    """
    df = res['_universe'].select('record_id')
    cols = ([f'i_{k}' for k in horizons] if 'i' in cd_types else []) \
        + (['bcites'] if 'bcites' in cd_types else [])
    for col in cols:
        df = (df.join(res[col], on='record_id', how='left')
              .with_columns(pl.col(col).fill_null(0).cast(pl.Int64)))
    df = df.sort('record_id')
    # A single-horizon table is almost always a mis-invocation (see the array guard
    # in main), and it silently replaces a complete one. Say so rather than write it.
    if len(cols) < 2:
        print(f'WARNING: writing only {cols} -- if you meant all horizons, drop '
              f'--time_horizons and do not submit this as an array',
              file=sys.stderr)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.write_parquet(out_path, compression='gzip')
    print(f'wrote {out_path}: {df.height:,} papers x {len(cols)} metrics '
          + ' | '.join(f'{c} sum={df[c].sum():,}' for c in cols), file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--citation_loc', default=CITATION_LOC)
    ap.add_argument('--save_loc', default=SAVE_LOC)
    ap.add_argument('--out', default=None,
                    help='single joined parquet (default: <save_loc>/'
                         'semanticscholar_outcomes.parquet.gz); this is what '
                         'denominator.py reads, so aggregate.py is not needed')
    ap.add_argument('--csvs', action='store_true',
                    help='also write per-metric CSVs, for aggregate.py')
    ap.add_argument('--cd_types', nargs='+', default=['i'],
                    choices=['i', 'bcites'])
    ap.add_argument('--time_horizons', nargs='+', type=int, default=list(HORIZONS))
    ap.add_argument('--end_year', type=int, default=END_YEAR)
    ap.add_argument('--n_buckets', type=int, default=N_BUCKETS)
    ap.add_argument('--work_dir', default=None,
                    help='scratch for bucketed edges (default: <save_loc>/_buckets)')
    ap.add_argument('--delete-buckets', action='store_true',
                    help='remove the bucket scratch when done (kept by default: '
                         'pass 1 costs ~13 min and is reusable via --reuse-buckets)')
    ap.add_argument('--reuse-buckets', action='store_true',
                    help='skip pass 1 and use an existing work_dir')
    args = ap.parse_args()

    # This computes EVERY horizon in one pass, so a SLURM array over horizons is
    # not just redundant, it is destructive: sibling tasks share `_buckets` (one
    # deletes it mid-read of another) and all write the same --out, so the last
    # finisher wins and the table ends up holding a single horizon. That is exactly
    # what happened on job 15097009: tasks 2/3/5 raced and left an i_3-only table.
    if os.environ.get('SLURM_ARRAY_TASK_ID') and not (args.out and args.work_dir):
        raise SystemExit(
            'refusing to run as a SLURM array task: this command already computes '
            f'all horizons ({", ".join(map(str, args.time_horizons))}) in one pass. '
            'Submit it as a single job (no --array), or if you really want parallel '
            'tasks, give each an explicit --out AND --work_dir.')

    work = args.work_dir or os.path.join(args.save_loc, '_buckets')
    if not args.reuse_buckets:
        bucket_edges(args.citation_loc, work, args.end_year, args.n_buckets)
    res = count_outcomes(work, tuple(args.time_horizons), args.n_buckets)
    out = args.out or os.path.join(args.save_loc,
                                   'semanticscholar_outcomes.parquet.gz')
    write_parquet(res, out, tuple(args.time_horizons), tuple(args.cd_types))
    if args.csvs:
        write_csvs(res, args.save_loc, tuple(args.time_horizons),
                   tuple(args.cd_types))
    if args.delete_buckets:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
