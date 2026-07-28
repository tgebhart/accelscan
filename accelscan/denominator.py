"""Stage 4a: paper-level analytic table (denominators + numerators + citations).

Three products, all keyed on corpusid:

`denominator` — one row per S2ORC full-text paper (the population): year,
primary/all fields, is_candidate. This is the denominator for every prevalence
share, so it must cover papers with NO accelerator mention.

`paper_flags` — accelerator numerators per paper: any ok mention, used-in-this-
work, model-specific vs generic-only, manufacturer set, newest/oldest model
vintage. Built from mentions + registry canonicalization.

`citations` — forward-citation and disruption outcomes for accelerator papers,
read from the lab's precomputed outcomes table (`i_k` = cumulative citations k
years after publication, so `i_5` is the 5-year window; `cd_k` = CD/disruption
index). Papers published after CITATION_CUTOFF have incomplete windows and are
flagged rather than dropped.

Big tables are handled part-by-part (the papers and citations tables are far
too large to materialize whole — see `_by_part`).

  python -m accelscan.denominator                    # all three -> S3
  python -m accelscan.denominator --skip-citations   # faster
"""

import argparse
import io
import sys

import polars as pl

from accelscan.config import BUCKET, OUT_PREFIX, PAPERS_PREFIX

# Precomputed citation/disruption outcomes (lab table): i_k = cumulative
# citations k years post-publication; cd_k = CD (disruption) index.
OUTCOMES_PATH = ('/projects/standard/funkr/gebhart/projects/rfunklab/novelty_corp/data'
                 '/semanticscholar/processed/citation_outcomes'
                 '/semanticscholar_outcomes.parquet.gz')
CITATION_WINDOW = 5
CITATION_CUTOFF = 2020        # focal papers <= this year have a complete 5-yr window
USED = 'used-in-this-work'


def _by_part(prefix: str, cols: list[str], ids: pl.DataFrame | None,
             on: str, so: dict, client, transform=None,
             label: str = '') -> pl.DataFrame:
    """Read one parquet part at a time, optionally semi-join to `ids` on `on`,
    apply `transform`, accumulate. Bounded memory regardless of table size."""
    from accelscan.s3 import list_keys
    keys = list_keys(prefix, suffix='.parquet', client=client)
    parts = []
    for i, k in enumerate(keys):
        df = pl.read_parquet(f's3://{BUCKET}/{k}', storage_options=so, columns=cols)
        if ids is not None:
            df = df.join(ids, on=on, how='semi')
        if transform is not None:
            df = transform(df)
        if df.height:
            parts.append(df)
        if label and ((i + 1) % 60 == 0 or i + 1 == len(keys)):
            print(f'  [{label}] {i+1}/{len(keys)} parts', file=sys.stderr)
    return pl.concat(parts) if parts else pl.DataFrame(schema={c: pl.Null for c in cols})


def build_denominator(so: dict, client) -> pl.DataFrame:
    """Full-text population with year + field. One row per paper."""
    inv = pl.read_parquet(
        f's3://{BUCKET}/{OUT_PREFIX}/inventory/parts/*.parquet', storage_options=so,
        columns=['corpusid', 'has_body', 'is_candidate', 'body_chars'])
    inv = inv.unique('corpusid')
    print(f'inventory: {inv.height:,} full-text papers', file=sys.stderr)

    ids = inv.select('corpusid')
    meta = _by_part(
        PAPERS_PREFIX, ['corpusid', 'year', 's2fieldsofstudy', 'citationcount'],
        ids, 'corpusid', so, client,
        transform=lambda df: df.with_columns(
            field=pl.col('s2fieldsofstudy').list.first().struct.field('category'),
            n_fields=pl.col('s2fieldsofstudy').list.len(),
        ).select('corpusid', 'year', 'field', 'n_fields', 'citationcount'),
        label='papers')
    return inv.join(meta, on='corpusid', how='left')


def build_paper_flags(so: dict, model_tag: str, prompt_version: str) -> pl.DataFrame:
    """Accelerator numerators per paper (from mentions + canonicalization)."""
    from accelscan.normalize import canonicalize_column
    from accelscan.registry import load_registry
    from accelscan.s3 import mentions_glob
    reg = load_registry()

    m = (pl.scan_parquet(mentions_glob(model_tag, prompt_version), storage_options=so)
         .filter(pl.col('status') == 'ok')
         .select('corpusid', 'model_normalized', 'manufacturer',
                 'accelerator_subtype', 'usage_context', 'device_count')
         .collect())
    m = canonicalize_column(m, reg)
    rel = {mid: int(mm.release[:4]) for mid, mm in reg.models.items() if mm.release}
    m = m.with_columns(
        release_year=pl.col('canonical_model').replace_strict(rel, default=None,
                                                             return_dtype=pl.Int32),
        is_model=(pl.col('canonical_kind') == 'model'),
        is_used=(pl.col('usage_context') == USED))

    return m.group_by('corpusid').agg(
        accel_any=pl.lit(True),
        accel_used=pl.col('is_used').any(),
        model_specific=pl.col('is_model').any(),
        model_specific_used=(pl.col('is_model') & pl.col('is_used')).any(),
        n_mentions=pl.len(),
        n_models=pl.col('canonical_model').filter(pl.col('is_model')).n_unique(),
        manufacturers=pl.col('manufacturer').filter(pl.col('manufacturer').is_in(
            ['nvidia', 'amd', 'intel', 'google', 'apple', 'huawei', 'graphcore',
             'cerebras', 'amazon'])).unique(),
        subtypes=pl.col('accelerator_subtype').unique(),
        newest_model_year=pl.col('release_year').filter(pl.col('is_used')).max(),
        oldest_model_year=pl.col('release_year').filter(pl.col('is_used')).min(),
        max_device_count=pl.col('device_count').filter(pl.col('is_used')).max(),
        usage_contexts=pl.col('usage_context').unique(),
    ).with_columns(generic_only=~pl.col('model_specific'))


def build_citations(so: dict, client, focal: pl.DataFrame) -> pl.DataFrame:
    """Citation + disruption outcomes for `focal` (corpusid, year).

    Uses the lab's precomputed outcomes table rather than re-deriving windows
    from the 361-part edge list: `i_k` is cumulative citations k years post
    publication (i_5 = the 5-year window used in the paper) and `cd_k` is the
    CD/disruption index. Left-joined, so papers absent from the outcomes build
    (mostly the newest ones) stay present with nulls and are flagged.
    """
    focal = focal.filter(pl.col('year').is_not_null()).select('corpusid', 'year')
    keep = ['i_1', 'i_3', 'i_5', 'i_10', 'cd_5', 'cd_10', 'bcites_300']
    oc = (pl.scan_parquet(OUTCOMES_PATH)
          .select(['record_id'] + keep)
          .with_columns(corpusid=pl.col('record_id').cast(pl.Int64, strict=False))
          .drop('record_id')
          .join(focal.lazy().select('corpusid'), on='corpusid', how='semi')
          .collect())
    print(f'outcomes matched: {oc.height:,} of {focal.height:,} focal papers '
          f'({100*oc.height/max(focal.height,1):.1f}%)', file=sys.stderr)
    return (focal.join(oc, on='corpusid', how='left')
            .rename({'i_5': 'citations_5y', 'cd_5': 'disruption_5y'})
            .with_columns(
                window_complete=pl.col('year') <= CITATION_CUTOFF,
                has_outcomes=pl.col('citations_5y').is_not_null()))


def _put(client, df: pl.DataFrame, key: str) -> None:
    buf = io.BytesIO(); df.write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f'wrote s3://{BUCKET}/{key} ({df.height:,} rows)', file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-tag', default='qwen3-14b')
    ap.add_argument('--prompt-version', default='p1')
    ap.add_argument('--skip-citations', action='store_true')
    ap.add_argument('--local-out')
    args = ap.parse_args()

    from accelscan.registry import load_registry
    from accelscan.s3 import make_s3_client, storage_options
    so, client = storage_options(), make_s3_client()
    rv = load_registry().version
    base = f'{OUT_PREFIX}/analytic/{rv}/{args.prompt_version}/{args.model_tag}'

    den = build_denominator(so, client)
    flags = build_paper_flags(so, args.model_tag, args.prompt_version)
    print(f'denominator {den.height:,} | flagged accelerator papers {flags.height:,}',
          file=sys.stderr)

    outputs = {'denominator': den, 'paper_flags': flags}
    if not args.skip_citations:
        focal = (den.join(flags.select('corpusid'), on='corpusid', how='semi')
                 .select('corpusid', 'year'))
        outputs['citations'] = build_citations(so, client, focal)

    if args.local_out:
        from pathlib import Path
        d = Path(args.local_out); d.mkdir(parents=True, exist_ok=True)
        for name, df in outputs.items():
            df.write_parquet(d / f'{name}.parquet')
            print(f'wrote {d}/{name}.parquet ({df.height:,})', file=sys.stderr)
    else:
        for name, df in outputs.items():
            _put(client, df, f'{base}/{name}.parquet')


if __name__ == '__main__':
    main()
