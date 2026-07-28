"""Stage 3.5: reported compute capacity per paper.

Joins per-device registry specs (scraped, never hardcoded — see
scripts/build_registry.py) to canonicalized used-in-this-work mentions and
sums device_count x spec:

    F_i^(p) = sum_g  c_ig * flops_p(g)      p in {fp32, fp64, fp16_tensor}
    M_i     = sum_g  c_ig * vram(g)
    W_i     = sum_g  c_ig * tdp(g)

**Interpretation (must accompany every figure): reported NAMEPLATE PEAK
capacity, not utilized FLOPs.** It measures the compute a paper had access to
and chose to report — an upper bound on work performed.

Device-count handling: `device_count` is winsorized (extraction junk reaches
13e6) at a per-year p99.9 cap and a hard cap; nulls become 1 under the primary
`floor1` policy (a floor), or drop the mention under `explicit` (robustness).
Papers whose used models all lack a spec on an axis get null for that axis plus
a `spec_missing_*` flag — never zero-filled.

  python -m accelscan.compute                     # -> S3 paper-capacity table
  python -m accelscan.compute --local-out output/compute_smoke --limit 50000
"""

import argparse
import io
import sys

import polars as pl

from accelscan.config import BUCKET, OUT_PREFIX
from accelscan.normalize import canonicalize_column
from accelscan.registry import CompiledRegistry, load_registry

HARD_DEVICE_CAP = 65_536      # beyond any real single-paper deployment
WINSOR_QUANTILE = 0.999
USED = 'used-in-this-work'
AXES = {'fp32': 'fp32_gflops', 'fp64': 'fp64_gflops', 'tensor': 'fp16_tensor_gflops'}


def spec_frame(reg: CompiledRegistry) -> pl.DataFrame:
    """Registry -> per-model spec table (the ONLY source of spec numbers)."""
    rows = [{
        'canonical_model': m.id,
        'display': m.display,
        'manufacturer': m.manufacturer,
        'segment': m.segment,
        'architecture': m.architecture,
        'release_year': int(m.release[:4]) if m.release else None,
        'fp32_gflops': m.fp32_gflops,
        'fp64_gflops': m.fp64_gflops,
        'fp16_tensor_gflops': m.fp16_tensor_gflops,
        'vram_gb': m.vram_gb,
        'tdp_w': m.tdp_w,
    } for m in reg.models.values() if m.kind == 'model']
    return pl.DataFrame(rows, schema={
        'canonical_model': pl.Utf8, 'display': pl.Utf8, 'manufacturer': pl.Utf8,
        'segment': pl.Utf8, 'architecture': pl.Utf8, 'release_year': pl.Int32,
        'fp32_gflops': pl.Float64, 'fp64_gflops': pl.Float64,
        'fp16_tensor_gflops': pl.Float64, 'vram_gb': pl.Float64, 'tdp_w': pl.Float64})


def winsorize_counts(mentions: pl.DataFrame, year_col: str = 'year',
                     policy: str = 'floor1') -> pl.DataFrame:
    """Add `n_dev`: device_count winsorized within year, per count policy."""
    if policy == 'explicit':
        mentions = mentions.filter(pl.col('device_count_basis') == 'explicit')
    caps = (mentions.filter(pl.col('device_count').is_not_null())
            .group_by(year_col)
            .agg(cap=pl.col('device_count').quantile(WINSOR_QUANTILE))
            .with_columns(cap=pl.col('cap').clip(1, HARD_DEVICE_CAP)))
    out = mentions.join(caps, on=year_col, how='left').with_columns(
        n_dev=pl.when(pl.col('device_count').is_null())
        .then(pl.lit(1.0) if policy == 'floor1' else None)
        .otherwise(pl.min_horizontal(
            pl.col('device_count').cast(pl.Float64),
            pl.col('cap').fill_null(HARD_DEVICE_CAP),
            pl.lit(float(HARD_DEVICE_CAP)))))
    return out.drop('cap')


def paper_capacity(mentions: pl.DataFrame, specs: pl.DataFrame) -> pl.DataFrame:
    """Mention-level (already canonicalized, usage-filtered, winsorized)
    -> one row per paper with capacity sums and spec-coverage flags."""
    d = mentions.join(specs, on='canonical_model', how='left')
    # dedup: one row per (paper, model) using the max reported device count,
    # so a model repeated across passages is not double-counted
    d = (d.group_by('corpusid', 'canonical_model')
         .agg(pl.col('n_dev').max(),
              pl.col('memory_gb').max().alias('stated_memory_gb'),
              *[pl.col(c).first() for c in
                ['fp32_gflops', 'fp64_gflops', 'fp16_tensor_gflops', 'vram_gb',
                 'tdp_w', 'release_year', 'manufacturer', 'segment', 'display']]))

    # stated per-device memory resolves variants (A100 40 vs 80 GB) when present
    d = d.with_columns(vram_eff=pl.coalesce('stated_memory_gb', 'vram_gb'))

    exprs = []
    for axis, col in AXES.items():
        exprs += [
            (pl.col('n_dev') * pl.col(col)).sum().alias(f'reported_{axis}_gflops'),
            pl.col(col).is_null().all().alias(f'spec_missing_{axis}'),
        ]
    agg = d.group_by('corpusid').agg(
        *exprs,
        reported_vram_gb=(pl.col('n_dev') * pl.col('vram_eff')).sum(),
        reported_tdp_w=(pl.col('n_dev') * pl.col('tdp_w')).sum(),
        spec_missing_vram=pl.col('vram_eff').is_null().all(),
        max_devices=pl.col('n_dev').max(),
        total_devices=pl.col('n_dev').sum(),
        n_models=pl.col('canonical_model').n_unique(),
        newest_release_year=pl.col('release_year').max(),
        oldest_release_year=pl.col('release_year').min(),
        any_datacenter=(pl.col('segment') == 'datacenter').any(),
        any_consumer=(pl.col('segment') == 'consumer').any(),
        tensor_capable=pl.col('fp16_tensor_gflops').is_not_null().any(),
    )
    # null out an axis when no used model on that axis had a spec
    for axis in AXES:
        agg = agg.with_columns(
            pl.when(pl.col(f'spec_missing_{axis}')).then(None)
            .otherwise(pl.col(f'reported_{axis}_gflops')).alias(f'reported_{axis}_gflops'))
    return agg.with_columns(
        pl.when(pl.col('spec_missing_vram')).then(None)
        .otherwise(pl.col('reported_vram_gb')).alias('reported_vram_gb'))


def build(model_tag: str, prompt_version: str, count_policy: str,
          limit: int | None = None,
          mentions_version: str | None = None) -> pl.DataFrame:
    from accelscan.config import PAPERS_PREFIX
    from accelscan.s3 import mentions_glob, storage_options
    reg = load_registry()
    so = storage_options()

    # mentions are namespaced by the registry version at EXTRACTION time,
    # which may lag the current registry (spec/model additions don't require
    # re-running the LLM) — discover it rather than assuming reg.version
    glob = mentions_glob(model_tag, prompt_version, mentions_version)
    m = (pl.scan_parquet(glob, storage_options=so)
         .filter((pl.col('status') == 'ok') & (pl.col('usage_context') == USED))
         .select('corpusid', 'model_normalized', 'device_count',
                 'device_count_basis', 'memory_gb')
         .collect())
    if limit:
        m = m.head(limit)
    m = canonicalize_column(m, reg)
    m = m.filter(pl.col('canonical_model').is_not_null())

    years = (pl.scan_parquet(f's3://{BUCKET}/{PAPERS_PREFIX}*.parquet', storage_options=so)
             .select('corpusid', 'year')
             .filter(pl.col('corpusid').is_in(m['corpusid'].unique().to_list()))
             .collect())
    m = m.join(years, on='corpusid', how='left')
    m = winsorize_counts(m, policy=count_policy)
    m = m.filter(pl.col('n_dev').is_not_null())

    cap = paper_capacity(m, spec_frame(reg))
    ext_v = glob.split('/mentions/')[1].split('/')[0]
    return cap.join(years, on='corpusid', how='left').with_columns(
        count_policy=pl.lit(count_policy),
        spec_registry_version=pl.lit(reg.version),      # specs used here
        mentions_registry_version=pl.lit(ext_v))        # extraction-time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-tag', default='qwen3-14b')
    ap.add_argument('--prompt-version', default='p1')
    ap.add_argument('--count-policy', default='floor1', choices=['floor1', 'explicit'])
    ap.add_argument('--limit', type=int, help='dev: cap mention rows')
    ap.add_argument('--mentions-version', help='registry version of the mentions to read '
                    '(default: highest available on S3)')
    ap.add_argument('--local-out', help='dev: write parquet locally instead of S3')
    args = ap.parse_args()

    cap = build(args.model_tag, args.prompt_version, args.count_policy,
                args.limit, args.mentions_version)
    n = cap.height
    print(f'{n:,} papers with reported capacity', file=sys.stderr)
    for axis in AXES:
        have = cap.filter(~pl.col(f'spec_missing_{axis}')).height
        print(f'  {axis}: {have:,} papers with spec ({100*have/max(n,1):.1f}%)', file=sys.stderr)

    if args.local_out:
        from pathlib import Path
        out = Path(args.local_out); out.mkdir(parents=True, exist_ok=True)
        cap.write_parquet(out / f'paper_capacity_{args.count_policy}.parquet')
        print(f'wrote {out}/paper_capacity_{args.count_policy}.parquet', file=sys.stderr)
    else:
        from accelscan.registry import load_registry as _lr
        from accelscan.s3 import make_s3_client
        rv = _lr().version
        key = (f'{OUT_PREFIX}/capacity/{rv}/{args.prompt_version}/{args.model_tag}'
               f'/paper_capacity_{args.count_policy}.parquet')
        buf = io.BytesIO(); cap.write_parquet(buf)
        make_s3_client().put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        print(f'wrote s3://{BUCKET}/{key}', file=sys.stderr)


if __name__ == '__main__':
    main()
