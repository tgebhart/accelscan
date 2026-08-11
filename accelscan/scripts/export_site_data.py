"""Export the public arXiv-results website's data bundle.

Two artifacts land under `--out` (default `site_data/`):

1. `charts/*.json` — the headline figures (capacity/trends/productivity),
   computed by `accelscan.webexport` against the same analytic/capacity tables
   the notebooks read. Same functions the notebooks call, so the website's
   numbers can never drift from the PDF figures. (Not every `webexport`
   function has a chart here -- the paper-reporting-extraction-funnel figure
   is PDF-only, dropped from the site as redundant with the drill-down page.)
2. `mentions/year=YYYY/part-0.parquet` (hive-partitioned) + `manifest.json` —
   every `status=='ok'` mention, canonicalized and joined to arXiv title/
   category metadata, for the paper-drill-down page's DuckDB-WASM queries.
   No `passage_text`: `evidence_quote` (the LLM's verbatim justifying span) is
   the drill-down's display text, which keeps the bundle two orders of
   magnitude smaller than shipping the surrounding passages.

    python -m accelscan.scripts.export_site_data --out site_data

Re-run this after each ~monthly bulk-release re-ingestion, then redeploy per
`docs/site_deploy.md`. Only reads from S3 — writes only to `--out` locally.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from accelscan.compute import spec_frame
from accelscan.meta import paper_fields
from accelscan.normalize import canonicalize_column
from accelscan.paths import (analytic_base, arxiv_metadata_key, capacity_key,
                            get_corpus, s3_uri)
from accelscan.registry import load_registry
from accelscan.s3 import mentions_glob, storage_options
from accelscan import webexport as we

# arXiv's group names are too long for legends/axis ticks in the notebooks; the
# website inherits the same short forms so labels agree everywhere.
FIELD_LABELS = {'Electrical Engineering and Systems Science': 'EE and Systems',
                'Quantitative Biology': 'Quant. Biology',
                'Quantitative Finance': 'Quant. Finance'}
OTHER, NOLAB, ALL = 'Other fields', 'No field label', 'All fields'


def short_fields(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col('field').replace(FIELD_LABELS))


def load_capacity(c, reg, so, prompt_version: str, model_tag: str, count_policy: str,
                  year_min: int, year_max: int, top_fields_n: int):
    """Mirrors `notebooks/arxiv/capacity.ipynb`'s load cell."""
    cap = pl.read_parquet(s3_uri(capacity_key(c, reg.version, prompt_version,
                                              model_tag, count_policy)),
                          storage_options=so)
    flag_ids = pl.read_parquet(
        f'{s3_uri(analytic_base(c, reg.version, prompt_version, model_tag))}/paper_flags.parquet',
        storage_options=so, columns=[c.key])
    need = pl.concat([cap.select(c.key), flag_ids]).unique().to_series().to_list()
    fields = short_fields(paper_fields(c, pl.DataFrame({c.key: need}), so)).select(c.key, 'field')
    cap = (cap.join(fields, on=c.key, how='left')
           .filter(pl.col('year').is_between(year_min, year_max)))
    top_fields = (flag_ids.join(fields, on=c.key, how='left')
                  .filter(pl.col('field').is_not_null())
                  .group_by('field').agg(n=pl.len()).sort('n', descending=True)
                  .head(top_fields_n)['field'].to_list())
    # Per-device counterparts: the frontier is a single-device spec. A handful
    # of rows (14 of 201,813 at v0.4.0) have total_devices == 0 -- a floored-
    # count artifact, not a real zero-device paper -- which would otherwise
    # divide 0/0 to NaN and poison every downstream sum() it touches.
    cap = cap.with_columns(
        per_device_fp32_gflops=pl.when(pl.col('total_devices') > 0)
            .then(pl.col('reported_fp32_gflops') / pl.col('total_devices')),
        per_device_vram_gb=pl.when(pl.col('total_devices') > 0)
            .then(pl.col('reported_vram_gb') / pl.col('total_devices')))
    return cap, top_fields


def load_papers(c, reg, so, prompt_version: str, model_tag: str,
                year_min: int, year_max: int, top_fields_n: int):
    """Mirrors `notebooks/arxiv/trends.ipynb`'s load cell."""
    base = s3_uri(analytic_base(c, reg.version, prompt_version, model_tag))
    den = short_fields(pl.read_parquet(f'{base}/denominator.parquet', storage_options=so))
    flags = pl.read_parquet(f'{base}/paper_flags.parquet', storage_options=so)
    papers = (den.join(flags, on=c.key, how='left')
              .with_columns(
                  accel_any=pl.col('accel_any').fill_null(False),
                  accel_used=pl.col('accel_used').fill_null(False),
                  model_specific=pl.col('model_specific').fill_null(False),
                  model_specific_used=pl.col('model_specific_used').fill_null(False),
                  is_candidate=pl.col('is_candidate').fill_null(False))
              .filter(pl.col('year').is_between(year_min, year_max)))
    fields_top = (papers.filter(pl.col('field').is_not_null() & pl.col('accel_any'))
                  .group_by('field').agg(n=pl.len()).sort('n', descending=True)
                  .head(top_fields_n)['field'].to_list())
    return papers, fields_top


def load_pm(c, reg, so, prompt_version: str, model_tag: str, year_min: int, year_max: int):
    """One row per (paper, model), used-in-this-work only. Mirrors
    `notebooks/arxiv/productivity.ipynb`'s load cell."""
    base = s3_uri(analytic_base(c, reg.version, prompt_version, model_tag))
    den = short_fields(pl.read_parquet(f'{base}/denominator.parquet',
                                       storage_options=so)).select(c.key, 'year', 'field')
    m = (pl.scan_parquet(mentions_glob(model_tag, prompt_version, corpus=c), storage_options=so)
         .filter((pl.col('status') == 'ok') & (pl.col('usage_context') == 'used-in-this-work'))
         .select(c.key, 'model_normalized').collect())
    m = canonicalize_column(m, reg).filter(pl.col('canonical_kind') == 'model')
    meta = pl.DataFrame(
        [{'canonical_model': x.id, 'display': x.display,
          'release_year': int(x.release[:4]) if x.release else None}
         for x in reg.models.values() if x.kind == 'model'],
        schema={'canonical_model': pl.Utf8, 'display': pl.Utf8, 'release_year': pl.Int32})
    return (m.select(c.key, 'canonical_model').unique()
            .join(den, on=c.key, how='inner')
            .filter(pl.col('year').is_between(year_min, year_max))
            .join(meta, on='canonical_model', how='left'))


def build_charts(cap: pl.DataFrame, top_fields: list, papers: pl.DataFrame,
                 fields_top: list, pm: pl.DataFrame, reg, *, year_min: int, year_max: int,
                 min_papers_per_year: int, min_papers: int, top_n_change: int,
                 min_papers_model: int) -> dict:
    charts = {}

    specs = spec_frame(reg)
    per_dev = we.capacity_by_year(cap, 'per_device_fp32_gflops', 'spec_missing_fp32', min_papers_per_year)
    per_paper = we.capacity_by_year(cap, 'reported_fp32_gflops', 'spec_missing_fp32', min_papers_per_year)
    per_dev_vram = we.capacity_by_year(cap, 'per_device_vram_gb', 'spec_missing_vram', min_papers_per_year)
    per_paper_vram = we.capacity_by_year(cap, 'reported_vram_gb', 'spec_missing_vram', min_papers_per_year)
    charts['compute_memory'] = {
        'compute': {'per_device': per_dev.to_dicts(), 'per_paper': per_paper.to_dicts(),
                    'frontier': we.frontier(specs, 'fp32_gflops', year_min, year_max).to_dicts()},
        'memory': {'per_device': per_dev_vram.to_dicts(), 'per_paper': per_paper_vram.to_dicts(),
                   'frontier': we.frontier(specs, 'vram_gb', year_min, year_max).to_dicts()},
    }

    seg = (cap.filter(pl.col('field').is_not_null() & pl.col('year').is_between(2014, year_max))
           .with_columns(seg=pl.when(pl.col('field').is_in(top_fields))
                              .then(pl.col('field')).otherwise(pl.lit(OTHER))))
    charts['compute_fp32_memory_vram_by_field'] = {
        'fields': top_fields + [OTHER],
        'compute': we.by_field(seg, 'reported_fp32_gflops', 'spec_missing_fp32', min_papers=10).to_dicts(),
        'memory': we.by_field(seg, 'reported_vram_gb', 'spec_missing_vram', min_papers=10).to_dicts(),
    }
    charts['compute_fp32_memory_vram_spread'] = {
        'fields': top_fields + [OTHER, ALL],
        'compute': we.spread_by_field(seg, 'reported_fp32_gflops', 'spec_missing_fp32',
                                      all_label=ALL).to_dicts(),
        'memory': we.spread_by_field(seg, 'reported_vram_gb', 'spec_missing_vram',
                                     all_label=ALL).to_dicts(),
    }

    charts['corpus_composition'] = {
        'fields': fields_top, 'other': OTHER, 'no_label': NOLAB,
        'series': we.corpus_composition(papers, fields_top, other=OTHER, nolab=NOLAB).to_dicts(),
    }
    charts['prevalence'] = {
        'fields': fields_top + [OTHER],
        'overall': we.prevalence_series(papers, min_papers=min_papers).to_dicts(),
        'by_field': we.field_prevalence(papers, fields_top, other=OTHER, min_papers=100).to_dicts(),
    }
    top_models = (pm.group_by('canonical_model', 'display', 'release_year')
                  .agg(papers=pl.len())
                  .filter(pl.col('papers') >= min_papers_model)
                  .sort('papers', descending=True).head(top_n_change))
    charts['device_life'] = {'series': we.device_life_series(pm, top_models).to_dicts()}

    return charts


def build_drilldown(c, reg, so, prompt_version: str, model_tag: str) -> pl.DataFrame:
    """One row per DISTINCT (paper, model, usage context, evidence quote) among
    `status=='ok'` ACCELERATOR mentions, joined to arXiv title/category
    metadata. No `passage_text` — `evidence_quote` is the drill-down's display
    text; see module docstring.

    `accelerator_subtype != 'not-an-accelerator'` is the load-bearing filter
    here: the extraction LLM's off-ramp for registry false positives is a
    real label on 10.1% of `ok` mentions corpus-wide (K80-the-antibody, and
    software model names like "Claude 3.7 Sonnet" or "Llama-3.3-70B-Instruct"
    that a bare-model-name-shaped alias can still surface as a registry
    candidate). Every chart already excludes these via the same column
    (`denominator.build_paper_flags`); the drill-down just hadn't been
    filtering on it.

    Two de-duplications, neither of which touches any chart's numbers (every
    chart aggregates at the paper or paper-model grain via group_by, which
    already collapses these):

    1. Overlapping passage windows ("matched paragraph +/- 1 neighbor
       paragraph", `scan.py`) mean two adjacent passages can share a sentence,
       and the LLM then extracts the identical `evidence_quote` from both --
       11.5% of `ok` mentions corpus-wide are such exact duplicates. Kept as
       distinct mentions upstream (they DO count once each toward device/model
       identification), but shown once in a paper's drill-down list.
    2. 78 paper_ids appear more than once in `arxiv_metadata.parquet` (an
       upstream snapshot artifact); an inner join without deduping first would
       silently double every one of that paper's rows.
    """
    m = (pl.scan_parquet(mentions_glob(model_tag, prompt_version, corpus=c), storage_options=so)
         .filter((pl.col('status') == 'ok')
                 & (pl.col('accelerator_subtype') != 'not-an-accelerator'))
         .select(c.key, 'passage_id', 'model_raw', 'model_normalized', 'manufacturer',
                 'accelerator_subtype', 'device_count', 'usage_context', 'evidence_quote')
         .collect())
    m = canonicalize_column(m, reg)
    m = m.unique(subset=[c.key, 'canonical_model', 'usage_context', 'evidence_quote'],
                 keep='first')

    meta = (pl.scan_parquet(s3_uri(arxiv_metadata_key()), storage_options=so)
            .select('paper_id', 'arxiv_id', 'title', 'year', 'primary_category')
            .unique(subset='paper_id', keep='first')
            .collect())
    fields = paper_fields(c, m.select(c.key).unique(), so).select(c.key, 'field')

    out = (m.join(meta, on=c.key, how='inner')
           .join(fields, on=c.key, how='left')
           .filter(pl.col('year').is_not_null())
           .rename({'canonical_display': 'display'}))
    return out


def _sanitize_nans(obj):
    """Replace float NaN/Infinity with None, recursively.

    Python's `json` module serializes these as bare `NaN`/`Infinity` tokens,
    which are not valid JSON and fail `JSON.parse` in every browser. The
    upstream data shouldn't produce them (see the total_devices==0 guard
    above), but this is the last line of defense against a future source of
    the same problem breaking the site instead of just this file's numbers.
    """
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float('inf'), float('-inf'))) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    return obj


def write_charts(charts: dict, out: Path) -> None:
    charts_dir = out / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in charts.items():
        with open(charts_dir / f'{name}.json', 'w') as f:
            json.dump(_sanitize_nans(payload), f, separators=(',', ':'))


def write_drilldown(drill: pl.DataFrame, key: str, out: Path) -> None:
    """Partitioned by year directory (so a year-range filter only fetches the
    relevant files over HTTP), but `year` is ALSO kept as an in-file column --
    each part is then self-describing and queryable on its own, with no
    dependency on a DuckDB build parsing hive-style directory names."""
    mentions_dir = out / 'mentions'
    mentions_dir.mkdir(parents=True, exist_ok=True)
    for (year,), part in drill.group_by('year'):
        part_dir = mentions_dir / f'year={int(year)}'
        part_dir.mkdir(parents=True, exist_ok=True)
        part.write_parquet(part_dir / 'part-0.parquet', compression='zstd')

    def counts(col: str) -> list:
        return (drill.group_by(col).agg(n=pl.len()).sort('n', descending=True)
                .rename({col: 'value'}).to_dicts())

    manifest = {
        'n_mentions': drill.height,
        'n_papers': drill.select(key).n_unique(),
        'years': sorted(drill['year'].unique().drop_nulls().cast(pl.Int64).to_list()),
        'by_field': counts('field'),
        'by_manufacturer': counts('manufacturer'),
        'by_usage_context': counts('usage_context'),
    }
    with open(out / 'mentions' / 'manifest.json', 'w') as f:
        json.dump(manifest, f, separators=(',', ':'))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', default='arxiv')
    p.add_argument('--prompt-version', default='p1')
    p.add_argument('--model-tag', default='qwen3-14b')
    p.add_argument('--count-policy', default='floor1', choices=['floor1', 'explicit'])
    p.add_argument('--year-min', type=int, default=2005)
    p.add_argument('--year-max', type=int, default=2026)
    p.add_argument('--top-fields', type=int, default=8)
    p.add_argument('--min-papers-per-year', type=int, default=25)
    p.add_argument('--min-papers', type=int, default=200)
    p.add_argument('--top-n-change', type=int, default=12)
    p.add_argument('--min-papers-model', type=int, default=100)
    p.add_argument('--out', default='site_data')
    args = p.parse_args()

    out = Path(args.out)
    c = get_corpus(args.corpus)
    reg = load_registry()
    so = storage_options()

    cap, top_fields = load_capacity(c, reg, so, args.prompt_version, args.model_tag,
                                    args.count_policy, args.year_min, args.year_max,
                                    args.top_fields)
    papers, fields_top = load_papers(c, reg, so, args.prompt_version, args.model_tag,
                                     args.year_min, args.year_max, args.top_fields)
    pm = load_pm(c, reg, so, args.prompt_version, args.model_tag, args.year_min, args.year_max)
    print(f'{cap.height:,} capacity rows | {papers.height:,} papers | '
          f'{pm.height:,} paper-model rows')

    charts = build_charts(cap, top_fields, papers, fields_top, pm, reg,
                          year_min=args.year_min, year_max=args.year_max,
                          min_papers_per_year=args.min_papers_per_year,
                          min_papers=args.min_papers, top_n_change=args.top_n_change,
                          min_papers_model=args.min_papers_model)
    write_charts(charts, out)
    print(f'wrote {len(charts)} chart JSON files to {out / "charts"}')

    drill = build_drilldown(c, reg, so, args.prompt_version, args.model_tag)
    print(f'{drill.height:,} mentions | {drill.select(c.key).n_unique():,} papers for drill-down')
    write_drilldown(drill, c.key, out)
    print(f'wrote hive-partitioned mentions + manifest.json to {out / "mentions"}')


if __name__ == '__main__':
    main()
