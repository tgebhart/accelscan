"""Section-3 (Methods) statistics for the paper, read from live pipeline output.

Every number in main.tex's Methods section that is not registry composition
(registry_stats.py) or a matching-convention exemplar (matching_table.py) is
computed here: stage-1 candidate/passage counts, stage-2 LLM status mix,
stage-3 normalization/unresolved-bucket shares, usage-context mix, stage-3.5
capacity/spec-coverage/device-count statistics, and stage-4a field-label
coverage. One script per corpus-independent statistic set, so the paper's
prose is never hand-typed against a notebook cell that can drift.

    python -m accelscan.scripts.methods_stats                    # both corpora
    python -m accelscan.scripts.methods_stats --corpus s2orc
"""

import argparse
import sys

import polars as pl

from accelscan.compute import AXES
from accelscan.config import BUCKET, PAPERS_PREFIX
from accelscan.normalize import canonicalize_column, classify_unresolved, is_reject
from accelscan.paths import ARXIV, S2ORC, analytic_base, candidates_glob, \
    capacity_key, get_corpus, inventory_glob
from accelscan.registry import load_registry
from accelscan.s3 import mentions_glob, storage_options

USED = 'used-in-this-work'


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else float('nan')


def stage1(c, candidates_version, so) -> dict:
    inv = pl.scan_parquet(inventory_glob(c), storage_options=so).collect()
    cand = pl.scan_parquet(candidates_glob(c, candidates_version), storage_options=so).collect()
    return {
        'n_papers_inventory': inv.height,
        'n_candidate_papers': inv.filter(pl.col('is_candidate')).height,
        'n_candidate_passages': cand.height,
        'n_model_specific_passages': cand.filter(pl.col('model_specific')).height,
        'n_capped_papers': inv.filter(pl.col('passages_truncated')).height,
    }


def stage2(mentions: pl.DataFrame) -> dict:
    n = mentions.height
    status = mentions['status'].value_counts().sort('status')
    counts = dict(zip(status['status'].to_list(), status['count'].to_list()))
    return {
        'n_rows': n,
        'n_ok': counts.get('ok', 0),
        'n_no_mention': counts.get('no_mention', 0),
        'n_truncated': counts.get('truncated', 0),
        'n_json_error': counts.get('json_error', 0),
    }


def usage_context(ok: pl.DataFrame) -> dict:
    n = ok.height
    vc = ok['usage_context'].value_counts()
    counts = dict(zip(vc['usage_context'].to_list(), vc['count'].to_list()))
    return {'n_confirmed': n, **{k: counts.get(k, 0) for k in
            ('used-in-this-work', 'comparison-or-related-work',
             'object-of-study', 'speculative-future')}}


def used_named_canon(ok: pl.DataFrame, reg) -> pl.DataFrame:
    """Used-in-this-work mentions naming a (non-rejected) model, canonicalized.
    Shared by normalization() and device_count_stats() so both read off the
    same population as the capacity table (compute.py's own filter is
    usage_context==used & canonical_model is not null) rather than each
    re-deriving it and silently drifting apart."""
    used = ok.filter(pl.col('usage_context') == USED)
    named = used.filter(pl.col('model_normalized').is_not_null()
                        & ~pl.col('model_normalized').map_elements(
                            is_reject, return_dtype=pl.Boolean))
    return canonicalize_column(named, reg)


def normalization(canon: pl.DataFrame) -> dict:
    named = canon
    resolved = canon.filter(pl.col('canonical_model').is_not_null())
    unresolved = canon.filter(pl.col('canonical_model').is_null())
    buckets = (unresolved
              .with_columns(bucket=pl.col('model_normalized')
                            .map_elements(classify_unresolved, return_dtype=pl.Utf8))
              ['bucket'].value_counts().sort('count', descending=True))
    return {
        'n_named': named.height,
        'n_resolved': resolved.height,
        'pct_resolved': pct(resolved.height, named.height),
        'unresolved_buckets': dict(zip(buckets['bucket'].to_list(),
                                       buckets['count'].to_list())),
        'pct_xilinx': pct(buckets.filter(pl.col('bucket') == 'fpga-xilinx')['count']
                          .sum() if 'fpga-xilinx' in buckets['bucket'].to_list() else 0,
                          named.height),
        'pct_unknown': pct(buckets.filter(pl.col('bucket') == 'unknown')['count']
                           .sum() if 'unknown' in buckets['bucket'].to_list() else 0,
                          named.height),
    }


def capacity_stats(c, reg, so, prompt_version: str, model_tag: str) -> dict:
    key = f's3://{BUCKET}/{capacity_key(c, reg.version, prompt_version, model_tag, "floor1")}'
    cap = pl.read_parquet(key, storage_options=so)
    n = cap.height
    out = {'n_papers': n}
    for axis in AXES:
        have = cap.filter(~pl.col(f'spec_missing_{axis}')).height
        out[f'pct_{axis}'] = pct(have, n)
    have_vram = cap.filter(~pl.col('spec_missing_vram')).height
    out['pct_vram'] = pct(have_vram, n)
    return out


def device_count_stats(resolved: pl.DataFrame, cap: pl.DataFrame) -> dict:
    """Device-count/memory stated shares are restricted to the SAME population
    as the capacity table: used-in-this-work mentions that resolve to a
    registry entry (`resolved` = used_named_canon() filtered to
    canonical_model is not null). This makes every row in the capacity-stats
    table apples-to-apples -- deliberately narrower than the raw
    'of confirmed mentions' framing in the prose, which spans all usage
    contexts and doesn't require resolution."""
    n = resolved.height
    stated = resolved.filter(pl.col('device_count').is_not_null())
    explicit = stated.filter(pl.col('device_count_basis') == 'explicit')
    stated_mem = resolved.filter(pl.col('memory_gb').is_not_null())
    q = cap['max_devices'].drop_nulls()
    return {
        'pct_device_count_stated': pct(stated.height, n),
        'n_explicit': explicit.height,
        'pct_memory_stated': pct(stated_mem.height, n),
        'median_max_devices': q.median(),
        'p90_max_devices': q.quantile(0.9),
        'p99_max_devices': q.quantile(0.99),
    }


def field_label_coverage(c, reg, so) -> float | None:
    """Share of the full in-window population (the denominator table, not just
    accelerator-flagged papers) that carries a field label. S2ORC only --
    arXiv's field always resolves at 100% by construction (its own taxonomy)."""
    if c is not S2ORC:
        return None
    denom = pl.read_parquet(
        f's3://{BUCKET}/{analytic_base(c, reg.version, "p1", "qwen3-14b")}'
        f'/denominator.parquet', storage_options=so, columns=['field'])
    return pct(denom.filter(pl.col('field').is_not_null()).height, denom.height)


def run(corpus_name: str, prompt_version: str, model_tag: str) -> dict:
    c = get_corpus(corpus_name)
    reg = load_registry()
    so = storage_options()

    mentions = pl.scan_parquet(mentions_glob(model_tag, prompt_version, corpus=c),
                               storage_options=so).collect()
    ok = mentions.filter(pl.col('status') == 'ok')
    used = ok.filter(pl.col('usage_context') == USED)
    extraction_version = mentions['registry_version'][0] if mentions.height else reg.version

    key = f's3://{BUCKET}/{capacity_key(c, reg.version, prompt_version, model_tag, "floor1")}'
    cap = pl.read_parquet(key, storage_options=so)
    canon = used_named_canon(ok, reg)
    resolved = canon.filter(pl.col('canonical_model').is_not_null())

    return {
        'corpus': corpus_name, 'registry_version': reg.version,
        'mentions_extraction_version': extraction_version,
        'stage1': stage1(c, extraction_version, so),
        'stage2': stage2(mentions),
        'usage_context': usage_context(ok),
        'normalization': normalization(canon),
        'capacity': capacity_stats(c, reg, so, prompt_version, model_tag),
        'device_counts': device_count_stats(resolved, cap),
        'field_label_coverage': field_label_coverage(c, reg, so),
    }


def report(st: dict) -> str:
    lines = [f'=== {st["corpus"]} (registry v{st["registry_version"]}, '
            f'mentions extracted under v{st["mentions_extraction_version"]}) ===']
    s1 = st['stage1']
    lines.append(f'  stage1: {s1["n_candidate_passages"]:,} candidate passages from '
                f'{s1["n_candidate_papers"]:,} papers ({s1["n_papers_inventory"]:,} in '
                f'inventory); {s1["n_model_specific_passages"]:,} model-specific; '
                f'{s1["n_capped_papers"]:,} papers hit the passage cap')
    s2 = st['stage2']
    lines.append(f'  stage2: {s2["n_rows"]:,} rows; ok={s2["n_ok"]:,} '
                f'({pct(s2["n_ok"], s2["n_rows"])}%) no_mention={s2["n_no_mention"]:,} '
                f'({pct(s2["n_no_mention"], s2["n_rows"])}%) '
                f'truncated={s2["n_truncated"]} json_error={s2["n_json_error"]}')
    uc = st['usage_context']
    lines.append(f'  usage_context (of {uc["n_confirmed"]:,} confirmed): '
                f'used={uc["used-in-this-work"]:,} ({pct(uc["used-in-this-work"], uc["n_confirmed"])}%) '
                f'comparison={uc["comparison-or-related-work"]:,} '
                f'({pct(uc["comparison-or-related-work"], uc["n_confirmed"])}%) '
                f'object={uc["object-of-study"]:,} ({pct(uc["object-of-study"], uc["n_confirmed"])}%) '
                f'speculative={uc["speculative-future"]:,} ({pct(uc["speculative-future"], uc["n_confirmed"])}%)')
    nz = st['normalization']
    lines.append(f'  normalization: {nz["pct_resolved"]}% resolved of {nz["n_named"]:,} named '
                f'used-in-this-work mentions; xilinx-fpga={nz["pct_xilinx"]}% unknown={nz["pct_unknown"]}%')
    lines.append(f'    unresolved buckets: {nz["unresolved_buckets"]}')
    cp = st['capacity']
    lines.append(f'  capacity: {cp["n_papers"]:,} papers; coverage ' +
                ' '.join(f'{k.replace("pct_", "")}={v}%' for k, v in cp.items() if k.startswith('pct_')))
    dc = st['device_counts']
    lines.append(f'  device counts (of used-in-this-work, resolved mentions): '
                f'{dc["pct_device_count_stated"]}% stated '
                f'({dc["n_explicit"]:,} explicit); memory stated {dc["pct_memory_stated"]}%; '
                f'max_devices median={dc["median_max_devices"]} p90={dc["p90_max_devices"]} '
                f'p99={dc["p99_max_devices"]}')
    if st['field_label_coverage'] is not None:
        lines.append(f'  field label coverage: {st["field_label_coverage"]}%')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['s2orc', 'arxiv'])
    ap.add_argument('--prompt-version', default='p1')
    ap.add_argument('--model-tag', default='qwen3-14b')
    args = ap.parse_args()

    corpora = [args.corpus] if args.corpus else ['s2orc', 'arxiv']
    for name in corpora:
        st = run(name, args.prompt_version, args.model_tag)
        print(report(st), file=sys.stderr)
        print(file=sys.stderr)


if __name__ == '__main__':
    main()
