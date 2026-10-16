"""Per-field growth-rate (CAGR) tables for reported use and reported capacity.

For each corpus, two things are computed per top field (ranked by papers with any
accelerator mention, same ranking `trends.ipynb`/`capacity.ipynb` use):

  - the OLS log-linear CAGR of reported-use share among full-text papers
    (`accel_used` mean by field-year, from the denominator + paper_flags tables);
  - the OLS log-linear CAGR of median reported per-paper capacity, on each of the
    FP32 / VRAM / TDP axes (from the `accelscan.compute` capacity table).

Both quantities and the growth-rate math (endpoint CAGR, OLS log-linear CAGR with
a 95% CI from the slope's standard error, implied doubling time) are exactly what
`notebooks/trends.ipynb` §4.1 and `notebooks/capacity.ipynb` §5.5 already compute
in-notebook (see their "Growth-rate summary" cells) -- mirrored here, rather than
imported, because the notebook cells operate on frames already filtered/joined
in-notebook; this script re-derives them from the same S3 tables so the two paths
can be diffed against each other, not so they can silently drift.

    python -m accelscan.scripts.growth_stats                  # both corpora, write tables
    python -m accelscan.scripts.growth_stats --corpus s2orc --stdout
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from accelscan.compute import AXES
from accelscan.config import BUCKET, OUT_PREFIX
from accelscan.meta import paper_fields
from accelscan.paths import analytic_base, capacity_key, get_corpus, s3_uri
from accelscan.registry import load_registry
from accelscan.s3 import storage_options

TOP_FIELDS = 8
YEAR_MIN, YEAR_MAX = 2010, 2025   # matches trends.ipynb/capacity.ipynb's S2ORC window;
                                  # applied to both corpora so the two tables are
                                  # measured over the same years (arXiv's notebooks
                                  # otherwise run through 2026, a partial year)
MIN_PAPERS_USE = 100         # field-year support floor, matches trends.ipynb `fd`
MIN_PAPERS_CAPACITY = 10     # matches capacity.ipynb `by_field()`
OTHER = 'Other fields'
# TDP is computed alongside FP32/VRAM (same growth_summary/growth_table
# machinery) but reported in a separate appendix table, not the main one --
# it is a nameplate power rating, not measured energy, and the least
# spec-complete of the three axes, so it is too ambiguous to sit next to
# FP32/VRAM with the same apparent confidence (matches the notebooks'
# Appendix: Reported power (TDP) sections).
CAP_AXES = [('reported_fp32_gflops', 'spec_missing_fp32', 'FP32'),
           ('reported_vram_gb', 'spec_missing_vram', 'VRAM'),
           ('reported_tdp_w', 'spec_missing_tdp', 'TDP')]
MAIN_AXES = ['FP32', 'VRAM']
APPENDIX_AXES = ['TDP']
OUT_DIR = Path('docs/tables')


def growth_summary(year, value) -> dict | None:
    """OLS log-linear CAGR (with 95% CI) + endpoint CAGR, over the given series.

    See the docstring math in the response this script accompanies: fit
    log(v) = a + b*t by OLS, CAGR = exp(b) - 1, CI from b's standard error
    (normal approximation), doubling time = ln(2) / b.
    """
    y = np.asarray(year, dtype=float); v = np.asarray(value, dtype=float)
    m = np.isfinite(v) & (v > 0)
    y, v = y[m], v[m]
    if y.size < 2:
        return None
    y0, y1, v0, v1 = y[0], y[-1], v[0], v[-1]
    span = y1 - y0
    fold = v1 / v0
    cagr = fold ** (1 / span) - 1 if span > 0 else float('nan')
    logv = np.log(v)
    slope, intercept = np.polyfit(y, logv, 1)
    ols_cagr = np.exp(slope) - 1
    doubling = np.log(2) / np.log(1 + ols_cagr) if ols_cagr > 0 else float('nan')
    n = y.size
    dof = n - 2
    if dof > 0:
        resid = logv - (slope * y + intercept)
        se_slope = np.sqrt(np.sum(resid ** 2) / dof / np.sum((y - y.mean()) ** 2))
        ols_cagr_lo = np.exp(slope - 1.96 * se_slope) - 1
        ols_cagr_hi = np.exp(slope + 1.96 * se_slope) - 1
    else:
        ols_cagr_lo = ols_cagr_hi = float('nan')
    return dict(y0=int(y0), y1=int(y1), n=n, fold=fold, cagr=cagr,
                ols_cagr=ols_cagr, ols_cagr_lo=ols_cagr_lo, ols_cagr_hi=ols_cagr_hi,
                doubling=doubling)


def fmt_cell(g: dict | None) -> str:
    if g is None:
        return '--'
    lo, hi = 100 * g['ols_cagr_lo'], 100 * g['ols_cagr_hi']
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f'{100 * g["ols_cagr"]:.1f}'
    return f'{100 * g["ols_cagr"]:.1f} ({lo:.1f}--{hi:.1f})'


def usage_growth_by_field(papers: pl.DataFrame, top_fields: list[str]) -> dict:
    fd = (papers.filter(pl.col('field').is_not_null())
          .with_columns(seg=pl.when(pl.col('field').is_in(top_fields))
                              .then(pl.col('field')).otherwise(pl.lit(OTHER)))
          .group_by('seg', 'year').agg(n=pl.len(), used=pl.col('accel_used').mean())
          .filter(pl.col('n') >= MIN_PAPERS_USE).sort('year'))
    out = {}
    for f in top_fields + [OTHER]:
        s = fd.filter(pl.col('seg') == f)
        out[f] = growth_summary(s['year'], s['used']) if s.height > 2 else None
    return out


def capacity_growth_by_field(cap: pl.DataFrame, top_fields: list[str]) -> dict:
    seg = (cap.filter(pl.col('field').is_not_null())
           .with_columns(seg=pl.when(pl.col('field').is_in(top_fields))
                              .then(pl.col('field')).otherwise(pl.lit(OTHER))))
    out = {axis_label: {} for _, _, axis_label in CAP_AXES}
    for col, flag, axis_label in CAP_AXES:
        fd = (seg.filter(~pl.col(flag) & pl.col(col).is_not_null())
              .group_by('seg', 'year').agg(n=pl.len(), med=pl.col(col).median())
              .filter(pl.col('n') >= MIN_PAPERS_CAPACITY).sort('year'))
        for f in top_fields + [OTHER]:
            s = fd.filter(pl.col('seg') == f)
            out[axis_label][f] = growth_summary(s['year'], s['med']) if s.height > 2 else None
    return out


def load_papers(corpus_name: str, prompt_version: str, model_tag: str) -> tuple:
    """(papers, top_fields) -- the denominator+paper_flags population used for
    the 'Reported use' growth column. Public so notebooks can call it directly
    (e.g. capacity.ipynb's by-field growth chart) instead of re-deriving the same
    join, which is what makes the chart's "Reported use" bars match this script's
    LaTeX table exactly rather than a second, independently-computed version."""
    c = get_corpus(corpus_name)
    reg = load_registry()
    so = storage_options()

    base = s3_uri(analytic_base(c, reg.version, prompt_version, model_tag))
    den = pl.read_parquet(f'{base}/denominator.parquet', storage_options=so)
    flags = pl.read_parquet(f'{base}/paper_flags.parquet', storage_options=so)
    key = c.key
    if 'field' not in den.columns:
        fields = paper_fields(c, den.select(key), so).select(key, 'field')
        den = den.join(fields, on=key, how='left')
    papers = den.join(flags, on=key, how='left').with_columns(
        accel_any=pl.col('accel_any').fill_null(False),
        accel_used=pl.col('accel_used').fill_null(False))

    top_fields = (papers.filter(pl.col('field').is_not_null() & pl.col('accel_any'))
                  .group_by('field').agg(n=pl.len()).sort('n', descending=True)
                  .head(TOP_FIELDS)['field'].to_list())
    return papers, top_fields


def collect(corpus_name: str, prompt_version: str, model_tag: str,
           count_policy: str) -> dict:
    c = get_corpus(corpus_name)
    reg = load_registry()
    so = storage_options()
    key = c.key

    # top_fields ranking stays over the full population (matches load_papers'
    # own convention, and capacity.ipynb's top_fields/flag_ids); only the growth
    # calculations below are restricted to YEAR_MIN-YEAR_MAX.
    papers, top_fields = load_papers(corpus_name, prompt_version, model_tag)
    papers = papers.filter(pl.col('year').is_between(YEAR_MIN, YEAR_MAX))

    cap_key = s3_uri(capacity_key(c, reg.version, prompt_version, model_tag, count_policy))
    cap = pl.read_parquet(cap_key, storage_options=so)
    if 'field' not in cap.columns:
        fields = paper_fields(c, cap.select(key), so).select(key, 'field')
        cap = cap.join(fields, on=key, how='left')
    cap = cap.filter(pl.col('year').is_between(YEAR_MIN, YEAR_MAX))

    return {
        'corpus': corpus_name, 'top_fields': top_fields,
        'usage': usage_growth_by_field(papers, top_fields),
        'capacity': capacity_growth_by_field(cap, top_fields),
    }


# arXiv's field names are archive-level group names (too long for a table column);
# shortened the same way notebooks/arxiv/*.ipynb shorten them for legends/axes.
FIELD_LABELS = {'Electrical Engineering and Systems Science': 'EE and Systems',
               'Quantitative Biology': 'Quant. Biology',
               'Quantitative Finance': 'Quant. Finance'}


def _sorted_fields(st: dict) -> list[str]:
    """top_fields ordered by Reported-use CAGR descending -- the row order every
    table (main and appendix) uses, so a field's row lines up across tables."""
    def usage_cagr_key(f: str) -> float:
        g = st['usage'].get(f)
        return -g['ols_cagr'] if g is not None else float('inf')
    return sorted(st['top_fields'], key=usage_cagr_key)


def _table(st: dict, axis_labels: list[str], *, file_suffix: str, kind: str) -> str:
    label_suffix = file_suffix.replace('_', '-')   # hyphenated \label, matches
                                                    # the repo's tab: label style
    corpus_label = {'s2orc': 'S2ORC', 'arxiv': 'arXiv'}[st['corpus']]
    corpus_id = st['corpus']

    def row_cells(f: str) -> list[str] | None:
        cells = [fmt_cell(st['usage'].get(f))] + \
            [fmt_cell(st['capacity'][label].get(f)) for label in axis_labels]
        return None if all(c == '--' for c in cells) else cells

    # sorted by Reported-use CAGR descending; "Other fields" stays a separate,
    # visually set-off residual row rather than competing for rank among peers
    top_rows = [(f, row_cells(f)) for f in _sorted_fields(st)]
    other_row = row_cells(OTHER)   # arXiv's top fields already exhaust its 8-group
                                    # taxonomy, so "Other fields" is empty -- drop it
                                    # rather than print an all-dash row.
    cols = 'l ' + 'r ' * (1 + len(axis_labels))
    header = ' & '.join(['Field', 'Reported use'] + axis_labels)

    if kind == 'main':
        caption = (
            rf'{corpus_label}: OLS log-linear CAGR (\%/yr, 95\% CI), '
            rf'{YEAR_MIN}--{YEAR_MAX}, of reported accelerator use and median '
            r'reported per-paper capacity, by field. Fields outside the top '
            rf'{TOP_FIELDS} (by any-mention papers) are pooled into '
            r'\emph{Other fields}. Reported power (TDP) is analyzed separately '
            rf'in Table~\ref{{tab:growth-{corpus_id}-tdp}} (Appendix) -- see '
            r'there for why it is not reported here.')
    else:
        caption = (
            rf'{corpus_label}: OLS log-linear CAGR (\%/yr, 95\% CI), '
            rf'{YEAR_MIN}--{YEAR_MAX}, of reported power (TDP), by field, in '
            rf'the same field order as Table~\ref{{tab:growth-{corpus_id}}}. '
            r'TDP is a nameplate power-draw rating, not measured energy '
            r'consumption, and the least spec-complete of the reported-capacity '
            r'axes, so it is reported here separately rather than alongside '
            r'FP32/VRAM.')

    lines = [
        r'% GENERATED by accelscan/scripts/growth_stats.py -- do not hand-edit.',
        rf'% \input{{tables/growth_{corpus_id}{file_suffix}}}',
        r'\begin{table}[t]',
        r'\centering\small',
        rf'\caption{{{caption}}}',
        rf'\label{{tab:growth-{corpus_id}{label_suffix}}}',
        rf'\begin{{tabular}}{{{cols.strip()}}}',
        r'\hline',
        rf'{header} \\',
        r'\hline',
    ]
    for f, cells in top_rows:
        if cells is None:
            continue
        lines.append(f'{FIELD_LABELS.get(f, f)} & ' + ' & '.join(cells) + r' \\')
    if other_row is not None:
        lines.append(r'\hline')
        lines.append('Other fields & ' + ' & '.join(other_row) + r' \\')
    lines += [
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


def latex(st: dict) -> str:
    return _table(st, MAIN_AXES, file_suffix='', kind='main')


def latex_appendix_tdp(st: dict) -> str:
    """Reported power (TDP) growth, split out of the main table: TDP is a
    nameplate power rating (not measured energy) and the least spec-complete
    of the three capacity axes, too ambiguous to report with the same apparent
    confidence as FP32/VRAM in the main text -- matches the notebooks'
    Appendix: Reported power (TDP) sections. Same field order as the main
    table (Reported-use CAGR descending), so rows line up across tables."""
    return _table(st, APPENDIX_AXES, file_suffix='_tdp', kind='tdp')


def summary(st: dict) -> str:
    lines = [f'=== {st["corpus"]} (top fields: {", ".join(st["top_fields"])}) ===']
    for f in st['top_fields'] + [OTHER]:
        cap_str = ' '.join(f'{label}={fmt_cell(st["capacity"][label].get(f))}%/yr'
                           for label in [l for _, _, l in CAP_AXES])
        lines.append(f'  {f:<24} use={fmt_cell(st["usage"].get(f))}%/yr | {cap_str}')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['s2orc', 'arxiv'])
    ap.add_argument('--prompt-version', default='p1')
    ap.add_argument('--model-tag', default='qwen3-14b')
    ap.add_argument('--count-policy', default='floor1')
    ap.add_argument('--stdout', action='store_true', help='print tables, write nothing')
    args = ap.parse_args()

    corpora = [args.corpus] if args.corpus else ['s2orc', 'arxiv']
    for name in corpora:
        st = collect(name, args.prompt_version, args.model_tag, args.count_policy)
        print(summary(st), file=sys.stderr)
        table = latex(st)
        tdp_table = latex_appendix_tdp(st)
        if args.stdout:
            print(table)
            print(tdp_table)
            continue
        out = OUT_DIR / f'growth_{name}.tex'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table)
        print(f'wrote {out}', file=sys.stderr)
        out_tdp = OUT_DIR / f'growth_{name}_tdp.tex'
        out_tdp.write_text(tdp_table)
        print(f'wrote {out_tdp}', file=sys.stderr)


if __name__ == '__main__':
    main()
