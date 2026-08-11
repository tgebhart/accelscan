"""Pure aggregation functions behind the public website's 7 headline charts.

Each function is lifted verbatim (same group-bys, same filters, same eligibility
floors) out of the `notebooks/arxiv/{capacity,trends,productivity}.ipynb` plot
cells that first computed it. The notebooks import these instead of inlining the
logic, and `accelscan/scripts/export_site_data.py` calls the same functions to
produce the website's chart JSON -- so the PDF figures and the website numbers
can never drift apart; both are one function call away from the same code.

No plotting, no I/O: every function takes an already-loaded polars DataFrame (the
same variables the notebooks build in their load cells: `cap`, `papers`, `pm`) and
returns a small tidy DataFrame or a dict of scalars. Callers do the S3 reads.
"""

import polars as pl


# ---------------------------------------------------------------------------
# capacity.ipynb
# ---------------------------------------------------------------------------

def capacity_by_year(cap: pl.DataFrame, col: str, missing_flag: str,
                      min_papers_per_year: int) -> pl.DataFrame:
    """Median / p90 / total of `col` by year, among papers with a spec for it.

    Source: `notebooks/arxiv/capacity.ipynb`, `capacity_by_year()` in the cell
    building figure 02 (compute-memory).
    """
    d = cap.filter(~pl.col(missing_flag) & pl.col(col).is_not_null())
    return (d.group_by('year').agg(n=pl.len(),
                                   med=pl.col(col).median(),
                                   p90=pl.col(col).quantile(0.9),
                                   total=pl.col(col).sum())
            .filter(pl.col('n') >= min_papers_per_year).sort('year'))


def frontier(specs: pl.DataFrame, axis_col: str, year_min: int, year_max: int,
             segment: str = 'datacenter') -> pl.DataFrame:
    """Best single-device spec available up to each year (the vendor frontier).

    `specs` is `accelscan.compute.spec_frame(reg)`. Source: capacity.ipynb's
    `frontier()`, same cell as `capacity_by_year`.
    """
    s = (specs.filter(pl.col('segment') == segment)
         .filter(pl.col(axis_col).is_not_null() & pl.col('release_year').is_not_null()))
    rows = []
    for y in range(year_min, year_max + 1):
        avail = s.filter(pl.col('release_year') <= y)
        if avail.height:
            rows.append({'year': y, 'frontier': avail[axis_col].max()})
    return pl.DataFrame(rows, schema={'year': pl.Int64, 'frontier': pl.Float64})


def by_field(seg: pl.DataFrame, col: str, flag: str,
             min_papers: int = 10) -> pl.DataFrame:
    """Median of `col` by (field segment, year), among papers with a spec for it.

    `seg` already carries the `seg` column (top-N fields + a pooled residual).
    Source: capacity.ipynb's `by_field()`, the cell building figure 10.
    """
    return (seg.filter(~pl.col(flag) & pl.col(col).is_not_null())
            .group_by('seg', 'year').agg(n=pl.len(), med=pl.col(col).median())
            .filter(pl.col('n') >= min_papers).sort('year'))


def _spread(d: pl.DataFrame, by: list, min_field_spread: int) -> pl.DataFrame:
    return (d.group_by(by)
            .agg(n=pl.len(), med=pl.col('_v').median(), p90=pl.col('_v').quantile(0.9))
            .filter((pl.col('n') >= min_field_spread) & (pl.col('med') > 0))
            .with_columns(ratio=pl.col('p90') / pl.col('med')))


def spread_by_field(seg: pl.DataFrame, col: str, flag: str, *,
                     all_label: str = 'All fields', min_field_spread: int = 10,
                     smooth: int = 3) -> pl.DataFrame:
    """p90/median of paper-level capacity per field-year, plus a pooled series,
    each smoothed with a centered `smooth`-year moving average.

    Source: capacity.ipynb's `spread_by_field()`, the cell building figure 11.
    """
    d = (seg.filter(~pl.col(flag) & pl.col(col).is_not_null() & (pl.col(col) > 0))
         .with_columns(_v=pl.col(col)))
    per = _spread(d, ['seg', 'year'], min_field_spread)
    pooled = _spread(d, ['year'], min_field_spread).with_columns(
        seg=pl.lit(all_label)).select(per.columns)
    return (pl.concat([per, pooled]).sort('seg', 'year')
            .with_columns(ratio_ma=pl.col('ratio')
                          .rolling_mean(smooth, min_samples=1, center=True).over('seg')))


# ---------------------------------------------------------------------------
# trends.ipynb
# ---------------------------------------------------------------------------

def corpus_composition(papers: pl.DataFrame, fields_top: list, *,
                        other: str = 'Other fields',
                        nolab: str = 'No field label') -> pl.DataFrame:
    """Full-text papers per year, split into top-N fields + pooled residual +
    unlabeled, in long form (`year, seg, n`).

    Source: trends.ipynb's `piv` cell (figure 02); returned long rather than
    pivoted-wide so JSON consumers don't need to know the field set up front.
    """
    segments = fields_top + [other, nolab]
    piv = (papers.with_columns(
               seg=pl.when(pl.col('field').is_null()).then(pl.lit(nolab))
                     .when(pl.col('field').is_in(fields_top)).then(pl.col('field'))
                     .otherwise(pl.lit(other)))
           .group_by('year', 'seg').agg(n=pl.len()))
    assert piv['n'].sum() == papers.height, 'segments must partition the corpus'
    full = pl.DataFrame({'seg': segments}).join(
        pl.DataFrame({'year': sorted(papers['year'].unique().drop_nulls().to_list())}),
        how='cross')
    return (full.join(piv, on=['year', 'seg'], how='left')
            .with_columns(n=pl.col('n').fill_null(0)).sort('year', 'seg'))


def prevalence_series(papers: pl.DataFrame, min_papers: int) -> pl.DataFrame:
    """Overall accelerator-reporting prevalence by year.

    Source: trends.ipynb's `pv` cell (figure 03, left panel).
    """
    return (papers.group_by('year').agg(
                n=pl.len(),
                any_share=pl.col('accel_any').mean(),
                used_share=pl.col('accel_used').mean(),
                modelspec_share=pl.col('model_specific_used').mean(),
                cand_share=pl.col('is_candidate').mean())
            .filter(pl.col('n') >= min_papers).sort('year'))


def field_prevalence(papers: pl.DataFrame, fields_top: list, *,
                      other: str = 'Other fields', min_papers: int = 100) -> pl.DataFrame:
    """Reported-use share by (field segment, year).

    Source: trends.ipynb's `fd` cell (figure 03, right panel).
    """
    return (papers.filter(pl.col('field').is_not_null())
            .with_columns(seg=pl.when(pl.col('field').is_in(fields_top))
                                .then(pl.col('field')).otherwise(pl.lit(other)))
            .group_by('seg', 'year').agg(n=pl.len(), used=pl.col('accel_used').mean())
            .filter(pl.col('n') >= min_papers).sort('year'))


def extraction_funnel(papers: pl.DataFrame) -> dict:
    """Five-stage paper count funnel: full-text -> candidate -> confirmed ->
    reported use -> named model.

    Source: trends.ipynb's `funnel` dict (figure 04, left panel).
    """
    return {
        'full-text papers': papers.height,
        'registry candidate mentions': int(papers['is_candidate'].sum()),
        'LLM-confirmed accelerator mention': int(papers['accel_any'].sum()),
        'reported use of accelerator': int(papers['accel_used'].sum()),
        'names and uses model': int(papers['model_specific_used'].sum()),
    }


def specificity_share(papers: pl.DataFrame, min_papers: int = 50) -> pl.DataFrame:
    """Share of accelerator-using papers naming a specific model, by year.

    Source: trends.ipynb's `sp` cell (figure 04, right panel).
    """
    return (papers.filter(pl.col('accel_used'))
            .group_by('year').agg(n=pl.len(),
                                  specific=pl.col('model_specific_used').mean())
            .filter(pl.col('n') >= min_papers).sort('year'))


# ---------------------------------------------------------------------------
# productivity.ipynb
# ---------------------------------------------------------------------------

def device_life_series(pm: pl.DataFrame, top_models: pl.DataFrame) -> pl.DataFrame:
    """Papers supported per model, indexed by years since release.

    `top_models` is a small frame with `canonical_model, display, release_year`
    (the top-N eligible models by paper support). Source: productivity.ipynb's
    `ts` cell (figure 02).
    """
    return (pm.filter(pl.col('canonical_model').is_in(top_models['canonical_model'].to_list()))
            .group_by('canonical_model', 'year').agg(papers=pl.len())
            .join(top_models.select('canonical_model', 'display', 'release_year'),
                  on='canonical_model')
            .filter(pl.col('release_year').is_not_null()
                    & (pl.col('year') >= pl.col('release_year'))
                    & (pl.col('papers') > 0))
            .with_columns(since_release=pl.col('year') - pl.col('release_year'))
            .sort('canonical_model', 'year'))
