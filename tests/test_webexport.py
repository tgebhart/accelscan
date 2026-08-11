"""Unit tests for `accelscan.webexport`'s chart-aggregation functions.

Each test builds a tiny synthetic frame and checks the aggregation against a
value computed by hand, so a refactor that changes a filter or a group-by key
is caught here rather than by diffing PDFs.
"""

import polars as pl

from accelscan import webexport as we


def test_capacity_by_year_respects_missing_flag_and_floor():
    cap = pl.DataFrame({
        'year': [2020, 2020, 2020, 2021, 2021],
        'x': [10.0, 20.0, 30.0, 5.0, None],
        'x_missing': [False, False, True, False, True],
    })
    out = we.capacity_by_year(cap, 'x', 'x_missing', min_papers_per_year=2)
    # 2020: two non-missing, non-null rows (10, 20) -> n=2, med=15
    # 2021: only one usable row -> dropped by the floor
    assert out['year'].to_list() == [2020]
    row = out.row(0, named=True)
    assert row['n'] == 2
    assert row['med'] == 15.0
    assert row['total'] == 30.0


def test_frontier_is_monotone_and_only_covers_available_years():
    specs = pl.DataFrame({
        'segment': ['datacenter', 'datacenter', 'consumer'],
        'spec': [10.0, 40.0, 999.0],
        'release_year': [2018, 2020, 2018],
    })
    out = we.frontier(specs, 'spec', year_min=2016, year_max=2021)
    # no datacenter part released by 2016/2017 -> no row for those years
    assert out['year'].to_list() == [2018, 2019, 2020, 2021]
    assert out['frontier'].to_list() == [10.0, 10.0, 40.0, 40.0]


def test_by_field_filters_missing_and_floor():
    seg = pl.DataFrame({
        'seg': ['CS'] * 12 + ['Physics'] * 5,
        'year': [2020] * 12 + [2020] * 5,
        'v': list(range(12)) + list(range(5)),
        'flag': [False] * 12 + [False] * 5,
    })
    out = we.by_field(seg, 'v', 'flag', min_papers=10)
    # CS has 12 (>=10) rows, Physics only 5 -> dropped
    assert out['seg'].to_list() == ['CS']
    assert out.row(0, named=True)['n'] == 12


def test_spread_by_field_adds_a_pooled_all_fields_series():
    seg = pl.DataFrame({
        'seg': ['CS'] * 15 + ['Physics'] * 15,
        'year': [2020] * 15 + [2020] * 15,
        'v': list(range(1, 16)) + list(range(1, 16)),
        'flag': [False] * 30,
    })
    out = we.spread_by_field(seg, 'v', 'flag', min_field_spread=10, smooth=3)
    assert set(out['seg'].unique().to_list()) == {'CS', 'Physics', 'All fields'}
    all_row = out.filter(pl.col('seg') == 'All fields').row(0, named=True)
    assert all_row['n'] == 30


def test_corpus_composition_partitions_the_corpus():
    papers = pl.DataFrame({
        'year': [2020, 2020, 2020, 2021],
        'field': ['CS', 'Physics', None, 'CS'],
    })
    out = we.corpus_composition(papers, fields_top=['CS'])
    assert out['n'].sum() == papers.height
    by_seg = dict(zip(out['seg'], out['n']))
    # can't sum duplicate segs directly since seg x year rows exist; check totals
    totals = out.group_by('seg').agg(n=pl.col('n').sum())
    d = dict(zip(totals['seg'], totals['n']))
    assert d['CS'] == 2
    assert d['Other fields'] == 1
    assert d['No field label'] == 1


def test_prevalence_series_applies_floor():
    papers = pl.DataFrame({
        'year': [2020] * 5 + [2021] * 1,
        'accel_any': [True, True, False, False, False, True],
        'accel_used': [True, False, False, False, False, True],
        'model_specific_used': [False] * 6,
        'is_candidate': [True] * 6,
    })
    out = we.prevalence_series(papers, min_papers=3)
    assert out['year'].to_list() == [2020]
    row = out.row(0, named=True)
    assert row['any_share'] == 2 / 5
    assert row['used_share'] == 1 / 5


def test_extraction_funnel_counts_are_monotone_decreasing():
    papers = pl.DataFrame({
        'is_candidate': [True, True, True, False],
        'accel_any': [True, True, False, False],
        'accel_used': [True, False, False, False],
        'model_specific_used': [True, False, False, False],
    })
    funnel = we.extraction_funnel(papers)
    vals = list(funnel.values())
    assert vals == [4, 3, 2, 1, 1]
    assert vals == sorted(vals, reverse=True)


def test_specificity_share_restricted_to_accel_used():
    papers = pl.DataFrame({
        'year': [2020] * 4,
        'accel_used': [True, True, True, False],
        'model_specific_used': [True, False, True, True],
    })
    out = we.specificity_share(papers, min_papers=1)
    row = out.row(0, named=True)
    assert row['n'] == 3
    assert row['specific'] == 2 / 3


def test_device_life_series_requires_year_at_or_after_release():
    pm = pl.DataFrame({
        'canonical_model': ['a100', 'a100', 'a100', 'v100'],
        'year': [2019, 2020, 2021, 2018],
    })
    top = pl.DataFrame({
        'canonical_model': ['a100', 'v100'],
        'display': ['NVIDIA A100', 'NVIDIA V100'],
        'release_year': [2020, 2017],
    })
    out = we.device_life_series(pm, top)
    # the 2019 a100 row predates its 2020 release -> dropped
    a100 = out.filter(pl.col('canonical_model') == 'a100')
    assert a100['year'].to_list() == [2020, 2021]
    assert a100['since_release'].to_list() == [0, 1]
