"""Tests for the pyfixest helpers in accelscan.regress.

Estimation is pyfixest's and is not re-tested here. What IS tested is the thin
layer the notebook depends on: the Arrow-free polars/pandas hand-off, the
result-tidying, the per-group slope parsing, the slope-equality contrast, and --
the reason the module exists -- the clustering guard, including a test that
demonstrates the degeneracy it exists to prevent actually happens in pyfixest.
"""

import numpy as np
import polars as pl
import pyfixest as pf
import pytest

from accelscan import regress as rg


def _panel(n=4000, seed=0):
    """Panel with known slopes, correlated fixed effects, and field-specific slopes."""
    rng = np.random.default_rng(seed)
    field = rng.choice([f'F{i}' for i in range(6)], n,
                       p=[0.45, 0.2, 0.15, 0.1, 0.06, 0.04])
    year = rng.choice(list(range(2012, 2023)), n)
    fe_f = dict(zip(sorted(set(field)), rng.normal(0, 1, 6)))
    fe_y = dict(zip(sorted(set(year)), rng.normal(0, 0.5, 11)))
    af = np.array([fe_f[f] for f in field])
    ay = np.array([fe_y[y] for y in year])
    x = 0.8 * af + 0.5 * ay + rng.normal(0, 1, n)      # x correlated with both FE
    y = 1.5 * x + af + ay + rng.normal(0, 1, n)
    return pl.DataFrame({'y': y, 'x': x, 'field': field,
                         'year_f': [str(v) for v in year],
                         'field_year': [f'{f}|{v}' for f, v in zip(field, year)],
                         'cnt': rng.poisson(np.exp(1 + 0.25 * x)).astype(float)})


def _feols(d, formula, cluster='field'):
    return pf.feols(formula, data=rg.to_pandas(d), vcov={'CRV1': cluster})


# --- polars -> pandas hand-off ---------------------------------------------

def test_to_pandas_roundtrips_without_pyarrow():
    d = _panel(n=50)
    p = rg.to_pandas(d)
    assert list(p.columns) == d.columns
    assert np.allclose(p['y'].to_numpy(), d['y'].to_numpy())
    assert p['field'].tolist() == d['field'].to_list()


def test_to_pandas_subsets_and_dedupes_columns():
    d = _panel(n=20)
    p = rg.to_pandas(d, ['y', 'x', 'y'])
    assert list(p.columns) == ['y', 'x']


def test_to_pandas_rejects_unknown_column():
    with pytest.raises(ValueError, match='columns not in frame'):
        rg.to_pandas(_panel(n=10), ['nope'])


def test_pyfixest_accepts_the_converted_frame():
    """The hand-off has to actually work end to end -- polars input needs pyarrow."""
    d = _panel(n=2000, seed=3)
    m = _feols(d, 'y ~ x | field + year_f')
    assert abs(m.coef()['x'] - 1.5) < 4 * m.se()['x']


# --- the clustering guard ---------------------------------------------------

def test_assert_cluster_finer_rejects_group_level_clustering():
    d = _panel(n=500)
    with pytest.raises(ValueError, match='sits inside a single'):
        rg.assert_cluster_finer(d, 'field', 'field')


def test_assert_cluster_finer_accepts_a_finer_cluster():
    d = _panel(n=500)
    rg.assert_cluster_finer(d, 'field', 'field_year')       # must not raise


def test_assert_cluster_finer_checks_columns_exist():
    with pytest.raises(ValueError, match='not in frame'):
        rg.assert_cluster_finer(_panel(n=10), 'field', 'nope')


def test_the_degeneracy_the_guard_prevents_is_real_in_pyfixest():
    """pyfixest reports nested-cluster interaction SEs without warning.

    This is the whole justification for `assert_cluster_finer`: the same
    specification, clustered at the interacting group's own level, returns SEs
    an order of magnitude smaller and a slope-equality chi2 inflated to
    nonsense. Nothing in the output marks it as broken.
    """
    d = _panel(n=3000, seed=5)
    f = 'y ~ i(field, x) | field + year_f'
    nested = _feols(d, f, cluster='field')
    finer = _feols(d, f, cluster='field_year')
    assert nested.se().max() < 0.2 * finer.se().max()
    assert rg.equality_test(nested)[0] > 50 * rg.equality_test(finer)[0]


# --- tidying ----------------------------------------------------------------

def test_tidy_columns_and_percent_transform():
    m = _feols(_panel(n=2000, seed=7), 'y ~ x | field + year_f')
    t = rg.tidy(m, label='headline')
    assert t.height == 1
    assert t['spec'][0] == 'headline'
    assert t['n'][0] == 2000
    assert t['fe'][0] == 'field + year_f'
    assert 'CRV1 field' in t['cluster'][0]
    assert abs(t['pct'][0] - 100 * (np.exp(t['beta'][0]) - 1)) < 1e-9
    assert abs(t['pct_lo'][0] - 100 * (np.exp(t['lo'][0]) - 1)) < 1e-9


def test_tidy_can_skip_the_percent_columns():
    m = _feols(_panel(n=500), 'y ~ x | field + year_f')
    assert 'pct' not in rg.tidy(m, pct=False).columns


def test_tidy_keep_filters_terms():
    d = _panel(n=2000, seed=11).with_columns(z=pl.col('x') * 0.5)
    m = _feols(d, 'y ~ x + z | field + year_f')
    assert rg.tidy(m, keep=['x'])['term'].to_list() == ['x']


def test_vcov_label_distinguishes_hc1_from_clustered():
    d = _panel(n=500)
    hc1 = pf.feols('y ~ x | field + year_f', data=rg.to_pandas(d), vcov='hetero')
    assert 'CRV' not in rg.vcov_label(hc1)
    assert 'CRV1 field_year' in rg.vcov_label(_feols(d, 'y ~ x | field',
                                                     cluster='field_year'))


def test_stack_labels_each_specification():
    d = _panel(n=1000, seed=13)
    t = rg.stack({'one FE': _feols(d, 'y ~ x | year_f'),
                  'two FE': _feols(d, 'y ~ x | field + year_f')}, keep=['x'])
    assert t['spec'].to_list() == ['one FE', 'two FE']
    assert t['fe'].to_list() == ['year_f', 'field + year_f']


# --- per-group slopes -------------------------------------------------------

def test_slopes_parses_every_level_with_no_reference_dropped():
    d = _panel(n=4000, seed=17)
    m = _feols(d, 'y ~ i(field, x) | field + year_f', cluster='field_year')
    s = rg.slopes(m, 'field', 'x')
    assert sorted(s['level'].to_list()) == sorted(d['field'].unique().to_list())


def test_slopes_match_per_group_regressions():
    """With only group FE, each interacted slope IS that group's own OLS slope."""
    d = _panel(n=6000, seed=19)
    m = _feols(d, 'y ~ i(field, x) | field', cluster='field_year')
    s = rg.slopes(m, 'field', 'x')
    for r in s.iter_rows(named=True):
        own = pf.feols('y ~ x', data=rg.to_pandas(d.filter(pl.col('field') == r['level'])))
        assert abs(r['beta'] - own.coef()['x']) < 1e-8


def test_slopes_raises_when_the_model_has_no_interaction():
    m = _feols(_panel(n=500), 'y ~ x | field + year_f')
    with pytest.raises(ValueError, match='no i\\(field, x\\)'):
        rg.slopes(m, 'field', 'x')


def test_equality_test_ignores_a_common_slope_and_catches_a_different_one():
    rng = np.random.default_rng(23)
    n = 8000
    g = rng.choice(['a', 'b'], n)
    x = rng.normal(0, 1, n)
    yr = rng.choice(['2010', '2011'], n)
    base = pl.DataFrame({'x': x, 'field': g, 'year_f': yr,
                         'field_year': [f'{a}|{b}' for a, b in zip(g, yr)]})
    same = base.with_columns(y=2 * pl.col('x') + pl.Series(rng.normal(0, 1, n)))
    m = _feols(same, 'y ~ i(field, x) | field + year_f', cluster='field_year')
    assert rg.equality_test(m)[2] > 0.05
    diff = base.with_columns(
        y=pl.when(pl.col('field') == 'a').then(2 * pl.col('x'))
        .otherwise(-pl.col('x')) + pl.Series(rng.normal(0, 1, n)))
    m2 = _feols(diff, 'y ~ i(field, x) | field + year_f', cluster='field_year')
    assert rg.equality_test(m2)[2] < 1e-6


def test_equality_test_needs_two_coefficients():
    m = _feols(_panel(n=500), 'y ~ x | field + year_f')
    with pytest.raises(ValueError, match='need >= 2'):
        rg.equality_test(m)


# --- dose-response ----------------------------------------------------------

def test_quantile_bins_partition_the_sample():
    d, summary = rg.quantile_bins(_panel(n=5000, seed=29), 'x', 10)
    assert summary.height == 10
    assert summary['n_bin'].sum() == d.height
    assert summary['x_mean'].to_list() == sorted(summary['x_mean'].to_list())


def test_quantile_bins_survive_heavy_ties():
    d = pl.DataFrame({'v': [1.0] * 900 + [2.0] * 50 + [3.0] * 50})
    binned, summary = rg.quantile_bins(d, 'v', 5)
    assert binned.height == 1000
    assert summary['n_bin'].sum() == 1000


def test_dose_table_has_an_explicit_zero_reference_and_bin_support():
    d = _panel(n=5000, seed=31)
    binned, summary = rg.quantile_bins(d, 'x', 6)
    m = _feols(binned, 'y ~ C(bin) | field + year_f', cluster='field_year')
    tab = rg.dose_table(m, summary)
    assert tab.height == 6
    assert tab['beta'][0] == 0.0 and tab['se'][0] == 0.0
    assert tab['n_bin'].sum() == binned.height
    assert tab['beta'].to_list() == sorted(tab['beta'].to_list())   # linear DGP


# --- Poisson ----------------------------------------------------------------

def test_fepois_tidies_like_feols():
    d = _panel(n=4000, seed=37)
    m = pf.fepois('cnt ~ x | field + year_f', data=rg.to_pandas(d),
                  vcov={'CRV1': 'field'})
    t = rg.tidy(m, label='PPML')
    assert t['spec'][0] == 'PPML'
    assert abs(t['beta'][0] - 0.25) < 4 * t['se'][0]


# --- citation re-scaling ----------------------------------------------------

def test_delta_citations_matches_the_formula_and_is_monotone():
    assert rg.delta_citations(0.0, 13) == 0.0
    assert abs(rg.delta_citations(0.113, 13) - 14 * (np.exp(0.113) - 1)) < 1e-12
    b = np.array([-0.2, 0.0, 0.1, 0.3])
    got = rg.delta_citations(b, 13)
    assert got.tolist() == sorted(got.tolist())
    assert got[0] < 0                     # a negative coefficient loses citations


def test_delta_citations_scales_with_the_baseline():
    """Proportional effects mean the citation conversion depends on where you stand."""
    lo = rg.delta_citations(0.113, 5)
    hi = rg.delta_citations(0.113, 100)
    assert hi > 10 * lo


def test_delta_citations_accepts_polars_expressions():
    """The same helper has to serve scalars, tick arrays, and per-row baselines."""
    d = pl.DataFrame({'beta': [0.0, 0.113], 'base': [13.0, 100.0]})
    got = d.with_columns(
        fixed=rg.delta_citations(pl.col('beta'), 13.0),
        per_row=rg.delta_citations(pl.col('beta'), pl.col('base')))
    assert got['fixed'].to_list() == pytest.approx([0.0, 14 * (np.exp(0.113) - 1)])
    assert got['per_row'][0] == 0.0
    assert got['per_row'][1] == pytest.approx(101 * (np.exp(0.113) - 1))
