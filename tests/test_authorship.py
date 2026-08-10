"""Author-level panel, jump detection, and the event-study estimator.

The fixture is a hand-built 6-author graph whose every expected number is
derivable by reading it, so a failure here says *which* piece of the panel logic
broke rather than that a 100M-row job produced surprising output. The cases are
the ones that would otherwise be silent:

- an author-year with no papers (densification) -- the event study lives or dies
  on zero-output years existing
- access recorded as null, never 0, when a paper reports no spec'd accelerator
- a non-full-text accelerator paper, which must count toward output but never
  toward access
- a ratio jump whose prior max comes from the *stock*, not the previous year
- a jump whose capacity was already held by a coauthor (`coauthor_sourced`)
"""

import numpy as np
import polars as pl
import pytest

from accelscan import authorship as au

# corpusid, authorid list, year, fulltext, accel_used, fp32 gflops, datacenter, i_3
PAPERS = [
    # a1: steady small-GPU work, then a 10x jump in 2016 to a datacenter part
    (1, ['a1'], 2012, True, True, 1_000.0, False, 3),
    (2, ['a1'], 2013, True, True, 1_500.0, False, 4),
    # 2014: no papers at all -> must exist in the panel as a zero row
    (3, ['a1'], 2015, True, True, 1_200.0, False, 2),
    (4, ['a1', 'a2'], 2016, True, True, 15_000.0, True, 20),
    (5, ['a1'], 2017, True, True, 15_000.0, True, 25),
    (6, ['a1'], 2018, True, True, 15_000.0, True, 30),
    # a2: already held the big part before a1's jump -> coauthor_sourced
    (7, ['a2'], 2013, True, True, 14_000.0, True, 5),
    (8, ['a2'], 2014, True, True, 14_000.0, True, 6),
    (9, ['a2'], 2015, True, True, 14_000.0, True, 7),
    (10, ['a2'], 2017, True, True, 14_000.0, True, 8),
    # a3: GPU user, never jumps -> risk-set control material
    (11, ['a3'], 2012, True, True, 1_000.0, False, 1),
    (12, ['a3'], 2013, True, True, 1_100.0, False, 2),
    (13, ['a3'], 2014, True, True, 1_000.0, False, 1),
    (14, ['a3'], 2015, True, True, 1_050.0, False, 2),
    (15, ['a3'], 2016, True, True, 1_000.0, False, 3),
    (16, ['a3'], 2017, True, True, 1_100.0, False, 2),
    (17, ['a3'], 2018, True, True, 1_000.0, False, 1),
    # a4: never touches an accelerator
    (18, ['a4'], 2012, True, False, None, False, 1),
    (19, ['a4'], 2013, True, False, None, False, 2),
    (20, ['a4'], 2016, True, False, None, False, 1),
    # a5: an accelerator paper OUTSIDE the full-text corpus. Counts as output,
    # must not count as access -- it was never scanned.
    (21, ['a5'], 2012, False, True, 9_000.0, True, 4),
    (22, ['a5'], 2013, True, False, None, False, 1),
    (23, ['a5'], 2014, True, False, None, False, 1),
    # a6: an accelerator user whose model has no spec -> access null, n_accel 1
    (24, ['a6'], 2013, True, True, None, False, 2),
    (25, ['a6'], 2014, True, True, None, False, 1),
    (26, ['a6'], 2015, True, True, None, False, 1),
]

COLS = ['corpusid', 'authors', 'year', 'fulltext', 'accel_used',
        'reported_fp32_gflops', 'any_datacenter', 'i_3']


@pytest.fixture
def rows() -> pl.DataFrame:
    """(author, paper) rows in the shape `author_year_panel` consumes."""
    df = pl.DataFrame(
        {c: [r[i] for r in PAPERS] for i, c in enumerate(COLS)},
        schema={'corpusid': pl.Int64, 'authors': pl.List(pl.Utf8),
                'year': pl.Int16, 'fulltext': pl.Boolean, 'accel_used': pl.Boolean,
                'reported_fp32_gflops': pl.Float64, 'any_datacenter': pl.Boolean,
                'i_3': pl.Int64})
    return (df.with_columns(n_authors=pl.col('authors').list.len().cast(pl.UInt16),
                            field=pl.lit('Computer Science'))
            .explode('authors', empty_as_null=True).rename({'authors': 'authorid'})
            .with_columns(
                any_consumer=~pl.col('any_datacenter'),
                reported_vram_gb=pl.col('reported_fp32_gflops') / 1000,
                reported_tensor_gflops=pl.col('reported_fp32_gflops') * 2,
                # paper 4 (a1's jump) is an 8-GPU cluster, everything else a single
                # card -- so the per-device column is genuinely exercised rather
                # than being a copy of the per-paper one
                total_devices=pl.when(pl.col('corpusid') == 4).then(8.0)
                .otherwise(1.0),
                newest_release_year=pl.when(pl.col('accel_used'))
                .then(pl.col('year').cast(pl.Int32) - 2).otherwise(None),
                # the pipeline gates access on full text before panel assembly
                accel_used=pl.col('accel_used') & pl.col('fulltext')))


@pytest.fixture
def panel(rows) -> pl.DataFrame:
    return au.add_outcomes(au.author_year_panel(rows))


# --- crosswalk --------------------------------------------------------------

def test_explode_authors_preserves_every_list_element():
    part = pl.DataFrame(
        {'corpusid': [1, 2, 3],
         'year': [2020, 2021, 2022],
         'authors': [['x', 'y', 'z'], None, ['q']],
         's2fieldsofstudy': [[{'category': 'Physics', 'source': 's2'}], None,
                             [{'category': 'Biology', 'source': 's2'}]]},
        schema={'corpusid': pl.Int64, 'year': pl.Int64,
                'authors': pl.List(pl.Utf8),
                's2fieldsofstudy': pl.List(
                    pl.Struct({'category': pl.Utf8, 'source': pl.Utf8}))})
    ft = pl.DataFrame({'corpusid': [1], 'fulltext': [True]})
    out = au.explode_authors(part, ft)
    assert out.height == 4, 'one row per author-list element; null list drops out'
    assert out.filter(pl.col('corpusid') == 1)['author_pos'].to_list() == [0, 1, 2]
    assert out.filter(pl.col('corpusid') == 1)['n_authors'].to_list() == [3, 3, 3]
    assert out.filter(pl.col('corpusid') == 1)['fulltext'].to_list() == [True] * 3
    assert out.filter(pl.col('corpusid') == 3)['fulltext'].to_list() == [False]
    assert out.filter(pl.col('corpusid') == 3)['field'].to_list() == ['Biology']


def test_explode_authors_handles_a_null_id_inside_a_byline():
    """~0.1% of real byline slots carry a null author id. That is a different
    thing from a paper with no byline, and conflating the two would silently
    shift co-authors into first position."""
    part = pl.DataFrame(
        {'corpusid': [1, 2],
         'year': [2020, 2021],
         'authors': [[None, 'y', 'z'], []],
         's2fieldsofstudy': [None, None]},
        schema={'corpusid': pl.Int64, 'year': pl.Int64,
                'authors': pl.List(pl.Utf8),
                's2fieldsofstudy': pl.List(
                    pl.Struct({'category': pl.Utf8, 'source': pl.Utf8}))})
    ft = pl.DataFrame({'corpusid': [1], 'fulltext': [True]})
    out = au.explode_authors(part, ft)
    assert out.height == 2, 'the null id drops, the empty byline contributes nothing'
    assert out['authorid'].to_list() == ['y', 'z']
    assert out['author_pos'].to_list() == [1, 2], 'positions must not close the gap'
    assert out['n_authors'].to_list() == [3, 3], 'team size counts the unresolved slot'


def test_explode_authors_rejects_a_mangled_author_column():
    """A duplicated/expanded authors list would multiply output silently."""
    part = pl.DataFrame(
        {'corpusid': [1], 'year': [2020], 'authors': [['x']],
         's2fieldsofstudy': [None]},
        schema={'corpusid': pl.Int64, 'year': pl.Int64, 'authors': pl.List(pl.Utf8),
                's2fieldsofstudy': pl.List(
                    pl.Struct({'category': pl.Utf8, 'source': pl.Utf8}))})
    ft = pl.DataFrame({'corpusid': [1], 'fulltext': [True]})
    au.explode_authors(part, ft)          # baseline passes
    bad = part.with_columns(authors=pl.lit(None, dtype=pl.List(pl.Utf8)))
    assert au.explode_authors(bad, ft).height == 0


def test_hash_pool_is_deterministic_and_rate_proportional():
    ids = pl.DataFrame({'authorid': [f'a{i}' for i in range(20_000)]})
    a = ids.filter(au.hash_pool(pl.col('authorid'), 0.10))
    b = ids.filter(au.hash_pool(pl.col('authorid'), 0.10))
    assert a['authorid'].to_list() == b['authorid'].to_list()
    assert 0.08 < a.height / ids.height < 0.12
    # a larger rate is a superset: an author never leaves the pool as it widens
    wide = set(ids.filter(au.hash_pool(pl.col('authorid'), 0.20))['authorid'])
    assert set(a['authorid']) <= wide


# --- panel ------------------------------------------------------------------

def test_panel_densifies_gap_years(panel):
    a1 = panel.filter(pl.col('authorid') == 'a1').sort('year')
    assert a1['year'].to_list() == list(range(2012, 2019))
    gap = a1.filter(pl.col('year') == 2014)
    assert gap['n_papers'].item() == 0 and gap['n_accel'].item() == 0
    assert gap['access_fp32_max'].item() is None, 'no papers means no new access'
    assert gap['access_fp32_stock'].item() == 1500.0, 'stock persists across the gap'
    assert a1.filter(pl.col('year') == 2012)['career_year'].item() == 1


def test_access_is_null_not_zero_when_unspecced(panel):
    a6 = panel.filter(pl.col('authorid') == 'a6').sort('year')
    assert a6['n_accel'].to_list() == [1, 1, 1]
    assert a6['access_fp32_max'].to_list() == [None] * 3
    assert a6['access_fp32_stock'].to_list() == [None] * 3
    assert a6['ever_gpu'].to_list() == [True] * 3


def test_nonfulltext_accel_paper_counts_as_output_not_access(panel):
    a5 = panel.filter(pl.col('authorid') == 'a5').sort('year')
    y12 = a5.filter(pl.col('year') == 2012)
    assert y12['n_papers'].item() == 1
    assert y12['n_fulltext'].item() == 0
    assert y12['n_accel'].item() == 0, 'unscanned paper is not accelerator evidence'
    assert y12['access_fp32_max'].item() is None
    assert a5['ever_gpu'].to_list() == [False] * 3


def test_stock_is_a_running_max_and_w3_is_a_window(panel):
    a1 = panel.filter(pl.col('authorid') == 'a1').sort('year')
    assert a1['access_fp32_stock'].to_list() == [
        1000.0, 1500.0, 1500.0, 1500.0, 15000.0, 15000.0, 15000.0]
    # the 3-year window still sees 2013's 1500 in 2015, but has forgotten it by
    # 2016 -- which is the whole point of the windowed robustness variant
    assert a1['access_fp32_stock_w3'].to_list() == [
        1000.0, 1500.0, 1500.0, 1500.0, 15000.0, 15000.0, 15000.0]


def test_counts_and_citations(panel):
    a1 = panel.filter((pl.col('authorid') == 'a1') & (pl.col('year') == 2016))
    assert a1['n_papers'].item() == 1 and a1['n_accel'].item() == 1
    assert a1['n_nonaccel'].item() == 0 and a1['n_solo'].item() == 0
    assert a1['cit_i3_mean'].item() == 20.0
    # fractional attribution halves the 2-author paper
    assert a1['access_frac_sum'].item() == pytest.approx(7_500.0)
    a4 = panel.filter((pl.col('authorid') == 'a4') & (pl.col('year') == 2012))
    assert a4['n_nonaccel'].item() == 1 and a4['n_solo'].item() == 1


def test_activity_filters_and_invariants(panel):
    kept = au.apply_activity_filters(panel)
    # every fixture author clears >=3 papers / >=3 years / first year <= 2021
    assert set(kept['authorid'].unique()) == {f'a{i}' for i in range(1, 7)}
    au._check_invariants(kept, au.find_jumps(kept), au.JUMP_RATIO)


def test_invariants_catch_a_zero_access_value(panel):
    broken = panel.with_columns(
        access_fp32_max=pl.col('access_fp32_max').fill_null(0.0))
    with pytest.raises(AssertionError, match='never 0'):
        au._check_invariants(broken, au.find_jumps(panel), au.JUMP_RATIO)


# --- jumps ------------------------------------------------------------------

def test_ratio_jump_uses_the_stock_not_the_previous_year(panel):
    j = au.find_jumps(panel, jump_ratio=4.0).filter(
        pl.col('event_type') == 'ratio_jump')
    a1 = j.filter(pl.col('authorid') == 'a1')
    assert a1.height == 1 and a1['event_year'].item() == 2016
    # prior year (2015) was 1200, but the career max was 1500 -> ratio 10, not 12.5
    assert a1['pre_max_fp32'].item() == 1500.0
    assert a1['jump_ratio'].item() == pytest.approx(10.0)
    assert a1['career_year_at_event'].item() == 5
    assert 'a3' not in j['authorid'].to_list(), 'flat access is not a jump'
    assert 'a2' not in j['authorid'].to_list(), 'no prior access to jump from'


def test_jump_ratio_threshold_binds(panel):
    lo = au.find_jumps(panel, jump_ratio=2.0).filter(
        pl.col('event_type') == 'ratio_jump')
    hi = au.find_jumps(panel, jump_ratio=20.0).filter(
        pl.col('event_type') == 'ratio_jump')
    assert 'a1' in lo['authorid'].to_list()
    assert 'a1' not in hi['authorid'].to_list()


def test_gpu_first_and_dc_first(panel):
    j = au.find_jumps(panel)
    gf = dict(zip(*j.filter(pl.col('event_type') == 'gpu_first')
                  .select('authorid', 'event_year').to_dict(as_series=False).values()))
    assert gf['a1'] == 2012 and gf['a6'] == 2013
    assert 'a4' not in gf and 'a5' not in gf
    dc = dict(zip(*j.filter(pl.col('event_type') == 'dc_first')
                  .select('authorid', 'event_year').to_dict(as_series=False).values()))
    assert dc['a1'] == 2016, 'a1 crosses into datacenter parts in 2016'
    assert dc['a2'] == 2013
    assert 'a3' not in dc


def test_coauthor_sourced_flags_the_well_equipped_collaborator(rows, panel):
    jumps = au.find_jumps(panel)
    jp = au.jump_papers(rows, jumps, panel)
    a1 = jp.filter((pl.col('authorid') == 'a1')
                   & (pl.col('event_type') == 'ratio_jump'))
    assert a1['jump_corpusids'].item().to_list() == [4]
    assert a1['n_jump_papers'].item() == 1
    assert a1['min_team_size'].item() == 2
    # a2 held 14000 in 2015 (stock), which is >= 15000/2 -> the jump is sourced
    assert a1['coauthor_prior_max'].item() == 14_000.0
    assert a1['coauthor_sourced'].item() is True
    # a solo jump cannot be coauthor-sourced
    a2 = jp.filter((pl.col('authorid') == 'a2') & (pl.col('event_type') == 'dc_first'))
    assert a2['coauthor_sourced'].item() is False


def test_pre_papers_mean_excludes_tau_minus_one(panel):
    ev = pl.DataFrame({'authorid': ['a3'], 'event_year': pl.Series([2017], dtype=pl.Int16)})
    got = au.pre_papers_mean(panel, ev)
    # tau in -5..-2 is 2012..2015: four papers; 2016 (tau=-1) excluded
    assert got['n_pre_years'].item() == 4
    assert got['pre_papers_mean'].item() == pytest.approx(1.0)


# --- event study ------------------------------------------------------------

def _synthetic_panel(n_jumpers=200, n_controls=1000, effect=1.0, seed=0,
                     noise=0.5):
    """Panel with a known post-jump effect, so theta is checkable against truth.

    `noise` is deliberately large relative to `effect`: with a near-noiseless DGP
    the bootstrap CI collapses to a width of ~0.01 and every residual imbalance
    reads as significant, which tests the DGP rather than the estimator. The
    control pool is 5x the jumper count so `match_controls` is not forced to
    reuse the same handful of controls for every jumper.
    """
    rng = np.random.default_rng(seed)
    rec = []
    for i in range(n_jumpers + n_controls):
        aid = f'j{i}' if i < n_jumpers else f'c{i}'
        jumper = i < n_jumpers
        level = rng.normal(4, 1)
        for year in range(2005, 2016):
            tau = year - 2010
            y = level + rng.normal(0, noise)
            if jumper and tau >= 0:
                y += effect
            rec.append({'authorid': aid, 'year': year, 'n_papers': y,
                        'modal_field': 'Computer Science', 'ever_gpu': True,
                        'y0': 2005, 'y1': 2015, 'career_year': year - 2004,
                        'control_sample': 'gpu_ever'})
    panel = pl.DataFrame(rec).with_columns(pl.col('year').cast(pl.Int16))
    jumps = pl.DataFrame({'authorid': [f'j{i}' for i in range(n_jumpers)]}) \
        .with_columns(event_year=pl.lit(2010, dtype=pl.Int16),
                      field=pl.lit('Computer Science'),
                      career_year_at_event=pl.lit(6))
    jumps = jumps.join(au.pre_papers_mean(panel, jumps.select('authorid', 'event_year')),
                       on=['authorid', 'event_year'], how='left')
    return panel, jumps


def test_event_study_recovers_a_known_effect():
    panel, jumps = _synthetic_panel(effect=1.0)
    ctrl = au.match_controls(panel, jumps, variant='riskset', seed=au.SEED)
    assert ctrl.height > 0, 'no controls matched'
    res = au.event_study(panel, jumps, ctrl, 'n_papers', b=200).sort('tau')
    pre = res.filter(pl.col('tau') < 0)
    post = res.filter(pl.col('tau') > 0)
    # baseline demeaning forces the pre-period to ~0 by construction, and the
    # post-period must land on the true effect -- an estimator that leaked the
    # effect into the pre-period (the failure that would invalidate every
    # pre-trend diagnostic in the notebook) shows up here as a nonzero pre theta
    assert pre['theta'].abs().max() < 0.15
    assert post['theta'].min() > 0.85 and post['theta'].max() < 1.15
    # the effect must be resolved well clear of zero; pre-period CIs are not
    # pinned (see the placebo test for why coverage is not the claim here)
    assert post['lo'].min() > 0.5


def test_event_study_is_null_under_a_placebo():
    panel, jumps = _synthetic_panel(effect=0.0)
    ctrl = au.match_controls(panel, jumps, seed=au.SEED)
    res = au.event_study(panel, jumps, ctrl, 'n_papers', b=200)
    # Magnitude, not significance: with controls shared across jumpers and matched
    # on a pre-period statistic, the block bootstrap over jumpers is mildly
    # anti-conservative, so exact CI coverage under the null is not a property
    # worth pinning. What must hold is that the null estimate is an order of
    # magnitude below the unit effect the previous test recovers, and that the CI
    # is not absurdly narrow relative to the scatter.
    assert res['theta'].abs().max() < 0.15
    assert res.filter(pl.col('tau') > 0)['theta'].abs().mean() < 0.1
    ratio = (res['theta'].abs() / ((res['hi'] - res['lo']) / 2)).max()
    assert ratio < 3, f'point estimate {ratio:.1f}x the CI half-width under a null'


def test_match_controls_excludes_already_jumped_authors():
    panel, jumps = _synthetic_panel(n_jumpers=40, n_controls=60)
    # make half the "controls" jump in 2008 -- before the pseudo-event
    early = pl.DataFrame({'authorid': [f'c{i}' for i in range(40, 70)]}) \
        .with_columns(event_year=pl.lit(2008, dtype=pl.Int16),
                      field=pl.lit('Computer Science'),
                      career_year_at_event=pl.lit(4),
                      pre_papers_mean=pl.lit(4.0))
    ctrl = au.match_controls(panel, pl.concat([jumps, early], how='diagonal'),
                             seed=au.SEED)
    used = set(ctrl.filter(pl.col('event_year') == 2010)['control_id'])
    assert used and not (used & set(early['authorid'])), \
        'an author who already jumped is not a valid risk-set control'


def test_match_controls_rejects_an_unknown_variant():
    panel, jumps = _synthetic_panel(n_jumpers=5, n_controls=5)
    with pytest.raises(ValueError, match='control variant'):
        au.match_controls(panel, jumps, variant='nope')


def test_add_outcomes_defines_every_named_outcome(panel):
    p = au.add_outcomes(panel)
    for o in au.OUTCOMES:
        assert o in p.columns, f'{o} listed in OUTCOMES but not built'
    with pytest.raises(ValueError, match='not in panel'):
        au.event_study(panel.drop('log_papers'), au.find_jumps(panel),
                       pl.DataFrame({'authorid': [], 'event_year': [],
                                     'control_id': []}), 'log_papers')


# --- control matching: the two defects the 6-year placebo exposed -------------

def _career_panel(seed=0):
    """Young jumpers, plus young and old candidate controls.

    Every author has the SAME pre-period output, so the field and output-quartile
    parts of the stratum key cannot tell the two control pools apart -- career-age
    bin is the only thing that can. Career ages at the 2010 event are 8 (bin 1)
    and 26 (bin 5), and the jumps table carries a stale 26.
    """
    rng = np.random.default_rng(seed)
    rec = []
    for i in range(120):
        aid, y0 = ((f'j{i}', 2003) if i < 40 else
                   (f'young{i}', 2003) if i < 80 else (f'old{i}', 1985))
        for year in range(2003, 2016):
            rec.append({'authorid': aid, 'year': year,
                        'n_papers': 5 + rng.normal(0, 0.1),
                        'modal_field': 'Computer Science', 'ever_gpu': True,
                        'y0': y0, 'y1': 2015, 'career_year': year - y0 + 1,
                        'control_sample': 'gpu_ever'})
    panel = pl.DataFrame(rec).with_columns(pl.col('year').cast(pl.Int16))
    jumps = pl.DataFrame({'authorid': [f'j{i}' for i in range(40)]}).with_columns(
        event_year=pl.lit(2010, dtype=pl.Int16), field=pl.lit('Computer Science'),
        event_type=pl.lit('ratio_jump'),
        # Stale on purpose: says career year 26 (the OLD cohort's age) when the
        # truth at 2010 is 8. This is the shape a re-timed jump table has -- the
        # 6-year-shifted placebo, or a threshold sweep.
        career_year_at_event=pl.lit(26))
    return panel, jumps.join(
        au.pre_papers_mean(panel, jumps.select('authorid', 'event_year')),
        on=['authorid', 'event_year'], how='left')


def test_career_age_is_derived_not_read_from_a_stale_jumps_column():
    """The jumps table claims career year 21; the panel says 3. If the stale
    column wins, every young jumper is matched to a much older control."""
    panel, jumps = _career_panel()
    ctrl = au.match_controls(panel, jumps, seed=au.SEED)
    assert ctrl.height, 'no controls matched'
    picked = set(ctrl['control_id'])
    assert not any(c.startswith('old') for c in picked), (
        'matched an old control to a young jumper -- career age came from the '
        'stale jumps column instead of the panel')
    assert all(c.startswith('young') for c in picked)


def test_risk_events_keeps_real_jumpers_out_of_a_placebo_control_pool():
    """A placebo's treated units are non-jumpers, so the real jumps must be what
    defines eligibility -- otherwise real jumpers serve as controls and carry the
    very effect the placebo should fail to find."""
    panel, jumps = _synthetic_panel(n_jumpers=40, n_controls=60)
    placebo = au.placebo_jumps(panel, jumps.with_columns(
        event_type=pl.lit('ratio_jump')), seed=au.SEED)
    assert placebo.height, 'no placebo events built'
    assert not (set(placebo['authorid']) & set(jumps['authorid'])), \
        'a real jumper was used as a placebo treated unit'
    ctrl = au.match_controls(panel, placebo, seed=au.SEED, risk_events=jumps)
    assert ctrl.height
    # The property is risk-set, not disjointness: an author who jumps LATER than
    # the pseudo-event year is a legitimate not-yet-treated control, and on real
    # data many are. What must never happen is a control who had already jumped.
    real = jumps.select(control_id='authorid', real_jump='event_year')
    early = (ctrl.join(real, on='control_id', how='inner')
             .filter(pl.col('real_jump') <= pl.col('event_year')))
    assert early.height == 0, (
        f'{early.height} controls had already jumped at the pseudo-event year')
    # without risk_events the real jumps are invisible, so already-jumped authors
    # are drawn in
    loose = au.match_controls(panel, placebo, seed=au.SEED)
    assert (loose.join(real, on='control_id', how='inner')
            .filter(pl.col('real_jump') <= pl.col('event_year')).height > 0)


def test_never_jumper_placebo_is_null():
    panel, jumps = _synthetic_panel(effect=1.0)
    placebo = au.placebo_jumps(panel, jumps.with_columns(
        event_type=pl.lit('ratio_jump')), seed=au.SEED)
    ctrl = au.match_controls(panel, placebo, seed=au.SEED, risk_events=jumps)
    res = au.event_study(panel, placebo, ctrl, 'n_papers', b=200)
    # the real effect is 1.0; the placebo must be an order of magnitude below it
    assert res['theta'].abs().max() < 0.15, res
    # and flat: no monotone drift, which is the signature the shifted-jumper
    # placebo showed
    post = res.filter(pl.col('tau') > 0).sort('tau')['theta'].to_list()
    assert abs(post[-1] - post[0]) < 0.15, f'placebo drifts across tau: {post}'


def test_as_share_of_baseline_is_an_exact_rescaling():
    """The reader-facing scale must be a linear rescaling of the count estimate,
    not a back-transform of the log one: exp() of a difference of mean logs is
    only an approximation (+1.49 vs +1.23 papers at tau=-1 on the full panel)."""
    res = pl.DataFrame({'tau': [-1, 0, 1], 'theta': [1.23, 0.5, 1.78],
                        'lo': [1.17, 0.4, 1.675], 'hi': [1.292, 0.6, 1.874],
                        'n': [10, 10, 10]})
    got = au.as_share_of_baseline(res, 4.33)
    assert got.filter(pl.col('tau') == 1)['theta_pct'].item() == pytest.approx(41.11, abs=0.01)
    # ordering is preserved, so the CI stays a CI
    assert (got['lo_pct'] <= got['theta_pct']).all()
    assert (got['theta_pct'] <= got['hi_pct']).all()
    # exact: rescaling twice by reciprocal factors returns the original
    back = au.as_share_of_baseline(got.drop('theta_pct', 'lo_pct', 'hi_pct'), 1.0)
    assert back['theta_pct'].to_list() == pytest.approx([123.0, 50.0, 178.0])
    with pytest.raises(ValueError, match='non-zero'):
        au.as_share_of_baseline(res, 0.0)


# --- calendar time and composition -------------------------------------------

def _year_shock_panel(n_jumpers=150, n_controls=600, effect=1.0, shock=3.0,
                      seed=0, noise=0.4):
    """Known effect PLUS a large calendar-year shock common to everyone.

    The shock is the thing the user's question is about: accelerator reporting
    rises over calendar time, so an estimator that fails to absorb year effects
    will attribute that rise to the event. Event years are spread, so calendar
    year and event time are not collinear -- with a single event year they would
    be, and the test would be vacuous.
    """
    rng = np.random.default_rng(seed)
    year_fx = {y: shock * (y - 2005) / 10 for y in range(2003, 2021)}
    rec, ev = [], []
    for i in range(n_jumpers + n_controls):
        jumper = i < n_jumpers
        aid = f'j{i}' if jumper else f'c{i}'
        e = 2010 + (i % 5)                       # event years 2010..2014
        level = rng.normal(4, 1)
        for year in range(2003, 2021):
            y = level + year_fx[year] + rng.normal(0, noise)
            if jumper and year >= e:
                y += effect
            rec.append({'authorid': aid, 'year': year, 'n_papers': y,
                        'modal_field': 'Computer Science', 'ever_gpu': True,
                        'y0': 2003, 'y1': 2020, 'career_year': year - 2002,
                        'control_sample': 'gpu_ever'})
        if jumper:
            ev.append({'authorid': aid, 'event_year': e})
    panel = pl.DataFrame(rec).with_columns(pl.col('year').cast(pl.Int16))
    jumps = pl.DataFrame(ev).with_columns(
        pl.col('event_year').cast(pl.Int16), field=pl.lit('Computer Science'),
        event_type=pl.lit('ratio_jump'))
    jumps = jumps.join(
        au.pre_papers_mean(panel, jumps.select('authorid', 'event_year')),
        on=['authorid', 'event_year'], how='left')
    return panel, jumps


def test_twoway_fe_recovers_the_effect_through_a_calendar_shock():
    panel, jumps = _year_shock_panel(effect=1.0, shock=3.0)
    ctrl = au.match_controls(panel, jumps, seed=au.SEED)
    assert ctrl.height
    fe = au.twoway_fe(panel, jumps, ctrl, 'n_papers').sort('tau')
    post = fe.filter(pl.col('tau') > 0)
    pre = fe.filter(pl.col('tau') < 0)          # only tau=-1; the rest is reference
    assert post['beta'].min() > 0.75 and post['beta'].max() < 1.25, fe
    assert pre['beta'].abs().max() < 0.25, fe
    # the year shock is 3.0 per decade and must not leak into the event-time terms
    assert abs(post['beta'].mean() - 1.0) < 0.15, fe


def test_twoway_fe_agrees_with_the_matched_difference():
    """The two estimators control calendar time differently -- year dummies vs a
    within-event-year matched difference. They should agree; if they ever stop,
    the matched cells are not absorbing calendar time."""
    panel, jumps = _year_shock_panel(effect=1.0, shock=3.0)
    ctrl = au.match_controls(panel, jumps, seed=au.SEED)
    fe = au.twoway_fe(panel, jumps, ctrl, 'n_papers')
    es = au.event_study(panel, jumps, ctrl, 'n_papers', b=100)
    j = fe.join(es.select('tau', 'theta'), on='tau', how='inner')
    assert j.height >= 5
    assert (j['beta'] - j['theta']).abs().max() < 0.2, j


def test_twoway_fe_reference_is_the_whole_baseline_window():
    panel, jumps = _year_shock_panel(n_jumpers=60, n_controls=200)
    ctrl = au.match_controls(panel, jumps, seed=au.SEED)
    fe = au.twoway_fe(panel, jumps, ctrl, 'n_papers')
    got = set(fe['tau'].to_list())
    assert got.isdisjoint(set(au.BASE_TAUS)), 'baseline taus must be the reference'
    assert got == set(range(au.TAU_MIN, au.TAU_MAX + 1)) - set(au.BASE_TAUS)
    with pytest.raises(ValueError, match='not in panel'):
        au.twoway_fe(panel, jumps, ctrl, 'nope')


def test_balanced_jumps_keeps_only_fully_observable_windows():
    panel, jumps = _year_shock_panel(n_jumpers=60, n_controls=60)
    # panel spans 2003-2020 and every career runs the whole span, so an event at
    # 2010-2014 has a full +/-5 window and all events survive
    assert au.balanced_jumps(panel, jumps).height == jumps.height
    # an event 3 years before the panel ends cannot have a +5 window
    late = jumps.with_columns(event_year=pl.lit(2018, dtype=pl.Int16))
    assert au.balanced_jumps(panel, late).height == 0
    # nor can one 2 years after it starts have a -5 window
    early = jumps.with_columns(event_year=pl.lit(2005, dtype=pl.Int16))
    assert au.balanced_jumps(panel, early).height == 0
    # a career that ends before tau_max is excluded: the panel stops at y1, so
    # those rows are missing rather than zero, and tau=+5 would be unobserved
    short = panel.filter((pl.col('authorid').str.starts_with('j').not_())
                         | (pl.col('year') <= 2013)).with_columns(
        y1=pl.when(pl.col('authorid').str.starts_with('j')).then(2013)
        .otherwise(pl.col('y1')))
    assert au.balanced_jumps(short, jumps).height == 0


# --- frontier distance --------------------------------------------------------

def test_per_device_access_column(panel):
    """The vendor frontier is a single-device spec, so the author side has to be
    per device too -- otherwise scale-out is mistaken for vintage."""
    a1 = panel.filter(pl.col('authorid') == 'a1').sort('year')
    # 2016: 15,000 GFLOPS across 8 devices
    y16 = a1.filter(pl.col('year') == 2016)
    assert y16['access_fp32_max'].item() == 15_000.0
    assert y16['access_percard_fp32_max'].item() == 15_000.0 / 8
    assert y16['access_vintage_max'].item() == 2014
    # single-device years: the two columns coincide
    y13 = a1.filter(pl.col('year') == 2013)
    assert y13['access_percard_fp32_max'].item() == y13['access_fp32_max'].item()
    # never above the per-paper figure, anywhere
    both = panel.filter(pl.col('access_fp32_max').is_not_null())
    assert (both['access_percard_fp32_max'] <= both['access_fp32_max']).all()


def _frontier_panel(seed=0, n=400):
    """Authors whose access grows, against a threshold that grows faster.

    The point of the design under test: because the threshold is a within-year
    quantile, an author whose absolute capacity rises steadily does NOT drift
    across it. Only a relative move does.
    """
    rng = np.random.default_rng(seed)
    rec = []
    for i in range(n):
        aid = f'a{i}'
        base = 10 ** rng.uniform(2, 4)
        # a1x authors make a big relative move in 2015; the rest track the median
        mover = i % 4 == 0
        for year in range(2008, 2021):
            g = base * (1.35 ** (year - 2008))       # everyone grows with the field
            if mover and year >= 2015:
                g *= 30
            rec.append({'authorid': aid, 'year': year, 'n_papers': 3.0,
                        'n_accel': 1, 'access_fp32_max': g,
                        'modal_field': 'Computer Science', 'ever_gpu': True,
                        'y0': 2008, 'y1': 2020, 'career_year': year - 2007,
                        'control_sample': 'gpu_ever'})
    return pl.DataFrame(rec).with_columns(pl.col('year').cast(pl.Int16))


def test_frontier_thresholds_rise_and_are_ordered():
    panel = _frontier_panel()
    thr = au.frontier_thresholds(panel, min_year_n=50)
    assert thr.height == 13, 'every year has 400 authors with access'
    assert (thr['thr_75'] <= thr['thr_90']).all()
    assert (thr['thr_90'] <= thr['thr_95']).all()
    # the threshold itself grows over time, which is what nets out secular growth
    assert thr.sort('year')['thr_90'].to_list() == sorted(thr['thr_90'].to_list())


def test_thin_years_get_no_threshold_and_so_no_events():
    panel = _frontier_panel(n=400)
    thin = panel.filter((pl.col('year') != 2012)
                        | (pl.col('authorid').str.slice(1).cast(pl.Int32) < 10))
    thr = au.frontier_thresholds(thin, min_year_n=50)
    assert 2012 not in thr['year'].to_list(), 'a 10-author year is not estimable'
    ev = au.find_frontier_events(thin, thresholds=thr)
    assert 2012 not in ev['event_year'].to_list()


def test_frontier_position_ranks_only_authors_with_access():
    """A null access year is not a position in the distribution. If nulls entered
    the rank denominator every percentile would be compressed toward zero."""
    panel = _frontier_panel(n=100)
    # blank out access for half the authors in 2015
    holed = panel.with_columns(
        access_fp32_max=pl.when((pl.col('year') == 2015)
                                & (pl.col('authorid').str.slice(1)
                                   .cast(pl.Int32) < 50))
        .then(None).otherwise(pl.col('access_fp32_max')))
    thr = au.frontier_thresholds(holed, min_year_n=10)
    pos = au.frontier_position(holed, thr)
    y = pos.filter((pl.col('year') == 2015) & pl.col('has_access'))
    assert y.height == 50
    assert y['frontier_pct'].max() == pytest.approx(1.0)
    assert y['frontier_pct'].min() == pytest.approx(1 / 50)
    assert pos.filter((pl.col('year') == 2015) & ~pl.col('has_access'))[
        'frontier_pct'].null_count() == 50


def test_frontier_cross_separates_from_frontier_entry():
    panel = _frontier_panel()
    ev = au.find_frontier_events(panel)
    assert set(ev['event_type'].unique()) <= set(au.FRONTIER_EVENT_TYPES)
    cross = ev.filter(pl.col('event_type') == 'frontier_cross')
    entry = ev.filter(pl.col('event_type') == 'frontier_entry')
    # the high-base authors are already above p90 in their FIRST observed year, so
    # they are entries, not crossings -- there is no prior year to have moved from
    assert entry.height and entry['event_year'].unique().to_list() == [2008]
    assert entry['pre_max_fp32'].null_count() == entry.height
    # everyone else crosses in the year they move, and only then
    assert cross.height and cross['event_year'].unique().to_list() == [2015]
    assert cross['pre_frontier_gap'].max() < 0, 'below the frontier beforehand'
    assert (cross['event_fp32'] >= cross['event_threshold']).all()
    # the two types partition: no author is both
    assert ev.group_by('authorid').agg(n=pl.len()).filter(pl.col('n') > 1).height == 0


def test_frontier_entry_is_an_author_with_no_prior_access():
    panel = _frontier_panel(n=200)
    # a2000 shows up only in 2016, already at the top
    late = pl.DataFrame([{'authorid': 'zz', 'year': y, 'n_papers': 3.0, 'n_accel': 1,
                          'access_fp32_max': 1e9 if y >= 2016 else None,
                          'modal_field': 'Computer Science', 'ever_gpu': True,
                          'y0': 2016, 'y1': 2020, 'career_year': y - 2015,
                          'control_sample': 'gpu_ever'}
                         for y in range(2016, 2021)]).with_columns(
        pl.col('year').cast(pl.Int16))
    ev = au.find_frontier_events(pl.concat([panel, late], how='diagonal'))
    zz = ev.filter(pl.col('authorid') == 'zz')
    assert zz['event_type'].to_list() == ['frontier_entry']
    assert zz['event_year'].item() == 2016
    assert zz['pre_max_fp32'].item() is None, 'no prior access by construction'


def test_absolute_growth_alone_does_not_cross_the_frontier():
    """The headline property. Everyone's capacity grows 1.35x/yr; only the relative
    movers cross. A ratio-based event would fire for the growth itself."""
    panel = _frontier_panel()
    ev = au.find_frontier_events(panel).filter(
        pl.col('event_type') == 'frontier_cross')
    movers = {a for a in panel['authorid'].unique().to_list()
              if int(a[1:]) % 4 == 0}
    crossers = set(ev['authorid'])
    assert crossers, 'the relative movers should cross'
    assert crossers <= movers, f'{len(crossers) - len(crossers & movers)} non-movers crossed'
    # Not ALL movers cross, and that is the design working: a quarter of the panel
    # multiplying by 30 drags the p90 threshold up with them, so only the movers who
    # end up in the top decile of the NEW distribution qualify. A threshold that
    # responds to the population is the whole point of a percentile event.
    assert len(crossers) < len(movers)


def test_near_miss_candidates_exclude_ever_crossers():
    panel = _frontier_panel()
    thr = au.frontier_thresholds(panel)
    nm = au.near_miss_candidates(panel, thr)
    ev = au.find_frontier_events(panel, thresholds=thr)
    assert nm.height
    assert not (set(nm['authorid']) & set(ev['authorid'])), \
        'a near-miss control must never have crossed the frontier'


def test_match_controls_candidates_restricts_the_pool():
    panel, jumps = _synthetic_panel(n_jumpers=40, n_controls=200)
    allowed = (panel.filter(pl.col('authorid').str.starts_with('c'))
               .select('authorid').unique().head(30)
               .with_columns(event_year=pl.lit(2010, dtype=pl.Int16)))
    wide = au.match_controls(panel, jumps, seed=au.SEED)
    narrow = au.match_controls(panel, jumps, seed=au.SEED, candidates=allowed)
    assert narrow.height
    assert set(narrow['control_id']) <= set(allowed['authorid'])
    # the unrestricted run does draw from outside the allowed pool, so the
    # restriction is doing something rather than being trivially satisfied
    assert set(wide['control_id']) - set(allowed['authorid'])


def test_match_controls_extra_stratum_changes_the_matching():
    panel, jumps = _synthetic_panel(n_jumpers=40, n_controls=200)
    panel = panel.with_columns(
        pos=pl.when(pl.col('authorid').str.starts_with('j')).then(1.0)
        .otherwise(pl.col('authorid').str.slice(1).cast(pl.Float64) % 7))
    base = au.match_controls(panel, jumps, seed=au.SEED)
    with_x = au.match_controls(panel, jumps, seed=au.SEED, extra_stratum='pos')
    assert with_x.height and with_x.height <= base.height
    assert set(map(tuple, with_x.rows())) != set(map(tuple, base.rows()))
    with pytest.raises(ValueError, match='not in panel'):
        au.match_controls(panel, jumps, extra_stratum='nope')


def test_vendor_frontier_is_monotone_and_registry_sourced():
    from accelscan.registry import load_registry
    vf = au.vendor_frontier(load_registry(), range(2008, 2026))
    assert vf.height > 10
    v = vf.sort('year')['vendor_frontier'].to_list()
    assert v == sorted(v), 'best-available-so-far cannot fall'
    yr = vf.sort('year')['vendor_frontier_year'].to_list()
    assert yr == sorted(yr)
    assert (vf['vendor_frontier_year'] <= vf['year']).all(), 'no unreleased cards'


def test_existing_estimators_are_unchanged_by_the_new_parameters():
    """Non-regression: authors.ipynb results must not move. The new arguments all
    default to None, so the old paths have to produce identical output."""
    panel, jumps = _synthetic_panel(n_jumpers=40, n_controls=100)
    a = au.match_controls(panel, jumps, seed=au.SEED)
    b = au.match_controls(panel, jumps, seed=au.SEED, candidates=None,
                          extra_stratum=None, risk_events=None)
    assert a.rows() == b.rows()
    p1 = au.placebo_jumps(panel, jumps.with_columns(event_type=pl.lit('ratio_jump')),
                          seed=au.SEED)
    p2 = au.placebo_jumps(panel, jumps.with_columns(event_type=pl.lit('ratio_jump')),
                          seed=au.SEED, require=None)
    assert p1.rows() == p2.rows()


def test_control_matching_is_reproducible():
    """A fixed seed must reproduce a matching exactly.

    It did not: `polars.group_by` orders groups and their rows nondeterministically
    (multi-threaded hashing), so the rng drew indices into a differently-ordered
    candidate array on each run and the control sets wobbled between runs of the
    same notebook. Every sampling site now sorts first.
    """
    panel, jumps = _synthetic_panel(n_jumpers=20, n_controls=80)
    j = jumps.with_columns(event_type=pl.lit('ratio_jump'))
    runs = [au.match_controls(panel, jumps, seed=au.SEED)
            .sort('authorid', 'control_id').rows() for _ in range(5)]
    assert all(r == runs[0] for r in runs), 'match_controls is not reproducible'
    pl_runs = [au.placebo_jumps(panel, j, seed=au.SEED).sort('authorid').rows()
               for _ in range(5)]
    assert all(r == pl_runs[0] for r in pl_runs), 'placebo_jumps is not reproducible'
    # and a different seed must actually give a different matching
    other = au.match_controls(panel, jumps, seed=au.SEED + 1).sort(
        'authorid', 'control_id').rows()
    assert other != runs[0]
