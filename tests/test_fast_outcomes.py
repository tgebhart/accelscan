"""The vectorized i_k/bcites must equal the lab's implementation, edge for edge.

This is the whole justification for `fast_outcomes.py`: it replaces a
dict-of-sets/year-loop computation with a group-by, so the only acceptable evidence
is that both produce the same numbers on the same graph. The reference here is
`outcomes.run()` itself, not a hand-derived expectation.
"""

import glob
import os

import polars as pl
import pytest

from accelscan.semanticscholar import fast_outcomes as fo
from accelscan.semanticscholar import outcomes as oc

# A graph with the cases that distinguish the implementations:
#  - 10 cited by 20,21,22 at lags 1,3,6  -> i_1=1, i_3=2, i_5=2, i_10=3
#  - 11 cited by nobody                  -> i_k = 0 for all k (must appear, as 0)
#  - 12 cited by 20 at lag 0             -> same year citation counts
#  - self-citation and a backwards edge   -> both dropped by the filters
#  - a duplicated edge (21->10)           -> counted once
EDGES = [
    (20, 10, 2011, 2010),
    (21, 10, 2013, 2010),
    (21, 10, 2013, 2010),        # duplicate of the previous row
    (22, 10, 2016, 2010),
    (20, 12, 2011, 2011),
    (20, 20, 2011, 2011),        # self-citation
    (20, 30, 2011, 2015),        # cites the future
    (23, 11, 2014, None),        # null cited year
    (21, 12, 2013, 2011),
]


@pytest.fixture
def edge_dir(tmp_path):
    d = tmp_path / 'edges'
    d.mkdir()
    # two parts, so cross-part dedup and bucketing are both exercised
    for i, chunk in enumerate((EDGES[:5], EDGES[5:])):
        pl.DataFrame(
            {'citingcorpusid': [r[0] for r in chunk],
             'citedcorpusid': [r[1] for r in chunk],
             'citingyear': [r[2] for r in chunk],
             'citedyear': [r[3] for r in chunk]},
            schema={'citingcorpusid': pl.Int64, 'citedcorpusid': pl.Int64,
                    'citingyear': pl.Int64, 'citedyear': pl.Int64},
        ).write_parquet(d / f'part{i}.parquet')
    return str(d)


def fast_result(edge_dir, tmp_path, horizons):
    work, save = str(tmp_path / 'work'), str(tmp_path / 'fast')
    fo.bucket_edges(edge_dir, work, end_year=2025, n_buckets=4)
    res = fo.count_outcomes(work, tuple(horizons), n_buckets=4)
    fo.write_csvs(res, save, tuple(horizons), ('i', 'bcites'))
    return {c: dict(zip(*pl.read_csv(f'{save}/{c}.csv.gz')
                        .select('record_id', c).to_dict(as_series=False).values()))
            for c in [f'i_{k}' for k in horizons] + ['bcites']}


def lab_result(edge_dir, tmp_path, horizons):
    """Run the lab implementation and read back its per-year appended CSVs."""
    save = str(tmp_path / 'lab')
    oc.run(citation_loc=edge_dir, save_loc=save, cd_types=['i'],
           time_horizons=list(horizons), thresholds=[1], start_year=None,
           end_year=2025, start_year_output=None,
           citing_var='citingcorpusid', cited_var='citedcorpusid',
           citing_var_type='int64', cited_var_type='int64',
           garbage_collect=False, reread_citations=False,
           date_var_citing='citingyear', date_var_cited='citedyear',
           date_type='year', nworkers=1, write_every=1, shuffle=0, sample=0)
    out = {}
    for k in horizons:
        f = sorted(glob.glob(os.path.join(save, f'i_{k}_*.csv.gz')))
        assert f, f'lab run produced no i_{k} file'
        df = pl.read_csv(f[-1])
        # the lab appends per year; last write per record_id wins
        out[f'i_{k}'] = {r['record_id']: r[df.columns[-1]]
                         for r in df.iter_rows(named=True)}
    return out


def test_matches_lab_implementation(edge_dir, tmp_path):
    horizons = (1, 3, 5, 10)
    fast = fast_result(edge_dir, tmp_path / 'a', horizons)
    lab = lab_result(edge_dir, tmp_path / 'b', horizons)
    for k in horizons:
        col = f'i_{k}'
        shared = set(fast[col]) & set(lab[col])
        assert shared, f'{col}: no overlapping record_ids to compare'
        for rid in sorted(shared):
            assert fast[col][rid] == lab[col][rid], (
                f'{col} disagrees for {rid}: fast={fast[col][rid]} '
                f'lab={lab[col][rid]}')


def test_windows_duplicates_and_zeros(edge_dir, tmp_path):
    """Pin the semantics directly, so a regression is legible without the lab run."""
    r = fast_result(edge_dir, tmp_path, (1, 3, 5, 10))
    assert r['i_1'][10] == 1                      # only the lag-1 citation
    assert r['i_3'][10] == 2                      # duplicate edge counted once
    assert r['i_5'][10] == 2                      # lag-6 citation still outside
    assert r['i_10'][10] == 3
    assert r['i_5'][12] == 2                      # lag-0 and lag-2 citations
    assert r['i_5'][20] == 0                      # cites others, cited by none
    assert 30 not in r['i_5'], 'the backwards edge must be dropped entirely'
    assert 11 not in r['i_5'], 'the null-cited-year edge must be dropped'
    assert r['bcites'][20] == 2                   # 10 and 12; self-cite dropped
    assert r['bcites'][21] == 2                   # 10 (deduped) and 12
