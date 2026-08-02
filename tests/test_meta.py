

def test_arxiv_years_come_from_the_id_not_the_snapshot():
    """compute.py must not block on the 5.4GB Kaggle download for a year."""
    import polars as pl
    from accelscan.meta import paper_years
    from accelscan.paths import ARXIV
    ids = pl.DataFrame({'paper_id': ['arxiv:2301.01234', 'arxiv:hep-th/9901001',
                                     'arxiv:9107.00001', 'arxiv:garbage']})
    out = paper_years(ARXIV, ids, so={}, metadata_uri='/nonexistent.parquet')
    got = dict(zip(out['paper_id'], out['year']))
    assert got['arxiv:2301.01234'] == 2023
    assert got['arxiv:hep-th/9901001'] == 1999      # YY>=91 -> 19YY
    assert got['arxiv:9107.00001'] == 1991
    assert 'arxiv:garbage' not in got               # unparseable dropped, not zero
