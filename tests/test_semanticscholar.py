"""Semantic Scholar preprocessing, on fixtures rather than the 2 TB release.

The network paths (dataset listing, part download) are not tested -- they are thin
wrappers over requests/urlretrieve. What is tested is everything that can silently
produce a wrong table: the declared schemas, the papers transform, and the
citations-to-years join, which is the input every citation outcome depends on.
"""

import gzip
import json

import polars as pl
import pytest

from accelscan.semanticscholar import preprocess as pp

PAPERS = [
    {'corpusid': 1, 'externalids': {'DOI': '10.1/a', 'MAG': '111'},
     'title': 'A GPU paper', 'referencecount': 3, 'citationcount': 7,
     'journal': {'name': 'J', 'volume': '1', 'pages': '1-9'},
     'conference': None, 'influentialcitationcount': 2,
     'publicationdate': '2019-04-01', 'year': 2019,
     'publicationtypes': ['JournalArticle'],
     's2fieldsofstudy': [{'category': 'Computer Science', 'source': 's2'}],
     'authors': [{'authorId': 'a1', 'name': 'X'}]},
    # a row whose optional fields are all null: the case that broke inference and
    # is the reason SCHEMAS is declared rather than sniffed
    {'corpusid': 2, 'externalids': None, 'title': None, 'referencecount': None,
     'citationcount': None, 'journal': None, 'conference': None,
     'influentialcitationcount': None, 'publicationdate': None, 'year': 2021,
     'publicationtypes': None, 's2fieldsofstudy': None, 'authors': None},
]


def write_ndjson_gz(path, rows):
    with gzip.open(path, 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def test_papers_parse_transform_and_delete(tmp_path):
    raw = tmp_path / 'part0.jsonl.gz'
    write_ndjson_gz(raw, PAPERS)
    out = tmp_path / 'part0.parquet'

    assert pp.parse_and_delete(str(raw), str(out), pp.TRANSFORMS['papers'],
                               pp.SCHEMAS['papers']) == str(out)
    assert not raw.exists(), 'raw part must be deleted to bound scratch use'

    df = pl.read_parquet(out)
    assert df.height == 2
    # the struct is flattened to the columns downstream reads
    assert 'externalids' not in df.columns
    assert df['doi'].to_list() == ['10.1/a', None]
    assert df['magid'].to_list() == ['111', None]
    assert df.schema['publicationdate'] == pl.Date
    # authors collapse to ids, and a null author list stays null (not [])
    assert df['authors'].to_list() == [['a1'], None]
    # field-of-study survives as a struct list, which meta.paper_fields indexes
    assert df['s2fieldsofstudy'].to_list()[0][0]['category'] == 'Computer Science'


def test_parse_reports_and_keeps_bad_part(tmp_path, capsys):
    """A corrupt part must not delete itself: it has to be retried next run."""
    raw = tmp_path / 'part0.jsonl.gz'
    with gzip.open(raw, 'wt') as f:
        f.write('{not json at all\n')
    assert pp.parse_and_delete(str(raw), str(tmp_path / 'o.parquet')) is None
    assert raw.exists()
    assert 'error parsing' in capsys.readouterr().err


def test_build_citation_edgelist_attaches_both_years(tmp_path):
    proc = tmp_path / '2026-02-10' / 'processed'
    (proc / 'papers' / 'parts').mkdir(parents=True)
    (proc / 'citations' / 'parts').mkdir(parents=True)
    pl.DataFrame({'corpusid': [1, 2, 3], 'year': [2015, 2019, 2021]}) \
        .write_parquet(proc / 'papers' / 'parts' / 'p0.parquet')
    pl.DataFrame({'citationid': [10, 11, 12],
                  'isinfluential': [True, False, True],
                  'citingcorpusid': [2, 3, 3],
                  # 99 is absent from papers: the year must be null, not dropped
                  'citedcorpusid': [1, 1, 99]}) \
        .write_parquet(proc / 'citations' / 'parts' / 'c0.parquet')

    out = pp.build_citation_edgelist(release_id='2026-02-10',
                                     data_dir=str(tmp_path),
                                     out_path=str(tmp_path / 'edges.parquet'))
    df = pl.read_parquet(out)
    assert set(df.columns) == {'citingcorpusid', 'citedcorpusid',
                              'citingyear', 'citedyear'}
    assert df.height == 3, 'a left join must not drop edges with unknown endpoints'
    rows = {(r['citingcorpusid'], r['citedcorpusid']): (r['citingyear'], r['citedyear'])
            for r in df.iter_rows(named=True)}
    assert rows[(2, 1)] == (2019, 2015)
    assert rows[(3, 1)] == (2021, 2015)
    assert rows[(3, 99)] == (2021, None)


AUTHORS = [
    {'authorid': 'a1', 'externalids': {'ORCID': '0000-0001', 'DBLP': ['X 0001']},
     'name': 'X', 'affiliations': ['UMN'], 'papercount': 12,
     'citationcount': 300, 'hindex': 7},
    {'authorid': 'a2', 'externalids': None, 'name': 'Y', 'affiliations': None,
     'papercount': None, 'citationcount': None, 'hindex': None},
]


def test_authors_parse_and_transform(tmp_path):
    raw = tmp_path / 'part0.jsonl.gz'
    write_ndjson_gz(raw, AUTHORS)
    out = tmp_path / 'part0.parquet'
    assert pp.parse_and_delete(str(raw), str(out), pp.TRANSFORMS['authors'],
                               pp.SCHEMAS['authors']) == str(out)
    df = pl.read_parquet(out)
    assert df.height == 2
    # flattened the same way papers flattens DOI/MAG, so the table joins cleanly
    # to the author ids papers already carries
    assert 'externalids' not in df.columns
    assert df['orcid'].to_list() == ['0000-0001', None]
    assert df['dblp'].to_list() == [['X 0001'], None]
    assert df['authorid'].to_list() == ['a1', 'a2']
    assert df['hindex'].to_list() == [7, None]


def test_authors_join_to_paper_author_ids(tmp_path):
    """The reason the table is downloaded at all: papers keeps only author ids."""
    praw, araw = tmp_path / 'p.jsonl.gz', tmp_path / 'a.jsonl.gz'
    write_ndjson_gz(praw, PAPERS)
    write_ndjson_gz(araw, AUTHORS)
    papers = pl.read_parquet(pp.parse_and_delete(
        str(praw), str(tmp_path / 'p.parquet'), pp.TRANSFORMS['papers'],
        pp.SCHEMAS['papers']))
    authors = pl.read_parquet(pp.parse_and_delete(
        str(araw), str(tmp_path / 'a.parquet'), pp.TRANSFORMS['authors'],
        pp.SCHEMAS['authors']))
    joined = (papers.select('corpusid', 'authors')
              .explode('authors', empty_as_null=True)
              .join(authors, left_on='authors', right_on='authorid', how='inner'))
    assert joined.select('corpusid', 'name').rows() == [(1, 'X')]


@pytest.mark.parametrize('dataset', ['papers', 'abstracts', 'citations', 'authors'])
def test_schema_declared_for_every_dataset(dataset):
    assert dataset in pp.SCHEMAS and pp.SCHEMAS[dataset]


def test_outcomes_defaults_match_the_edgelist_columns():
    """The outcome defaults must name the columns build_citation_edgelist emits,
    or a SLURM run silently computes nothing."""
    from accelscan.semanticscholar import outcomes
    for col in (outcomes.CITING_VAR, outcomes.CITED_VAR,
                outcomes.DATE_VAR_CITING, outcomes.DATE_VAR_CITED):
        assert col in {'citingcorpusid', 'citedcorpusid', 'citingyear', 'citedyear'}
