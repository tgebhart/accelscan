"""arXiv category mapping + metadata record parsing.

The completeness test is the important one: an unmapped archive yields a null
`field`, which would silently drop those papers out of every field-level figure
rather than failing.
"""

import json

import polars as pl
import pytest

from accelscan.arxiv_meta import (archive_of, field_of, load_category_map,
                                 primary_category_of)
from accelscan.arxiv_source import OLD_ARCHIVES
from accelscan.scripts.build_arxiv_metadata import SCHEMA, build, parse_record

MAP = load_category_map()

# every archive of the modern (post-2007) scheme
CURRENT = ['cs', 'econ', 'eess', 'math', 'astro-ph', 'cond-mat', 'gr-qc', 'hep-ex',
           'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin', 'nucl-ex', 'nucl-th',
           'physics', 'quant-ph', 'q-bio', 'q-fin', 'stat']


@pytest.mark.parametrize('archive', CURRENT)
def test_every_current_archive_is_mapped(archive):
    assert MAP.get(archive), f'{archive} missing from registry/arxiv_categories.yaml'


@pytest.mark.parametrize('archive', sorted(OLD_ARCHIVES))
def test_every_legacy_archive_is_mapped(archive):
    """Pre-2007 archives must land in the group that absorbed them, so a 1995 and a
    2025 paper share a field label."""
    assert MAP.get(archive), f'legacy {archive} missing from arxiv_categories.yaml'


def test_subject_class_is_reduced_to_its_archive():
    assert archive_of('cs.LG') == 'cs'
    assert archive_of('astro-ph.GA') == 'astro-ph'
    assert archive_of('hep-th') == 'hep-th'
    assert archive_of('') == ''


def test_field_labels():
    assert field_of('cs.LG') == 'Computer Science'
    assert field_of('hep-th') == 'Physics'
    assert field_of('math.AG') == 'Mathematics'
    assert field_of('cmp-lg') == 'Computer Science'     # legacy -> cs
    assert field_of('supr-con') == 'Physics'            # legacy -> cond-mat
    assert field_of('alg-geom') == 'Mathematics'
    assert field_of('q-bio.NC') == 'Quantitative Biology'
    assert field_of('eess.SP') == 'Electrical Engineering and Systems Science'


def test_unknown_archive_is_none_not_a_bucket():
    """None so a test catches the gap; 'Other' would hide it in a figure."""
    assert field_of('zz-nonsense') is None


def test_primary_category_from_space_separated_string():
    # the Kaggle snapshot stores categories as a string, not a list
    assert primary_category_of('cs.LG cs.AI stat.ML') == 'cs.LG'
    assert primary_category_of(['hep-th', 'gr-qc']) == 'hep-th'
    assert primary_category_of('') is None


REC_NEW = {'id': '2301.01234', 'categories': 'cs.LG cs.AI',
           'title': 'A  Study\n of GPUs', 'abstract': ' We used  8 A100s.\n',
           'versions': [{'version': 'v1', 'created': 'Mon, 2 Jan 2023 19:18:42 GMT'},
                        {'version': 'v2', 'created': 'Tue, 3 Jan 2023 19:18:42 GMT'}],
           'license': 'http://creativecommons.org/licenses/by/4.0/',
           'doi': '10.1000/x', 'journal-ref': 'JMLR 24'}
REC_OLD = {'id': 'hep-th/9901001', 'categories': 'hep-th',
           'title': 'Old Paper', 'abstract': 'Lattice sums.',
           'versions': [{'version': 'v1', 'created': 'Fri, 1 Jan 1999 00:00:00 GMT'}]}


def test_parse_record_new_style():
    r = parse_record(REC_NEW, MAP)
    assert r['paper_id'] == 'arxiv:2301.01234'
    assert r['primary_category'] == 'cs.LG' and r['field'] == 'Computer Science'
    assert r['categories'] == ['cs.LG', 'cs.AI']
    assert r['title'] == 'A Study of GPUs'              # whitespace normalised
    assert r['abstract'] == 'We used 8 A100s.'
    assert (r['year'], r['month']) == (2023, 1)         # from the id, not the snapshot
    assert r['submitted'].isoformat() == '2023-01-02'
    assert r['n_versions'] == 2 and r['journal_ref'] == 'JMLR 24'


def test_parse_record_old_style_year_agrees_with_v1_date():
    r = parse_record(REC_OLD, MAP)
    assert r['paper_id'] == 'arxiv:hep-th/9901001'
    assert r['year'] == 1999 and r['snapshot_year'] == 1999
    assert r['field'] == 'Physics'


def test_parse_record_rejects_idless_and_survives_bad_dates():
    assert parse_record({'categories': 'cs.LG'}, MAP) is None
    r = parse_record({'id': '2301.01234', 'categories': 'cs.LG',
                      'versions': [{'version': 'v1', 'created': 'not a date'}]}, MAP)
    assert r['submitted'] is None and r['year'] == 2023


def test_build_reads_jsonl_and_types_the_frame(tmp_path):
    p = tmp_path / 'snap.json'
    p.write_text('\n'.join([json.dumps(REC_NEW), json.dumps(REC_OLD),
                            'this line is not json', json.dumps({'no': 'id'})]))
    df = build(str(p))
    assert df.height == 2
    assert df.schema == SCHEMA
    assert set(df['field']) == {'Computer Science', 'Physics'}
    assert df['paper_id'].to_list() == ['arxiv:2301.01234', 'arxiv:hep-th/9901001']
