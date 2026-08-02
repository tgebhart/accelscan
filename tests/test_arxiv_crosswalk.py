"""arXiv<->corpusid crosswalk extraction. No network: records are built inline."""

import orjson
import polars as pl

from accelscan.scripts.build_arxiv_crosswalk import (SCHEMA, classify, extract,
                                                     normalise)


def _rec(corpusid, arxiv):
    return orjson.dumps({
        'corpusid': corpusid, 'title': 't',
        'openaccessinfo': {'externalids': {'DOI': '10.1/x', 'MAG': '1', 'ArXiv': arxiv}},
    })


def test_id_shapes_are_classified():
    assert classify('2503.22397') == 'modern'
    assert classify('hep-th/9901001') == 'old'
    # old-style cross-lists carry a lowercase hyphenated subject class
    assert classify('cond-mat.str-el/0512345') == 'old'
    assert classify('math.AG/0001001') == 'old'
    assert classify('nonsense') == 'unrecognised'


def test_version_suffix_is_stripped_to_match_our_paper_id():
    # our arXiv key is version-less, so 2301.01234v3 must join 2301.01234
    assert normalise('2301.01234v3') == '2301.01234'
    assert normalise('hep-th/9901001v12') == 'hep-th/9901001'
    assert normalise(' 2301.01234 ') == '2301.01234'


def test_extract_keeps_only_records_with_an_arxiv_id():
    lines = [_rec(1, '2503.22397'), _rec(2, None), _rec(3, 'hep-th/9901001v2'),
             _rec(4, ''), b'{not json', _rec(5, '2401.11391')]
    rows, stats = extract(lines)
    assert stats['records'] == 5            # the unparseable line is not counted
    assert stats['with_arxiv'] == 3
    df = pl.DataFrame(rows, schema=SCHEMA, orient='row')
    assert df['paper_id'].to_list() == ['arxiv:2503.22397', 'arxiv:hep-th/9901001',
                                       'arxiv:2401.11391']
    assert df['corpusid'].to_list() == [1, 3, 5]
    assert df['id_style'].to_list() == ['modern', 'old', 'modern']


def test_missing_openaccessinfo_does_not_raise():
    rows, stats = extract([orjson.dumps({'corpusid': 9, 'title': 't'}),
                           orjson.dumps({'corpusid': 10, 'openaccessinfo': None})])
    assert rows == [] and stats == {'records': 2, 'with_arxiv': 0}
