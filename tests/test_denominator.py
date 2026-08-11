"""Stage 4a: paper_flags numerators.

`paper_flags_from_mentions` is the pure aggregation `build_paper_flags` wraps
around an S3 read of the mentions table. The case that matters here: a paper
whose only `ok` mention is a registry false positive the extraction schema
already flagged as `not-an-accelerator` (K80 the antibody, "Claude 3.7
Sonnet") must not be counted as having ANY accelerator mention.
"""

import polars as pl

from accelscan.denominator import paper_flags_from_mentions
from accelscan.registry import load_registry

REG = load_registry()

SCHEMA = {'corpusid': pl.Int64, 'status': pl.Utf8, 'model_normalized': pl.Utf8,
          'manufacturer': pl.Utf8, 'accelerator_subtype': pl.Utf8,
          'usage_context': pl.Utf8, 'device_count': pl.Int32}


def _mentions(rows: list) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA, orient='row')


def test_paper_whose_only_mention_is_not_an_accelerator_gets_no_row():
    m = _mentions([
        (1, 'ok', 'Claude 3.7 Sonnet', 'unknown', 'not-an-accelerator',
         'used-in-this-work', None),
    ])
    out = paper_flags_from_mentions(m, REG, 'corpusid')
    assert out.height == 0


def test_paper_with_one_real_and_one_false_positive_mention_still_counts():
    m = _mentions([
        (1, 'ok', 'NVIDIA Tesla V100', 'nvidia', 'datacenter-gpu',
         'used-in-this-work', 8),
        (1, 'ok', 'Claude 3.7 Sonnet', 'unknown', 'not-an-accelerator',
         'used-in-this-work', None),
    ])
    out = paper_flags_from_mentions(m, REG, 'corpusid')
    assert out.height == 1
    row = out.row(0, named=True)
    assert row['accel_any'] and row['accel_used'] and row['model_specific_used']
    # the false positive must not survive into any per-paper aggregate
    assert row['n_mentions'] == 1
    assert 'unknown' not in row['manufacturers']


def test_non_ok_status_is_excluded_same_as_before():
    m = _mentions([
        (1, 'no_mention', None, None, None, None, None),
        (1, 'json_error', None, None, None, None, None),
    ])
    out = paper_flags_from_mentions(m, REG, 'corpusid')
    assert out.height == 0
