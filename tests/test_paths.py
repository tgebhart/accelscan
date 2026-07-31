"""Guard the legacy alias: S2ORC keys must not move when arXiv is added.

The S2ORC literals below are transcribed from the pre-refactor inline f-strings.
If one of these fails, previously-computed S2ORC outputs (339k mentions, the
analytic tables, the capacity table) have been orphaned at their old keys --
fix the code, never the expectation.

The second half pins the namespace-leak property that lets both corpora share
`accelscan/`: no arXiv key may be found by an S2ORC prefix listing or vice versa.
That is the sole reason `discover_mentions_version`, `precision.py`'s candidates
version scan, and `scan.run_s3`'s `.done` diff stay corpus-safe.
"""

import pytest

from accelscan.paths import (ARXIV, S2ORC, analytic_base, arxiv_metadata_key,
                            candidates_glob, candidates_parts, candidates_root,
                            capacity_key, clusters_base, embeddings_chunk,
                            embeddings_parts, get_corpus, ingest_stats_key,
                            inventory_glob, inventory_parts, mentions_parts,
                            mentions_root, passages_key, passages_manifest,
                            passages_prefix, precision_key, s3_uri)

RV, PV, TAG = '0.2.0', 'p1', 'qwen3-14b'

# (builder result, expected key) for every S2ORC product.
S2ORC_EXPECTED = [
    (inventory_parts(S2ORC), 'accelscan/inventory/parts'),
    (candidates_root(S2ORC), 'accelscan/candidates/'),
    (candidates_parts(S2ORC, RV), 'accelscan/candidates/0.2.0/parts'),
    (passages_prefix(S2ORC, RV), 'accelscan/passages/0.2.0'),
    (passages_key(S2ORC, RV, 3), 'accelscan/passages/0.2.0/shard_0003.parquet'),
    (passages_manifest(S2ORC, RV), 'accelscan/passages/0.2.0/manifest.parquet'),
    (mentions_root(S2ORC), 'accelscan/mentions/'),
    (mentions_parts(S2ORC, '0.1.0', PV, TAG),
     'accelscan/mentions/0.1.0/p1/qwen3-14b/parts'),
    (capacity_key(S2ORC, RV, PV, TAG, 'floor1'),
     'accelscan/capacity/0.2.0/p1/qwen3-14b/paper_capacity_floor1.parquet'),
    (analytic_base(S2ORC, RV, PV, TAG), 'accelscan/analytic/0.2.0/p1/qwen3-14b'),
    (precision_key(S2ORC, '0.1.0'), 'accelscan/precision/0.1.0/paper_precision.parquet'),
    (embeddings_parts(S2ORC, 'specter2-base'),
     'accelscan/embeddings/specter2-base/parts'),
    (embeddings_chunk(S2ORC, 'specter2-base', 12),
     'accelscan/embeddings/specter2-base/parts/chunk_0012'),
    (clusters_base(S2ORC, 'specter2-base-mcs50-ms5-ro'),
     'accelscan/clusters/specter2-base-mcs50-ms5-ro'),
]


@pytest.mark.parametrize('got,want', S2ORC_EXPECTED)
def test_s2orc_keys_are_byte_identical(got, want):
    assert got == want


def test_globs_are_s3_uris():
    assert inventory_glob(S2ORC) == 's3://scirec-embeddings/accelscan/inventory/parts/*.parquet'
    assert candidates_glob(S2ORC, RV) == (
        's3://scirec-embeddings/accelscan/candidates/0.2.0/parts/*.parquet')
    assert s3_uri('a/b') == 's3://scirec-embeddings/a/b'


ARXIV_KEYS = [
    inventory_parts(ARXIV), candidates_root(ARXIV), candidates_parts(ARXIV, RV),
    passages_prefix(ARXIV, RV), passages_key(ARXIV, RV, 3),
    passages_manifest(ARXIV, RV), mentions_root(ARXIV),
    mentions_parts(ARXIV, RV, PV, TAG), capacity_key(ARXIV, RV, PV, TAG, 'floor1'),
    analytic_base(ARXIV, RV, PV, TAG), precision_key(ARXIV, RV),
    embeddings_parts(ARXIV, 'specter2-base'),
    embeddings_chunk(ARXIV, 'specter2-base', 12),
    clusters_base(ARXIV, 'v1'), arxiv_metadata_key(),
    ingest_stats_key('arXiv_src_2301_001'),
]
S2ORC_KEYS = [k for k, _ in S2ORC_EXPECTED]


@pytest.mark.parametrize('key', ARXIV_KEYS)
def test_arxiv_keys_are_under_the_arxiv_prefix(key):
    assert key.startswith('accelscan/arxiv/')


@pytest.mark.parametrize('arxiv_key', ARXIV_KEYS)
def test_no_arxiv_key_is_reachable_from_an_s2orc_listing(arxiv_key):
    """An S3 list_objects_v2 on any S2ORC prefix must not return arXiv objects."""
    for s2orc_key in S2ORC_KEYS:
        assert not arxiv_key.startswith(s2orc_key), (
            f'{arxiv_key} would be listed under the S2ORC prefix {s2orc_key}')


@pytest.mark.parametrize('s2orc_key', S2ORC_KEYS)
def test_no_s2orc_key_is_reachable_from_an_arxiv_listing(s2orc_key):
    for arxiv_key in ARXIV_KEYS:
        assert not s2orc_key.startswith(arxiv_key)


def test_mentions_version_discovery_parse_still_works_per_corpus():
    """`discover_mentions_version` slices keys relative to `mentions_root` and
    reads rest[0]=version, rest[1]=prompt, rest[2]=model_tag. Adding the corpus
    segment must not shift those indices."""
    for c in (S2ORC, ARXIV):
        root = mentions_root(c)
        key = f'{mentions_parts(c, "0.1.0", PV, TAG)}/shard_0000.parquet'
        rest = key[len(root):].split('/')
        assert len(rest) >= 4
        assert (rest[0], rest[1], rest[2]) == ('0.1.0', PV, TAG)


def test_corpus_identity_columns():
    assert S2ORC.key == 'corpusid' and ARXIV.key == 'paper_id'
    assert get_corpus() is S2ORC
    assert get_corpus('arxiv') is ARXIV
    with pytest.raises(ValueError):
        get_corpus('nope')
