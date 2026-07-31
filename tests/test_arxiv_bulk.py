"""Manifest parsing and forward-only tar streaming, with no network access.

The streaming test deliberately wraps the tar in a shim that raises on `seek`,
because `tarfile` mode 'r|' silently works on a seekable BytesIO and would hide a
regression that only appears against a real HTTP body.
"""

import gzip
import io
import tarfile

import pytest

from accelscan.arxiv_bulk import (ARXIV_BUCKET, EGRESS_USD_PER_GB, iter_members,
                                 load_manifest, manifest_bytes_total, open_stream,
                                 parse_manifest, select_tars, yymm_to_year_month)

MANIFEST = b'''<?xml version="1.0" encoding="UTF-8"?>
<arXivSRC>
  <file>
    <content_md5sum>aaa</content_md5sum>
    <filename>src/arXiv_src_9108_001.tar</filename>
    <first_item>hep-th9108001</first_item>
    <last_item>hep-th9108010</last_item>
    <md5sum>bbb</md5sum>
    <num_items>10</num_items>
    <seq_num>1</seq_num>
    <size>1048576</size>
    <timestamp>2010-12-23 00:00:00</timestamp>
    <yymm>9108</yymm>
  </file>
  <file>
    <filename>src/arXiv_src_2301_002.tar</filename>
    <first_item>2301.05000</first_item>
    <last_item>2301.06000</last_item>
    <md5sum>ddd</md5sum>
    <num_items>2000</num_items>
    <seq_num>2</seq_num>
    <size>524288000</size>
    <timestamp>2023-02-01 00:00:00</timestamp>
    <yymm>2301</yymm>
  </file>
  <file>
    <filename>src/arXiv_src_2301_001.tar</filename>
    <num_items>1999</num_items>
    <seq_num>1</seq_num>
    <size>524288000</size>
    <md5sum>ccc</md5sum>
    <yymm>2301</yymm>
    <timestamp>2023-02-01 00:00:00</timestamp>
  </file>
</arXivSRC>'''


def test_parse_manifest_fields_and_order():
    entries = parse_manifest(MANIFEST)
    assert [e.filename for e in entries] == [
        'src/arXiv_src_9108_001.tar',        # sorted by (yymm, seq_num)
        'src/arXiv_src_2301_001.tar',
        'src/arXiv_src_2301_002.tar']
    first = entries[0]
    assert (first.shard_id, first.yymm, first.num_items, first.size, first.md5sum) == \
        ('arXiv_src_9108_001', '9108', 10, 1048576, 'bbb')
    assert first.first_item == 'hep-th9108001'


def test_shard_id_is_collision_free_against_s2orc():
    """S2ORC shard ids look like '20260313_122437_00106_k45gi_<uuid>'."""
    for e in parse_manifest(MANIFEST):
        assert e.shard_id.startswith('arXiv_src_')


def test_year_from_yymm_handles_the_1990s():
    entries = {e.yymm: e for e in parse_manifest(MANIFEST)}
    assert entries['9108'].year == 1991          # arXiv's first month
    assert entries['2301'].year == 2023


def test_yymm_ordering_crosses_the_century_boundary():
    """Regression: raw string compare put 1991 after 2023 and made the default
    full-history range ('9108','2512') select nothing at all."""
    assert yymm_to_year_month('9108') == (1991, 8)
    assert yymm_to_year_month('2301') == (2023, 1)
    assert yymm_to_year_month('9108') < yymm_to_year_month('2301')
    entries = parse_manifest(MANIFEST)
    assert len(select_tars(entries, yymm_range=('9108', '2512'))) == 3


def test_select_tars_by_range_and_limit():
    entries = parse_manifest(MANIFEST)
    assert len(select_tars(entries, yymm_range=('2301', '2512'))) == 2
    assert len(select_tars(entries, yymm_range=('9108', '9108'))) == 1
    assert len(select_tars(entries, limit=2)) == 2
    assert select_tars(entries, yymm_range=('9901', '9912')) == []


def test_manifest_bytes_total_and_cost_estimate():
    total = manifest_bytes_total(parse_manifest(MANIFEST))
    assert total == 1048576 + 524288000 * 2
    # the number this guards is the ~$260 full-pass egress bill
    assert total / 1e9 * EGRESS_USD_PER_GB == pytest.approx(0.0944, rel=0.01)


def test_malformed_entries_are_skipped_not_raised():
    bad = b'''<arXivSRC>
      <file><filename>src/ok.tar</filename><size>10</size><seq_num>1</seq_num>
            <yymm>2301</yymm><md5sum>x</md5sum><num_items>1</num_items></file>
      <file><size>nonsense</size><filename>src/bad.tar</filename></file>
      <file><size>5</size></file>
    </arXivSRC>'''
    entries = parse_manifest(bad)
    assert [e.filename for e in entries] == ['src/ok.tar']


def test_load_manifest_uses_cache_without_network(tmp_path):
    cached = tmp_path / 'manifest.xml'
    cached.write_bytes(MANIFEST)

    class Boom:
        def get_object(self, **kw):
            raise AssertionError('must not hit the network when cached')

    assert len(load_manifest(client=Boom(), cache=cached)) == 3


def test_load_manifest_writes_the_cache(tmp_path):
    calls = {}

    class FakeClient:
        def get_object(self, Bucket, Key, RequestPayer=None):
            calls.update(bucket=Bucket, key=Key, payer=RequestPayer)
            return {'Body': io.BytesIO(MANIFEST)}

    dst = tmp_path / 'sub' / 'manifest.xml'
    entries = load_manifest(client=FakeClient(), cache=dst)
    assert len(entries) == 3
    assert dst.read_bytes() == MANIFEST
    assert calls['bucket'] == ARXIV_BUCKET
    # requester-pays: without this header every call 403s
    assert calls['payer'] == 'requester'


# --- forward-only streaming ------------------------------------------------

class NonSeekable(io.RawIOBase):
    """Raises on seek/tell, like a real HTTP response body."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def readable(self):
        return True

    def seekable(self):
        return False

    def readinto(self, b):
        return self._buf.readinto(b)

    def seek(self, *a):
        raise OSError('stream is not seekable')

    def tell(self):
        raise OSError('stream is not seekable')


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


TEX = b'\\begin{document}An NVIDIA V100 GPU run.\\end{document}'


def test_iter_members_over_a_non_seekable_stream():
    inner = _tar({'main.tex': TEX, 'refs.bbl': b'\\bibitem{x} K80 study.'})
    payload = _tar({
        '2301/2301.00001.gz': gzip.compress(TEX),        # single-file submission
        '2301/2301.00002.gz': gzip.compress(inner),      # nested-tar submission
        '2301/2301.00003.pdf': b'%PDF-1.5 binary',       # PDF-only
        '2301/2301.00004.gz': gzip.compress(b'%!PS-Adobe-2.0'),
    })
    with open_stream(NonSeekable(payload)) as tf:
        names = [name for name, _ in iter_members(tf)]
    assert names == ['2301/2301.00001.gz', '2301/2301.00002.gz',
                     '2301/2301.00003.pdf', '2301/2301.00004.gz']


def test_stream_mode_forbids_reaching_back():
    """Pins the constraint that shapes the driver: a member the stream has passed
    cannot be re-read, so each one must be consumed in order (a forward
    getmembers() scan is fine -- it just reads to EOF)."""
    with open_stream(NonSeekable(_tar({'a.tex': TEX, 'b.tex': TEX}))) as tf:
        members = tf.getmembers()
        with pytest.raises(tarfile.StreamError):
            tf.extractfile(members[0]).read()


def test_members_classify_end_to_end_through_unpack():
    from accelscan.arxiv_source import arxiv_id_from_member, unpack_member
    payload = _tar({
        '2301/2301.00001.gz': gzip.compress(TEX),
        '2301/2301.00003.pdf': b'%PDF-1.5 binary',
        '2301/2301.00004.gz': gzip.compress(b'%!PS-Adobe-2.0'),
    })
    seen = {}
    with open_stream(NonSeekable(payload)) as tf:
        for name, data in iter_members(tf):
            files, skip = unpack_member(name, data)
            seen[arxiv_id_from_member(name)] = skip or f'ok:{len(files)}'
    assert seen == {'2301.00001': 'ok:1', '2301.00003': 'pdf_only',
                    '2301.00004': 'postscript'}
