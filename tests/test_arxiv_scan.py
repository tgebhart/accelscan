"""arXiv stage-1 driver: tar stream -> the shared scan core -> frames.

Uses an in-memory tar behind a non-seekable shim; no network, no S3.
"""

import gzip
import io
import tarfile
import time

import polars as pl
import pytest

from accelscan.arxiv_bulk import TarEntry, open_stream
from accelscan.arxiv_scan import INGEST_SCHEMA, scan_tar_stream
from accelscan.registry import load_registry
from tests.test_arxiv_bulk import NonSeekable, _tar

FILLER = 'This filler sentence pads the paragraph out to a plausible length here.'


@pytest.fixture(scope='module')
def reg():
    return load_registry()


@pytest.fixture
def entry():
    return TarEntry(filename='src/arXiv_src_2301_001.tar', seq_num=1, yymm='2301',
                    num_items=6, first_item='2301.00001', last_item='2301.00006',
                    size=1000, md5sum='deadbeef', timestamp='2023-02-01')


def _paper(body: str) -> bytes:
    return gzip.compress(
        f'\\documentclass{{article}}\\begin{{document}}\n{body}\n\\end{{document}}'
        .encode())


def test_scan_tar_stream_end_to_end(reg, entry):
    payload = _tar({
        # model-specific hit, with a device count that must survive inline math
        '2301/2301.00001.gz': _paper(
            f'\\section{{Setup}}\nWe trained on $8$ NVIDIA V100 GPUs. {FILLER}'),
        # generic-only hit
        '2301/2301.00002.gz': _paper(f'The code ran on a GPU cluster. {FILLER}'),
        # no accelerator at all
        '2301/2301.00003.gz': _paper(f'We prove a bound on the spectral gap. {FILLER}'),
        # PDF-only submission
        '2301/2301.00004.pdf': b'%PDF-1.5 binary',
        # a hit that must NOT be found: it is inside the bibliography
        '2301/2301.00005.gz': _paper(
            f'Our proof is combinatorial. {FILLER}\n'
            '\\begin{thebibliography}{9}\n'
            '\\bibitem{x} A. Author. A K80-based renderer. Proc. IEEE, 2015.\n'
            '\\end{thebibliography}'),
        # unrecognisable member name
        '2301/README.txt': b'not a paper',
    })
    with open_stream(NonSeekable(payload)) as tf:
        inv, cand, stats = scan_tar_stream(tf, entry, reg)

    assert inv.height == 4                      # 4 papers reached the converter
    assert set(inv['paper_id']) == {'arxiv:2301.00001', 'arxiv:2301.00002',
                                   'arxiv:2301.00003', 'arxiv:2301.00005'}
    assert inv['corpusid'].null_count() == inv.height        # arXiv carries no corpusid

    cands = dict(zip(cand['paper_id'], cand['passage_text']))
    assert 'arxiv:2301.00001' in cands and 'NVIDIA V100' in cands['arxiv:2301.00001']
    assert '8 NVIDIA V100 GPUs' in cands['arxiv:2301.00001']   # device count kept
    assert 'arxiv:2301.00002' in cands                        # generic hit kept
    assert 'arxiv:2301.00003' not in cands                    # no hardware
    assert 'arxiv:2301.00005' not in cands, 'bibliography leaked into candidates'

    # ids and sections flow through the shared core
    row = cand.filter(pl.col('paper_id') == 'arxiv:2301.00001').row(0, named=True)
    assert row['passage_id'].startswith('arxiv:2301.00001:')
    assert row['section_header'] == 'Setup'
    assert row['shard_id'] == 'arXiv_src_2301_001'
    assert 'nvidia-v100' in row['matched_models']


def test_ingest_stats_record_skips_and_conversion(reg, entry):
    payload = _tar({
        '2301/2301.00001.gz': _paper(f'We used an NVIDIA A100 GPU. {FILLER}'),
        '2301/2301.00004.pdf': b'%PDF-1.5 binary',
        '2301/2301.00007.gz': gzip.compress(b'%!PS-Adobe-2.0'),
        '2301/README.txt': b'not a paper',
    })
    with open_stream(NonSeekable(payload)) as tf:
        _, _, stats = scan_tar_stream(tf, entry, reg)

    assert stats.schema == INGEST_SCHEMA
    by_reason = dict(zip(stats['arxiv_id'].fill_null('-'), stats['skip_reason']))
    assert by_reason['2301.00004'] == 'pdf_only'
    assert by_reason['2301.00007'] == 'postscript'
    assert by_reason['-'] == 'unknown_id'
    assert by_reason['2301.00001'] == ''
    ok = stats.filter(pl.col('arxiv_id') == '2301.00001').row(0, named=True)
    assert (ok['year'], ok['month']) == (2023, 1)        # year is free from the id
    assert ok['encoding'] == 'utf-8' and ok['body_found'] is True
    assert ok['n_paragraphs'] >= 1


def test_empty_tar_yields_empty_typed_frames(reg, entry):
    with open_stream(NonSeekable(_tar({}))) as tf:
        inv, cand, stats = scan_tar_stream(tf, entry, reg)
    assert inv.height == cand.height == stats.height == 0
    assert stats.schema == INGEST_SCHEMA


def test_old_style_ids_flow_through(reg, entry):
    payload = _tar({'9901/hep-th9901001.gz': _paper(
        f'Lattice sums were computed on an NVIDIA GPU here. {FILLER}')})
    with open_stream(NonSeekable(payload)) as tf:
        inv, cand, stats = scan_tar_stream(tf, entry, reg)
    assert inv['paper_id'].to_list() == ['arxiv:hep-th/9901001']
    assert stats.row(0, named=True)['year'] == 1999
    assert cand['passage_id'][0].startswith('arxiv:hep-th/9901001:')


# --- run_pool: a killed worker must not burn the rest of the run -----------------
#
# The 2026-07-31 full-history run died here: one worker was killed, every queued
# future then raised BrokenProcessPool, and the old loop logged them all as tar
# failures and exited 0 with ~5,600 tars unscanned. These tests pin the retry.

def _entries(n: int) -> list[TarEntry]:
    return [TarEntry(filename=f'src/arXiv_src_2301_{i:03d}.tar', seq_num=i, yymm='2301',
                     num_items=1, first_item='x', last_item='y', size=1, md5sum='m',
                     timestamp='2023-02-01')
            for i in range(1, n + 1)]


def _fake_pool(monkeypatch, fate):
    """Swap the process pool for a synchronous one; `fate(entry)` decides each tar.

    A real ProcessPoolExecutor would pickle `process_tar` into a child, where a
    monkeypatch is invisible, so the executor itself is what gets faked. Records the
    width of every pool built, which is how the halving is asserted.
    """
    from concurrent.futures import Future
    widths: list[int] = []

    class FakeExec:
        def __init__(self, max_workers):
            widths.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, entry, *rest):
            fut = Future()
            try:
                fut.set_result(fate(entry))
            except BaseException as exc:            # noqa: BLE001 - mirrors a worker
                fut.set_exception(exc)
            return fut

    monkeypatch.setattr('accelscan.arxiv_scan.ProcessPoolExecutor', FakeExec)
    return widths


def test_broken_pool_retries_unfinished_tars_at_half_width(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    from accelscan.arxiv_scan import run_pool
    seen: list[str] = []

    def fate(entry):
        seen.append(entry.shard_id)
        # the pool breaks the first time each of the last two tars is submitted
        if entry.seq_num > 2 and seen.count(entry.shard_id) == 1:
            raise BrokenProcessPool('A process in the process pool was terminated')
        return f'{entry.shard_id}: ok'

    widths = _fake_pool(monkeypatch, fate)
    ok, failed = run_pool(_entries(4), '0.2.0', max_workers=8)
    assert (ok, failed) == (4, 0)                     # nothing lost
    assert widths == [8, 4]                           # rebuilt once, half as wide
    assert seen.count('arXiv_src_2301_003') == 2      # only the broken tars retried
    assert seen.count('arXiv_src_2301_001') == 1


def test_one_bad_tar_is_logged_and_not_retried(monkeypatch):
    from accelscan.arxiv_scan import run_pool
    seen: list[str] = []

    def fate(entry):
        seen.append(entry.shard_id)
        if entry.seq_num == 2:
            raise OSError('read timeout on the tar stream')
        return f'{entry.shard_id}: ok'

    widths = _fake_pool(monkeypatch, fate)
    ok, failed = run_pool(_entries(3), '0.2.0', max_workers=4)
    assert (ok, failed) == (2, 1)
    assert widths == [4]                              # no rebuild for a tar failure
    assert len(seen) == 3


def test_persistent_breakage_gives_up_and_reports_failure(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    from accelscan.arxiv_scan import run_pool

    def fate(entry):
        raise BrokenProcessPool('killed again')

    widths = _fake_pool(monkeypatch, fate)
    ok, failed = run_pool(_entries(2), '0.2.0', max_workers=8, max_rounds=3)
    assert (ok, failed) == (0, 2)                     # counted, so main() exits 1
    assert widths == [8, 4, 2]


def test_pace_reports_a_rate_and_eta():
    from accelscan.arxiv_scan import _pace
    out = _pace(60, 600, time.time() - 60)            # 60 tars in one minute
    assert 'tars/min' in out and 'eta' in out


def test_stage_timings_are_filled_in_place(reg, entry):
    """The breakdown is what tells a bandwidth stall apart from a converter stall."""
    from accelscan.arxiv_scan import STAGES
    payload = _tar({'2301/2301.00001.gz': _paper(
        f'Training ran on 4 NVIDIA V100 GPUs for a week. {FILLER}')})
    timings: dict = {}
    with open_stream(NonSeekable(payload)) as tf:
        inv, cand, stats = scan_tar_stream(tf, entry, reg, timings)
    assert inv.height == 1
    assert set(timings) == set(STAGES)
    assert timings['put'] == 0.0                      # process_tar fills that one
    assert all(v >= 0 for v in timings.values())
    assert timings['tex'] > 0 and timings['match'] > 0
