"""arXiv stage-1 driver: tar stream -> the shared scan core -> frames.

Uses an in-memory tar behind a non-seekable shim; no network, no S3.
"""

import gzip
import io
import tarfile

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
