"""arXiv id parsing and tar-member unpacking.

Id parsing is load-bearing in a quiet way: a wrong pre-2007 archive split produces a
`paper_id` that matches nothing in the metadata snapshot, so the paper silently drops
out of every join rather than failing loudly. Hence the exhaustive id table.
"""

import gzip
import io
import tarfile

import pytest

from accelscan.arxiv_source import (OLD_ARCHIVES, arxiv_id_from_member,
                                   member_kind, paper_id_from_arxiv_id,
                                   unpack_member, year_month_from_id)

IDS = [
    # new-style (2007-04 onward), 4- and 5-digit sequence numbers
    ('0704/0704.0001.gz', '0704.0001'),
    ('2301/2301.01234.gz', '2301.01234'),
    ('2301.01234v3.gz', '2301.01234'),                 # version suffix stripped
    ('1412.6980.pdf', '1412.6980'),
    # old-style: archive and number are concatenated in the member name
    ('0001/hep-th0001001.gz', 'hep-th/0001001'),
    ('math0001001.gz', 'math/0001001'),
    ('9108/hep-th9108001.gz', 'hep-th/9108001'),       # the very first month
    ('cond-mat9901001v2.gz', 'cond-mat/9901001'),
    ('cs9907023.gz', 'cs/9907023'),
    ('quant-ph0201001.gz', 'quant-ph/0201001'),
    ('q-bio0402001.gz', 'q-bio/0402001'),
    ('cmp-lg9503001.gz', 'cmp-lg/9503001'),            # defunct archive
    ('adap-org9401001.gz', 'adap-org/9401001'),        # defunct archive
    # subject-class form used in some old ids
    ('math.GT0001001.gz', 'math.GT/0001001'),
    # unrecognisable
    ('readme.txt', None),
    ('0704/notanid.gz', None),
    ('', None),
]


@pytest.mark.parametrize('member,want', IDS)
def test_arxiv_id_from_member(member, want):
    assert arxiv_id_from_member(member) == want


YEARS = [('9108.0001', 1991, 8), ('hep-th/9108001', 1991, 8),
         ('9901.0001', 1999, 1), ('math/9901001', 1999, 1),
         ('0704.0001', 2007, 4), ('2301.01234', 2023, 1),
         ('2512.00001', 2025, 12)]


@pytest.mark.parametrize('aid,year,month', YEARS)
def test_year_month_from_id(aid, year, month):
    """Year comes free from the id, so stage 1 needs no metadata lookup."""
    assert year_month_from_id(aid) == (year, month)


def test_year_month_rejects_nonsense():
    assert year_month_from_id('abc') == (None, None)
    assert year_month_from_id('0013.0001') == (None, None)      # month 13


def test_paper_id_is_namespaced():
    assert paper_id_from_arxiv_id('2301.01234') == 'arxiv:2301.01234'
    assert paper_id_from_arxiv_id('hep-th/9901001') == 'arxiv:hep-th/9901001'


def test_old_archive_list_covers_the_real_ones():
    for a in ('hep-th', 'cond-mat', 'cs', 'math', 'q-bio', 'cmp-lg', 'supr-con'):
        assert a in OLD_ARCHIVES


# --- member unpacking ------------------------------------------------------

def _gz(data: bytes) -> bytes:
    return gzip.compress(data)


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


TEX = b'\\documentclass{article}\\begin{document}An NVIDIA V100 GPU.\\end{document}'


def test_single_file_submission():
    files, skip = unpack_member('2301.01234.gz', _gz(TEX))
    assert skip == ''
    assert list(files) == ['2301.01234.tex']
    assert files['2301.01234.tex'] == TEX


def test_nested_tar_submission_excludes_bbl_and_binaries():
    inner = _tar_bytes({'main.tex': TEX, 'intro.tex': b'Intro text.',
                        'refs.bbl': b'\\bibitem{x} A K80 study. Proc. IEEE, 2015.',
                        'fig1.pdf': b'%PDF-1.4 binary', 'logo.png': b'\x89PNG'})
    files, skip = unpack_member('2301.01234.gz', _gz(inner))
    assert skip == ''
    assert set(files) == {'main.tex', 'intro.tex'}      # .bbl excluded: bib layer 1
    assert not any(n.endswith('.bbl') for n in files)


def test_pdf_only_submission():
    assert unpack_member('2301.01234.pdf', b'%PDF-1.5 ...') == ({}, 'pdf_only')


def test_gzipped_pdf_is_still_pdf_only():
    """Classification must survive one layer of compression."""
    assert unpack_member('9901001.gz', _gz(b'%PDF-1.2 ...')) == ({}, 'pdf_only')


def test_postscript_submission():
    assert unpack_member('9901001.gz', _gz(b'%!PS-Adobe-2.0'))[1] == 'postscript'


def test_corrupt_gzip_and_empty():
    assert unpack_member('x.gz', b'\x1f\x8b' + b'garbage')[1] == 'corrupt'
    assert unpack_member('x.gz', b'')[1] == 'corrupt'


def test_oversize_member_is_skipped_not_read():
    assert unpack_member('x.gz', b'\x1f\x8b' + b'\x00' * (65 << 20))[1] == 'oversize'


def test_nested_tar_without_tex_is_no_tex():
    inner = _tar_bytes({'fig.eps': b'%!PS', 'data.csv': b'a,b\n1,2\n'})
    assert unpack_member('x.gz', _gz(inner))[1] == 'no_tex'


def test_member_kind_uses_magic_bytes_not_extension():
    assert member_kind('paper.tex', b'%PDF-1.4') == 'pdf'          # lying extension
    assert member_kind('paper.gz', b'\x1f\x8b\x08') == 'gzip'
    assert member_kind('paper.tex', b'\\documentclass') == 'text'
    assert member_kind('mystery.dat', b'\x00\x01\x02\x03') == 'unknown'


def test_unpacked_files_flow_into_the_converter():
    """End-to-end seam: tar member -> paragraphs the shared scan core can use."""
    from accelscan.latex import latex_to_paragraphs
    files, skip = unpack_member('2301.01234.gz', _gz(
        _tar_bytes({'main.tex': b'\\begin{document}\n'
                                b'We trained on $4$ NVIDIA A100 GPUs for a week.\n'
                                b'\\end{document}'})))
    assert skip == ''
    paras, stats = latex_to_paragraphs(files)
    assert len(paras) == 1
    assert '4 NVIDIA A100 GPUs' in paras[0].text


def test_bib_and_bbl_are_never_surfaced():
    """A `.bib` is a pure reference database; a `.bbl` is a formatted one.

    Either would turn every cited title into candidate text -- and cited titles say
    "GPU-accelerated ..." and name specific cards constantly, so this is a large
    inflation of the mention series, not a slight one. The filter is a *whitelist*
    (`.tex/.ltx/.txt`), which is the kind of thing that drifts, hence this test.
    """
    bib = (b'@article{smith2019,\n title = {GPU-accelerated Monte Carlo on the K80 '
           b'and Tesla V100 throughput},\n journal = {Proc. IEEE}, year = {2019}\n}\n')
    bbl = (b'\\begin{thebibliography}{9}\n\\bibitem{x} A. Author. An H100 renderer. '
           b'Proc. SIGGRAPH, 2023.\n\\end{thebibliography}\n')
    payload = _tar_bytes({'main.tex': TEX, 'refs.bib': bib, 'Refs.BIB': bib,
                          'ms.bbl': bbl, 'notes.bib.tex': b'ignored-name-shape'})
    files, skip = unpack_member('2301.01234.gz', _gz(payload))
    assert skip == ''
    assert 'refs.bib' not in files and 'Refs.BIB' not in files, 'bib database surfaced'
    assert 'ms.bbl' not in files, 'formatted bibliography surfaced'
    assert 'main.tex' in files
    blob = b' '.join(files.values())
    for card in (b'K80', b'V100 throughput', b'H100'):
        assert card not in blob, f'{card!r} reached the matcher from a bibliography file'
