"""arXiv tar members -> per-paper source files, plus id/date parsing.

One member of an `arXiv_src_YYMM_NNN.tar` is one submission: either `<id>.gz`
(gzipped single .tex, or a gzipped tar of a multi-file source) or `<id>.pdf` for a
PDF-only submission. Nothing here touches the network.

Two rules that matter:

**Classify by magic bytes, never by extension.** Real tars contain PostScript with
a `.gz` name, HTML masquerading as TeX, and truncated gzip streams. Every unknown
shape is *counted* into a skip reason rather than raised, because a crash costs the
whole 500 MB tar on re-download.

**Never surface a `.bbl`.** Bibliography exclusion layer 1 (see `accelscan.latex`):
if a `.bbl` reached the converter, formatted reference lists would become candidate
passages and a reference titled "GPU-accelerated Monte Carlo..." would be extracted
as if the paper used a K80.

Year is derivable from the id for *every* arXiv paper, 1991->present, so stage 1
needs no metadata lookup: new ids are `YYMM.NNNNN`, old ones `archive/YYMMNNN`,
and YY >= 91 means 19YY.
"""

import gzip
import io
import re
import tarfile

# Archives used by the pre-2007 identifier scheme, including the defunct ones.
# Member names concatenate archive and number (`hep-th0001001`), so splitting
# requires this list; a wrong split yields a paper_id that matches nothing.
OLD_ARCHIVES = frozenset("""
    acc-phys adap-org alg-geom ao-sci astro-ph atom-ph bayes-an chao-dyn chem-ph
    cmp-lg comp-gas cond-mat cs dg-ga funct-an gr-qc hep-ex hep-lat hep-ph hep-th
    math math-ph mtrl-th nlin nucl-ex nucl-th patt-sol physics plasm-ph q-alg
    q-bio quant-ph solv-int supr-con
""".split())

# Read from a submission; `.bbl` is deliberately absent (see module docstring).
TEXT_EXTS = ('.tex', '.ltx', '.txt', '.sty', '.cls', '.tikz')
MAX_MEMBER_BYTES = 64 << 20          # arXiv's historical submission cap is ~50 MB
MAX_NESTED_FILES = 400
MAX_NESTED_BYTES = 24 << 20

NEW_ID = re.compile(r'(\d{4})\.(\d{4,5})')
OLD_ID = re.compile(r'([a-z-]+(?:\.[A-Z]{2})?)(\d{7})')
_VERSION = re.compile(r'v\d+$')

SKIP_REASONS = ('pdf_only', 'postscript', 'not_gzip', 'oversize', 'no_tex',
                'corrupt', 'unknown_id')


def arxiv_id_from_member(name: str) -> str | None:
    """Member path -> canonical arXiv id, or None if unrecognisable.

    `0704/0704.0001.gz`      -> `0704.0001`
    `0001/hep-th0001001.gz`  -> `hep-th/0001001`
    `math0001001v2.gz`       -> `math/0001001`   (version suffix stripped)
    """
    stem = name.rsplit('/', 1)[-1]
    for ext in ('.gz', '.pdf', '.tar', '.abs'):
        stem = stem.removesuffix(ext)
    stem = _VERSION.sub('', stem)

    m = NEW_ID.fullmatch(stem)
    if m:
        return f'{m.group(1)}.{m.group(2)}'
    m = OLD_ID.fullmatch(stem.replace('/', ''))
    if m:
        archive = m.group(1)
        if archive.split('.')[0] in OLD_ARCHIVES:
            return f'{archive}/{m.group(2)}'
    return None


def year_month_from_id(arxiv_id: str) -> tuple[int, int] | tuple[None, None]:
    """(year, month) from the id itself. arXiv started in 1991, so YY>=91 is 19YY."""
    digits = arxiv_id.split('/')[-1].split('.')[0] if '/' in arxiv_id \
        else arxiv_id.split('.')[0]
    if len(digits) < 4 or not digits[:4].isdigit():
        return None, None
    yy, mm = int(digits[:2]), int(digits[2:4])
    if not 1 <= mm <= 12:
        return None, None
    return (1900 + yy if yy >= 91 else 2000 + yy), mm


def paper_id_from_arxiv_id(arxiv_id: str) -> str:
    """Namespaced universal key: `arxiv:2301.01234`, `arxiv:hep-th/9901001`."""
    return f'arxiv:{arxiv_id}'


def member_kind(name: str, data: bytes) -> str:
    """'pdf' | 'postscript' | 'gzip' | 'tar' | 'text' | 'unknown', by magic bytes."""
    head = data[:8]
    if head.startswith(b'%PDF'):
        return 'pdf'
    if head.startswith((b'%!PS', b'\x04%!')):
        return 'postscript'
    if head.startswith(b'\x1f\x8b'):
        return 'gzip'
    if len(data) > 262 and data[257:262] == b'ustar':
        return 'tar'
    if name.lower().endswith(TEXT_EXTS) or head.startswith((b'\\doc', b'%', b'\\begin')):
        return 'text'
    return 'unknown'


def _is_tar(data: bytes) -> bool:
    if len(data) > 262 and data[257:262] == b'ustar':
        return True
    try:                                    # older tars lack the ustar magic
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:'):
            return True
    except Exception:
        return False


def _from_nested_tar(data: bytes) -> dict[str, bytes]:
    """Text members of a multi-file submission; `.bbl` and binaries excluded."""
    out, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        for m in tf.getmembers():
            if not m.isfile() or len(out) >= MAX_NESTED_FILES:
                continue
            low = m.name.lower()
            if low.endswith('.bbl') or not low.endswith(TEXT_EXTS):
                continue
            if m.size > MAX_NESTED_BYTES or total + m.size > MAX_NESTED_BYTES:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            out[m.name] = f.read()
            total += m.size
    return out


def unpack_member(name: str, data: bytes) -> tuple[dict[str, bytes], str]:
    """Tar member -> ({filename: bytes}, skip_reason).

    `skip_reason` is '' on success and one of SKIP_REASONS otherwise; callers count
    them into the ingest-stats product rather than failing the tar.
    """
    if not data:
        return {}, 'corrupt'
    if len(data) > MAX_MEMBER_BYTES:
        return {}, 'oversize'

    kind = member_kind(name, data)
    if kind == 'pdf':
        return {}, 'pdf_only'
    if kind == 'postscript':
        return {}, 'postscript'

    if kind == 'gzip':
        try:
            data = gzip.decompress(data)
        except Exception:
            return {}, 'corrupt'
        if len(data) > MAX_MEMBER_BYTES:
            return {}, 'oversize'
        # the decompressed payload is itself a PDF/PS surprisingly often
        inner = member_kind(name.removesuffix('.gz'), data)
        if inner == 'pdf':
            return {}, 'pdf_only'
        if inner == 'postscript':
            return {}, 'postscript'
    elif kind not in ('tar', 'text'):
        return {}, 'not_gzip'

    if _is_tar(data):
        try:
            files = _from_nested_tar(data)
        except Exception:
            return {}, 'corrupt'
        return (files, '') if files else ({}, 'no_tex')

    # single-file submission: a bare .tex under the paper's own name
    stem = name.rsplit('/', 1)[-1].removesuffix('.gz') or 'main'
    if not stem.lower().endswith(TEXT_EXTS):
        stem += '.tex'
    return {stem: data}, ''
