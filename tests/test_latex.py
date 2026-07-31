"""LaTeX -> paragraph conversion, driven by tests/fixtures/latex_cases.yaml.

Text quality is the whole justification for reading arXiv source rather than
S2ORC's GROBID output, and the converter has no grammar to guarantee it -- these
fixtures are the guarantee. Add a case for every real-world failure found.
"""

from pathlib import Path

import pytest
import yaml

from accelscan.config import SPLIT_LONG_PARA_CHARS
from accelscan.latex import (assemble_source, latex_to_paragraphs,
                            looks_like_reference)

ROOT = Path(__file__).resolve().parents[1]
CASES = yaml.safe_load((ROOT / 'tests/fixtures/latex_cases.yaml').read_text())['cases']


def _files(case) -> dict[str, bytes]:
    if 'files' in case:
        return {k: v.encode() for k, v in case['files'].items()}
    if 'tex_latin1' in case:
        # exercise the cp1252/latin-1 decode fallback with real high bytes
        return {'main.tex': case['tex_latin1'].encode('latin-1')}
    return {'main.tex': case['tex'].encode()}


@pytest.mark.parametrize('case', CASES, ids=[c['name'] for c in CASES])
def test_latex_case(case):
    paras, stats = latex_to_paragraphs(_files(case))
    joined = '\n'.join(p.text for p in paras)

    for want in case.get('contains', []):
        assert want in joined, f'missing {want!r} in {joined!r}'
    for bad in case.get('absent', []):
        assert bad not in joined, f'leaked {bad!r} in {joined!r}'
    if 'sections' in case:
        seen, out = set(), []
        for p in paras:
            if p.section and p.section not in seen:
                seen.add(p.section)
                out.append(p.section)
        assert out == case['sections']
    if 'n_paragraphs' in case:
        assert len(paras) == case['n_paragraphs']


def test_paragraph_idx_is_monotonic_and_offsets_are_consistent():
    tex = ('\\begin{document}\n' + '\n\n'.join(
        f'Paragraph number {i} describes an NVIDIA V100 GPU run in some detail.'
        for i in range(5)) + '\n\\end{document}')
    paras, _ = latex_to_paragraphs({'main.tex': tex.encode()})
    assert [p.idx for p in paras] == sorted(p.idx for p in paras)
    assert all(p.end > p.start for p in paras)
    assert all(p.end - p.start == len(p.text) for p in paras)


def test_long_paragraph_is_split_at_sentence_boundaries():
    sentence = 'We ran the model on an NVIDIA A100 GPU for a very long time indeed. '
    body = sentence * 120                       # ~8k chars, one TeX paragraph
    paras, _ = latex_to_paragraphs(
        {'main.tex': f'\\begin{{document}}\n{body}\n\\end{{document}}'.encode()})
    assert len(paras) > 1
    assert all(len(p.text) <= SPLIT_LONG_PARA_CHARS for p in paras)
    assert all(p.text.endswith('.') for p in paras)      # cut on sentence ends


def test_unbalanced_braces_do_not_raise():
    """Truncated source is normal on arXiv; the converter must degrade, not crash."""
    for tex in ('\\begin{document}\n\\emph{unclosed on an A100 GPU',
                '\\begin{document}\n\\section{Open',
                '\\begin{document}\n$8 unclosed math on a V100 GPU',
                '\\begin{figure}\nno end marker at all'):
        paras, stats = latex_to_paragraphs({'main.tex': tex.encode()})
        assert isinstance(paras, list)


def test_empty_and_binary_input():
    assert latex_to_paragraphs({})[0] == []
    assert latex_to_paragraphs({'a.png': b'\x89PNG\r\n\x1a\n'})[0] == []


def test_root_file_selection_prefers_the_document():
    files = {'appendix.tex': b'Some appendix text about an NVIDIA A100 GPU here.',
             'main.tex': b'\\documentclass{article}\\begin{document}'
                         b'The body mentions an NVIDIA V100 GPU explicitly here.'
                         b'\\end{document}'}
    src, stats = assemble_source(files)
    assert stats['root'] == 'main.tex'
    assert 'V100' in src


def test_looks_like_reference():
    assert looks_like_reference('[12] A. Author et al. Title. Proc. IEEE, 2015.')
    assert looks_like_reference('Smith et al. (2019) IEEE Trans. pp. 1-9.')
    assert not looks_like_reference(
        'We trained the network on eight NVIDIA V100 GPUs for three weeks, '
        'reaching 95% of peak throughput on the cluster described above.')
