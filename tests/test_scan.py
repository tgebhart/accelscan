from pathlib import Path

import pytest

from accelscan.config import MAX_PASSAGES_GENERIC_ONLY
from accelscan.records import Paragraph
from accelscan.registry import load_registry
from accelscan.scan import scan_paragraphs, scan_record

ROOT = Path(__file__).resolve().parents[1]
FILLER = 'This filler sentence pads the paragraph to a plausible length.'


@pytest.fixture(scope='session')
def reg():
    return load_registry(ROOT / 'registry')


def record_from_paragraphs(paragraphs):
    return {'corpusid': 42, 'body': {'text': '\n\n'.join(paragraphs),
                                     'annotations': {}}}


def test_non_candidate(reg):
    rec = record_from_paragraphs([f'Nothing about hardware here. {FILLER}'] * 3)
    s = scan_record(rec, reg, 'shard0')
    assert s.inventory['is_candidate'] is False
    assert s.candidates == []


def test_candidate_with_neighbor_context(reg):
    rec = record_from_paragraphs([
        f'Introduction to the broader research problem. {FILLER}',
        f'We trained the network on an NVIDIA V100 GPU for ten hours. {FILLER}',
        f'Evaluation was carried out on held-out data. {FILLER}',
    ])
    s = scan_record(rec, reg, 'shard0')
    assert s.inventory['is_candidate'] is True
    assert len(s.candidates) == 1
    c = s.candidates[0]
    assert 'nvidia-v100' in c['matched_models']
    assert c['model_specific'] is True
    # passage includes neighbors and match offsets point at the surface
    for surf, a, b in zip(c['matched_surfaces'], c['match_starts'], c['match_ends']):
        assert c['passage_text'][a:b] == surf


def test_paper_level_gate_rescue(reg):
    # bare 'A100' with no local GPU context is rescued by a tier-A hit elsewhere
    bare = f'Throughput measured on the A100 was twice as high. {FILLER}'
    rec_alone = record_from_paragraphs([bare])
    assert scan_record(rec_alone, reg, 's').inventory['is_candidate'] is False

    rec_anchored = record_from_paragraphs([
        f'We used an NVIDIA A100 GPU for training runs. {FILLER}',
        f'Unrelated middle paragraph about datasets and metrics. {FILLER}',
        bare,
    ])
    s = scan_record(rec_anchored, reg, 's')
    assert len(s.candidates) == 2
    assert 'nvidia-a100' in s.candidates[1]['matched_models']


def test_generic_only_cap(reg):
    para = f'The computation ran on a GPU cluster for several days. {FILLER}'
    rec = record_from_paragraphs([para] * (MAX_PASSAGES_GENERIC_ONLY + 3))
    s = scan_record(rec, reg, 's')
    assert s.inventory['is_candidate'] is True
    assert len(s.candidates) == MAX_PASSAGES_GENERIC_ONLY
    assert s.inventory['passages_truncated'] is True
    assert all(c['model_specific'] is False for c in s.candidates)


def test_shared_core_is_corpus_agnostic(reg):
    """The matcher must be reached by exactly one code path.

    Same prose, once via the S2ORC record adapter and once via `scan_paragraphs`
    with hand-built Paragraphs (the seam the arXiv LaTeX driver uses). Every
    measurement-bearing field must agree; only the ids may differ. If someone
    forks the match/gate/cap logic for a second corpus, this fails.
    """
    paras = [f'We trained on 8 NVIDIA A100 GPUs for two weeks. {FILLER}',
             f'An unrelated paragraph about the dataset and metrics. {FILLER}',
             f'Inference used a single RTX 3090 card. {FILLER}']
    s2 = scan_record(record_from_paragraphs(paras), reg, 'shard-a')
    ax = scan_paragraphs(
        [Paragraph(idx=i, start=0, end=len(t), text=t, section=None)
         for i, t in enumerate(paras)],
        reg, paper_id='arxiv:2301.01234', corpusid=None, shard_id='shard-a',
        body_chars=len('\n\n'.join(paras)))

    assert len(s2.candidates) == len(ax.candidates) > 0
    measured = ('para_idx', 'passage_text', 'section_header', 'matched_models',
                'matched_surfaces', 'match_starts', 'match_ends', 'gated_only',
                'model_specific')
    for a, b in zip(s2.candidates, ax.candidates):
        assert {k: a[k] for k in measured} == {k: b[k] for k in measured}
    for k in ('is_candidate', 'n_paragraphs', 'n_candidate_passages',
              'passages_truncated'):
        assert s2.inventory[k] == ax.inventory[k]

    # ids: S2ORC derives paper_id from corpusid; arXiv carries a string id only
    assert s2.inventory['paper_id'] == '42' and s2.inventory['corpusid'] == 42
    assert ax.inventory['paper_id'] == 'arxiv:2301.01234'
    assert ax.inventory['corpusid'] is None
    assert ax.candidates[0]['passage_id'].startswith('arxiv:2301.01234:')


def test_match_offsets_index_the_stored_passage(reg):
    """match_starts/ends must index passage_text after neighbour padding."""
    s = scan_record(record_from_paragraphs(
        [f'Baseline paragraph with no hardware. {FILLER}',
         f'We ran the model on an NVIDIA V100 GPU. {FILLER}',
         f'Trailing paragraph for context. {FILLER}']), reg, 's')
    c = s.candidates[0]
    for start, end, surface in zip(c['match_starts'], c['match_ends'],
                                   c['matched_surfaces']):
        assert c['passage_text'][start:end] == surface
