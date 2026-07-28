from pathlib import Path

import pytest

from accelscan.config import MAX_PASSAGES_GENERIC_ONLY
from accelscan.registry import load_registry
from accelscan.scan import scan_record

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
