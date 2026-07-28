from pathlib import Path

import pytest
import yaml

from accelscan.registry import load_registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def reg():
    return load_registry(ROOT / 'registry')


def kept_ids(reg, text):
    return sorted({m.model_id for m in reg.match_paragraph(text)
                   if not m.gate_required or m.gate_ok})


CASES = yaml.safe_load((ROOT / 'tests/fixtures/registry_cases.yaml').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['text'][:40])
def test_case(reg, case):
    assert kept_ids(reg, case['text']) == sorted(case['expect'])


def test_registry_integrity(reg):
    assert len(reg.models) > 1000
    for m in reg.models.values():
        assert m.id and m.display and m.manufacturer
        if m.release is not None:
            assert 2000 <= int(m.release[:4]) <= 2027, m.id
    # every model-kind entry from the generated file has a release date
    dated = [m for m in reg.models.values() if m.kind == 'model' and m.release]
    assert len(dated) > 1000


def test_case_insensitive_normalization(reg):
    a = kept_ids(reg, 'trained on a TESLA V100 GPU')
    b = kept_ids(reg, 'trained on a Tesla V100 gpu')
    assert 'nvidia-v100' in a and 'nvidia-v100' in b


def test_plural_and_hyphen_forms(reg):
    assert 'nvidia-v100' in kept_ids(reg, 'eight V100s were used with CUDA')
    assert 'nvidia-geforce-rtx-3090' in kept_ids(reg, 'an RTX-3090 GPU')
    assert 'nvidia-geforce-rtx-3090' in kept_ids(reg, 'an RTX3090 GPU')


def test_boundaries(reg):
    assert kept_ids(reg, 'the CV100x sensor and V1000 pump') == []
