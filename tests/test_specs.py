"""Spec-scrape validation: the FLOPS/VRAM estimands are only meaningful if the
scraped per-device specs are right and precision axes are not conflated.

Expected values are published dense (non-sparse) peaks, used ONLY as test
oracles — the pipeline itself never hardcodes specs, it reads the registry.
"""

from pathlib import Path

import pytest

from accelscan.registry import load_registry

ROOT = Path(__file__).resolve().parents[1]

# id -> (fp32_gflops, fp64_gflops, vram_gb); None = don't check
PUBLISHED = {
    'nvidia-v100': (14900, 7450, 32),
    'nvidia-a100': (19500, 9700, 80),
    'nvidia-h100': (66900, 33500, None),
    'nvidia-k80': (8736, 2912, None),
    'nvidia-geforce-gtx-1080': (8873, None, 8),
    'nvidia-geforce-gtx-1080-ti': (11340, None, 11),
    'nvidia-geforce-rtx-3090': (35581, None, 24),
    'nvidia-geforce-rtx-4090': (82575, None, 24),
    'nvidia-geforce-8800-gtx': (518, None, None),
    'amd-mi250x': (47870, 47870, 128),
}


@pytest.fixture(scope='session')
def reg():
    return load_registry(ROOT / 'registry')


@pytest.mark.parametrize('mid', sorted(PUBLISHED))
def test_published_specs(reg, mid):
    m = reg.models[mid]
    e32, e64, evram = PUBLISHED[mid]
    for got, exp, name in [(m.fp32_gflops, e32, 'fp32'),
                           (m.fp64_gflops, e64, 'fp64'),
                           (m.vram_gb, evram, 'vram')]:
        if exp is None:
            continue
        assert got is not None, f'{mid}: {name} missing'
        assert abs(got - exp) / exp < 0.05, f'{mid}: {name}={got}, expected ~{exp}'


def test_tensor_not_conflated_with_fp32(reg):
    """The data-center table names its tensor column 'Half precision Tensor
    Core FP32 Accumulate'; it must NOT land in fp32_gflops (~8x inflation)."""
    for mid in ['nvidia-v100', 'nvidia-a100', 'nvidia-h100']:
        m = reg.models[mid]
        assert m.fp16_tensor_gflops is not None, f'{mid}: tensor spec missing'
        assert m.fp16_tensor_gflops > m.fp32_gflops * 3, \
            f'{mid}: tensor should far exceed fp32'


def test_no_tensor_specs_on_pre_volta(reg):
    """Tensor cores arrived with Volta (2017); earlier cards must be null."""
    for mid in ['nvidia-geforce-gtx-1080', 'nvidia-k80', 'nvidia-geforce-8800-gtx']:
        assert reg.models[mid].fp16_tensor_gflops is None, f'{mid} has tensor spec'


# Apple's own published GPU figures (TFLOPS -> GFLOPS), for the top GPU bin of
# each chip. Oracles only; the registry reads them from the Apple silicon page.
APPLE_PUBLISHED = {'apple-m1': 2600, 'apple-m1-ultra': 21000,
                   'apple-m2-ultra': 27200, 'apple-m4-max': 16160}


@pytest.mark.parametrize('mid', sorted(APPLE_PUBLISHED))
def test_apple_gpu_fp32(reg, mid):
    got = reg.models[mid].fp32_gflops
    exp = APPLE_PUBLISHED[mid]
    assert got is not None, f'{mid}: fp32 missing'
    assert abs(got - exp) / exp < 0.05, f'{mid}: fp32={got}, expected ~{exp}'


def test_apple_variants_are_separate_entries(reg):
    """Pro/Max/Ultra differ by up to 8x in GPU throughput, so collapsing a
    generation into one entry (as the pre-0.3.0 hand entries did) would put an
    M1 Ultra's 21 TFLOPS and an M1's 2.6 into the same capacity bucket."""
    for gen in ('m1', 'm2', 'm3'):
        base, ultra = reg.models[f'apple-{gen}'], reg.models[f'apple-{gen}-ultra']
        assert ultra.fp32_gflops > 5 * base.fp32_gflops
        assert ultra.vram_gb > base.vram_gb
    # longest-span resolution must prefer the variant over the bare generation
    ms = reg.match_paragraph('we used an M4 Max in a MacBook Pro')
    assert [m.model_id for m in ms] == ['apple-m4-max']


def test_apple_axes_left_null(reg):
    """Metal exposes no FP64, and the Neural Engine's integer TOPS is not GPU
    dense FP16 throughput -- both axes must stay null rather than be filled."""
    apple = [m for m in reg.models.values() if m.manufacturer == 'apple']
    assert len(apple) >= 18
    for m in apple:
        assert m.fp64_gflops is None, f'{m.id}: unexpected fp64'
        assert m.fp16_tensor_gflops is None, f'{m.id}: Neural Engine TOPS leaked in'
        assert m.fp32_gflops is not None and m.vram_gb is not None, f'{m.id}: no specs'
        assert 'unified memory' in (m.notes or ''), f'{m.id}: missing vram caveat'


def test_spec_coverage_floor(reg):
    """Guard against a scrape regression silently emptying the spec columns."""
    models = [m for m in reg.models.values() if m.kind == 'model']
    have32 = sum(m.fp32_gflops is not None for m in models)
    havevram = sum(m.vram_gb is not None for m in models)
    assert have32 / len(models) > 0.5, f'fp32 coverage collapsed: {have32}/{len(models)}'
    assert havevram / len(models) > 0.9, f'vram coverage collapsed: {havevram}/{len(models)}'


def test_spec_source_present_when_specs_present(reg):
    for m in reg.models.values():
        if m.fp32_gflops is not None or m.vram_gb is not None:
            assert m.spec_source, f'{m.id}: specs without spec_source provenance'
