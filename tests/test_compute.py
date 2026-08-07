"""Stage 3.5: device-count handling and per-paper capacity sums.

The capacity table is where a bad number stops being visible: once
`reported_fp32_gflops` is a float, nothing downstream can tell a real 8-GPU
cluster from an extraction artifact. So the cases here are the ones that produce
a *plausible* wrong number rather than an error:

- `device_count = 0` from a fractional hardware report ("one-half K80",
  "0.04 GPU days"), which without a floor makes every axis exactly 0 while
  `spec_missing_*` stays False
- the 13e6-device outlier, which must be winsorized, not trusted
- a used model with no spec on an axis, which must be null and flagged, never 0
- the same model mentioned in several passages, which must not be double-counted
"""

import polars as pl
import pytest

from accelscan import compute as cp

MENTIONS = [
    # corpusid, canonical_model, device_count, basis, memory_gb, year
    (1, 'nvidia-v100', 8, 'explicit', None, 2019),
    (1, 'nvidia-v100', 4, 'explicit', None, 2019),      # same model, 2nd passage
    (2, 'nvidia-v100', None, None, 32.0, 2019),          # unstated count
    (3, 'nvidia-v100', 0, 'explicit', None, 2019),       # "one-half a K80"
    (4, 'nvidia-v100', 13_000_000, 'explicit', None, 2019),   # extraction junk
    (5, 'no-spec-model', 2, 'explicit', None, 2019),
]

SPECS = pl.DataFrame({
    'canonical_model': ['nvidia-v100', 'no-spec-model'],
    'display': ['NVIDIA Tesla V100', 'Mystery'],
    'manufacturer': ['nvidia', 'nvidia'],
    'segment': ['datacenter', 'datacenter'],
    'architecture': ['volta', None],
    'release_year': [2017, None],
    'fp32_gflops': [14_000.0, None],
    'fp64_gflops': [7_000.0, None],
    'fp16_tensor_gflops': [112_000.0, None],
    'vram_gb': [16.0, None],
    'tdp_w': [300.0, None],
}, schema={'canonical_model': pl.Utf8, 'display': pl.Utf8, 'manufacturer': pl.Utf8,
           'segment': pl.Utf8, 'architecture': pl.Utf8, 'release_year': pl.Int32,
           'fp32_gflops': pl.Float64, 'fp64_gflops': pl.Float64,
           'fp16_tensor_gflops': pl.Float64, 'vram_gb': pl.Float64,
           'tdp_w': pl.Float64})


@pytest.fixture
def mentions() -> pl.DataFrame:
    cols = ['corpusid', 'canonical_model', 'device_count', 'device_count_basis',
            'memory_gb', 'year']
    return pl.DataFrame({c: [r[i] for r in MENTIONS] for i, c in enumerate(cols)},
                        schema={'corpusid': pl.Int64, 'canonical_model': pl.Utf8,
                                'device_count': pl.Int64, 'device_count_basis': pl.Utf8,
                                'memory_gb': pl.Float64, 'year': pl.Int32})


def _n_dev(w: pl.DataFrame, corpusid: int) -> list[float]:
    return w.filter(pl.col('corpusid') == corpusid)['n_dev'].to_list()


def test_zero_device_count_is_floored_to_one(mentions):
    """A used device implies >=1 device. A 0 would zero out every axis while
    leaving spec_missing False, i.e. publish an exact 0 TFLOPS as a real value."""
    w = cp.winsorize_counts(mentions, policy='floor1')
    assert _n_dev(w, 3) == [1.0]
    cap = cp.paper_capacity(w, SPECS)
    row = cap.filter(pl.col('corpusid') == 3)
    assert row['reported_fp32_gflops'].item() == 14_000.0
    assert row['spec_missing_fp32'].item() is False
    # the property that was violated in production: no axis is ever exactly 0
    for col in [f'reported_{a}_gflops' for a in cp.AXES] + ['reported_vram_gb',
                                                            'reported_tdp_w']:
        assert cap.filter(pl.col(col) == 0).height == 0, f'{col} has an exact zero'


def test_outlier_count_is_winsorized(mentions):
    w = cp.winsorize_counts(mentions, policy='floor1')
    assert _n_dev(w, 4)[0] <= cp.HARD_DEVICE_CAP
    assert _n_dev(w, 4)[0] < 13_000_000


def test_null_count_policies(mentions):
    """floor1 assumes one device (a floor); explicit drops the mention."""
    assert _n_dev(cp.winsorize_counts(mentions, policy='floor1'), 2) == [1.0]
    w = cp.winsorize_counts(mentions, policy='explicit')
    assert _n_dev(w, 2) == [], 'an unstated count is not explicit evidence'
    assert w.filter(pl.col('corpusid') == 1).height == 2


def test_repeated_model_is_not_double_counted(mentions):
    """Paper 1 mentions the V100 twice (8 and 4 devices) across passages."""
    cap = cp.paper_capacity(cp.winsorize_counts(mentions), SPECS)
    row = cap.filter(pl.col('corpusid') == 1)
    assert row['total_devices'].item() == 8.0, 'the max, not 8+4'
    assert row['reported_fp32_gflops'].item() == 8 * 14_000.0
    assert row['n_models'].item() == 1


def test_missing_spec_is_null_and_flagged_never_zero(mentions):
    cap = cp.paper_capacity(cp.winsorize_counts(mentions), SPECS)
    row = cap.filter(pl.col('corpusid') == 5)
    for axis in cp.AXES:
        assert row[f'spec_missing_{axis}'].item() is True
        assert row[f'reported_{axis}_gflops'].item() is None
    assert row['spec_missing_vram'].item() is True
    assert row['reported_vram_gb'].item() is None
    # polars sums a column of nulls to 0, so vram and tdp need the same explicit
    # null-out the FLOPS axes get -- otherwise an unspec'd model publishes 0 W
    assert row['spec_missing_tdp'].item() is True
    assert row['reported_tdp_w'].item() is None


def test_stated_memory_resolves_the_variant(mentions):
    """Paper 2 states 32 GB, so the 16 GB registry default must not win."""
    cap = cp.paper_capacity(cp.winsorize_counts(mentions), SPECS)
    assert cap.filter(pl.col('corpusid') == 2)['reported_vram_gb'].item() == 32.0
    # paper 1 states nothing, so it falls back to the registry config
    assert cap.filter(pl.col('corpusid') == 1)['reported_vram_gb'].item() == 8 * 16.0


def test_capacity_is_monotone_in_device_count(mentions):
    """Doubling every count must not lower any axis -- the sanity invariant that
    catches a broken join or a sum over the wrong grouping."""
    base = cp.paper_capacity(cp.winsorize_counts(mentions), SPECS).sort('corpusid')
    more = cp.paper_capacity(
        cp.winsorize_counts(mentions.with_columns(
            device_count=pl.col('device_count') * 2)), SPECS).sort('corpusid')
    j = base.join(more, on='corpusid', how='inner', suffix='_more')
    grew = j.filter(pl.col('reported_fp32_gflops').is_not_null())
    assert (grew['reported_fp32_gflops_more'] >= grew['reported_fp32_gflops']).all()
