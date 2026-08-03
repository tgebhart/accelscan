"""Stage-2 output parsing: status assignment and the Int32 device-count clamp.

`accelscan.infer` imports vLLM lazily (inside `build_llm`/`sampling_params`), so
these run on a CPU box with no GPU stack installed.
"""

import orjson
import polars as pl
import pytest

from accelscan.infer import INT32_MAX, MENTION_SCHEMA, parse_output


def _payload(**over) -> str:
    m = {'model_raw': 'A100', 'model_normalized': 'NVIDIA A100',
         'manufacturer': 'nvidia', 'accelerator_subtype': 'datacenter-gpu',
         'device_count': 4, 'device_count_basis': 'explicit', 'memory_gb': 80.0,
         'usage_context': 'used-in-this-work', 'evidence_quote': 'on 4 A100s'}
    m.update(over)
    return orjson.dumps({'mentions': [m]}).decode()


def _frame(mentions: list[dict]) -> pl.DataFrame:
    """Build one row per mention exactly as `run_inference` does."""
    base = {'passage_id': 'p:1', 'paper_id': 'p', 'corpusid': 1,
            'passage_shard': 0, 'status': 'ok', 'raw_json': '{}',
            'registry_version': '0.1.0', 'prompt_version': 'p1',
            'llm_model': 'Qwen/Qwen3-14B', 'code_version': 'abc123'}
    rows = [{**base, 'mention_idx': j, **m} for j, m in enumerate(mentions)]
    return pl.DataFrame(rows, schema=MENTION_SCHEMA, orient='row')


def test_status_codes():
    assert parse_output('', truncated=True)[0] == 'truncated'
    assert parse_output('not json', truncated=False)[0] == 'json_error'
    assert parse_output('{"mentions": []}', truncated=False)[0] == 'no_mention'
    status, mentions = parse_output(_payload(), truncated=False)
    assert status == 'ok'
    assert mentions[0]['device_count'] == 4


@pytest.mark.parametrize('raw', [5_000_000_000_000, 2**31, 10**18])
def test_out_of_range_device_count_is_clamped_not_fatal(raw):
    """An absurd count must not abort the shard.

    The guided-JSON schema's `int` is unbounded, MENTION_SCHEMA's column is
    Int32, and one such value cost a full hour of inference on S2ORC shard 27
    before the frame was built. It is clamped rather than nulled: the mention did
    state a count, and every value here is orders of magnitude above
    `compute.HARD_DEVICE_CAP`, so winsorization is unaffected.
    """
    status, mentions = parse_output(_payload(device_count=raw), truncated=False)
    assert status == 'ok'
    assert mentions[0]['device_count'] == INT32_MAX
    assert _frame(mentions)['device_count'][0] == INT32_MAX


def test_in_range_counts_are_untouched():
    for n in (1, 8, 65_536, 150_000_000, INT32_MAX):
        _, mentions = parse_output(_payload(device_count=n), truncated=False)
        assert mentions[0]['device_count'] == n
        assert _frame(mentions)['device_count'][0] == n


def test_device_count_column_stays_int32():
    """Widening the column would make one shard unreadable in the same scan as
    the 33 already written under Int32."""
    assert MENTION_SCHEMA['device_count'] == pl.Int32
