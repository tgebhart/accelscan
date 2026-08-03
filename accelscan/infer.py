"""Stage 2: vLLM offline batched extraction over passage shards.

The model is loaded once per process; pass multiple shards to amortize the
load + torch.compile cost across them on a single GPU. Each shard is
independently restartable via its `.done` marker (skipped if present), so a
preempted or rerun process just redoes the unfinished shards.

  python -m accelscan.infer --shard-index $SLURM_ARRAY_TASK_ID   # SLURM array: 1 shard/task
  python -m accelscan.infer --shards 0-35                        # one GPU, all shards, one load
  python -m accelscan.infer --shards 0-9,20,25                   # ranges + explicit indices
  python -m accelscan.infer --local-parquet output/smoke/x.candidates.parquet \
      --out-dir output/infer_smoke --limit 200                   # dev/smoke
"""

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

# Greedy decoding (temperature=0) never needs FlashInfer's top-k/top-p sampler,
# whose JIT compile at warmup requires nvcc/CUDA_HOME (absent on bare compute
# nodes). Force the torch-native sampler so startup doesn't depend on a CUDA
# toolkit. Must be set before vllm is imported.
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')

import orjson
import polars as pl

from accelscan.config import BUCKET
from accelscan.paths import S2ORC, Corpus, get_corpus, mentions_parts, passages_key
from accelscan.llm_schema import EXTRACTION_JSON_SCHEMA, PassageExtraction
from accelscan.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from accelscan.registry import load_registry

DEFAULT_MODEL = 'Qwen/Qwen3-14B'
MAX_OUTPUT_TOKENS = 1024

MENTION_SCHEMA = {
    'passage_id': pl.Utf8, 'paper_id': pl.Utf8, 'corpusid': pl.Int64,
    'passage_shard': pl.Int32,
    'status': pl.Utf8, 'mention_idx': pl.Int32,
    'model_raw': pl.Utf8, 'model_normalized': pl.Utf8, 'manufacturer': pl.Utf8,
    'accelerator_subtype': pl.Utf8, 'device_count': pl.Int32,
    'device_count_basis': pl.Utf8, 'memory_gb': pl.Float32,
    'usage_context': pl.Utf8, 'evidence_quote': pl.Utf8, 'raw_json': pl.Utf8,
    'registry_version': pl.Utf8, 'prompt_version': pl.Utf8,
    'llm_model': pl.Utf8, 'code_version': pl.Utf8,
}


def code_version() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def model_tag(model: str) -> str:
    return model.split('/')[-1].lower().replace('.', '-')


def build_llm(model: str, max_model_len: int):
    from vllm import LLM
    return LLM(model=model, dtype='bfloat16', max_model_len=max_model_len,
               gpu_memory_utilization=0.92, enable_prefix_caching=True)


def sampling_params():
    from vllm import SamplingParams
    try:  # vllm >= 0.9
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS,
                              structured_outputs=StructuredOutputsParams(
                                  json=EXTRACTION_JSON_SCHEMA))
    except ImportError:  # vllm < 0.9
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS,
                              guided_decoding=GuidedDecodingParams(
                                  json=EXTRACTION_JSON_SCHEMA))


INT32_MAX = 2**31 - 1


def _clamp(m: dict) -> dict:
    """Clamp `device_count` into Int32.

    The schema's `int` is unbounded but `MENTION_SCHEMA` stores Int32, so a
    passage reading "5 x 10^12 operations" that the model transcribes as a device
    count aborted a whole shard at frame construction -- after an hour of
    inference, and only for the one shard containing it. Clamping keeps the value
    obviously junk (it is orders of magnitude above `compute.HARD_DEVICE_CAP`, so
    winsorization treats it exactly as before) without widening the column, which
    would make this shard's parquet unreadable in the same scan as the others.
    """
    n = m.get('device_count')
    if n is not None and not -INT32_MAX <= n <= INT32_MAX:
        m['device_count'] = INT32_MAX if n > 0 else 0
    return m


def parse_output(text: str, truncated: bool) -> tuple[str, list[dict]]:
    if truncated:
        return 'truncated', []
    try:
        ext = PassageExtraction.model_validate(orjson.loads(text))
    except Exception:
        return 'json_error', []
    if not ext.mentions:
        return 'no_mention', []
    return 'ok', [_clamp(m.model_dump()) for m in ext.mentions]


def parse_shard_spec(spec: str) -> list[int]:
    """'0-35' -> [0..35]; '0,2,5' -> [0,2,5]; '0-3,7' -> [0,1,2,3,7]."""
    out: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def run_inference(llm, params, passages: pl.DataFrame, shard_index: int,
                  provenance: dict) -> pl.DataFrame:
    messages = [
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': build_user_prompt(
             row['section_header'], row['matched_surfaces'], row['passage_text'])}]
        for row in passages.iter_rows(named=True)
    ]
    outputs = llm.chat(messages, params,
                       chat_template_kwargs={'enable_thinking': False})

    rows = []
    for row, out in zip(passages.iter_rows(named=True), outputs):
        text = out.outputs[0].text
        truncated = out.outputs[0].finish_reason == 'length'
        status, mentions = parse_output(text, truncated)
        # passage shards written before paper_id existed only carry corpusid
        base = {'passage_id': row['passage_id'],
                'paper_id': row.get('paper_id') or str(row['corpusid']),
                'corpusid': row['corpusid'],
                'passage_shard': shard_index, 'status': status,
                'raw_json': text, **provenance}
        if not mentions:
            rows.append({**base, 'mention_idx': None, 'model_raw': None,
                         'model_normalized': None, 'manufacturer': None,
                         'accelerator_subtype': None, 'device_count': None,
                         'device_count_basis': None, 'memory_gb': None,
                         'usage_context': None, 'evidence_quote': None})
        for j, m in enumerate(mentions):
            rows.append({**base, 'mention_idx': j, **m})
    return pl.DataFrame(rows, schema=MENTION_SCHEMA, orient='row')


def process_shard(client, llm, params, shard_index: int, tag: str,
                  registry_version: str, provenance: dict,
                  corpus: Corpus = S2ORC) -> None:
    """Fetch one passage shard, extract, upload output + `.done` marker.
    Skips if the marker already exists."""
    src = passages_key(corpus, registry_version, shard_index)
    dst_base = (f'{mentions_parts(corpus, registry_version, PROMPT_VERSION, tag)}'
                f'/shard_{shard_index:04d}')
    done_key = f'{dst_base}.done'
    if client.list_objects_v2(Bucket=BUCKET, Prefix=done_key).get('KeyCount', 0):
        print(f'shard {shard_index} already done, skipping', file=sys.stderr)
        return

    body = client.get_object(Bucket=BUCKET, Key=src)['Body'].read()
    passages = pl.read_parquet(io.BytesIO(body))
    df = run_inference(llm, params, passages, shard_index, provenance)
    buf = io.BytesIO()
    df.write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=f'{dst_base}.parquet', Body=buf.getvalue())
    client.put_object(Bucket=BUCKET, Key=done_key, Body=b'')
    print(f'shard {shard_index}: {passages.height} passages -> {df.height} rows',
          file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, help='single shard (SLURM array)')
    ap.add_argument('--shards', help='multiple shards in one process, e.g. "0-35" or "0-9,20"')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--local-parquet', help='dev mode: candidates parquet on disk')
    ap.add_argument('--out-dir', default='output/infer_local')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--corpus', default='s2orc', choices=['s2orc', 'arxiv'])
    ap.add_argument('--registry-version', help='registry version the passage shards '
                    'were repacked under (default: discovered from S3)')
    args = ap.parse_args()

    reg = load_registry()
    provenance = {'registry_version': reg.version, 'prompt_version': PROMPT_VERSION,
                  'llm_model': args.model, 'code_version': code_version()}

    if args.local_parquet:
        passages = pl.read_parquet(args.local_parquet)
        if args.limit:
            passages = passages.head(args.limit)
        llm = build_llm(args.model, args.max_model_len)
        df = run_inference(llm, sampling_params(), passages, -1, provenance)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / 'mentions.parquet')
        print(df.group_by('status').len().sort('len', descending=True))
        return

    if args.shards:
        shards = parse_shard_spec(args.shards)
    elif args.shard_index is not None:
        shards = [args.shard_index]
    else:
        ap.error('pass --shards or --shard-index in S3 mode')

    from accelscan.s3 import discover_passages_version, make_s3_client
    client = make_s3_client()
    tag = model_tag(args.model)
    c = get_corpus(args.corpus)
    # The version that namespaces input and output is the one stage 1.5 wrote, NOT
    # the locally installed registry: a bump between the scan and this run would
    # otherwise 404 on the passage shard and split the mentions table in two.
    pv = args.registry_version or discover_passages_version(c, client)
    if pv != reg.version:
        print(f'passages are under registry {pv}, local registry is {reg.version}; '
              f'reading and writing {pv}', file=sys.stderr)
        provenance['registry_version'] = pv
    llm = build_llm(args.model, args.max_model_len)  # loaded once, reused across shards
    params = sampling_params()
    for shard_index in shards:
        process_shard(client, llm, params, shard_index, tag, pv, provenance, c)
    print(f'done: {len(shards)} shard(s) processed', file=sys.stderr)


if __name__ == '__main__':
    main()
