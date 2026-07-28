"""Stage 1.6: numerical-precision flags from stored candidate passages.

Complements the *capability* view from registry specs (what precision the
hardware could do) with the *chosen* representation the authors report
(fp32/fp16/bf16/tf32/int8/mixed). The divergence between the two is a finding
in itself: mixed-precision adoption can race ahead of hardware turnover.

Pure CPU regex over the candidate passages already on S3 — no LLM, no GPU, and
no re-extraction. One row per paper with boolean flags + the matched surfaces
for auditability.

**Coverage caveat (report with any precision figure):** the sweep sees only
candidate passages (the hardware paragraph ± 1 neighbour), not full text.
Precision language often sits in a separate "implementation details" paragraph,
so these flags are a LOWER BOUND on reported precision (~1-3% of accelerator
papers hit a pattern here). They are still comparable across years and fields
because the passage-selection rule is constant; treat levels as conservative
and read the trends. A complete-coverage version would require re-running the
stage-1 full-text scan with these patterns as additional capture rules.

  python -m accelscan.precision                       # -> S3
  python -m accelscan.precision --shards 4 --local-out output/precision_smoke
"""

import argparse
import io
import re
import sys
from pathlib import Path

import polars as pl

from accelscan.config import BUCKET, OUT_PREFIX

# Word-boundary patterns; case-insensitive except where caps disambiguate.
# Ordered dict: flag -> pattern. Keep each pattern narrow — these become
# paper-level claims about reported numerics.
PATTERNS: dict[str, str] = {
    'fp64': r'(?<!\w)(?:fp64|float64|double[- ]precision|binary64)(?!\w)',
    'fp32': r'(?<!\w)(?:fp32|float32|single[- ]precision|full[- ]precision)(?!\w)',
    'tf32': r'(?<!\w)(?:tf32|tensorfloat[- ]?32)(?!\w)',
    'fp16': r'(?<!\w)(?:fp16|float16|half[- ]precision)(?!\w)',
    'bf16': r'(?<!\w)(?:bf16|bfloat16|brain[- ]float)(?!\w)',
    'fp8': r'(?<!\w)(?:fp8|float8|e4m3|e5m2)(?!\w)',
    'int8': r'(?<!\w)(?:int8|8[- ]bit integer|uint8)(?!\w)',
    'int4': r'(?<!\w)(?:int4|4[- ]bit)(?!\w)',
    'mixed_precision': r'(?<!\w)(?:mixed[- ]precision|amp\b|automatic mixed precision'
                       r'|apex\.amp|torch\.cuda\.amp|bitsandbytes)(?!\w)',
    'quantization': r'(?<!\w)(?:quanti[sz](?:ed|ation)|post[- ]training quanti'
                    r'|qat\b|gptq|awq|qlora)(?!\w)',
    'tensor_core': r'(?<!\w)(?:tensor cores?|tensorcores?)(?!\w)',
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in PATTERNS.items()}

SCHEMA = {'corpusid': pl.Int64,
          **{f'prec_{k}': pl.Boolean for k in PATTERNS},
          'prec_surfaces': pl.List(pl.Utf8), 'n_passages_scanned': pl.Int32}


def flags_for_text(text: str) -> tuple[dict[str, bool], list[str]]:
    flags, surfaces = {}, []
    for name, rx in COMPILED.items():
        m = rx.search(text)
        flags[name] = m is not None
        if m:
            surfaces.append(m.group(0).lower())
    return flags, surfaces


def scan_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """candidates shard (corpusid, passage_text) -> per-paper precision flags."""
    per_paper: dict[int, dict] = {}
    for cid, text in zip(df['corpusid'].to_list(), df['passage_text'].to_list()):
        rec = per_paper.setdefault(cid, {'corpusid': cid, 'surfaces': set(), 'n': 0})
        rec['n'] += 1
        if not text:
            continue
        flags, surfaces = flags_for_text(text)
        rec['surfaces'].update(surfaces)
        for k, v in flags.items():
            rec[k] = rec.get(k, False) or v
    rows = [{
        'corpusid': r['corpusid'],
        **{f'prec_{k}': bool(r.get(k, False)) for k in PATTERNS},
        'prec_surfaces': sorted(r['surfaces']),
        'n_passages_scanned': r['n'],
    } for r in per_paper.values()]
    return pl.DataFrame(rows, schema=SCHEMA) if rows else pl.DataFrame(schema=SCHEMA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates-version', help='registry version of candidates '
                    '(default: highest available)')
    ap.add_argument('--shards', type=int, help='dev: only first N candidate shards')
    ap.add_argument('--local-out', help='dev: write locally instead of S3')
    args = ap.parse_args()

    from accelscan.s3 import list_keys, make_s3_client, storage_options
    client = make_s3_client()
    so = storage_options()

    # candidates are namespaced by the registry version at SCAN time
    prefix = f'{OUT_PREFIX}/candidates/'
    keys = list_keys(prefix, suffix='.parquet', client=client)
    versions = sorted({k[len(prefix):].split('/')[0] for k in keys},
                      key=lambda v: [int(x) for x in v.split('.')])
    version = args.candidates_version or versions[-1]
    keys = [k for k in keys if k.startswith(f'{prefix}{version}/')]
    if args.shards:
        keys = keys[:args.shards]
    print(f'candidates v{version}: {len(keys)} shards', file=sys.stderr)

    parts = []
    for i, k in enumerate(keys):
        body = client.get_object(Bucket=BUCKET, Key=k)['Body'].read()
        df = pl.read_parquet(io.BytesIO(body), columns=['corpusid', 'passage_text'])
        parts.append(scan_candidates(df))
        if (i + 1) % 25 == 0 or i + 1 == len(keys):
            print(f'  {i+1}/{len(keys)} shards', file=sys.stderr)

    out = pl.concat(parts) if parts else pl.DataFrame(schema=SCHEMA)
    # a paper can appear in several shards; OR the flags together
    out = out.group_by('corpusid').agg(
        *[pl.col(f'prec_{k}').any() for k in PATTERNS],
        prec_surfaces=pl.col('prec_surfaces').list.explode(
            keep_nulls=False, empty_as_null=False).unique(),
        n_passages_scanned=pl.col('n_passages_scanned').sum())
    print(f'{out.height:,} papers with precision flags', file=sys.stderr)
    for k in PATTERNS:
        n = out[f'prec_{k}'].sum()
        print(f'  {k:16} {n:7,} papers ({100*n/max(out.height,1):5.1f}%)', file=sys.stderr)

    if args.local_out:
        d = Path(args.local_out); d.mkdir(parents=True, exist_ok=True)
        out.write_parquet(d / 'paper_precision.parquet')
        print(f'wrote {d}/paper_precision.parquet', file=sys.stderr)
    else:
        key = f'{OUT_PREFIX}/precision/{version}/paper_precision.parquet'
        buf = io.BytesIO(); out.write_parquet(buf)
        client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        print(f'wrote s3://{BUCKET}/{key}', file=sys.stderr)


if __name__ == '__main__':
    main()
