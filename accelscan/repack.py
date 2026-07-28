"""Stage 1.5: repack per-S2ORC-shard candidate passages into fixed-size
passage shards (~25k rows) so stage-2 GPU array tasks have uniform runtimes.

Reads  s3://{BUCKET}/{OUT_PREFIX}/candidates/{registry_version}/parts/*.parquet
Writes s3://{BUCKET}/{OUT_PREFIX}/passages/{registry_version}/shard_{i:04d}.parquet
       + manifest.parquet

Runs in minutes on a login node: python -m accelscan.repack
"""

import argparse
import io
import sys

import polars as pl

from accelscan.config import BUCKET, OUT_PREFIX
from accelscan.registry import load_registry
from accelscan.s3 import list_keys, make_s3_client, storage_options

PASSAGES_PER_SHARD = 25_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-size', type=int, default=PASSAGES_PER_SHARD)
    args = ap.parse_args()

    reg = load_registry()
    src = f's3://{BUCKET}/{OUT_PREFIX}/candidates/{reg.version}/parts/*.parquet'
    dst_prefix = f'{OUT_PREFIX}/passages/{reg.version}'
    df = (pl.scan_parquet(src, storage_options=storage_options())
          .sort('corpusid', 'para_idx')
          .collect())
    print(f'{df.height} passages from candidates/{reg.version}', file=sys.stderr)

    client = make_s3_client()
    n_shards = (df.height + args.shard_size - 1) // args.shard_size
    manifest_rows = []
    for i in range(n_shards):
        shard = df.slice(i * args.shard_size, args.shard_size)
        buf = io.BytesIO()
        shard.write_parquet(buf)
        key = f'{dst_prefix}/shard_{i:04d}.parquet'
        client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        manifest_rows.append({'shard': i, 'key': key, 'n_passages': shard.height,
                              'n_papers': shard.n_unique('corpusid')})
    manifest = pl.DataFrame(manifest_rows)
    buf = io.BytesIO()
    manifest.write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=f'{dst_prefix}/manifest.parquet',
                      Body=buf.getvalue())
    print(f'{n_shards} passage shards -> s3://{BUCKET}/{dst_prefix}/', file=sys.stderr)


if __name__ == '__main__':
    main()
