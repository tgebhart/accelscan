"""Stage 1.5: repack per-S2ORC-shard candidate passages into fixed-size
passage shards (~25k rows) so stage-2 GPU array tasks have uniform runtimes.

Reads  {corpus}/candidates/{registry_version}/parts/*.parquet
Writes {corpus}/passages/{registry_version}/shard_{i:04d}.parquet + manifest.parquet

Runs in minutes on a login node:
  python -m accelscan.repack                    # s2orc
  python -m accelscan.repack --corpus arxiv
"""

import argparse
import io
import sys

import polars as pl

from accelscan.config import BUCKET
from accelscan.paths import (candidates_glob, get_corpus, passages_key,
                            passages_manifest, passages_prefix)
from accelscan.registry import load_registry
from accelscan.s3 import make_s3_client, storage_options

PASSAGES_PER_SHARD = 25_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-size', type=int, default=PASSAGES_PER_SHARD)
    ap.add_argument('--corpus', default='s2orc', choices=['s2orc', 'arxiv'])
    args = ap.parse_args()

    c = get_corpus(args.corpus)
    reg = load_registry()
    df = (pl.scan_parquet(candidates_glob(c, reg.version),
                          storage_options=storage_options())
          .sort(c.key, 'para_idx')
          .collect())
    print(f'{df.height} passages from {c.name} candidates/{reg.version}',
          file=sys.stderr)

    client = make_s3_client()
    n_shards = (df.height + args.shard_size - 1) // args.shard_size
    manifest_rows = []
    for i in range(n_shards):
        shard = df.slice(i * args.shard_size, args.shard_size)
        buf = io.BytesIO()
        shard.write_parquet(buf)
        key = passages_key(c, reg.version, i)
        client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        manifest_rows.append({'shard': i, 'key': key, 'n_passages': shard.height,
                              'n_papers': shard.n_unique(c.key)})
    manifest = pl.DataFrame(manifest_rows)
    buf = io.BytesIO()
    manifest.write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=passages_manifest(c, reg.version),
                      Body=buf.getvalue())
    print(f'{n_shards} passage shards -> s3://{BUCKET}/'
          f'{passages_prefix(c, reg.version)}/', file=sys.stderr)


if __name__ == '__main__':
    main()
