"""Stage 1.5: repack per-S2ORC-shard candidate passages into fixed-size
passage shards (~25k rows) so stage-2 GPU array tasks have uniform runtimes.

Reads  {corpus}/candidates/{registry_version}/parts/*.parquet
Writes {corpus}/passages/{registry_version}/shard_{i:04d}.parquet + manifest.parquet

Runs in minutes on a login node:
  python -m accelscan.repack                    # s2orc
  python -m accelscan.repack --corpus arxiv
  python -m accelscan.repack --corpus arxiv --append --registry-version 0.4.0

`--append` packs only candidates whose `passage_id` is not already in an existing
shard, as new shards numbered after the last one, and extends the manifest. A full
repack re-sorts everything and moves every shard boundary, so all of stage 2 would
have to re-run; appending leaves existing shards (and their `.done` mentions)
untouched, so only the new shard indices need inference.
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
    ap.add_argument('--registry-version',
                    help='candidates/passages version (default: installed registry)')
    ap.add_argument('--append', action='store_true',
                    help='add only new passages, as shards after the existing ones')
    args = ap.parse_args()

    c = get_corpus(args.corpus)
    rv = args.registry_version or load_registry().version
    so = storage_options()
    df = (pl.scan_parquet(candidates_glob(c, rv), storage_options=so)
          .sort(c.key, 'para_idx')
          .collect())
    print(f'{df.height} passages from {c.name} candidates/{rv}', file=sys.stderr)

    start, manifest_rows = 0, []
    if args.append:
        old = pl.read_parquet(f's3://{BUCKET}/{passages_manifest(c, rv)}',
                              storage_options=so)
        seen = (pl.scan_parquet([f's3://{BUCKET}/{k}' for k in old['key']],
                                storage_options=so)
                .select('passage_id').collect())
        df = df.join(seen, on='passage_id', how='anti')
        start, manifest_rows = int(old['shard'].max()) + 1, old.to_dicts()
        print(f'append: {seen.height} passages already in {old.height} shards; '
              f'{df.height} new, starting at shard {start}', file=sys.stderr)
        if not df.height:
            return

    client = make_s3_client()
    n_shards = (df.height + args.shard_size - 1) // args.shard_size
    for j in range(n_shards):
        i = start + j
        shard = df.slice(j * args.shard_size, args.shard_size)
        buf = io.BytesIO()
        shard.write_parquet(buf)
        key = passages_key(c, rv, i)
        client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        manifest_rows.append({'shard': i, 'key': key, 'n_passages': shard.height,
                              'n_papers': shard.n_unique(c.key)})
    manifest = pl.DataFrame(manifest_rows)
    buf = io.BytesIO()
    manifest.write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=passages_manifest(c, rv),
                      Body=buf.getvalue())
    print(f'{n_shards} passage shards (from {start}) -> s3://{BUCKET}/'
          f'{passages_prefix(c, rv)}/', file=sys.stderr)


if __name__ == '__main__':
    main()
