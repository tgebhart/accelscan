"""Compare pilot candidates under a new registry against the old run, same shards.

Answers the only question that decides whether a full re-scan is worth its cost:
how many passages and papers the new vocabulary adds, and how many it loses to
descoping. Join key is `corpusid`, not `paper_id`: candidate parquet written
before the corpus refactor has no `paper_id` column.

  python scripts/rescan_comp.py [OLD_VERSION] [NEW_VERSION]
"""

import sys

import polars as pl

from accelscan.paths import S2ORC, candidates_parts, s3_uri
from accelscan.s3 import list_keys, make_s3_client, storage_options

OLD, NEW = (sys.argv[1:3] + ['0.1.0', '0.3.0'])[:2]
KEY = 'corpusid'

so, cl = storage_options(), make_s3_client()
new_keys = list_keys(f'{candidates_parts(S2ORC, NEW)}/', suffix='.parquet', client=cl)
if not new_keys:
    raise SystemExit(f'no candidates under {NEW}: run the pilot scan first')
shards = [k.rsplit('/', 1)[-1] for k in new_keys]
old_keys = [f'{candidates_parts(S2ORC, OLD)}/{s}' for s in shards]

cols = [KEY, 'matched_models', 'model_specific']
n = pl.read_parquet([s3_uri(k) for k in new_keys], storage_options=so, columns=cols)
o = pl.read_parquet([s3_uri(k) for k in old_keys], storage_options=so, columns=cols)

print(f'{len(shards)} shards')
print(f'passages {o.height:,} -> {n.height:,} ({n.height - o.height:+,}, '
      f'{100 * (n.height / o.height - 1):+.1f}%)')
op, np_ = set(o[KEY].to_list()), set(n[KEY].to_list())
print(f'papers   {len(op):,} -> {len(np_):,} ({len(np_) - len(op):+,})')
print(f'  gained {len(np_ - op):,} papers, lost {len(op - np_):,}')
print(f'model-specific passages {o["model_specific"].sum():,} -> '
      f'{n["model_specific"].sum():,}')


def top(df: pl.DataFrame, label: str) -> None:
    if not df.height:
        return
    print(f'\n{label}:')
    with pl.Config(tbl_rows=15):
        print(df.explode('matched_models').group_by('matched_models')
              .agg(passages=pl.len(), papers=pl.col(KEY).n_unique())
              .sort('papers', descending=True).head(15))


# which vocabulary drives the change -- the concrete argument for or against
# paying for a full re-scan
top(n.filter(pl.col(KEY).is_in(list(np_ - op))), 'models triggering newly-found papers')
top(o.filter(pl.col(KEY).is_in(list(op - np_))), 'models in papers that dropped out')
