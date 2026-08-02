"""arXiv id <-> corpusid crosswalk, from `openaccessinfo.externalids.ArXiv`.

The id is *only* in the S2ORC v2 shards -- the processed `papers` parquet carries
`doi` and `magid` but no arXiv id, and there is no sibling externalids table (both
checked) -- so building this needs one decompressing pass over all 214 shards. Same
I/O as stage 1, far less CPU, and free because it is MSI-local.

The original plan dropped this crosswalk. It is worth having for three things, in
descending order of value to the paper:

1. **Same-paper LaTeX-vs-GROBID validation.** The arXiv corpus matches raw LaTeX
   source, which knowingly includes captions, comments and spec tables that GROBID
   drops. On papers present in *both* corpora that bias can be measured instead of
   described -- and it is the direct test of the text-fidelity premise for adding
   arXiv at all.
2. **Overlap quantification** by year and field, which is the obvious reviewer
   question about two corpora reported side by side.
3. **Citations for arXiv papers**, via corpusid into the lab's outcomes table. Real
   but selective: only papers S2ORC indexed have one, so it is a subset analysis and
   must not be presented as an arXiv-corpus-level result.

`arxiv_id` is stored verbatim as S2 supplies it (`'2503.22397'`), plus `paper_id` in
our namespaced form so it joins the arXiv tables directly. **Whether S2 stores
pre-2007 ids as `hep-th/9901001` is not documented anywhere we can rely on, so the
report breaks the harvest down by id shape** -- roughly a third of arXiv predates the
modern scheme, and if those do not join we need to know rather than discover it in a
figure.

  python -m accelscan.scripts.build_arxiv_crosswalk --max-workers 24
  python -m accelscan.scripts.build_arxiv_crosswalk --shards 4      # smoke test
  python -m accelscan.scripts.build_arxiv_crosswalk --compact       # parts -> one table
"""

import argparse
import gzip
import io
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from accelscan.config import BUCKET, S2ORC_PREFIX
from accelscan.paths import OUT_PREFIX

CROSSWALK_PREFIX = f'{OUT_PREFIX}/arxiv/crosswalk'
PARTS = f'{CROSSWALK_PREFIX}/parts'
TABLE = f'{CROSSWALK_PREFIX}/arxiv_corpusid.parquet'

SCHEMA = {'paper_id': pl.Utf8, 'arxiv_id': pl.Utf8, 'corpusid': pl.Int64,
          'shard_id': pl.Utf8, 'id_style': pl.Utf8}

# '2503.22397' / '2503.22397v2' post-2007; 'hep-th/9901001' pre-2007.
_MODERN = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')
# old-style subject classes are lowercase and hyphenated (cond-mat.str-el/0512345),
# not two uppercase letters as in the modern scheme
_OLD = re.compile(r'^[a-z-]+(\.[A-Za-z-]{2,})?/\d{7}(v\d+)?$')


def classify(arxiv_id: str) -> str:
    if _MODERN.match(arxiv_id):
        return 'modern'
    if _OLD.match(arxiv_id):
        return 'old'
    return 'unrecognised'


def normalise(arxiv_id: str) -> str:
    """Strip a version suffix: our arXiv ids are version-less (`arxiv:2301.01234`)."""
    return re.sub(r'v\d+$', '', arxiv_id.strip())


def extract(lines) -> tuple[list[dict], dict]:
    """One shard's records -> (crosswalk rows, counts). Only two fields are read."""
    rows, stats = [], {'records': 0, 'with_arxiv': 0}
    for line in lines:
        try:
            rec = orjson.loads(line)
        except Exception:
            continue
        stats['records'] += 1
        oa = rec.get('openaccessinfo') or {}
        ext = oa.get('externalids') or {}
        raw = ext.get('ArXiv')
        cid = rec.get('corpusid')
        if not raw or cid is None:
            continue
        aid = normalise(str(raw))
        if not aid:
            continue
        stats['with_arxiv'] += 1
        rows.append({'paper_id': f'arxiv:{aid}', 'arxiv_id': aid, 'corpusid': int(cid),
                     'shard_id': '', 'id_style': classify(aid)})
    return rows, stats


def process_shard(key: str) -> str:
    """Worker: stream one shard, upload its crosswalk part + `.done` marker."""
    from accelscan.s3 import make_s3_client
    shard_id = Path(key).name.removesuffix('.gz')
    client = make_s3_client()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True)
    def fetch() -> bytes:
        return client.get_object(Bucket=BUCKET, Key=key)['Body'].read()

    t0 = time.time()
    with gzip.open(io.BytesIO(fetch())) as f:
        rows, stats = extract(f)
    for r in rows:
        r['shard_id'] = shard_id
    df = pl.DataFrame(rows, schema=SCHEMA, orient='row') if rows \
        else pl.DataFrame(schema=SCHEMA)
    buf = io.BytesIO()
    df.write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=f'{PARTS}/{shard_id}.parquet',
                      Body=buf.getvalue())
    client.put_object(Bucket=BUCKET, Key=f'{PARTS}/{shard_id}.done', Body=b'')
    return (f'{shard_id}: {stats["records"]:,} records, {stats["with_arxiv"]:,} '
            f'with arXiv id ({100 * stats["with_arxiv"] / max(stats["records"], 1):.1f}%), '
            f'{time.time() - t0:.0f}s')


def compact(client) -> pl.DataFrame:
    """Concatenate the parts into one table and report id-shape coverage."""
    from accelscan.s3 import list_keys
    keys = [k for k in list_keys(f'{PARTS}/', suffix='.parquet', client=client)]
    frames = []
    for i, k in enumerate(keys, 1):
        b = client.get_object(Bucket=BUCKET, Key=k)['Body'].read()
        frames.append(pl.read_parquet(io.BytesIO(b)))
        if i % 50 == 0:
            print(f'  read {i}/{len(keys)} parts', file=sys.stderr)
    df = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)

    print(f'\n{df.height:,} S2ORC papers carry an arXiv id', file=sys.stderr)
    print(df.group_by('id_style').len().sort('len', descending=True), file=sys.stderr)
    dupes = df.height - df['paper_id'].n_unique()
    print(f'duplicate paper_id (>1 corpusid per arXiv id): {dupes:,}', file=sys.stderr)
    if dupes:
        print('  keeping the smallest corpusid per arXiv id (earliest S2 record)',
              file=sys.stderr)
        df = df.sort('corpusid').unique(subset='paper_id', keep='first')
    if (df['id_style'] == 'unrecognised').any():
        print(df.filter(pl.col('id_style') == 'unrecognised')
                .select('arxiv_id').head(10), file=sys.stderr)

    buf = io.BytesIO()
    df.write_parquet(buf)
    client.put_object(Bucket=BUCKET, Key=TABLE, Body=buf.getvalue())
    print(f'wrote s3://{BUCKET}/{TABLE} ({df.height:,} rows)', file=sys.stderr)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, help='smoke test: first N shards only')
    ap.add_argument('--max-workers', type=int, default=24)
    ap.add_argument('--compact', action='store_true',
                    help='skip extraction; just concatenate existing parts')
    args = ap.parse_args()

    from accelscan.s3 import list_keys, make_s3_client
    client = make_s3_client()

    if not args.compact:
        keys = list(list_keys(S2ORC_PREFIX, suffix='.gz', client=client))
        if args.shards:
            keys = keys[:args.shards]
        done = {k.rsplit('/', 1)[-1].removesuffix('.done')
                for k in list_keys(f'{PARTS}/', suffix='.done', client=client)}
        todo = [k for k in keys if Path(k).name.removesuffix('.gz') not in done]
        print(f'{len(keys)} shards, {len(keys) - len(todo)} done, {len(todo)} to read',
              file=sys.stderr)
        ok = failed = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(process_shard, k): k for k in todo}
            for fut in as_completed(futures):
                try:
                    print(f'[{ok + failed + 1}/{len(todo)}] {fut.result()}',
                          file=sys.stderr)
                    ok += 1
                except Exception as exc:
                    failed += 1
                    print(f'FAILED {futures[fut]}: {exc!r}', file=sys.stderr)
        print(f'{ok} shards read, {failed} failed', file=sys.stderr)
        if failed:
            raise SystemExit(1)

    compact(client)


if __name__ == '__main__':
    main()
