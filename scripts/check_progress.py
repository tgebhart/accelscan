import datetime as dt
from collections import Counter
from accelscan.paths import ARXIV, candidates_parts
from accelscan.registry import load_registry
from accelscan.s3 import make_s3_client
from accelscan.config import BUCKET

c = make_s3_client(); rv = load_registry().version
prefix = f'{candidates_parts(ARXIV, rv)}/'
objs = []
for page in c.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix):
    objs += page.get('Contents', [])
done = sorted((o['LastModified'], o['Key'].rsplit('/', 1)[-1].removesuffix('.done'))
              for o in objs if o['Key'].endswith('.done'))
parq = [o for o in objs if o['Key'].endswith('.parquet')]
print(f'{len(done)} tars done | {len(parq)} parquet objects written')
if len(done) >= 2:
    t0, t1 = done[0][0], done[-1][0]
    span = (t1 - t0).total_seconds()
    now = dt.datetime.now(dt.timezone.utc)
    print(f'first {t0:%H:%M:%S}Z  latest {t1:%H:%M:%S}Z  span {span/60:.1f} min')
    print(f'age of latest marker: {(now - t1).total_seconds()/60:.1f} min')
    if span > 0:
        rate = (len(done) - 1) / span
        print(f'\nmeasured rate: {rate*60:.1f} tars/min ({rate*3600:.0f} tars/hour)')
        for total in (5800, 6500):
            rem = (total - len(done)) / rate / 3600
            print(f'  if the manifest holds ~{total} tars: {rem:.1f} h remaining')
    print('\nby YYMM slice (which parallel branches are progressing):')
    for k, n in sorted(Counter(d[1].split("_")[2] for d in done).items()):
        print(f'  {k}: {n}', end='   ')
    print()