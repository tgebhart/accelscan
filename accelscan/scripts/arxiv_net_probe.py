"""Measure the S3 read bandwidth actually available to this box. Reads, discards.

The full-history scan is network-bound, so its wall time is decided by one number
this prints and nothing else. Run it before resizing an instance or tuning
`--max-workers`: if one stream already saturates the box, more workers only slice
the same pipe thinner and make each tar (and each lost tar) slower.

Reads bytes and throws them away -- no parsing, no S3 writes, no `.done` markers --
so it isolates the network from `latex.py` and from MSI. In us-east-1 the bytes are
free; elsewhere they are egress, hence the small default budget and the region note.

  python -m accelscan.scripts.arxiv_net_probe                 # 1, 4, 16 streams
  python -m accelscan.scripts.arxiv_net_probe --mb 512 --streams 1,8,32
"""

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from accelscan.arxiv_bulk import (ARXIV_BUCKET, arxiv_client, in_arxiv_region,
                                  load_manifest)

IMDS = 'http://169.254.169.254/latest/meta-data'
CHUNK = 8 << 20


def imds(path: str, timeout: float = 1.0) -> str:
    """One IMDS field, or '?' off EC2. IMDSv2 needs a token; v1 is a fallback."""
    try:
        tok = urllib.request.urlopen(urllib.request.Request(
            'http://169.254.169.254/latest/api/token', method='PUT',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'}),
            timeout=timeout).read()
        req = urllib.request.Request(f'{IMDS}/{path}',
                                     headers={'X-aws-ec2-metadata-token': tok})
        return urllib.request.urlopen(req, timeout=timeout).read().decode()
    except Exception:
        try:
            return urllib.request.urlopen(f'{IMDS}/{path}', timeout=timeout).read().decode()
        except Exception:
            return '?'


def read_range(client, key: str, nbytes: int) -> int:
    """GET the first `nbytes` of a key, discarding them. -> bytes actually read."""
    body = client.get_object(Bucket=ARXIV_BUCKET, Key=key, RequestPayer='requester',
                             Range=f'bytes=0-{nbytes - 1}')['Body']
    got = 0
    while True:
        buf = body.read(CHUNK)
        if not buf:
            return got
        got += len(buf)


def probe(keys: list[str], streams: int, mb: int) -> tuple[float, float]:
    """-> (aggregate MB/s, per-stream MB/s) for `streams` concurrent range GETs."""
    per = mb * (1 << 20)
    client = arxiv_client()                  # botocore clients are thread-safe
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=streams) as ex:
        got = sum(ex.map(lambda k: read_range(client, k, per), keys[:streams]))
    dt = max(time.time() - t0, 1e-6)
    mbps = got / (1 << 20) / dt
    return mbps, mbps / streams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mb', type=int, default=256, help='bytes per stream, MiB')
    ap.add_argument('--streams', default='1,4,16', help='comma-separated stream counts')
    ap.add_argument('--manifest-cache', default='output/arxiv_src_manifest.xml')
    args = ap.parse_args()

    counts = [int(s) for s in args.streams.split(',')]
    entries = load_manifest(cache=args.manifest_cache)
    # biggest tars first: a 2 MB 1991 tar measures startup latency, not bandwidth
    keys = [e.filename for e in sorted(entries, key=lambda e: -e.size)][:max(counts)]

    print(f'instance {imds("instance-type")}  az {imds("placement/availability-zone")}  '
          f'in us-east-1: {in_arxiv_region()}', file=sys.stderr)
    if not in_arxiv_region():
        print(f'NOTE: outside us-east-1 this probe is billable egress '
              f'(~{sum(counts) * args.mb / 1024:.1f} GB)', file=sys.stderr)
    print(f'\n{"streams":>8} {"aggregate":>12} {"per stream":>12}', file=sys.stderr)
    for n in counts:
        agg, each = probe(keys, n, args.mb)
        print(f'{n:>8} {agg:>9.0f} MB/s {each:>9.1f} MB/s', file=sys.stderr)

    total_tb = sum(e.size for e in entries) / 1e12
    print(f'\nfull history is {total_tb:.2f} TB over {len(entries)} tars; at the best '
          f'rate above that is {total_tb * 1e6 / (agg * 3600):.1f} h of pure transfer',
          file=sys.stderr)
    print('aggregate flat as streams rise => instance network cap: resize, do not '
          'add workers.', file=sys.stderr)


if __name__ == '__main__':
    main()
