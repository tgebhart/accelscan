"""Streaming reader for the arXiv bulk source tars on AWS S3.

`s3://arxiv/src/arXiv_src_YYMM_NNN.tar` (~500 MB each, ~2.9 TB total) is
**requester-pays**, so every call carries `RequestPayer='requester'` and the bill
lands on whoever holds the credentials.

**Cost discipline is a feature of this module, not an afterthought.** Reading these
bytes from EC2 *in us-east-1* is free; reading them from MSI (on-premise: verified
`s3.msi.umn.edu` -> 128.101.135.x, a UMN Ceph endpoint) is AWS internet egress at
~$0.09/GB, i.e. ~$260 per full pass. Hence `--dry-run`, `--max-bytes`, the
`manifest_bytes_total()` report before any bulk GET, and `warn_if_not_in_region()`.

**Never list the bucket.** Requester-pays listing needs the payer header that
`accelscan.s3.list_keys` does not send, and it would 403. The manifest XML is the
index -- it also carries per-tar md5 and item counts, which the listing would not.

**Streaming implies no seeking.** `tarfile` mode `'r|'` reads forward only: a member
the stream has already passed cannot be re-read (`StreamError`), so each one must be
consumed in order. A socket failure mid-tar loses the whole tar (there is no
resumption), which is an accepted ~$5 at a 2% failure rate -- far cheaper than the
complexity of ranged restarts.

This module never writes: `arxiv_scan` owns output and `.done` markers.
"""

import io
import os
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

ARXIV_BUCKET = 'arxiv'
ARXIV_REGION = 'us-east-1'
SRC_MANIFEST_KEY = 'src/arXiv_src_manifest.xml'
REQUEST_PAYER = 'requester'
STREAM_BUFFER = 8 << 20              # tarfile reads 512-byte headers; buffer them
EGRESS_USD_PER_GB = 0.09             # us-east-1 -> internet, first-10TB tier


def yymm_to_year_month(yymm: str) -> tuple[int, int]:
    """'9108' -> (1991, 8); '2301' -> (2023, 1). arXiv began 1991-08."""
    yy, mm = int(yymm[:2]), int(yymm[2:4])
    return (1900 + yy if yy >= 91 else 2000 + yy), mm


@dataclass(frozen=True)
class TarEntry:
    """One row of the source manifest."""

    filename: str            # 'src/arXiv_src_2301_001.tar'
    seq_num: int
    yymm: str
    num_items: int
    first_item: str
    last_item: str
    size: int
    md5sum: str
    timestamp: str

    @property
    def shard_id(self) -> str:
        """Output shard id / `.done` marker name: 'arXiv_src_2301_001'."""
        return Path(self.filename).name.removesuffix('.tar')

    @property
    def year(self) -> int:
        return yymm_to_year_month(self.yymm)[0]

    @property
    def chrono(self) -> tuple[int, int, int]:
        """Chronological sort key. NOT the raw yymm string: lexicographically
        '9108' (1991) sorts after '2301' (2023), which silently reversed history
        and made a 1991-2025 range select nothing."""
        y, m = yymm_to_year_month(self.yymm)
        return y, m, self.seq_num


def arxiv_client(profile: str | None = None):
    """Real-AWS, requester-pays S3 client for the `arxiv` bucket.

    NOT `accelscan.s3.make_s3_client`: that one is pinned to the MSI endpoint with
    MSI keys and cannot reach AWS. Credentials come from `ARXIV_AWS_*` if set, else
    the ambient AWS chain (instance role on EC2).
    """
    import boto3
    from botocore.config import Config as BotoConfig

    cfg = BotoConfig(region_name=ARXIV_REGION, read_timeout=900, connect_timeout=30,
                     retries={'max_attempts': 10, 'mode': 'adaptive'},
                     max_pool_connections=32)
    key = os.environ.get('ARXIV_AWS_ACCESS_KEY_ID')
    secret = os.environ.get('ARXIV_AWS_SECRET_ACCESS_KEY')
    if key and secret:
        return boto3.client('s3', aws_access_key_id=key,
                            aws_secret_access_key=secret, config=cfg)
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client('s3', config=cfg)


def in_arxiv_region() -> bool:
    """True when running inside us-east-1, where reading these tars is free."""
    for var in ('AWS_REGION', 'AWS_DEFAULT_REGION'):
        if os.environ.get(var):
            return os.environ[var] == ARXIV_REGION
    try:                                   # IMDSv2, EC2 only, fails fast elsewhere
        import urllib.request
        req = urllib.request.Request(
            'http://169.254.169.254/latest/api/token', method='PUT',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})
        token = urllib.request.urlopen(req, timeout=1).read().decode()
        req = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/placement/region',
            headers={'X-aws-ec2-metadata-token': token})
        return urllib.request.urlopen(req, timeout=1).read().decode() == ARXIV_REGION
    except Exception:
        return False


def warn_if_not_in_region(n_bytes: int) -> None:
    """Print the egress bill when reading from outside us-east-1."""
    if in_arxiv_region():
        print(f'[arxiv] in {ARXIV_REGION}: {n_bytes / 1e12:.2f} TB reads at $0 egress',
              file=sys.stderr)
        return
    print(f'[arxiv] WARNING: not in {ARXIV_REGION}. {n_bytes / 1e12:.2f} TB of '
          f'requester-pays egress ≈ ${n_bytes / 1e9 * EGRESS_USD_PER_GB:,.0f}. '
          f'Run stage 1 on EC2 in {ARXIV_REGION} to pay $0, or pass --yes-i-know.',
          file=sys.stderr)


def parse_manifest(xml_bytes: bytes) -> list[TarEntry]:
    """arXiv_src_manifest.xml -> TarEntry list in chronological order.

    Sorted on `TarEntry.chrono`, not the raw `yymm` string: '9108' (1991) is
    lexicographically greater than '2301' (2023).
    """
    root = ET.fromstring(xml_bytes)
    out = []
    for f in root.iter('file'):
        get = lambda tag: (f.findtext(tag) or '').strip()  # noqa: E731
        filename = get('filename')
        if not filename:
            continue
        try:
            out.append(TarEntry(
                filename=filename, seq_num=int(get('seq_num') or 0),
                yymm=get('yymm'), num_items=int(get('num_items') or 0),
                first_item=get('first_item'), last_item=get('last_item'),
                size=int(get('size') or 0), md5sum=get('md5sum'),
                timestamp=get('timestamp')))
        except ValueError:
            continue
    return sorted(out, key=lambda e: e.chrono)


def load_manifest(client=None, cache: str | Path | None = None) -> list[TarEntry]:
    """Fetch (or reuse a cached copy of) the source manifest."""
    path = Path(cache) if cache else None
    if path and path.exists():
        return parse_manifest(path.read_bytes())
    client = client or arxiv_client()
    body = client.get_object(Bucket=ARXIV_BUCKET, Key=SRC_MANIFEST_KEY,
                            RequestPayer=REQUEST_PAYER)['Body'].read()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return parse_manifest(body)


def select_tars(entries: list[TarEntry], yymm_range: tuple[str, str] | None = None,
                limit: int | None = None) -> list[TarEntry]:
    """Filter by inclusive YYMM range (e.g. ('9108', '2512')) and cap the count."""
    out = entries
    if yymm_range:
        lo = yymm_to_year_month(yymm_range[0])
        hi = yymm_to_year_month(yymm_range[1])
        out = [e for e in out if lo <= yymm_to_year_month(e.yymm) <= hi]
    return out[:limit] if limit else out


def manifest_bytes_total(entries: list[TarEntry]) -> int:
    return sum(e.size for e in entries)


class _StreamAdapter(io.RawIOBase):
    """Non-seekable file object over a boto3 StreamingBody.

    `tarfile` in stream mode issues many small reads for 512-byte headers, so this
    is wrapped in a BufferedReader by the caller -- unbuffered small reads against
    an HTTP response cost several times the throughput.
    """

    def __init__(self, body):
        self._body = body

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, b) -> int:
        chunk = self._body.read(len(b))
        if not chunk:
            return 0
        b[:len(chunk)] = chunk
        return len(chunk)

    def close(self) -> None:
        try:
            self._body.close()
        finally:
            super().close()


def open_stream(fileobj) -> tarfile.TarFile:
    """Buffered, forward-only TarFile over a non-seekable stream."""
    return tarfile.open(fileobj=io.BufferedReader(fileobj, STREAM_BUFFER), mode='r|')


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=120), reraise=True)
def open_tar(client, entry: TarEntry) -> tarfile.TarFile:
    """Open one manifest entry as a forward-only tar stream.

    Retry is whole-tar by necessity: a broken stream cannot be resumed mid-file.
    """
    body = client.get_object(Bucket=ARXIV_BUCKET, Key=entry.filename,
                             RequestPayer=REQUEST_PAYER)['Body']
    return open_stream(_StreamAdapter(body))


def iter_members(tf: tarfile.TarFile) -> Iterator[tuple[str, bytes]]:
    """(name, bytes) per regular file, in tar order.

    Each member is read fully before advancing: `mode='r|'` permits a forward
    `getmembers()` scan (it just reads to EOF) but raises `StreamError` on any
    attempt to extract a member the stream has already passed.
    """
    for member in tf:
        if not member.isfile():
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        yield member.name, f.read()
