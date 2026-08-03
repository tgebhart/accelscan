"""MSI S3 helpers.

`request_checksum_calculation='when_required'` is required against the MSI
endpoint — the boto3 default fails with MissingContentLength on multipart
uploads.
"""

import boto3
from botocore.config import Config as BotoConfig

from accelscan.config import BUCKET, s3_config
from accelscan.paths import S2ORC, Corpus, mentions_parts, mentions_root, s3_uri


def make_s3_client():
    return boto3.client(
        's3', **s3_config(),
        config=BotoConfig(
            request_checksum_calculation='when_required',
            read_timeout=600,
            max_pool_connections=32,
        ),
    )


def list_keys(prefix: str, *, suffix: str = '', bucket: str = BUCKET, client=None) -> list[str]:
    client = client or make_s3_client()
    keys = []
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith(suffix):
                keys.append(obj['Key'])
    return keys


def discover_mentions_version(model_tag: str, prompt_version: str,
                              client=None, corpus: Corpus = S2ORC) -> str:
    """Return the registry version under which mentions were EXTRACTED.

    Mentions are namespaced by the registry version current at extraction
    time. Later registry edits (new models, spec fields) bump the version
    without requiring re-extraction — so downstream stages must not assume
    `load_registry().version` matches the mentions path. Picks the highest
    version that actually has parts for this model/prompt.

    Corpus-scoped: the listing prefix is `{corpus}/mentions/`, and because S3
    prefix matching is literal the two corpora cannot see each other's versions
    (`accelscan/arxiv/mentions/…` is not under `accelscan/mentions/`).
    """
    prefix = mentions_root(corpus)
    keys = list_keys(prefix, suffix='.parquet', client=client)
    versions = set()
    for k in keys:
        rest = k[len(prefix):].split('/')
        if len(rest) >= 4 and rest[1] == prompt_version and rest[2] == model_tag:
            versions.add(rest[0])
    if not versions:
        raise FileNotFoundError(
            f'no mentions found under s3://{BUCKET}/{prefix} for '
            f'{corpus.name}/{model_tag}/{prompt_version}')
    return max(versions, key=lambda v: [int(x) for x in v.split('.')])


def discover_passages_version(corpus: Corpus = S2ORC, client=None) -> str:
    """Return the registry version the repacked passage shards live under.

    Stage 2 must read the version that *stage 1.5 wrote*, never
    `load_registry().version`: a registry bump between the scan and the
    inference run (specs added, models appended) leaves the passages where they
    are, so keying off the local registry sends `get_object` at a nonexistent
    `passages/{new}/shard_NNNN.parquet` and would also split the mentions table
    across two version prefixes. Errors rather than guessing when several
    versions coexist.
    """
    prefix = f'{corpus.out_prefix}/passages/'
    versions = {k[len(prefix):].split('/')[0]
                for k in list_keys(prefix, suffix='.parquet', client=client)}
    if not versions:
        raise FileNotFoundError(f'no passage shards under s3://{BUCKET}/{prefix} '
                                f'— run `repack --corpus {corpus.name}` first')
    if len(versions) > 1:
        raise RuntimeError(
            f'{corpus.name} has passages under {sorted(versions)}; pass '
            f'--registry-version to choose one')
    return versions.pop()


def mentions_glob(model_tag: str, prompt_version: str,
                  mentions_version: str | None = None, client=None,
                  corpus: Corpus = S2ORC) -> str:
    v = mentions_version or discover_mentions_version(model_tag, prompt_version,
                                                     client, corpus)
    return s3_uri(f'{mentions_parts(corpus, v, prompt_version, model_tag)}/*.parquet')


def storage_options() -> dict:
    """polars scan_parquet/read_parquet storage_options for the MSI bucket."""
    cfg = s3_config()
    return {
        'aws_access_key_id': cfg['aws_access_key_id'],
        'aws_secret_access_key': cfg['aws_secret_access_key'],
        'endpoint_url': cfg['endpoint_url'],
        'aws_region': 'us-east-1',
    }
