"""MSI S3 helpers.

`request_checksum_calculation='when_required'` is required against the MSI
endpoint — the boto3 default fails with MissingContentLength on multipart
uploads.
"""

import boto3
from botocore.config import Config as BotoConfig

from accelscan.config import BUCKET, s3_config


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


def storage_options() -> dict:
    """polars scan_parquet/read_parquet storage_options for the MSI bucket."""
    cfg = s3_config()
    return {
        'aws_access_key_id': cfg['aws_access_key_id'],
        'aws_secret_access_key': cfg['aws_secret_access_key'],
        'endpoint_url': cfg['endpoint_url'],
        'aws_region': 'us-east-1',
    }
