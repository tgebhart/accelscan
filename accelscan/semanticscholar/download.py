"""Semantic Scholar bulk-dataset download (papers, abstracts, citations).

Ported from novelty_corp/semanticscholar/download.py. Two changes: the API key
comes from the environment rather than being hardcoded in the module, and the
release id defaults to `config.S2_RELEASE_ID` so the download and the prefixes the
analytic stages read cannot drift apart.

The bulk API hands out pre-signed part URLs that expire after roughly an hour, so
a long download has to be able to re-list a dataset mid-run; `preprocess.py` does
that and this module exposes the pieces it needs.

    python -m accelscan.semanticscholar.download            # all three datasets
    python -m accelscan.semanticscholar.download --datasets papers abstracts
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

import requests
from tenacity import (retry, retry_if_exception, stop_after_attempt,
                      wait_exponential)
from tqdm import tqdm

from accelscan.config import S2_API_BASE, S2_DATA_DIR, S2_RELEASE_ID, s2_api_key

DATASETS = ('papers', 'abstracts', 'citations')
_RETRY = dict(wait=wait_exponential(multiplier=1, min=2, max=60),
              stop=stop_after_attempt(5), reraise=True)


def _headers() -> dict:
    key = s2_api_key()
    return {'X-API-KEY': key} if key else {}


@retry(retry=retry_if_exception(requests.RequestException), **_RETRY)
def get_release_id(release_id: str = S2_RELEASE_ID) -> str:
    """Resolve 'latest' to a concrete release id; pass anything else through."""
    if release_id != 'latest':
        return release_id
    r = requests.get(os.path.join(S2_API_BASE, 'latest'), headers=_headers(),
                     timeout=60)
    r.raise_for_status()
    return r.json()['release_id']


@retry(retry=retry_if_exception(requests.RequestException), **_RETRY)
def get_dataset_parts(dataset: str, release_id: str = S2_RELEASE_ID) -> list[str]:
    """Pre-signed URLs for every part of `dataset`. They expire; re-call to refresh."""
    r = requests.get(os.path.join(S2_API_BASE, f'{release_id}/dataset/{dataset}'),
                     headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()['files']


def _is_transient(exc: BaseException) -> bool:
    """Retry network and 5xx failures only.

    An expired pre-signed URL comes back 400/403, and retrying that just burns the
    remaining attempts on a URL that will never work again -- the caller has to
    re-list the dataset instead (see preprocess.download_process_delete).
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, urllib.error.ContentTooShortError))


@retry(retry=retry_if_exception(_is_transient), **_RETRY)
def download_dataset_part(url: str, savename: str) -> str:
    urllib.request.urlretrieve(url, savename)
    return savename


def part_savename(idx: int) -> str:
    return f'part{idx}.jsonl.gz'


def download_full_dataset(dataset: str, release_id: str = S2_RELEASE_ID,
                          data_dir: str = S2_DATA_DIR) -> str:
    """Download every part of `dataset` to disk without parsing.

    `preprocess.download_process_delete` is usually what you want instead: it
    parses and deletes each part as it lands, which is the difference between
    needing ~200 GB of scratch and needing ~2 TB for the citations dataset.
    """
    saveloc = os.path.join(data_dir, release_id, 'raw', dataset)
    os.makedirs(saveloc, exist_ok=True)
    parts = get_dataset_parts(dataset, release_id=release_id)
    for idx in tqdm(range(len(parts)), desc=f'downloading {dataset}'):
        download_dataset_part(parts[idx],
                              os.path.join(saveloc, part_savename(idx)))
    return saveloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument('--release-id', default=S2_RELEASE_ID)
    ap.add_argument('--data-dir', default=S2_DATA_DIR)
    args = ap.parse_args()
    release_id = get_release_id(args.release_id)
    if not s2_api_key():
        print('warning: no S2_API_KEY in the environment; the bulk API will '
              'rate-limit a full download hard', file=sys.stderr)
    for dataset in args.datasets:
        print(f'{dataset} -> {download_full_dataset(dataset, release_id, args.data_dir)}',
              file=sys.stderr)


if __name__ == '__main__':
    main()
