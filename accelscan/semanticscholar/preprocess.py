"""Semantic Scholar bulk release -> partitioned parquet, one part at a time.

Ported from novelty_corp/semanticscholar/preprocessing.py. The loop is
download -> parse -> write parquet -> delete the .jsonl.gz, so peak scratch is one
part rather than the whole dataset; `citations` alone is ~2 TB decompressed, which
is why this is not a two-pass "download everything, then parse" design.

Schemas are declared rather than inferred: polars infers from the first rows of a
part, and the S2 dumps have enough null-only columns early on that inference
disagrees between parts and the resulting parquet will not concatenate.

Two things the original did in a commented-out `__main__` block are proper
functions here: joining citations to publication years (`build_citation_edgelist`),
which is the input `outcomes.py` needs, and uploading the processed parts to S3
(`upload_parts`), which is where every accelscan stage actually reads them from.

    python -m accelscan.semanticscholar.preprocess --datasets papers abstracts authors
    python -m accelscan.semanticscholar.preprocess --datasets citations
    python -m accelscan.semanticscholar.preprocess --edgelist
    python -m accelscan.semanticscholar.preprocess --upload papers abstracts authors
"""

import argparse
import glob
import os
import sys
import urllib.error
from dataclasses import dataclass

import polars as pl
from tqdm import tqdm

from accelscan.config import (BUCKET, S2_DATA_DIR, S2_PROCESSED_PREFIX,
                              S2_RELEASE_ID)
from accelscan.semanticscholar.download import (download_dataset_part,
                                                get_dataset_parts,
                                                get_release_id, part_savename)

DATASETS = ('papers', 'abstracts', 'authors', 'citations')
URL_EXPIRED = (400, 403)


@dataclass
class Part:
    url: str
    saveloc: str


SCHEMAS: dict[str, pl.Schema] = {
    'papers': pl.Schema({
        'corpusid': pl.Int64,
        'externalids': pl.Struct({'DOI': pl.String, 'MAG': pl.String}),
        'title': pl.String,
        'referencecount': pl.Int64,
        'citationcount': pl.Int64,
        'journal': pl.Struct({'name': pl.String, 'volume': pl.String,
                              'pages': pl.String}),
        'conference': pl.Struct({'name': pl.String, 'id': pl.String}),
        'influentialcitationcount': pl.Int64,
        'publicationdate': pl.String,
        'year': pl.Int64,
        'publicationtypes': pl.List(pl.String),
        's2fieldsofstudy': pl.List(pl.Struct({'category': pl.String,
                                              'source': pl.String})),
        'authors': pl.List(pl.Struct({'authorId': pl.String, 'name': pl.String})),
    }),
    'authors': pl.Schema({
        'authorid': pl.String,
        'externalids': pl.Struct({'ORCID': pl.String, 'DBLP': pl.List(pl.String)}),
        'name': pl.String,
        'affiliations': pl.List(pl.String),
        'papercount': pl.Int64,
        'citationcount': pl.Int64,
        'hindex': pl.Int64,
    }),
    'citations': pl.Schema({
        'citationid': pl.Int64,
        'isinfluential': pl.Boolean,
        'citingcorpusid': pl.Int64,
        'citedcorpusid': pl.Int64,
    }),
    'abstracts': pl.Schema({
        'corpusid': pl.Int64,
        'abstract': pl.String,
    }),
}


def _transform_papers(df: pl.DataFrame) -> pl.DataFrame:
    """Flatten what downstream reads: date, author ids, DOI/MAG out of the struct."""
    exprs = []
    if 'publicationdate' in df.columns:
        exprs.append(pl.col('publicationdate').str.to_date(strict=False))
    if 'authors' in df.columns and df.schema['authors'] != pl.Null:
        exprs.append(pl.col('authors').list.eval(
            pl.element().struct.field('authorId')))
    if exprs:
        df = df.with_columns(exprs)
    if 'externalids' in df.columns and df.schema['externalids'] != pl.Null:
        df = df.with_columns(
            pl.col('externalids').struct.field('DOI').alias('doi'),
            pl.col('externalids').struct.field('MAG').alias('magid'),
        ).drop('externalids')
    return df


def _transform_authors(df: pl.DataFrame) -> pl.DataFrame:
    """Lift ORCID/DBLP out of the struct, matching what papers does with DOI/MAG.

    A struct column is awkward to filter or join on, and the whole point of the
    authors table is joining it to the author ids `papers` already carries.
    """
    if 'externalids' in df.columns and df.schema['externalids'] != pl.Null:
        df = df.with_columns(
            pl.col('externalids').struct.field('ORCID').alias('orcid'),
            pl.col('externalids').struct.field('DBLP').alias('dblp'),
        ).drop('externalids')
    return df


TRANSFORMS = {'papers': _transform_papers, 'authors': _transform_authors}


def raw_dir(dataset: str, release_id: str, data_dir: str) -> str:
    d = os.path.join(data_dir, release_id, 'raw', dataset)
    os.makedirs(d, exist_ok=True)
    return d


def parts_dir(dataset: str, release_id: str, data_dir: str) -> str:
    d = os.path.join(data_dir, release_id, 'processed', dataset, 'parts')
    os.makedirs(d, exist_ok=True)
    return d


def parse_and_delete(saveloc: str, out_path: str, transform=None,
                     schema: pl.Schema | None = None) -> str | None:
    try:
        df = pl.read_ndjson(saveloc, schema=schema)
        if transform is not None:
            df = transform(df)
        os.remove(saveloc)
        df.write_parquet(out_path)
        return out_path
    except Exception as e:                       # noqa: BLE001 - report and continue
        print(f'error parsing {saveloc}: {e}', file=sys.stderr)
        return None


def download_process_delete(dataset: str, release_id: str = S2_RELEASE_ID,
                            data_dir: str = S2_DATA_DIR) -> list[str]:
    """One part at a time: download, parse to parquet, delete the raw file.

    Restartable: an existing output part is skipped, so a killed run resumes. Note
    that this is the *only* completion signal -- a part that failed to parse leaves
    no output and is retried on the next run.
    """
    release_id = get_release_id(release_id)
    raw = raw_dir(dataset, release_id, data_dir)
    out_dir = parts_dir(dataset, release_id, data_dir)
    transform, schema = TRANSFORMS.get(dataset), SCHEMAS.get(dataset)

    urls = get_dataset_parts(dataset, release_id=release_id)
    queue = [Part(u, os.path.join(raw, part_savename(i)))
             for i, u in enumerate(urls)]
    print(f'{dataset}: {len(queue)} parts -> {out_dir}', file=sys.stderr)

    written = []
    for idx, part in enumerate(tqdm(queue, desc=f'{dataset}')):
        stem = os.path.basename(part.saveloc).replace('.jsonl.gz', '')
        out_path = os.path.join(out_dir, f'{stem}.parquet')
        if os.path.exists(out_path):
            written.append(out_path)
            continue
        for attempt in range(3):
            try:
                download_dataset_part(part.url, part.saveloc)
                break
            except urllib.error.HTTPError as e:
                # pre-signed URLs expire after ~1 h, and a long run will outlive
                # the listing it started from; re-list rather than retry the URL
                if e.code not in URL_EXPIRED or attempt == 2:
                    raise
                print(f'URL expired (HTTP {e.code}); re-listing {dataset} for '
                      f'part {idx}', file=sys.stderr)
                part = Part(get_dataset_parts(dataset, release_id=release_id)[idx],
                            part.saveloc)
        path = parse_and_delete(part.saveloc, out_path, transform, schema)
        if path:
            written.append(path)
    return written


def build_citation_edgelist(release_id: str = S2_RELEASE_ID,
                            data_dir: str = S2_DATA_DIR,
                            out_dir: str | None = None) -> str:
    """Citation edges with the citing and cited publication years attached.

    This is the input `outcomes.py` consumes, and the reason it exists separately:
    the raw citations dump carries only corpus ids, while every window-based
    outcome (i_k, cd_k) needs both endpoints' years.

    Written as a directory of parts, one per input part, NOT a single file. A
    single lazy join sink over all 403 parts dies with "row count (4294972130)
    exceeded maximum supported of 4294967295" -- polars' streaming engine carries a
    32-bit row counter, and 2**32-1 rows is below this edge list. (That 4.29e9 is an
    engine-internal count, not the join's output: the release has 1,416,420,846
    edges, `corpusid` is unique across the papers parts, and both joins are
    row-preserving, which this function asserts per part.)

    Peak memory is the years table (~3.6 GB for 236.8M rows) plus one part.
    Restartable: an existing output part is skipped.
    """
    release_id = get_release_id(release_id)
    proc = os.path.join(data_dir, release_id, 'processed')
    out_dir = out_dir or os.path.join(proc, 'citations_edges', 'parts')
    os.makedirs(out_dir, exist_ok=True)

    years = (pl.scan_parquet(os.path.join(proc, 'papers', 'parts', '*.parquet'))
             .select('corpusid', 'year').collect())
    if years['corpusid'].n_unique() != years.height:
        raise SystemExit(f'papers has duplicate corpusid ({years.height:,} rows, '
                         f'{years["corpusid"].n_unique():,} unique): the year joins '
                         f'below would multiply edges rather than annotate them')
    citing = years.rename({'corpusid': 'citingcorpusid', 'year': 'citingyear'})
    cited = years.rename({'corpusid': 'citedcorpusid', 'year': 'citedyear'})
    print(f'years: {years.height:,} papers', file=sys.stderr)

    parts = sorted(glob.glob(os.path.join(proc, 'citations', 'parts', '*.parquet')))
    total = 0
    for path in tqdm(parts, desc='edgelist'):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            total += pl.scan_parquet(out_path).select(pl.len()).collect().item()
            continue
        e = pl.read_parquet(path, columns=['citingcorpusid', 'citedcorpusid'])
        out = (e.join(citing, on='citingcorpusid', how='left')
               .join(cited, on='citedcorpusid', how='left'))
        if out.height != e.height:
            raise SystemExit(f'{path}: {e.height:,} edges in, {out.height:,} out -- '
                             f'the year join multiplied rows, so every outcome '
                             f'computed from it would be wrong')
        out.write_parquet(out_path)
        total += out.height
    print(f'{len(parts)} parts, {total:,} edges -> {out_dir}', file=sys.stderr)
    return out_dir


def upload_parts(dataset: str, release_id: str = S2_RELEASE_ID,
                 data_dir: str = S2_DATA_DIR) -> int:
    """Push processed parts to S3, where every accelscan stage reads them.

    `config.PAPERS_PREFIX` / `ABSTRACTS_PREFIX` point at the bucket, not at local
    disk, so preprocessing alone does not make a new release usable -- this step
    does. Existing keys are overwritten, which is what makes it re-runnable.
    """
    from accelscan.s3 import make_s3_client
    client = make_s3_client()
    local = parts_dir(dataset, get_release_id(release_id), data_dir)
    files = sorted(f for f in os.listdir(local) if f.endswith('.parquet'))
    for f in tqdm(files, desc=f'upload {dataset}'):
        key = f'{S2_PROCESSED_PREFIX}/{dataset}/parts/{f}'
        client.upload_file(os.path.join(local, f), BUCKET, key)
    print(f'{len(files)} parts -> s3://{BUCKET}/{S2_PROCESSED_PREFIX}/{dataset}/parts/',
          file=sys.stderr)
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=[], choices=list(DATASETS),
                    help='download + parse these datasets')
    ap.add_argument('--edgelist', action='store_true',
                    help='join citations to publication years (needs papers first)')
    ap.add_argument('--upload', nargs='*', default=[], choices=list(DATASETS),
                    help='upload already-processed parts to S3')
    ap.add_argument('--release-id', default=S2_RELEASE_ID)
    ap.add_argument('--data-dir', default=S2_DATA_DIR)
    args = ap.parse_args()

    if not (args.datasets or args.edgelist or args.upload):
        ap.error('nothing to do: pass --datasets, --edgelist and/or --upload')
    for dataset in args.datasets:
        n = len(download_process_delete(dataset, args.release_id, args.data_dir))
        print(f'{dataset}: {n} parts written', file=sys.stderr)
    if args.edgelist:
        build_citation_edgelist(args.release_id, args.data_dir)
    for dataset in args.upload:
        upload_parts(dataset, args.release_id, args.data_dir)


if __name__ == '__main__':
    main()
