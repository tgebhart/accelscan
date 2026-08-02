"""arXiv OAI metadata snapshot (JSONL) -> `arxiv_metadata.parquet`.

Supplies everything stage 1 cannot derive from an id: `primary_category` and the
`field` label for the field-level analyses, plus title and abstract for the
embedding/topic pipeline. Year is *not* taken from here -- it comes free from the
arXiv id (`arxiv_source.year_month_from_id`) -- but the snapshot's v1 submission
date is carried so the two can be cross-checked.

Source: the Kaggle dataset `Cornell-University/arxiv`
(`arxiv-metadata-oai-snapshot.json`, ~5.4 GB, CC0, refreshed weekly). It is
metadata only: its "full text PDFs" refers to a separate, stale GCS mirror with no
LaTeX source, which is why the pipeline reads the requester-pays `src/` tars.

    kaggle datasets download -d Cornell-University/arxiv -p /tmp && unzip ...
    python -m accelscan.scripts.build_arxiv_metadata \\
        --input /tmp/arxiv-metadata-oai-snapshot.json --upload

Streams the file in batches (a 2.7M-row frame with abstracts is several GB, so the
batches are concatenated once at the end rather than held per-record as Python
dicts).
"""

import argparse
import io
import sys
from email.utils import parsedate_to_datetime
from pathlib import Path

import orjson
import polars as pl

from accelscan.arxiv_meta import field_of, load_category_map, primary_category_of
from accelscan.arxiv_source import paper_id_from_arxiv_id, year_month_from_id
from accelscan.config import BUCKET
from accelscan.paths import arxiv_metadata_key

BATCH = 200_000

SCHEMA = {
    'paper_id': pl.Utf8, 'arxiv_id': pl.Utf8, 'primary_category': pl.Utf8,
    'categories': pl.List(pl.Utf8), 'field': pl.Utf8, 'title': pl.Utf8,
    'abstract': pl.Utf8, 'submitted': pl.Date, 'year': pl.Int32, 'month': pl.Int32,
    'snapshot_year': pl.Int32, 'n_versions': pl.Int32, 'license': pl.Utf8,
    'doi': pl.Utf8, 'journal_ref': pl.Utf8,
}


def _v1_date(versions) -> object:
    """Date of version 1, from an RFC-2822 string ('Mon, 2 Apr 2007 19:18:42 GMT')."""
    if not versions:
        return None
    created = (versions[0] or {}).get('created')
    if not created:
        return None
    try:
        return parsedate_to_datetime(created).date()
    except Exception:
        return None


def parse_record(rec: dict, mapping: dict[str, str]) -> dict | None:
    arxiv_id = (rec.get('id') or '').strip()
    if not arxiv_id:
        return None
    cats = (rec.get('categories') or '').split()
    primary = primary_category_of(cats)
    year, month = year_month_from_id(arxiv_id)
    submitted = _v1_date(rec.get('versions'))
    return {
        'paper_id': paper_id_from_arxiv_id(arxiv_id),
        'arxiv_id': arxiv_id,
        'primary_category': primary,
        'categories': cats,
        'field': field_of(primary, mapping) if primary else None,
        'title': ' '.join((rec.get('title') or '').split()) or None,
        'abstract': ' '.join((rec.get('abstract') or '').split()) or None,
        'submitted': submitted,
        'year': year,
        'month': month,
        'snapshot_year': submitted.year if submitted else None,
        'n_versions': len(rec.get('versions') or []),
        'license': rec.get('license'),
        'doi': rec.get('doi'),
        'journal_ref': rec.get('journal-ref'),
    }


def build(path: str, limit: int | None = None) -> pl.DataFrame:
    mapping = load_category_map()
    frames, rows, n, bad, unmapped = [], [], 0, 0, {}
    with open(path, 'rb') as f:
        for line in f:
            try:
                rec = orjson.loads(line)
            except Exception:
                bad += 1
                continue
            parsed = parse_record(rec, mapping)
            if parsed is None:
                bad += 1
                continue
            if parsed['primary_category'] and parsed['field'] is None:
                # a gap in registry/arxiv_categories.yaml: count, do not paper over
                arch = parsed['primary_category'].split('.')[0]
                unmapped[arch] = unmapped.get(arch, 0) + 1
            rows.append(parsed)
            n += 1
            if len(rows) >= BATCH:
                frames.append(pl.DataFrame(rows, schema=SCHEMA, orient='row'))
                rows = []
                print(f'  {n:,} records', file=sys.stderr)
            if limit and n >= limit:
                break
    if rows:
        frames.append(pl.DataFrame(rows, schema=SCHEMA, orient='row'))
    df = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)

    print(f'{df.height:,} records ({bad:,} unparseable)', file=sys.stderr)
    if unmapped:
        print(f'UNMAPPED ARCHIVES (add to registry/arxiv_categories.yaml): '
              f'{dict(sorted(unmapped.items(), key=lambda kv: -kv[1]))}', file=sys.stderr)
    return df


def report(df: pl.DataFrame) -> None:
    """Coverage and the id-vs-snapshot year agreement check."""
    print(f'  field coverage      : {100 * df["field"].is_not_null().mean():.2f}%',
          file=sys.stderr)
    print(f'  abstract coverage   : {100 * df["abstract"].is_not_null().mean():.2f}%',
          file=sys.stderr)
    both = df.filter(pl.col('year').is_not_null() & pl.col('snapshot_year').is_not_null())
    if both.height:
        agree = (both['year'] == both['snapshot_year']).mean()
        print(f'  id year == v1 year  : {100 * agree:.2f}%  '
              f'(<99% means year_month_from_id is wrong somewhere)', file=sys.stderr)
    with pl.Config(tbl_rows=12):
        print(df.group_by('field').len().sort('len', descending=True), file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='arxiv-metadata-oai-snapshot.json')
    ap.add_argument('--limit', type=int, help='dev: first N records')
    ap.add_argument('--local-out', help='write here instead of S3')
    ap.add_argument('--upload', action='store_true', help='write to MSI S3')
    args = ap.parse_args()

    df = build(args.input, args.limit)
    report(df)

    if args.local_out:
        out = Path(args.local_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        print(f'wrote {out}', file=sys.stderr)
    if args.upload:
        from accelscan.s3 import make_s3_client
        buf = io.BytesIO()
        df.write_parquet(buf)
        key = arxiv_metadata_key()
        make_s3_client().put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        print(f'wrote s3://{BUCKET}/{key} ({df.height:,} rows)', file=sys.stderr)
    if not args.local_out and not args.upload:
        print('nothing written: pass --local-out and/or --upload', file=sys.stderr)


if __name__ == '__main__':
    main()
