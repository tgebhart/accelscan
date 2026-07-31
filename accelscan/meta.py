"""Per-paper metadata (year, field, title, abstract), dispatched by corpus.

Every corpus-specific metadata dependency in the pipeline funnels through here, so
downstream stages take a `Corpus` and stay otherwise identical. Sources:

- **s2orc** — the Semantic Scholar snapshot parquet (`PAPERS_PREFIX`,
  `ABSTRACTS_PREFIX`), read part-by-part because the tables are tens of millions of
  rows and a whole-table join OOMs.
- **arxiv** — one `arxiv_metadata.parquet` built from the arXiv OAI snapshot. `field`
  is the archive-level arXiv grouping (`registry/arxiv_categories.yaml`), NOT
  `s2fieldsofstudy`; `citationcount` is null because arXiv distributes no citation
  data and the design deliberately does not map arXiv ids onto Semantic Scholar ids.

`read_by_part` replaces the two near-identical readers that used to live in
`denominator._by_part` and `embed._matches_by_part`.
"""

import sys

import polars as pl

from accelscan.config import ABSTRACTS_PREFIX, BUCKET, PAPERS_PREFIX
from accelscan.paths import ARXIV, Corpus, arxiv_metadata_key, s3_uri


def read_by_part(prefix: str, cols: list[str], *, so: dict, client=None,
                 ids: pl.DataFrame | None = None, on: str = 'corpusid',
                 transform=None, text_col: str | None = None,
                 n_parts: int | None = None, stop_at: int | None = None,
                 label: str = '') -> pl.DataFrame:
    """Read one parquet part at a time, optionally semi-joined to `ids`.

    Peak memory = one part + accumulated matches, never the whole table.
    `text_col` drops null/near-empty text (>20 chars, the abstract-quality bar).
    `stop_at` stops early once enough rows are collected (smoke runs).
    """
    from accelscan.s3 import list_keys
    keys = list_keys(prefix, suffix='.parquet', client=client)
    if n_parts:
        keys = keys[:n_parts]
    parts, total = [], 0
    for i, k in enumerate(keys):
        part = pl.read_parquet(s3_uri(k), storage_options=so, columns=cols)
        if ids is not None:
            part = part.join(ids, on=on, how='semi')
        if text_col is not None:
            part = part.filter(pl.col(text_col).is_not_null()
                               & (pl.col(text_col).str.len_chars() > 20))
        if transform is not None:
            part = transform(part)
        if part.height:
            parts.append(part)
            total += part.height
        if label and ((i + 1) % 60 == 0 or i + 1 == len(keys)):
            print(f'  [{label}] {i+1}/{len(keys)} parts, {total:,} rows', file=sys.stderr)
        if stop_at and total >= stop_at:
            break
    if parts:
        return pl.concat(parts)
    return pl.DataFrame(schema={c: (pl.Int64 if c == on else pl.Utf8) for c in cols})


def _arxiv_meta(cols: list[str], ids: pl.DataFrame | None, so: dict,
                metadata_uri: str | None = None) -> pl.DataFrame:
    """Columns from the single arXiv metadata parquet, semi-joined to `ids`."""
    lf = pl.scan_parquet(metadata_uri or s3_uri(arxiv_metadata_key()),
                         storage_options=so).select(cols)
    if ids is not None:
        lf = lf.join(ids.lazy().select('paper_id'), on='paper_id', how='semi')
    return lf.collect()


def paper_years(c: Corpus, ids: pl.DataFrame, so: dict, client=None,
                metadata_uri: str | None = None) -> pl.DataFrame:
    """(key, year). Drives the per-year winsorization cap in `compute.py`."""
    if c is ARXIV:
        return _arxiv_meta(['paper_id', 'year'], ids, so, metadata_uri)
    return read_by_part(PAPERS_PREFIX, ['corpusid', 'year'], so=so, client=client,
                        ids=ids, label='years')


def paper_fields(c: Corpus, ids: pl.DataFrame, so: dict, client=None,
                 metadata_uri: str | None = None) -> pl.DataFrame:
    """(key, year, field, n_fields, citationcount) for the denominator table.

    arXiv `field` is its own archive-level taxonomy and `primary_category` is
    carried alongside for the fine-grained breakdowns that have no S2 analogue;
    `citationcount` is null there by design.
    """
    if c is ARXIV:
        m = _arxiv_meta(['paper_id', 'year', 'field', 'primary_category', 'categories'],
                        ids, so, metadata_uri)
        return m.with_columns(
            n_fields=pl.col('categories').list.len().cast(pl.UInt32),
            citationcount=pl.lit(None, dtype=pl.Int64)).drop('categories')
    return read_by_part(
        PAPERS_PREFIX, ['corpusid', 'year', 's2fieldsofstudy', 'citationcount'],
        so=so, client=client, ids=ids, label='papers',
        transform=lambda df: df.with_columns(
            field=pl.col('s2fieldsofstudy').list.first().struct.field('category'),
            n_fields=pl.col('s2fieldsofstudy').list.len(),
        ).select('corpusid', 'year', 'field', 'n_fields', 'citationcount'))


def paper_abstracts(c: Corpus, ids: pl.DataFrame, so: dict, client=None,
                    metadata_uri: str | None = None) -> pl.DataFrame:
    """(key, abstract) only -- for c-TF-IDF keywords, which never use the title."""
    if c is ARXIV:
        return _arxiv_meta(['paper_id', 'abstract'], ids, so, metadata_uri)
    return read_by_part(ABSTRACTS_PREFIX, ['corpusid', 'abstract'], so=so,
                        client=client, ids=ids, label='abstracts')


def paper_titles_abstracts(c: Corpus, ids: pl.DataFrame, so: dict, client=None,
                           n_parts: int | None = None, stop_at: int | None = None,
                           metadata_uri: str | None = None) -> pl.DataFrame:
    """(key, title, abstract) for the embedding worklist.

    arXiv ships author-written abstracts for ~100% of papers, versus the partial
    coverage of the S2 abstracts table -- a quality gain for the topic model.
    """
    if c is ARXIV:
        m = _arxiv_meta(['paper_id', 'title', 'abstract'], ids, so, metadata_uri)
        m = m.filter(pl.col('abstract').is_not_null()
                     & (pl.col('abstract').str.len_chars() > 20))
        if stop_at:
            m = m.head(stop_at)
        return m.sort('paper_id')
    abstracts = read_by_part(ABSTRACTS_PREFIX, ['corpusid', 'abstract'], so=so,
                            client=client, ids=ids, text_col='abstract',
                            n_parts=n_parts, stop_at=stop_at)
    if stop_at:
        abstracts = abstracts.head(stop_at)
    matched = abstracts.select('corpusid')
    # one title row per id, so stop as soon as every matched id is covered
    titles = read_by_part(PAPERS_PREFIX, ['corpusid', 'title'], so=so, client=client,
                          ids=matched, stop_at=matched.height)
    return abstracts.join(titles, on='corpusid', how='left').sort('corpusid')
