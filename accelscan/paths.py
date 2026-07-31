"""Corpus-aware S3 key builders.

Every output key in the pipeline is built here rather than as an inline f-string,
so a second corpus is a parameter instead of a fork.

**Legacy alias — the load-bearing invariant.** S2ORC keys are byte-identical to the
ones the pipeline has always written (`accelscan/{product}/...`), so nothing moves
and nothing re-runs; arXiv lives under `accelscan/arxiv/{product}/...`. The two
namespaces cannot leak into each other because S3 prefix matching is literal and
`'accelscan/arxiv/mentions/'.startswith('accelscan/mentions/')` is False — that is
what keeps `discover_mentions_version`, `precision`'s version scan, and the `.done`
shard diffing corpus-safe. `tests/test_paths.py` pins both properties; treat a
failure there as a data-corruption bug, not a test to update.

Pure string building: no S3 calls, no imports from `accelscan.s3` (which imports
config), so this module is safe to import anywhere.
"""

from dataclasses import dataclass

import polars as pl

from accelscan.config import BUCKET, OUT_PREFIX


@dataclass(frozen=True)
class Corpus:
    """A text corpus and its identity column.

    `key` is the per-paper primary key downstream stages group and join on:
    `corpusid` (Int64, Semantic Scholar) for S2ORC, `paper_id` (Utf8, e.g.
    `arxiv:2301.01234`) for arXiv, whose ids are strings and have no corpusid.
    """

    name: str
    out_prefix: str
    key: str
    key_dtype: pl.DataType


S2ORC = Corpus('s2orc', OUT_PREFIX, 'corpusid', pl.Int64)
ARXIV = Corpus('arxiv', f'{OUT_PREFIX}/arxiv', 'paper_id', pl.Utf8)
CORPORA = {c.name: c for c in (S2ORC, ARXIV)}
DEFAULT_CORPUS = 's2orc'


def get_corpus(name: str = DEFAULT_CORPUS) -> Corpus:
    if name not in CORPORA:
        raise ValueError(f'unknown corpus {name!r}; expected one of {sorted(CORPORA)}')
    return CORPORA[name]


def s3_uri(key: str) -> str:
    return f's3://{BUCKET}/{key}'


# --- stage 1: inventory + candidates ---------------------------------------
# inventory is deliberately NOT registry-versioned (it is the paper population,
# independent of the matcher) -- preserved from the original layout.

def inventory_parts(c: Corpus) -> str:
    return f'{c.out_prefix}/inventory/parts'


def inventory_glob(c: Corpus) -> str:
    return s3_uri(f'{inventory_parts(c)}/*.parquet')


def candidates_root(c: Corpus) -> str:
    """Prefix above the registry-version segment (for version discovery)."""
    return f'{c.out_prefix}/candidates/'


def candidates_parts(c: Corpus, registry_version: str) -> str:
    return f'{c.out_prefix}/candidates/{registry_version}/parts'


def candidates_glob(c: Corpus, registry_version: str) -> str:
    return s3_uri(f'{candidates_parts(c, registry_version)}/*.parquet')


# --- stage 1.5: repacked passage shards ------------------------------------

def passages_prefix(c: Corpus, registry_version: str) -> str:
    return f'{c.out_prefix}/passages/{registry_version}'


def passages_key(c: Corpus, registry_version: str, shard_index: int) -> str:
    return f'{passages_prefix(c, registry_version)}/shard_{shard_index:04d}.parquet'


def passages_manifest(c: Corpus, registry_version: str) -> str:
    return f'{passages_prefix(c, registry_version)}/manifest.parquet'


# --- stage 2: mentions -----------------------------------------------------

def mentions_root(c: Corpus) -> str:
    """Prefix above the registry-version segment (for version discovery)."""
    return f'{c.out_prefix}/mentions/'


def mentions_parts(c: Corpus, registry_version: str, prompt_version: str,
                   model_tag: str) -> str:
    return (f'{c.out_prefix}/mentions/{registry_version}/{prompt_version}'
            f'/{model_tag}/parts')


# --- stages 3.5 / 4a -------------------------------------------------------

def capacity_key(c: Corpus, registry_version: str, prompt_version: str,
                 model_tag: str, count_policy: str) -> str:
    return (f'{c.out_prefix}/capacity/{registry_version}/{prompt_version}'
            f'/{model_tag}/paper_capacity_{count_policy}.parquet')


def analytic_base(c: Corpus, registry_version: str, prompt_version: str,
                  model_tag: str) -> str:
    return (f'{c.out_prefix}/analytic/{registry_version}/{prompt_version}'
            f'/{model_tag}')


def precision_key(c: Corpus, candidates_version: str) -> str:
    return f'{c.out_prefix}/precision/{candidates_version}/paper_precision.parquet'


# --- topic pipeline --------------------------------------------------------

def embeddings_parts(c: Corpus, embed_tag: str) -> str:
    return f'{c.out_prefix}/embeddings/{embed_tag}/parts'


def embeddings_chunk(c: Corpus, embed_tag: str, chunk_index: int) -> str:
    """Base key for a chunk; callers append '.parquet' / '.done'."""
    return f'{embeddings_parts(c, embed_tag)}/chunk_{chunk_index:04d}'


def clusters_base(c: Corpus, cluster_version: str) -> str:
    return f'{c.out_prefix}/clusters/{cluster_version}'


# --- arXiv-only products ---------------------------------------------------

def arxiv_metadata_key() -> str:
    """Category/title/abstract/date snapshot (Kaggle OAI dump -> parquet)."""
    return f'{ARXIV.out_prefix}/metadata/arxiv_metadata.parquet'


def ingest_stats_parts() -> str:
    """Per-tar unpack skip reasons + LaTeX conversion stats."""
    return f'{ARXIV.out_prefix}/ingest/parts'


def ingest_stats_key(shard_id: str) -> str:
    return f'{ingest_stats_parts()}/{shard_id}.parquet'
