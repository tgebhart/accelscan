"""Stage E1: SPECTER2 embeddings of accelerator-paper abstracts.

Work-list = distinct corpusids with a `status==ok` mention (any usage_context),
joined to the abstracts parquet (title+abstract) via a lazy pushdown join so
only the matched rows materialize. Encodes `title [SEP] abstract` with SPECTER2
(CLS pooling — SPECTER's native convention), the true `allenai/specter2` base +
proximity adapter when the `adapters` lib is importable, else the plain
`allenai/specter2_base` encoder (recorded in the embed tag / provenance).

Model loads once; chunks are the unit of restart (zero-byte `.done` markers,
same convention as scan/infer). One GPU is ample (~minutes for ~300k).

  python -m accelscan.embed                       # all chunks, one process
  python -m accelscan.embed --limit 500 --local-out output/embed_smoke   # dev
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import polars as pl

from accelscan.config import BUCKET
from accelscan.meta import paper_titles_abstracts
from accelscan.paths import S2ORC, Corpus, embeddings_chunk, get_corpus
from accelscan.s3 import mentions_glob

SPECTER2_BASE = 'allenai/specter2_base'
PROXIMITY_ADAPTER = 'allenai/specter2'
CHUNK_SIZE = 50_000
MAX_LENGTH = 512
BATCH_SIZE = 64
EMB_DIM = 768


def load_worklist(mentions_glob: str, storage_options: dict,
                  limit: int | None = None, n_abstract_parts: int | None = None,
                  s3_client=None, corpus: Corpus = S2ORC) -> pl.DataFrame:
    """Build (key, title, abstract) for accelerator papers, memory-safely.

    Reads metadata part-by-part via `accelscan.meta` (bounded memory: a whole-table
    join OOMs on the tens-of-millions-row abstracts/papers tables).
    `n_abstract_parts` restricts the scan (=1 for a fast smoke; None = all)."""
    ids = (pl.scan_parquet(mentions_glob, storage_options=storage_options)
           .filter(pl.col('status') == 'ok')
           .select(corpus.key).unique().collect())          # ~270k, tiny
    return paper_titles_abstracts(corpus, ids, storage_options, s3_client,
                                  n_parts=n_abstract_parts, stop_at=limit)


class Specter2Encoder:
    """SPECTER2 document encoder (CLS pooling). Uses the proximity adapter when
    `adapters` is importable, else the base encoder. `.adapter` records which."""

    def __init__(self, device: str | None = None, precision: str = 'fp16'):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = {'fp32': torch.float32, 'fp16': torch.float16,
                      'bf16': torch.bfloat16}[precision]
        self.tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE, use_fast=True)
        self.sep = self.tokenizer.sep_token or '[SEP]'
        try:
            from adapters import AutoAdapterModel
            model = AutoAdapterModel.from_pretrained(SPECTER2_BASE, dtype=self.dtype)
            model.load_adapter(PROXIMITY_ADAPTER, source='hf',
                               load_as='proximity', set_active=True)
            self.adapter = 'proximity'
        except Exception as e:  # adapters missing / incompatible → base encoder
            from transformers import AutoModel
            print(f'[embed] proximity adapter unavailable ({e!r}); using base encoder',
                  file=sys.stderr)
            model = AutoModel.from_pretrained(SPECTER2_BASE, dtype=self.dtype)
            self.adapter = 'base'
        self.model = model.to(self.device).eval()

    def embed_tag(self) -> str:
        return f'specter2-{self.adapter}'

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(texts), EMB_DIM), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start:start + BATCH_SIZE]
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=MAX_LENGTH, return_tensors='pt').to(self.device)
                hidden = self.model(**enc).last_hidden_state  # (B, L, D)
                out[start:start + len(batch)] = hidden[:, 0].float().cpu().numpy()  # CLS
        return out


def build_texts(df: pl.DataFrame, sep: str) -> list[str]:
    texts = []
    for r in df.iter_rows(named=True):
        title = (r['title'] or '').strip()
        abstract = (r['abstract'] or '').strip()
        texts.append(f'{title}{sep}{abstract}' if title else abstract)
    return texts


def _emb_frame(ids: list, embs: np.ndarray, corpus: Corpus = S2ORC) -> pl.DataFrame:
    return pl.DataFrame({corpus.key: ids, 'emb': embs.tolist()},
                        schema={corpus.key: corpus.key_dtype,
                                'emb': pl.Array(pl.Float32, EMB_DIM)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-tag', default='qwen3-14b', help='mentions model tag to source ids from')
    ap.add_argument('--prompt-version', default='p1')
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--precision', default='fp16')
    ap.add_argument('--limit', type=int, help='dev: cap work-list size')
    ap.add_argument('--abstract-parts', type=int,
                    help='dev: scan only the first N abstract part-files (=1 for a fast smoke)')
    ap.add_argument('--local-out', help='dev: write chunks to a local dir instead of S3')
    ap.add_argument('--corpus', default='s2orc', choices=['s2orc', 'arxiv'])
    args = ap.parse_args()
    c = get_corpus(args.corpus)

    from accelscan.registry import load_registry
    from accelscan.s3 import make_s3_client, storage_options
    reg = load_registry()
    so = storage_options()
    client = make_s3_client()
    # discover the EXTRACTION-time registry version; building this from
    # reg.version silently yielded an empty glob once the registry passed 0.1.0
    glob = mentions_glob(args.model_tag, args.prompt_version, client=client, corpus=c)

    work = load_worklist(glob, so, limit=args.limit,
                         n_abstract_parts=args.abstract_parts, s3_client=client,
                         corpus=c)
    print(f'work-list: {work.height:,} papers with usable abstract', file=sys.stderr)

    enc = Specter2Encoder(precision=args.precision)
    tag = enc.embed_tag()
    print(f'encoder: {tag} on {enc.device}', file=sys.stderr)

    if args.local_out:
        Path(args.local_out).mkdir(parents=True, exist_ok=True)

    n_chunks = (work.height + args.chunk_size - 1) // args.chunk_size
    for i in range(n_chunks):
        chunk = work.slice(i * args.chunk_size, args.chunk_size)
        if args.local_out:
            dst = Path(args.local_out) / f'chunk_{i:04d}.parquet'
            if dst.exists():
                continue
        else:
            base = embeddings_chunk(c, tag, i)
            done_key = f'{base}.done'
            if client.list_objects_v2(Bucket=BUCKET, Prefix=done_key).get('KeyCount', 0):
                print(f'chunk {i} done, skipping', file=sys.stderr)
                continue
        embs = enc.encode(build_texts(chunk, enc.sep))
        frame = _emb_frame(chunk[c.key].to_list(), embs, c)
        if args.local_out:
            frame.write_parquet(dst)
        else:
            buf = io.BytesIO(); frame.write_parquet(buf)
            client.put_object(Bucket=BUCKET, Key=f'{base}.parquet', Body=buf.getvalue())
            client.put_object(Bucket=BUCKET, Key=done_key, Body=b'')
        print(f'chunk {i}/{n_chunks}: {frame.height} embeddings', file=sys.stderr)
    print(f'done: {work.height:,} papers, tag={tag}', file=sys.stderr)


if __name__ == '__main__':
    main()
