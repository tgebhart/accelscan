"""Stage 1: one CPU pass over S2ORC shards -> inventory + candidate passages.

Per paper: parse body paragraphs, match the registry against each paragraph
(gate window = paragraph ± neighbors), apply paper-level gate rescue (a tier-A
model hit anywhere in the paper un-gates tier-B matches), assemble capped
candidate passages, and emit one inventory row per paper plus one candidates
row per matched paragraph.

Restartability: the output shard is the unit of work; a zero-byte
`<shard>.done` S3 marker is the unit of completion. Rerunning skips shards
whose marker exists.

CLI:
  python -m accelscan.scan --local-file <path> --out-dir <dir>      # dev/smoke
  python -m accelscan.scan --shards 8 --seed 0                      # pilot (random shards)
  python -m accelscan.scan                                          # all shards
"""

import argparse
import gzip
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from accelscan.config import (BUCKET, GATE_WINDOW_CHARS,
                              MAX_PASSAGES_GENERIC_ONLY,
                              MAX_PASSAGES_MODEL_SPECIFIC, OUT_PREFIX,
                              PASSAGE_CHAR_CAP, S2ORC_PREFIX)
from accelscan.records import Paragraph, parse_body
from accelscan.registry import CompiledRegistry, Match, load_registry

INVENTORY_SCHEMA = {
    'corpusid': pl.Int64, 'shard_id': pl.Utf8, 'has_body': pl.Boolean,
    'body_chars': pl.Int32, 'n_paragraphs': pl.Int32,
    'is_candidate': pl.Boolean, 'n_candidate_passages': pl.Int32,
    'passages_truncated': pl.Boolean,
}
CANDIDATE_SCHEMA = {
    'passage_id': pl.Utf8, 'corpusid': pl.Int64, 'shard_id': pl.Utf8,
    'para_idx': pl.Int32, 'passage_text': pl.Utf8, 'section_header': pl.Utf8,
    'matched_models': pl.List(pl.Utf8), 'matched_surfaces': pl.List(pl.Utf8),
    'match_starts': pl.List(pl.Int32), 'match_ends': pl.List(pl.Int32),
    'gated_only': pl.Boolean, 'model_specific': pl.Boolean,
}


@dataclass
class PaperScan:
    inventory: dict
    candidates: list[dict] = field(default_factory=list)


def _assemble_passage(paras: list[Paragraph], i: int,
                      matches: list[Match]) -> tuple[str, int]:
    """Return (passage_text, offset of matched paragraph within it)."""
    core = paras[i].text[:PASSAGE_CHAR_CAP]
    budget = PASSAGE_CHAR_CAP - len(core)
    before = paras[i - 1].text[-(budget // 2):] if i > 0 and budget > 40 else ''
    after = paras[i + 1].text[:budget - len(before)] if i + 1 < len(paras) and budget > 40 else ''
    prefix = before + '\n' if before else ''
    text = prefix + core + ('\n' + after if after else '')
    return text, len(prefix)


def scan_record(record: dict, reg: CompiledRegistry, shard_id: str) -> PaperScan:
    corpusid = record.get('corpusid')
    paras = parse_body(record)
    body = record.get('body') or {}
    body_text = body.get('text') or '' if isinstance(body, dict) else ''

    inv = {
        'corpusid': corpusid, 'shard_id': shard_id, 'has_body': bool(body_text),
        'body_chars': len(body_text), 'n_paragraphs': len(paras),
        'is_candidate': False,
        'n_candidate_passages': 0, 'passages_truncated': False,
    }
    scan = PaperScan(inventory=inv)
    if not paras:
        return scan

    per_para: list[tuple[int, list[Match]]] = []
    for i, p in enumerate(paras):
        before = paras[i - 1].text[-GATE_WINDOW_CHARS:] if i > 0 else ''
        after = paras[i + 1].text[:GATE_WINDOW_CHARS] if i + 1 < len(paras) else ''
        ms = reg.match_paragraph(p.text, context_before=before, context_after=after)
        if ms:
            per_para.append((i, ms))
    if not per_para:
        return scan

    # Paper-level gate rescue: any ungated *model* hit un-gates tier-B matches.
    anchored = any(m.kind == 'model' and not m.gate_required
                   for _, ms in per_para for m in ms)
    kept_per_para: list[tuple[int, list[Match]]] = []
    for i, ms in per_para:
        kept = [m for m in ms if (not m.gate_required) or m.gate_ok or anchored]
        if kept:
            kept_per_para.append((i, kept))
    if not kept_per_para:
        return scan

    model_specific = any(m.kind != 'generic' for _, ms in kept_per_para for m in ms)
    cap = MAX_PASSAGES_MODEL_SPECIFIC if model_specific else MAX_PASSAGES_GENERIC_ONLY

    inv['is_candidate'] = True
    inv['passages_truncated'] = len(kept_per_para) > cap
    for i, ms in kept_per_para[:cap]:
        text, offset = _assemble_passage(paras, i, ms)
        scan.candidates.append({
            'passage_id': f'{corpusid}:{paras[i].idx}',
            'corpusid': corpusid, 'shard_id': shard_id, 'para_idx': paras[i].idx,
            'passage_text': text, 'section_header': paras[i].section,
            'matched_models': [m.model_id for m in ms],
            'matched_surfaces': [m.surface for m in ms],
            'match_starts': [m.start + offset for m in ms],
            'match_ends': [m.end + offset for m in ms],
            'gated_only': all(m.gate_required for m in ms),
            'model_specific': any(m.kind != 'generic' for m in ms),
        })
    inv['n_candidate_passages'] = len(scan.candidates)
    return scan


def scan_stream(lines, reg: CompiledRegistry,
                shard_id: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    inv_rows, cand_rows = [], []
    for line in lines:
        try:
            record = orjson.loads(line)
        except Exception:
            continue
        s = scan_record(record, reg, shard_id)
        inv_rows.append(s.inventory)
        cand_rows.extend(s.candidates)
    inv = pl.DataFrame(inv_rows, schema=INVENTORY_SCHEMA, orient='row') if inv_rows \
        else pl.DataFrame(schema=INVENTORY_SCHEMA)
    cand = pl.DataFrame(cand_rows, schema=CANDIDATE_SCHEMA, orient='row') if cand_rows \
        else pl.DataFrame(schema=CANDIDATE_SCHEMA)
    return inv, cand


# ---------------------------------------------------------------------------
# Shard workers
# ---------------------------------------------------------------------------

def _out_keys(shard_id: str, registry_version: str) -> dict[str, str]:
    base = f'{OUT_PREFIX}/candidates/{registry_version}'
    return {
        'inventory': f'{OUT_PREFIX}/inventory/parts/{shard_id}.parquet',
        'candidates': f'{base}/parts/{shard_id}.parquet',
        'done': f'{base}/parts/{shard_id}.done',
    }


def process_s3_shard(key: str, registry_version: str) -> str:
    """Worker: fetch one gzipped S2ORC shard, scan, upload outputs + marker."""
    from accelscan.registry import load_registry as _load
    from accelscan.s3 import make_s3_client

    shard_id = Path(key).name.removesuffix('.gz')
    client = make_s3_client()
    reg = _load()
    if reg.version != registry_version:
        raise RuntimeError(f'registry {reg.version} != requested {registry_version}')

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True)
    def fetch() -> bytes:
        return client.get_object(Bucket=BUCKET, Key=key)['Body'].read()

    with gzip.open(io.BytesIO(fetch())) as f:
        inv, cand = scan_stream(f, reg, shard_id)

    keys = _out_keys(shard_id, registry_version)
    for name, df in [('inventory', inv), ('candidates', cand)]:
        buf = io.BytesIO()
        df.write_parquet(buf)
        client.put_object(Bucket=BUCKET, Key=keys[name], Body=buf.getvalue())
    client.put_object(Bucket=BUCKET, Key=keys['done'], Body=b'')
    return f'{shard_id}: {inv.height} papers, {cand.height} passages'


def run_local(path: str, out_dir: str, limit: int | None) -> None:
    reg = load_registry()
    shard_id = Path(path).name.removesuffix('.gz')
    opener = gzip.open if path.endswith('.gz') else open
    t0 = time.time()
    with opener(path, 'rb') as f:
        lines = (line for i, line in enumerate(f) if limit is None or i < limit)
        inv, cand = scan_stream(lines, reg, shard_id)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv.write_parquet(out / f'{shard_id}.inventory.parquet')
    cand.write_parquet(out / f'{shard_id}.candidates.parquet')
    print(f'{inv.height} papers, {inv["is_candidate"].sum()} candidates, '
          f'{cand.height} passages in {time.time() - t0:.1f}s -> {out}')


def run_s3(n_shards: int | None, seed: int, max_workers: int) -> None:
    import random

    from accelscan.s3 import list_keys, make_s3_client
    reg = load_registry()
    keys = list_keys(S2ORC_PREFIX, suffix='.gz')
    if n_shards:
        random.Random(seed).shuffle(keys)
        keys = keys[:n_shards]

    client = make_s3_client()
    done_prefix = f'{OUT_PREFIX}/candidates/{reg.version}/parts/'
    done = {Path(k).name.removesuffix('.done')
            for k in list_keys(done_prefix, suffix='.done', client=client)}
    todo = [k for k in keys if Path(k).name.removesuffix('.gz') not in done]
    print(f'{len(keys)} shards requested, {len(done)} done, {len(todo)} to scan',
          file=sys.stderr)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_s3_shard, k, reg.version): k for k in todo}
        for fut in as_completed(futures):
            try:
                print(fut.result(), file=sys.stderr)
            except Exception as e:
                print(f'FAILED {futures[fut]}: {e}', file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--local-file', help='scan one local shard (dev/smoke)')
    ap.add_argument('--out-dir', default='output/scan_local')
    ap.add_argument('--limit', type=int, help='max records (local mode)')
    ap.add_argument('--shards', type=int, help='random shard subsample size (pilot)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-workers', type=int, default=24)
    args = ap.parse_args()
    if args.local_file:
        run_local(args.local_file, args.out_dir, args.limit)
    else:
        run_s3(args.shards, args.seed, args.max_workers)


if __name__ == '__main__':
    main()
