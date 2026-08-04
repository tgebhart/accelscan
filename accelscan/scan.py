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
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from accelscan.config import (BUCKET, GATE_WINDOW_CHARS,
                              MAX_PASSAGES_GENERIC_ONLY,
                              MAX_PASSAGES_MODEL_SPECIFIC,
                              PASSAGE_CHAR_CAP, S2ORC_PREFIX)
from accelscan.paths import S2ORC, Corpus, candidates_parts, inventory_parts
from accelscan.records import Paragraph, parse_body
from accelscan.registry import CompiledRegistry, Match, load_registry

# `paper_id` is the corpus-independent key (str(corpusid) for S2ORC,
# 'arxiv:2301.01234' for arXiv, whose ids are strings); `corpusid` stays for the
# S2ORC joins and is null for arXiv. NOTE: these dicts are consumed with
# orient='row', so the row dicts below must be built in this exact key order.
INVENTORY_SCHEMA = {
    'paper_id': pl.Utf8, 'corpusid': pl.Int64,
    'shard_id': pl.Utf8, 'has_body': pl.Boolean,
    'body_chars': pl.Int32, 'n_paragraphs': pl.Int32,
    'is_candidate': pl.Boolean, 'n_candidate_passages': pl.Int32,
    'passages_truncated': pl.Boolean,
}
CANDIDATE_SCHEMA = {
    'passage_id': pl.Utf8, 'paper_id': pl.Utf8, 'corpusid': pl.Int64,
    'shard_id': pl.Utf8,
    'para_idx': pl.Int32, 'passage_text': pl.Utf8, 'section_header': pl.Utf8,
    'matched_models': pl.List(pl.Utf8), 'matched_surfaces': pl.List(pl.Utf8),
    'match_starts': pl.List(pl.Int32), 'match_ends': pl.List(pl.Int32),
    'gated_only': pl.Boolean, 'gate_rescued': pl.Boolean,
    'model_specific': pl.Boolean,
}


@dataclass
class PaperScan:
    inventory: dict
    candidates: list[dict] = field(default_factory=list)


def _assemble_passage(paras: list[Paragraph], i: int,
                      matches: list[Match]) -> tuple[str, int]:
    """Return (passage_text, offset to add to a match position within it).

    The core window is centred on the matched span, not taken from the head of the
    paragraph. Head truncation silently cut the *trigger itself* out of the passage
    whenever a match sat past `PASSAGE_CHAR_CAP`: the recorded `match_starts` then
    indexed past the end of `passage_text`, and -- worse -- the LLM received a passage
    with no hardware mention in it and correctly reported none. Measured on the arXiv
    run at 0.56% of passages guaranteed to lose their trigger this way, and arXiv's
    long TeX paragraphs make it common in a way GROBID's shorter ones did not.
    """
    para = paras[i].text
    core_start = 0
    if len(para) > PASSAGE_CHAR_CAP and matches:
        mid = (min(m.start for m in matches) + max(m.end for m in matches)) // 2
        core_start = max(0, min(mid - PASSAGE_CHAR_CAP // 2,
                                len(para) - PASSAGE_CHAR_CAP))
    core = para[core_start:core_start + PASSAGE_CHAR_CAP]
    budget = PASSAGE_CHAR_CAP - len(core)
    before = paras[i - 1].text[-(budget // 2):] if i > 0 and budget > 40 else ''
    after = paras[i + 1].text[:budget - len(before)] if i + 1 < len(paras) and budget > 40 else ''
    prefix = before + '\n' if before else ''
    text = prefix + core + ('\n' + after if after else '')
    return text, len(prefix) - core_start


def scan_record(record: dict, reg: CompiledRegistry, shard_id: str) -> PaperScan:
    """S2ORC adapter: pull id + paragraphs off a record, then scan generically."""
    corpusid = record.get('corpusid')
    paras = parse_body(record)
    body = record.get('body') or {}
    body_text = body.get('text') or '' if isinstance(body, dict) else ''
    return scan_paragraphs(paras, reg, paper_id=str(corpusid), corpusid=corpusid,
                           shard_id=shard_id, body_chars=len(body_text))


def scan_paragraphs(paras: list[Paragraph], reg: CompiledRegistry, *,
                    paper_id: str, corpusid: int | None, shard_id: str,
                    body_chars: int) -> PaperScan:
    """Corpus-agnostic core: paragraphs in, inventory + candidate rows out.

    Owns everything that defines the measurement -- per-paragraph matching with
    +/-GATE_WINDOW_CHARS neighbour context, paper-level gate rescue, the
    passage cap, passage assembly and match-offset rebasing -- so every corpus
    is measured by identical code. Callers supply only the id and paragraphs.
    """
    inv = {
        'paper_id': paper_id, 'corpusid': corpusid,
        'shard_id': shard_id, 'has_body': bool(body_chars),
        'body_chars': body_chars, 'n_paragraphs': len(paras),
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
        # invariant: every reported span must index passage_text. Matches spread
        # wider than PASSAGE_CHAR_CAP cannot all fit in one window; keep those in
        # the window rather than emit a dangling offset.
        kept_ms = [m for m in ms
                   if 0 <= m.start + offset and m.end + offset <= len(text)
                   and text[m.start + offset:m.end + offset] == m.surface]
        ms = kept_ms or ms[:1]
        scan.candidates.append({
            'passage_id': f'{paper_id}:{paras[i].idx}',
            'paper_id': paper_id,
            'corpusid': corpusid, 'shard_id': shard_id, 'para_idx': paras[i].idx,
            'passage_text': text, 'section_header': paras[i].section,
            'matched_models': [m.model_id for m in ms],
            'matched_surfaces': [m.surface for m in ms],
            'match_starts': [m.start + offset for m in ms],
            'match_ends': [m.end + offset for m in ms],
            'gated_only': all(m.gate_required for m in ms),
            # this passage exists ONLY because of the paper-level rescue: every
            # match needed a gate and none was satisfied in the local window.
            # Recorded so the rescue's contribution is measurable after the fact
            # (it is applied above, and was otherwise invisible in the output).
            'gate_rescued': all(m.gate_required and not m.gate_ok for m in ms),
            'model_specific': any(m.kind != 'generic' for m in ms),
        })
    inv['n_candidate_passages'] = len(scan.candidates)
    return scan


def frames_from_scans(scans: Iterable[PaperScan]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(inventory, candidates) frames from per-paper scans. Corpus-agnostic."""
    inv_rows, cand_rows = [], []
    for s in scans:
        inv_rows.append(s.inventory)
        cand_rows.extend(s.candidates)
    inv = pl.DataFrame(inv_rows, schema=INVENTORY_SCHEMA, orient='row') if inv_rows \
        else pl.DataFrame(schema=INVENTORY_SCHEMA)
    cand = pl.DataFrame(cand_rows, schema=CANDIDATE_SCHEMA, orient='row') if cand_rows \
        else pl.DataFrame(schema=CANDIDATE_SCHEMA)
    return inv, cand


def scan_stream(lines, reg: CompiledRegistry,
                shard_id: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """S2ORC gzip-JSONL stream -> frames. arXiv has its own driver (arxiv_scan)."""
    def _scans():
        for line in lines:
            try:
                record = orjson.loads(line)
            except Exception:
                continue
            yield scan_record(record, reg, shard_id)
    return frames_from_scans(_scans())


# ---------------------------------------------------------------------------
# Shard workers
# ---------------------------------------------------------------------------

def _out_keys(shard_id: str, registry_version: str,
              corpus: Corpus = S2ORC) -> dict[str, str]:
    parts = candidates_parts(corpus, registry_version)
    return {
        'inventory': f'{inventory_parts(corpus)}/{shard_id}.parquet',
        'candidates': f'{parts}/{shard_id}.parquet',
        'done': f'{parts}/{shard_id}.done',
    }


def write_shard_outputs(client, shard_id: str, registry_version: str,
                        inv: pl.DataFrame, cand: pl.DataFrame,
                        corpus: Corpus = S2ORC, done_body: bytes = b'') -> None:
    """Upload one shard's outputs, then its `.done` marker last.

    Marker-last ordering is what makes a crash mid-upload safe: the shard simply
    re-runs and overwrites. `done_body` lets a caller stamp provenance in the
    marker (arXiv stores the manifest md5 so a re-issued tar isn't skipped).
    """
    keys = _out_keys(shard_id, registry_version, corpus)
    for name, df in [('inventory', inv), ('candidates', cand)]:
        buf = io.BytesIO()
        df.write_parquet(buf)
        client.put_object(Bucket=BUCKET, Key=keys[name], Body=buf.getvalue())
    client.put_object(Bucket=BUCKET, Key=keys['done'], Body=done_body)


def todo_shards(shard_ids: list[str], registry_version: str, client,
                corpus: Corpus = S2ORC) -> list[str]:
    """Shard ids without a `.done` marker, in the given order."""
    from accelscan.s3 import list_keys
    prefix = f'{candidates_parts(corpus, registry_version)}/'
    done = {Path(k).name.removesuffix('.done')
            for k in list_keys(prefix, suffix='.done', client=client)}
    return [s for s in shard_ids if s not in done]


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

    write_shard_outputs(client, shard_id, registry_version, inv, cand, S2ORC)
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
    by_id = {Path(k).name.removesuffix('.gz'): k for k in keys}
    todo = [by_id[s] for s in todo_shards(list(by_id), reg.version, client, S2ORC)]
    print(f'{len(keys)} shards requested, {len(keys) - len(todo)} done, '
          f'{len(todo)} to scan', file=sys.stderr)

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
