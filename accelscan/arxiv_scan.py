"""Stage 1 for arXiv: stream source tars -> inventory + candidate passages.

The arXiv counterpart of `accelscan.scan.run_s3`, and deliberately thin: tar
streaming (`arxiv_bulk`), member unpacking (`arxiv_source`) and LaTeX conversion
(`latex`) feed `scan.scan_paragraphs`, which is the *same* match / gate-rescue /
cap / passage-assembly code the S2ORC pass uses. Nothing about the measurement is
reimplemented here; if it were, `tests/test_scan.py` would fail.

Outputs go to MSI S3 under `accelscan/arxiv/...` via `scan.write_shard_outputs`, so
this can run on EC2 (where reading the tars is free) while every later stage reads
them on MSI exactly where it already looks. One extra product per tar --
`ingest/parts/{shard_id}.parquet` -- records skip reasons and LaTeX conversion
stats, which is how converter drift across eras gets caught before it is mistaken
for a trend.

Restartability: one tar = one unit of work, `.done` marker written last and holding
the manifest md5, so a re-issued tar with an unchanged name is not silently skipped
(`--verify-md5`). A killed worker breaks the executor rather than the tar, so
`run_pool` rebuilds the pool and retries the unfinished tars instead of letting the
rest of the run collapse into FAILED lines; every log line carries `n/total` and a
rate so a stall shows up in the log and not only in an S3 listing.

  python -m accelscan.arxiv_scan --tars 20 --yymm 2301-2312   # pilot
  python -m accelscan.arxiv_scan --dry-run                    # what it would cost
  python -m accelscan.arxiv_scan --max-workers 16             # full history
"""

import argparse
import io
import sys
import time
from concurrent.futures import (BrokenExecutor, CancelledError, ProcessPoolExecutor,
                                as_completed)

import polars as pl

from accelscan.arxiv_bulk import (TarEntry, arxiv_client, iter_members, load_manifest,
                                 manifest_bytes_total, open_tar, select_tars,
                                 warn_if_not_in_region)
from accelscan.arxiv_source import (arxiv_id_from_member, paper_id_from_arxiv_id,
                                   unpack_member, year_month_from_id)
from accelscan.config import BUCKET
from accelscan.latex import latex_to_paragraphs
from accelscan.paths import ARXIV, ingest_stats_key
from accelscan.registry import load_registry
from accelscan.scan import frames_from_scans, scan_paragraphs, write_shard_outputs

MANIFEST_CACHE = 'output/arxiv_src_manifest.xml'

INGEST_SCHEMA = {
    'shard_id': pl.Utf8, 'paper_id': pl.Utf8, 'arxiv_id': pl.Utf8,
    'year': pl.Int32, 'month': pl.Int32, 'skip_reason': pl.Utf8,
    'n_files': pl.Int32, 'n_paragraphs': pl.Int32, 'plain_chars': pl.Int32,
    'encoding': pl.Utf8, 'body_found': pl.Boolean, 'had_bibliography': pl.Boolean,
    'includes_resolved': pl.Int32, 'includes_missing': pl.Int32,
    'macros_expanded': pl.Int32, 'n_ref_paras_filtered': pl.Int32,
}


def _ingest_row(shard_id: str, paper_id: str | None, arxiv_id: str | None,
                skip: str, stats: dict) -> dict:
    year, month = year_month_from_id(arxiv_id) if arxiv_id else (None, None)
    return {
        'shard_id': shard_id, 'paper_id': paper_id, 'arxiv_id': arxiv_id,
        'year': year, 'month': month, 'skip_reason': skip,
        'n_files': stats.get('n_files'), 'n_paragraphs': stats.get('n_paragraphs'),
        'plain_chars': stats.get('plain_chars'), 'encoding': stats.get('encoding'),
        'body_found': stats.get('body_found'),
        'had_bibliography': stats.get('had_bibliography'),
        'includes_resolved': stats.get('includes_resolved'),
        'includes_missing': stats.get('includes_missing'),
        'macros_expanded': stats.get('macros_expanded'),
        'n_ref_paras_filtered': stats.get('n_ref_paras_filtered'),
    }


def scan_tar_stream(tf, entry: TarEntry, reg) -> tuple[pl.DataFrame, pl.DataFrame,
                                                      pl.DataFrame]:
    """One open tar stream -> (inventory, candidates, ingest_stats).

    Members are consumed strictly in order: the stream cannot go back.
    """
    scans, ingest = [], []
    for name, data in iter_members(tf):
        arxiv_id = arxiv_id_from_member(name)
        if arxiv_id is None:
            ingest.append(_ingest_row(entry.shard_id, None, None, 'unknown_id', {}))
            continue
        paper_id = paper_id_from_arxiv_id(arxiv_id)
        files, skip = unpack_member(name, data)
        if skip:
            ingest.append(_ingest_row(entry.shard_id, paper_id, arxiv_id, skip, {}))
            continue
        paras, stats = latex_to_paragraphs(files)
        scans.append(scan_paragraphs(
            paras, reg, paper_id=paper_id, corpusid=None,
            shard_id=entry.shard_id, body_chars=stats.get('plain_chars') or 0))
        ingest.append(_ingest_row(entry.shard_id, paper_id, arxiv_id,
                                  '' if paras else 'no_tex', stats))

    inv, cand = frames_from_scans(scans)
    stats_df = pl.DataFrame(ingest, schema=INGEST_SCHEMA, orient='row') if ingest \
        else pl.DataFrame(schema=INGEST_SCHEMA)
    return inv, cand, stats_df


def process_tar(entry: TarEntry, registry_version: str, verify_md5: bool = False) -> str:
    """Worker: stream one tar, scan it, upload outputs + `.done` marker."""
    from accelscan.s3 import make_s3_client
    reg = load_registry()
    if reg.version != registry_version:
        raise RuntimeError(f'registry {reg.version} != requested {registry_version}')

    msi = make_s3_client()                     # outputs land on MSI
    aws = arxiv_client()                       # source is requester-pays AWS
    t0 = time.time()
    with open_tar(aws, entry) as tf:
        inv, cand, stats_df = scan_tar_stream(tf, entry, reg)

    write_shard_outputs(msi, entry.shard_id, registry_version, inv, cand, ARXIV,
                        done_body=entry.md5sum.encode())
    buf = io.BytesIO()
    stats_df.write_parquet(buf)
    msi.put_object(Bucket=BUCKET, Key=ingest_stats_key(entry.shard_id),
                   Body=buf.getvalue())
    skipped = int(stats_df.filter(pl.col('skip_reason') != '').height) if stats_df.height else 0
    return (f'{entry.shard_id}: {inv.height} papers, {cand.height} passages, '
            f'{skipped} skipped, {time.time() - t0:.0f}s')


def _todo(entries: list[TarEntry], registry_version: str, client,
          verify_md5: bool) -> list[TarEntry]:
    """Entries without a `.done` marker (or whose marker holds a stale md5)."""
    from accelscan.paths import candidates_parts
    from accelscan.s3 import list_keys
    prefix = f'{candidates_parts(ARXIV, registry_version)}/'
    done = {k.rsplit('/', 1)[-1].removesuffix('.done')
            for k in list_keys(prefix, suffix='.done', client=client)}
    out = []
    for e in entries:
        if e.shard_id not in done:
            out.append(e)
        elif verify_md5:
            body = client.get_object(Bucket=BUCKET,
                                     Key=f'{prefix}{e.shard_id}.done')['Body'].read()
            if body.decode(errors='replace').strip() != e.md5sum:
                print(f'{e.shard_id}: manifest md5 changed, re-scanning',
                      file=sys.stderr)
                out.append(e)
    return out


def run_pool(todo: list[TarEntry], registry_version: str, max_workers: int,
             verify_md5: bool = False, max_rounds: int = 6) -> tuple[int, int]:
    """Scan `todo` in a process pool, rebuilding the pool if it breaks. -> (ok, failed).

    A per-tar exception is logged and skipped -- one lost tar out of thousands is
    not worth stopping for. But a worker *killed* (the OOM killer, most often, when
    too many tar streams are in flight at once) breaks the executor itself, and then
    every future still queued raises `BrokenProcessPool`. Treating that like a tar
    failure would burn the whole remaining run in a burst of FAILED lines and exit
    with status 0 -- which is exactly how the 2026-07-31 run died silently. So a
    broken pool retries its unfinished tars in a fresh pool at half the width,
    on the theory that whatever killed a worker was contention.

    Re-running a tar is safe and cheap in the only sense that matters: a killed
    worker wrote no `.done` marker, so nothing is double-counted, and only the tars
    that genuinely never returned are re-downloaded.
    """
    total = len(todo)
    pending, workers, ok, failed = list(todo), max_workers, 0, 0
    t0 = time.time()
    for round_no in range(1, max_rounds + 1):
        if not pending:
            break
        if round_no > 1:
            print(f'-- round {round_no}: pool broke, retrying {len(pending)} tars '
                  f'with {workers} workers', file=sys.stderr)
        retry, broke = [], False
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_tar, e, registry_version, verify_md5): e
                       for e in pending}
            for fut in as_completed(futures):
                entry = futures[fut]
                try:
                    msg = fut.result()
                except (BrokenExecutor, CancelledError):
                    retry.append(entry)          # worker died: the tar never ran
                    broke = True
                    continue
                except Exception as exc:         # one lost tar must not stop the run
                    failed += 1
                    print(f'[{ok + failed}/{total}] FAILED {entry.shard_id}: {exc!r}',
                          file=sys.stderr)
                    continue
                ok += 1
                print(f'[{ok + failed}/{total}] {msg}  |  {_pace(ok + failed, total, t0)}',
                      file=sys.stderr)
        pending = retry            # empty on a clean round, so nothing is left over
        if not broke:
            break
        workers = max(2, workers // 2)
    if pending:
        failed += len(pending)
        print(f'giving up on {len(pending)} tars after {max_rounds} rounds; '
              f'their .done markers are absent, so re-running picks them up',
              file=sys.stderr)
    return ok, failed


def _pace(n: int, total: int, t0: float) -> str:
    """'9.6 tars/min, eta 9.9h' -- so a stall is visible in the log, not only in S3."""
    mins = (time.time() - t0) / 60
    if mins < 0.05 or not n:
        return ''                  # too early to divide by; would print nonsense
    rate = n / mins
    eta = (total - n) / rate / 60
    return f'{rate:.1f} tars/min, eta {eta:.1f}h'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tars', type=int, help='cap the number of tars (pilot)')
    ap.add_argument('--yymm', help='inclusive range, e.g. 9108-2512')
    ap.add_argument('--max-workers', type=int, default=8)
    ap.add_argument('--verify-md5', action='store_true',
                    help='re-scan shards whose .done md5 no longer matches the manifest')
    ap.add_argument('--dry-run', action='store_true',
                    help='list the work and the egress bill, download nothing')
    ap.add_argument('--max-bytes', type=float,
                    help='refuse to start if the selection exceeds this many bytes')
    ap.add_argument('--yes-i-know', action='store_true',
                    help='proceed outside us-east-1 despite the egress cost')
    ap.add_argument('--manifest-cache', default=MANIFEST_CACHE)
    args = ap.parse_args()

    reg = load_registry()
    entries = load_manifest(cache=args.manifest_cache)
    yymm = tuple(args.yymm.split('-')) if args.yymm else None
    if yymm and len(yymm) != 2:
        ap.error('--yymm must look like 9108-2512')
    entries = select_tars(entries, yymm_range=yymm, limit=args.tars)
    if not entries:
        print('no tars selected', file=sys.stderr)
        return

    from accelscan.s3 import make_s3_client
    msi = make_s3_client()
    todo = _todo(entries, reg.version, msi, args.verify_md5)
    n_bytes = manifest_bytes_total(todo)
    print(f'{len(entries)} tars selected ({entries[0].yymm}..{entries[-1].yymm}), '
          f'{len(entries) - len(todo)} already done, {len(todo)} to scan, '
          f'{n_bytes / 1e12:.2f} TB, {sum(e.num_items for e in todo):,} submissions',
          file=sys.stderr)
    warn_if_not_in_region(n_bytes)

    if args.max_bytes and n_bytes > args.max_bytes:
        ap.error(f'selection is {n_bytes:,.0f} bytes > --max-bytes {args.max_bytes:,.0f}')
    if args.dry_run:
        for e in todo[:20]:
            print(f'  {e.shard_id}  {e.size / 1e6:6.0f} MB  {e.num_items:>5} items')
        if len(todo) > 20:
            print(f'  ... and {len(todo) - 20} more')
        return
    from accelscan.arxiv_bulk import in_arxiv_region
    if not in_arxiv_region() and not args.yes_i_know:
        ap.error('refusing to pay egress outside us-east-1; pass --yes-i-know to override')

    ok, failed = run_pool(todo, reg.version, args.max_workers, args.verify_md5)
    print(f'{ok} tars scanned, {failed} failed', file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
