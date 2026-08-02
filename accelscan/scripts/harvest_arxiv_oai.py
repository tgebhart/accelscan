"""Harvest arXiv metadata from arXiv's own OAI-PMH endpoint -> JSONL.

Replaces the Kaggle `Cornell-University/arxiv` snapshot as the metadata source.
OAI-PMH is arXiv's *own*, always-current interface: no credentials, no weekly-refresh
lag, and it is the route arXiv documents for bulk metadata (what they ask people not
to do over HTTP is bulk *full text* -- that still comes from the requester-pays
`src/` tars). Measured: 1,300 records per page at ~3 MB, so ~2,080 requests and
~6.4 GB for the full ~2.7M records.

Output is deliberately the *same record shape as the Kaggle snapshot*
(`id`, `categories`, `title`, `abstract`, `versions[].created`, `license`, `doi`,
`journal-ref`), so `build_arxiv_metadata.py` consumes it unchanged and there is only
ever one parser for this data.

Resumable: the resumption token is a plain `skip=N`, checkpointed to
`<out>.state.json` after every page, and the JSONL is appended. A killed harvest
restarts where it stopped rather than from zero.

A `.gz` output is written gzipped, which is the default: measured 3.2x on real
records, so ~3.0 GB plain against ~0.9 GB compressed for the full corpus, and this file is scratch that gets deleted afterwards.
Resuming appends a new gzip member, which `gzip.open` reads through transparently.

  python -m accelscan.scripts.harvest_arxiv_oai                  # -> output/…jsonl.gz
  python -m accelscan.scripts.harvest_arxiv_oai --max-pages 3     # smoke test
"""

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

BASE = 'https://oaipmh.arxiv.org/oai'
OAI = '{http://www.openarchives.org/OAI/2.0/}'
ARX = '{http://arxiv.org/OAI/arXiv/}'
UA = 'accelscan/0.1 (metascience research; contact gebhart@umn.edu)'
# arXiv asks harvesters to be gentle and honours Retry-After on 503.
DELAY = 3.0
MAX_RETRIES = 8        # consecutive network errors before giving up
MAX_THROTTLES = 40     # consecutive 503s; arXiv throttling is expected, not fatal
READ_TIMEOUT = 120     # a page normally takes ~2s, so this is 60x headroom; keeping it
                       # short means a stalled read is abandoned and retried quickly
                       # instead of looking like a hang
# ETA denominator only. OAI does not advertise completeListSize (checked), and
# ListIdentifiers stalls rather than answering, so this is our own measured figure:
# stage 1 saw 3,091,651 submissions in the src tars through 2026-06. Override with
# --total; being wrong skews the ETA and nothing else.
EST_TOTAL_RECORDS = 3_100_000
# One source of truth for the intermediate path, imported by build_arxiv_metadata so
# the two halves cannot disagree. Relative: everything here runs from the repo root,
# and output/ is gitignored.
OAI_JSONL = 'output/arxiv_oai.jsonl.gz'


def _hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f'{s // 3600}h{(s % 3600) // 60:02d}m' if s >= 3600 else f'{s // 60}m{s % 60:02d}s'


def fetch(params: dict, timeout: int = READ_TIMEOUT, label: str = '') -> str:
    """One OAI request, honouring 503 Retry-After (arXiv's documented throttle).

    503s get their own budget: they are arXiv telling us to wait, not a failure, and
    sharing one counter with network errors let a sustained throttle exhaust the
    retries and kill an hour-long harvest. Read timeouts are the common transient
    here -- pages are ~3 MB and the connection sometimes stalls mid-body.
    """
    url = f'{BASE}?{urllib.parse.urlencode(params)}'
    net_errors, throttles = 0, 0
    while True:
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode('utf-8', errors='replace')
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise
            throttles += 1
            if throttles > MAX_THROTTLES:
                raise RuntimeError(
                    f'throttled {throttles} times in a row; rerun to resume') from e
            wait = min(300, int(e.headers.get('Retry-After', 20) or 20))
            print(f'  {label} 503 throttle {throttles}/{MAX_THROTTLES}, '
                  f'sleeping {wait}s', file=sys.stderr, flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            net_errors += 1
            if net_errors > MAX_RETRIES:
                raise RuntimeError(
                    f'{net_errors} network errors on {url}; the page checkpoint is '
                    f'intact, so rerunning resumes from here') from e
            wait = min(60, 5 * 2 ** (net_errors - 1))
            print(f'  {label} {type(e).__name__}: {e} -- retry '
                  f'{net_errors}/{MAX_RETRIES} in {wait}s',
                  file=sys.stderr, flush=True)
            time.sleep(wait)


def _text(node, tag: str) -> str | None:
    el = node.find(f'{ARX}{tag}')
    return ' '.join(el.text.split()) if el is not None and el.text else None


def parse_page(xml: str) -> tuple[list[dict], str | None]:
    """-> (records in Kaggle shape, resumption token or None).

    Deleted records carry a `status="deleted"` header and no metadata; they are
    skipped rather than emitted as rows with null everything.
    """
    root = ET.fromstring(xml)
    err = root.find(f'{OAI}error')
    if err is not None and (err.get('code') or '') not in ('noRecordsMatch',):
        raise RuntimeError(f"OAI error {err.get('code')}: {err.text}")
    out = []
    for rec in root.iter(f'{OAI}record'):
        header = rec.find(f'{OAI}header')
        if header is not None and header.get('status') == 'deleted':
            continue
        meta = rec.find(f'{OAI}metadata')
        arx = meta.find(f'{ARX}arXiv') if meta is not None else None
        if arx is None:
            continue
        aid = _text(arx, 'id')
        if not aid:
            continue
        created = _text(arx, 'created')
        out.append({
            'id': aid,
            'categories': _text(arx, 'categories') or '',
            'title': _text(arx, 'title'),
            'abstract': _text(arx, 'abstract'),
            # same key the Kaggle snapshot uses, so the builder is unchanged
            'versions': [{'created': created}] if created else [],
            'license': _text(arx, 'license'),
            'doi': _text(arx, 'doi'),
            'journal-ref': _text(arx, 'journal-ref'),
        })
    tok_el = root.find(f'.//{OAI}resumptionToken')
    tok = tok_el.text.strip() if tok_el is not None and tok_el.text else None
    return out, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OAI_JSONL,
                    help=f'JSONL output, appended on resume (default {OAI_JSONL})')
    ap.add_argument('--max-pages', type=int, help='smoke test: stop after N pages')
    ap.add_argument('--delay', type=float, default=DELAY)
    ap.add_argument('--total', type=int, default=EST_TOTAL_RECORDS,
                    help=f'ETA denominator only (default {EST_TOTAL_RECORDS:,})')
    ap.add_argument('--restart', action='store_true', help='ignore the checkpoint')
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(out.suffix + '.state.json')
    token, written = None, 0
    if state_path.exists() and not args.restart:
        st = json.loads(state_path.read_text())
        token, written = st.get('token'), st.get('written', 0)
        print(f'resuming: {written:,} records already written, token={token}',
              file=sys.stderr)
    elif args.restart and out.exists():
        out.unlink()

    mode = 'a' if written else 'w'
    opener = (lambda: gzip.open(out, mode + 't', encoding='utf-8')) \
        if out.name.endswith('.gz') else (lambda: open(out, mode, encoding='utf-8'))
    t0, pages, start_written = time.time(), 0, written
    with opener() as fh:
        while True:
            params = ({'verb': 'ListRecords', 'resumptionToken': token} if token
                      else {'verb': 'ListRecords', 'metadataPrefix': 'arXiv'})
            label = f'[page {pages + 1}]'
            # announced BEFORE the request, because arXiv throttles by accepting the
            # connection and sending nothing: a stalled read is silent for the whole
            # timeout, and without this line that is indistinguishable from a hang.
            print(f'{label} requesting (skip={written:,})...', file=sys.stderr,
                  flush=True)
            t_req = time.time()
            recs, token = parse_page(fetch(params, label=label))
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            fh.flush()
            written += len(recs)
            pages += 1
            state_path.write_text(json.dumps({'token': token, 'written': written}))

            elapsed = time.time() - t0
            done_now = written - start_written
            rate = done_now / max(elapsed, 1e-9)
            pct = 100 * written / args.total if args.total else 0
            # Withhold the ETA until there is enough signal: over the first page or
            # two the elapsed window is near zero, which produced "eta ~7m" on a
            # 2-3 hour job. Same trap as the tar scan's pace line.
            eta = (f'~{_hms((args.total - written) / rate)}'
                   if pages >= 3 and elapsed >= 20 and rate > 0 and args.total
                   else 'settling')
            print(f'{label} +{len(recs):,} -> {written:,} recs ({pct:.1f}% of '
                  f'~{args.total / 1e6:.1f}M) | {time.time() - t_req:.1f}s/page '
                  f'{rate:.0f} rec/s | elapsed {_hms(elapsed)} eta {eta}',
                  file=sys.stderr, flush=True)
            if not token:
                print(f'harvest complete: {written:,} records in {_hms(elapsed)} '
                      f'({pages} pages)', file=sys.stderr, flush=True)
                state_path.unlink(missing_ok=True)
                break
            if args.max_pages and pages >= args.max_pages:
                print(f'stopping at --max-pages {args.max_pages}; rerun to resume',
                      file=sys.stderr)
                break
            time.sleep(args.delay)
    print(f'wrote {written:,} records -> {out}', file=sys.stderr)


if __name__ == '__main__':
    main()
