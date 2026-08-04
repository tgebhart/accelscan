"""External validation of candidate generation against SH-NER (Anjum et al. 2025).

The pipeline's own validity checks are structural (pre-release rate, offset
integrity, converter drift by era); none of them measures whether the registry
finds the hardware a human annotator would mark. SH-NER is an independent,
manually annotated span-level corpus of infrastructure entities in NLP papers
(5,287 train / 663 test sentences, five entity types), so running our matcher
over it converts the "candidate generation is a lower bound" caveat into a
number we did not produce.

What is measured, and what it is not: SH-NER labels `Hardware-device` spans in
single sentences, while our unit is a *paragraph* that generates a candidate
passage. So sentence-level recall here is a conservative reading of the pipeline
-- the real scan sees the paragraph plus its neighbours (a wider gate window) and
applies a paper-level rescue that this script reproduces only within a paper's
annotated sentences. Precision is deliberately NOT reported against these labels:
SH-NER's `Hardware-device` excludes CPUs and FPGAs, which our registry matches on
purpose and hands to the LLM, so every such match would count as a false positive
against a definition we do not share.

Misses are bucketed rather than counted, using the same
`normalize.classify_unresolved` vocabulary the unresolved audit uses, which
separates "out of our scope by decision" (CPU, FPGA, embedded module) from a
genuine registry gap.

    python -m accelscan.scripts.eval_shner --data data/shner/Test.json

Get the data (CC-BY, ~300 KB; `data/` is gitignored):
    mkdir -p data/shner && curl -sL -o data/shner/Test.json \\
      https://raw.githubusercontent.com/coderhub84/SH-NER/HEAD/data/Test.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from accelscan.normalize import classify_unresolved
from accelscan.registry import load_registry

DEFAULT_DATA = Path('data/shner/Test.json')
# SH-NER's label for a physical accelerator/processor. Its casing is inconsistent
# in the released files ('Hardware-device' and 'Hardware-Device' both occur).
HARDWARE = 'hardware-device'


def load_sentences(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        r['paper_id'] = (r.get('Comments') or {}).get('paper_id', '?')
        r['gold'] = [(s, e) for s, e, lab in r.get('label', [])
                     if lab.lower() == HARDWARE]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATA))
    ap.add_argument('--show-misses', type=int, default=25)
    args = ap.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f'{path} not found -- see this module\'s docstring for the '
                         f'one-line download.')

    reg = load_registry()
    rows = load_sentences(path)

    # matches per sentence, and the papers carrying an ungated model hit anywhere
    # (the paper-level rescue the scan applies, restricted to annotated sentences)
    matches, anchored = {}, set()
    for i, r in enumerate(rows):
        ms = reg.match_paragraph(r['text'])
        matches[i] = ms
        if any(m.kind == 'model' and not m.gate_required for m in ms):
            anchored.add(r['paper_id'])

    def kept(i: int, rescue: bool) -> list:
        r = rows[i]
        return [m for m in matches[i]
                if not m.gate_required or m.gate_ok
                or (rescue and r['paper_id'] in anchored)]

    total = sum(len(r['gold']) for r in rows)
    hit = {False: 0, True: 0}
    model_hit = 0
    missed: Counter = Counter()
    buckets: Counter = Counter()
    sent_with_gold = sent_candidate = 0

    for i, r in enumerate(rows):
        if not r['gold']:
            continue
        sent_with_gold += 1
        ks = kept(i, rescue=True)
        if ks:
            sent_candidate += 1
        for (gs, ge) in r['gold']:
            surface = r['text'][gs:ge].strip()
            over = {rescue: [m for m in kept(i, rescue)
                             if m.start < ge and gs < m.end]
                    for rescue in (False, True)}
            for rescue in (False, True):
                hit[rescue] += bool(over[rescue])
            if any(m.kind == 'model' for m in over[True]):
                model_hit += 1
            elif not over[True]:
                missed[surface] += 1
                buckets[classify_unresolved(surface)] += 1

    papers = {r['paper_id'] for r in rows}
    print(f'SH-NER {path.name}: {len(rows)} sentences, {total} Hardware-device spans '
          f'in {len(papers)} papers', file=sys.stderr)
    if len(papers) < 2:
        print('WARNING: this split carries no usable paper_id, so the paper-level '
              'rescue below degrades to a corpus-level one and overstates recall. '
              'Read the local-context line instead.', file=sys.stderr)
    print(f'\nspan recall (any registry match overlapping the gold span)')
    print(f'  local context only : {hit[False]:5d} / {total}  '
          f'({100 * hit[False] / total:.1f}%)')
    print(f'  + paper rescue     : {hit[True]:5d} / {total}  '
          f'({100 * hit[True] / total:.1f}%)')
    print(f'  resolved to a model entry (not generic/brand only): '
          f'{model_hit} ({100 * model_hit / total:.1f}%)')
    print(f'\ncandidate generation: {sent_candidate} / {sent_with_gold} sentences '
          f'carrying a gold span would generate a passage '
          f'({100 * sent_candidate / max(sent_with_gold, 1):.1f}%)')
    print('\nmisses by bucket (classify_unresolved):')
    for b, n in buckets.most_common():
        print(f'  {b:28s} {n:5d}  ({100 * n / total:.1f}% of spans)')
    print(f'\ntop {args.show_misses} missed surfaces:')
    for s, n in missed.most_common(args.show_misses):
        print(f'  {n:4d}  {s[:70]!r}  [{classify_unresolved(s)}]')


if __name__ == '__main__':
    main()
