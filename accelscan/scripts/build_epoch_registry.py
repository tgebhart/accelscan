"""Build registry/generated/epoch.yaml from Epoch AI's ML-hardware table.

Source: https://epoch.ai/data/machine-learning-hardware, exported by hand to
`registry/cache/ml_hardware.csv` (176 accelerators, 2008--2026). That export is
committed, so this generator re-runs from a clean checkout and its output can be
reproduced byte-for-byte; `registry/generated/epoch.yaml` additionally carries the
`spec_source` URL on every entry and the SHA-256 of the CSV it was built from. It
covers the long tail the Wikipedia GPU lists do not: Chinese domestic accelerators (Huawei
Ascend variants, Cambricon MLU, Baidu Kunlun, Biren, MetaX, Iluvatar, Moore
Threads), first-party cloud silicon (Meta MTIA, Microsoft Maia, AWS Trainium
generations), and one-off HPC parts (Sunway, NUDT MT-3000, PEZY, Groq, Tesla
Dojo). That tail used to be hand-authored in `registry/hardware.yaml` with
release dates typed in from vendor announcements and no specs at all.

**Only models not already in the registry are emitted.** Coverage is decided by
running the registry matcher over each CSV name and requiring a `kind == 'model'`
hit, so "NVIDIA H100 NVL" is recognised as the existing `nvidia-h100` and
skipped, while "NVIDIA GB200" -- which currently resolves only to the
`vendor-nvidia` brand entry -- is not treated as covered. The exclusion registry
is loaded from `hardware.yaml` + `wikipedia.yaml` ONLY: including this script's
own previous output would make every entry look covered and empty the file on the
second run.

Epoch's FLOPS convention matches ours, which is why the columns can be used
directly: verified dense, non-sparse (H100 SXM tensor FP16 = 989.4 TFLOPS, not
the 1979 sparse figure; A100 = 312, V100 = 125), and its FP32/FP64 figures
reproduce the values this registry scrapes from Wikipedia for the same parts.

  python -m accelscan.scripts.build_epoch_registry
"""

import argparse
import hashlib
import re
import sys
import tempfile
from pathlib import Path

import polars as pl
import yaml

from accelscan.registry import load_registry
from accelscan.scripts.alias_rules import bare_code_aliases, vendor_qualified_case
from accelscan.scripts.build_registry import slugify

CSV_PATH = Path('registry/cache/ml_hardware.csv')
OUT_PATH = Path('registry/generated/epoch.yaml')
SOURCE_URL = 'https://epoch.ai/data/machine-learning-hardware'

COL = {
    'name': 'Hardware name', 'mfr': 'Manufacturer', 'type': 'Type',
    'release': 'Release date', 'tdp_w': 'TDP (W)',
    'fp32': 'FP32 (single precision) performance (FLOP/s)',
    'fp64': 'FP64 (double precision) performance (FLOP/s)',
    'tensor': 'Tensor-FP16/BF16 performance (FLOP/s)',
    'memory': 'Memory (bytes)',
}

MANUFACTURER = {
    'NVIDIA': 'nvidia', 'AMD': 'amd', 'Intel': 'intel', 'Google': 'google',
    'Huawei': 'huawei', 'Amazon AWS': 'amazon', 'Cambricon': 'cambricon',
    'Kunlunxin Baidu': 'baidu', 'MetaX': 'metax', 'Meta': 'meta',
    'Microsoft': 'microsoft', 'Biren': 'biren', 'Iluvatar CoreX': 'iluvatar',
    'Moore Threads (Tencent)': 'moore-threads', 'Tesla': 'tesla',
    'Sunway': 'sunway', 'Alibaba': 'alibaba', 'Hygon': 'hygon', 'PEZY': 'pezy',
    'National University of Defense Technology': 'nudt',
}

# Epoch's `Type` is the vendor's own marketing label and calls several NPUs
# "GPU" (every Huawei Ascend row, for instance). Vendors whose parts are
# unambiguously fixed-function ML ASICs override it; the rest take the column.
ASIC_MANUFACTURERS = {'huawei', 'cambricon', 'baidu', 'alibaba', 'meta',
                      'microsoft', 'amazon', 'tesla', 'pezy', 'unknown'}
TYPE_SUBTYPE = {'GPU': 'datacenter-gpu', 'GPGPU': 'datacenter-gpu',
                'DCU (GPGPU)': 'datacenter-gpu', 'TPU': 'tpu', 'NPU': 'npu-asic',
                'ASIC': 'npu-asic', 'XPU': 'npu-asic', 'XPU-R': 'npu-asic',
                'LPU': 'npu-asic', 'Other': 'npu-asic', 'Hybrid CPU': 'manycore'}
# supercomputer many-core parts whose Type cell is blank; 'generic-accelerator' is
# reserved for unnamed mentions and must not label a specific model
MANYCORE_MANUFACTURERS = {'sunway', 'nudt'}
# first-party silicon that is only ever rented, never sold as a card
CLOUD_MANUFACTURERS = {'amazon', 'google', 'microsoft', 'meta', 'alibaba'}

# Board-SKU tokens: Epoch lists one row per orderable board, so 'A800 SXM',
# 'A800 PCIe 40 GB' and 'A800 PCIe 80 GB' are three rows for one model. Stripped
# to a key name, they collapse into one entry the way the Wikipedia parser
# collapses its variant rows -- otherwise a paper saying "A800" would match none
# of the three, and the model's paper support would be split three ways.
# 'PCle' is Epoch's own typo in the L20/L2 rows.
FORM_FACTOR = re.compile(r'^(PCIe|PCle|SXM\d*|NVL\d*|NVLink|DGXS|OAM|FHHL'
                         r'|mezzanine|OEM)$', re.IGNORECASE)
MEM_TOKEN = re.compile(r'^(\d+\s?)?(GB|GiB)$', re.IGNORECASE)
WATT_TOKEN = re.compile(r'^\d+\s?W$', re.IGNORECASE)
# A trailing lowercase letter on a short code is also a board revision
# ('Tesla K20c' -> K20, 'K40t' -> K40), which the registry already carries.
SHORT_CODE_REV = re.compile(r'^([A-Z]{1,2}\d{2,4})[a-z]{1,2}$')

PAREN = re.compile(r'\s*\(([^)]*)\)')
# a parenthetical is a second product name only with real letters AND digits:
# keeps "(MTIA 200)", "(BI-V100)"; drops "(PCIe)", "(150W)", "(v1)", "(M100)"
PAREN_ALIAS = re.compile(r'^(?=(?:[^\d]*\d){2,})(?=(?:[^A-Za-z]*[A-Za-z]){2,})[\w\s-]+$')
GB_PER_BYTE = 1e-9
GFLOP_PER_FLOP = 1e-9


def _num(row: dict, key: str, factor: float) -> float | None:
    v = row[COL[key]]
    if v is None or v != v or v <= 0:      # None / NaN / nonpositive
        return None
    return round(float(v) * factor, 3)


def base_registry():
    """Registry from hardware.yaml + wikipedia.yaml only (see module docstring)."""
    tmp = tempfile.mkdtemp(prefix='accelscan-reg-')
    d = Path(tmp)
    (d / 'generated').mkdir()
    (d / 'hardware.yaml').symlink_to(Path('registry/hardware.yaml').resolve())
    (d / 'generated' / 'wikipedia.yaml').symlink_to(
        Path('registry/generated/wikipedia.yaml').resolve())
    return load_registry(d)


def clean_name(name: str) -> tuple[str, list[str]]:
    """-> (name without parentheticals or board tokens, aliasable parentheticals)."""
    extra = [m.strip() for m in PAREN.findall(name) if PAREN_ALIAS.match(m.strip())]
    tokens = PAREN.sub('', name).split()
    while len(tokens) > 1:
        if MEM_TOKEN.match(tokens[-1]):
            tokens.pop()
            # '80 GB' is two tokens; only a bare number *before* a memory unit is
            # a capacity. Popping trailing digits unconditionally would turn
            # 'Maia 100' into 'Maia' and merge it with 'Maia 200'.
            if len(tokens) > 1 and tokens[-1].isdigit():
                tokens.pop()
        elif FORM_FACTOR.match(tokens[-1]) or WATT_TOKEN.match(tokens[-1]):
            tokens.pop()
        else:
            break
    return ' '.join(tokens).strip(), extra


def coverage_forms(name: str) -> list[str]:
    """Name spellings to test against the existing registry before emitting.

    Epoch's board SKUs are finer-grained than our model entries, so the raw name
    alone under-reports coverage: 'NVIDIA Tesla K20c' fails the matcher's
    right-boundary check against the `K20` alias and would be emitted as a second
    K20 entry, splitting that model's paper support.
    """
    key, _ = clean_name(name)
    forms = [name, key]
    tokens = key.split()
    if tokens:
        rev = SHORT_CODE_REV.match(tokens[-1])
        if rev:
            forms.append(' '.join(tokens[:-1] + [rev.group(1)]))
    return forms


def collides(mid: str, taken: set[str]) -> bool:
    """True if `mid` is an existing id, or a hyphen-segment prefix/extension of
    one -- which is how a near-duplicate like `nvidia-rtx-pro-6000-blackwell`
    against `...-blackwell-workstation` presents."""
    return any(mid == t or mid.startswith(f'{t}-') or t.startswith(f'{mid}-')
               for t in taken)


def vendor_tokens(mfr: str) -> set[str]:
    """Every spelling of the vendor that can lead a product name.

    'Amazon AWS' appears as both 'AWS Inferentia2' and 'Amazon Trainium2', so one
    canonical spelling is not enough: without both, 'Inferentia2' never gets a
    bare alias and 'Amazon Trainium1' gets a redundant 'AWS Amazon ...' one.
    """
    out = {mfr, *mfr.split('-'), 't-head'}
    if mfr in VENDOR_DISPLAY:
        out.add(slugify(VENDOR_DISPLAY[mfr]))
    out |= VENDOR_SYNONYMS.get(mfr, set())
    return out


def strip_manufacturer(name: str, mfr: str) -> str:
    """'Huawei Ascend 910B' -> 'Ascend 910B'; 'Tesla D1 Dojo' -> 'D1 Dojo'.

    Dropping the vendor token matters most for Tesla: a bare 'Tesla' alias would
    collide with NVIDIA's retired datacenter brand, which the hand registry gates
    for exactly that reason.
    """
    tokens = name.split()
    vt = vendor_tokens(mfr)
    while tokens and slugify(tokens[0]) in vt:
        tokens.pop(0)
    return ' '.join(tokens) or name


VENDOR_DISPLAY = {'nvidia': 'NVIDIA', 'amd': 'AMD', 'google': 'Google',
                  'intel': 'Intel', 'huawei': 'Huawei', 'amazon': 'AWS',
                  'microsoft': 'Microsoft', 'meta': 'Meta', 'cambricon': 'Cambricon',
                  'baidu': 'Baidu', 'metax': 'MetaX', 'biren': 'Biren',
                  'sunway': 'Sunway', 'hygon': 'Hygon', 'alibaba': 'Alibaba',
                  'tesla': 'Tesla', 'pezy': 'PEZY', 'iluvatar': 'Iluvatar',
                  'moore-threads': 'Moore Threads', 'nudt': 'NUDT'}
VENDOR_SYNONYMS = {'amazon': {'aws'}, 'alibaba': {'t-head'},
                   'moore-threads': {'moore', 'threads'}}   # not 'mtt': it is the product line, and bare 'S4000' collides with workstation SKUs
# product-line words that sit between the vendor and the model code
BRAND_WORD = re.compile(r'^(Tesla|Radeon|Instinct|HGX|GeForce|Quadro|Ascend|Habana)$',
                        re.IGNORECASE)
# 'Gaudi2' -> also 'Gaudi 2': the vendor writes it closed, papers write it either
# way, and a single closed token never expands through the [\s-]* separator rule.
SPLIT_CODE = re.compile(r'^([A-Za-z]{3,})(\d+)$')
# Generation-agnostic names the table cannot yield, attached to the earliest
# member of the family (the same hook build_registry uses for Trillium/Ironwood).
EXTRA_ALIASES = {
    'intel-habana-gaudi-hl-205': ['Habana Gaudi',
                                  {'pattern': 'Gaudi', 'gate': 'gpu', 'case': 'sensitive'}],
}


def build_aliases(name: str, mfr: str, extra: list[str],
                  entries: dict | None = None) -> list:
    """Full name, vendor-stripped form, vendor+code form, and a gated bare code.

    The bare-code decision is `alias_rules.bare_code_aliases`, the single rule
    shared with `build_registry.py`: gated bare alias unless the code carries a
    named non-hardware meaning (`T4`, `L4`, `H20`, ...) or another entry already
    claims that string. Denied codes stay reachable through a vendor-qualified
    form ("NVIDIA L4", "Tesla T4").
    """
    out: list = []
    seen: set[str] = set()
    entries = entries if entries is not None else {}

    def add(pattern: str, gate: str | None = None):
        pattern = ' '.join(re.escape(t) for t in pattern.split())
        if pattern in seen or not pattern:
            return
        seen.add(pattern)
        case = vendor_qualified_case(pattern)
        if gate is None and case is None:
            out.append(pattern)
            return
        alias = {'pattern': pattern}
        if gate:
            alias['gate'] = gate
        if case:
            alias['case'] = case
        out.append(alias)

    def add_form(text: str):
        if len(text.split()) > 1 or len(text) >= 6:
            add(text)                       # multi-word or long: distinctive
            return
        for a in bare_code_aliases(text, entries):
            if a['pattern'] not in seen:
                seen.add(a['pattern'])
                out.append(a)

    add(name)
    vendor = VENDOR_DISPLAY.get(mfr)
    if vendor and slugify(name.split()[0]) not in vendor_tokens(mfr):
        add(f'{vendor} {name}')          # 'Maia 100' -> also 'Microsoft Maia 100'
    stripped = strip_manufacturer(name, mfr)
    if stripped != name:
        add_form(stripped)
    # 'Tesla T4' -> also 'NVIDIA T4', the spelling papers use as often
    core = ' '.join(t for t in stripped.split() if not BRAND_WORD.match(t))
    if core and core != stripped:
        add(f'{vendor or mfr.title()} {core}')
        add_form(core)
    for e in extra:
        add_form(e)
    for text in [a for a in (stripped, core) if a]:
        m = SPLIT_CODE.match(text)
        if m:
            add_form(f'{m.group(1)} {m.group(2)}')
    return out


def entry_id(name: str, mfr: str) -> str:
    slug = slugify(name)
    return slug if slug.startswith(f'{mfr}-') or mfr == 'unknown' else f'{mfr}-{slug}'


def build(csv_path: Path = CSV_PATH) -> tuple[list[dict], dict]:
    reg = base_registry()
    taken = set(reg.models)
    # alias view of the existing registry, so the shared bare-code rule can see
    # a string another entry already claims (see alias_rules.claimed_elsewhere)
    claimed: dict[str, dict] = {}
    for a in reg.aliases:
        claimed.setdefault(a.model_id, {'aliases': []})['aliases'].append(a.pattern)
    df = pl.read_csv(csv_path)
    stats = {'rows': df.height, 'covered': 0, 'no_date': 0, 'multi_variant': 0,
             'id_collision': 0, 'merged_boards': 0, 'emitted': 0}
    entries: dict[str, dict] = {}

    SPECS = (('fp32_gflops', 'fp32', GFLOP_PER_FLOP),
             ('fp64_gflops', 'fp64', GFLOP_PER_FLOP),
             ('fp16_tensor_gflops', 'tensor', GFLOP_PER_FLOP),
             ('vram_gb', 'memory', GB_PER_BYTE), ('tdp_w', 'tdp_w', 1.0))

    for row in df.iter_rows(named=True):
        raw = (row[COL['name']] or '').strip()
        if not raw:
            continue
        # gates are satisfiable here: the name itself carries the vendor token
        if any(m.kind == 'model' and (not m.gate_required or m.gate_ok)
               for form in coverage_forms(raw) for m in reg.match_paragraph(form)):
            stats['covered'] += 1
            continue
        release = (row[COL['release']] or '')[:7]
        if not re.fullmatch(r'\d{4}-\d{2}', release):
            # no release date -> no frontier lag, no vintage; same rule the
            # Wikipedia parser applies, counted rather than silently kept
            stats['no_date'] += 1
            continue
        if '/' in raw:
            # 'Cambricon MLU370-S4/S8' is a combined row whose members are also
            # listed separately; an escaped '/' alias would match nothing
            stats['multi_variant'] += 1
            continue

        mfr = MANUFACTURER.get(row[COL['mfr']], slugify(row[COL['mfr']] or 'unknown'))
        name, extra = clean_name(raw)
        mid = entry_id(name, mfr)
        if mid not in entries and collides(mid, taken):
            stats['id_collision'] += 1
            continue

        prev = entries.get(mid)
        if prev is None:
            subtype = ('manycore' if mfr in MANYCORE_MANUFACTURERS else
                       'npu-asic' if mfr in ASIC_MANUFACTURERS else
                       TYPE_SUBTYPE.get(row[COL['type']], 'npu-asic'))
            entries[mid] = prev = {
                'id': mid, 'display': name, 'manufacturer': mfr,
                'architecture': None, 'subtype': subtype,
                'segment': 'cloud' if mfr in CLOUD_MANUFACTURERS else 'datacenter',
                'release': release, 'release_source': SOURCE_URL,
                **{f: None for f, _, _ in SPECS},
                'spec_source': SOURCE_URL,
                'aliases': build_aliases(name, mfr, extra, {**claimed, **entries})
                           + EXTRA_ALIASES.get(mid, []),
                'notes': None,
            }
            stats['emitted'] += 1
        else:
            stats['merged_boards'] += 1
            prev['release'] = min(prev['release'], release)
        # top configuration across the model's boards, per the registry's
        # existing variant-merge rule
        for field, key, factor in SPECS:
            v = _num(row, key, factor)
            if v is not None:
                prev[field] = v if prev[field] is None else max(prev[field], v)

    add_families(entries)
    for e in entries.values():
        if e['vram_gb'] is not None and e['vram_gb'] < 1:
            e['notes'] = 'on-package SRAM only, no external memory'
    return sorted(entries.values(), key=lambda m: (m['release'], m['id'])), stats


# Generation-agnostic families the table has no row for. Papers write "AWS
# Trainium" or "Inferentia" with no number far more often than "Trainium2", and
# without these the mention resolves to nothing. Specs stay null: an unnumbered
# mention does not tell us which generation was used, and attributing it to the
# earliest one would be a guess dressed as data.
FAMILY_ENTRIES = {
    'amazon-trainium': {
        'display': 'AWS Trainium', 'manufacturer': 'amazon', 'subtype': 'npu-asic',
        'segment': 'cloud', 'after': 'amazon-trainium1',
        'aliases': ['AWS Trainium', {'pattern': 'Trainium', 'gate': 'gpu'},
                    {'pattern': 'Trn1', 'gate': 'gpu'}],
    },
    'amazon-inferentia': {
        'display': 'AWS Inferentia', 'manufacturer': 'amazon', 'subtype': 'npu-asic',
        'segment': 'cloud', 'after': 'amazon-aws-inferentia2',
        'aliases': ['AWS Inferentia', {'pattern': 'Inferentia', 'gate': 'gpu'},
                    {'pattern': 'Inf[12]', 'gate': 'gpu'}],
    },
}


def add_families(entries: dict) -> None:
    """Attach family entries, dated from the earliest member present."""
    for mid, spec in FAMILY_ENTRIES.items():
        members = [e for e in entries.values() if e['id'].startswith(spec['after'][:-1])]
        if not members or mid in entries:
            continue
        spec = dict(spec)
        spec.pop('after')
        entries[mid] = {
            'id': mid, **spec, 'architecture': None,
            'release': min(m['release'] for m in members),
            'release_source': SOURCE_URL,
            'fp32_gflops': None, 'fp64_gflops': None, 'fp16_tensor_gflops': None,
            'vram_gb': None, 'tdp_w': None, 'spec_source': None, 'notes': None,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(CSV_PATH))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    models, stats = build(csv_path)
    if not models:
        raise SystemExit('no models emitted -- refusing to write an empty registry '
                         'file (is the exclusion registry picking up epoch.yaml?)')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # The CSV itself is gitignored, so the committed artefact carries its digest:
    # that is what lets a re-export be checked against the input this file was
    # built from without redistributing Epoch's table.
    raw = csv_path.read_bytes()
    OUT_PATH.write_text(yaml.safe_dump(
        {'source': SOURCE_URL, 'source_file': str(csv_path),
         'source_sha256': hashlib.sha256(raw).hexdigest(),
         'source_bytes': len(raw), 'source_rows': stats['rows'],
         'models': models},
        sort_keys=False, allow_unicode=True, width=100))
    print(f'{len(models)} models -> {OUT_PATH} {stats}', file=sys.stderr)
    for f in ('fp32_gflops', 'fp64_gflops', 'fp16_tensor_gflops', 'vram_gb', 'tdp_w'):
        n = sum(m[f] is not None for m in models)
        print(f'  {f}={n} ({100 * n / len(models):.0f}%)', file=sys.stderr)


if __name__ == '__main__':
    main()
