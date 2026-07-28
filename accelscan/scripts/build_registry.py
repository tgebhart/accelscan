"""Build registry/generated/wikipedia.yaml model entries from Wikipedia tables.

One-off generator (rerun to refresh): fetches the Wikipedia GPU/accelerator
list pages, parses the spec tables, collapses per-variant rows (memory size,
PCIe/SXM form factor, editions) into canonical models, merges duplicates
across pages, and emits YAML consumed by accelscan.registry.load_registry.
Hand curation lives in registry/hardware.yaml, which overrides generated
entries by id.

Deterministic rules (documented here, enforced below):
  - Only rows with a parseable launch date >= 2000 are kept (corpus starts
    2005; undated rows are counted and dropped).
  - Mobile / IGP / console / chipset sections and Mobile/Max-Q rows are
    skipped.
  - Alias tiers: full cleaned name and brand-stripped distinctive forms are
    tier A; bare short codes (A100, K80, MI250X, GV100) are gated on the
    'gpu' context vocabulary.
  - Same canonical id from multiple pages/rows merges: earliest launch,
    union of aliases, first non-null architecture, elementwise-max specs
    (variant rows like V100 16/32GB collapse to the top configuration).
  - Spec fields are scraped from the SAME tables (data-derived, never from
    memory): fp32_gflops / fp64_gflops from the "Processing power"
    single/double sub-columns; fp16_tensor_gflops ONLY from dense tensor
    columns ("Tensor ... Dense", "Tensor Core ... Accumulate") — sparse
    columns are explicitly excluded per the paper's FLOPS convention;
    vram_gb from "Memory ... Size"; tdp_w from "TDP". Units normalized
    (TFLOPS -> GFLOPS, MB -> GB). Multi-number cells (base/boost, memory
    configs) take the max (boost clock / top config). spec_source records
    the page + table heading per entry.

Usage: python -m accelscan.scripts.build_registry [--no-cache]
"""

import argparse
import datetime
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

CACHE_DIR = Path('registry/cache')
OUT_PATH = Path('registry/generated/wikipedia.yaml')
UA = {'User-Agent': 'accelscan-registry-builder/0.1 (research; contact tom@amerton.org)'}

SOURCES = {
    'nvidia': ('nvidia', 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units'),
    'amd_instinct': ('amd', 'https://en.wikipedia.org/wiki/AMD_Instinct'),
    'amd': ('amd', 'https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units'),
    'tpu': ('google', 'https://en.wikipedia.org/wiki/Tensor_Processing_Unit'),
}

SKIP_HEADING = re.compile(
    r'Mobility|\bGo\b|\dM\b|M series|MX series|IGP|NVS|Console|GRID|Chipset'
    r'|All-in-Wonder|Rage|Mach series|Wonder series|Quadro Plex|laptop',
    re.IGNORECASE)
SKIP_MODEL = re.compile(r'\b(Mobile|Laptop|Max-?Q|Embedded|Deskside)\b', re.IGNORECASE)

MONTHS = ('Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?'
          '|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?')
DATE_FULL = re.compile(rf'({MONTHS})\.?\s+\d{{1,2}},?\s+(\d{{4}})')
DATE_MONTH = re.compile(rf'({MONTHS})\.?\s+(\d{{4}})')
DATE_QUARTER = re.compile(r'Q([1-4])[\s,]*(\d{4})')
DATE_YEAR = re.compile(r'\b(19|20)\d{2}\b')
MONTH_NUM = {m: i + 1 for i, m in enumerate(
    'jan feb mar apr may jun jul aug sep oct nov dec'.split())}

FOOTNOTE = re.compile(r'\[[^\]]*\]')
PAREN = re.compile(r'\([^)]*\)')
QUOTED = re.compile(r'["“][^"”]*["”]')
DC_SUFFIX = re.compile(
    r'\b(GPU accelerator|GPU Computing (Module|Server)|Visual Computing System'
    r'|GPU Computing)\b', re.IGNORECASE)
VARIANT_TOKEN = re.compile(
    r'^(PCIe|SXM\d*|NVL|NVLink|FHHL|HGX|mezzanine|OEM|rev\.?|\d+\s?GB|GB|\d+G'
    r'|\d{3,4}SP|Eyefinity|Edition|Founders)$', re.IGNORECASE)
BRAND_TOKENS = {'amd', 'ati', 'nvidia', 'instinct'}
SHORT_CODE = re.compile(r'^[A-Z]{1,2}I?\d{2,4}[A-Za-z]{0,2}$')
DISTINCTIVE_PREFIX = re.compile(
    r'^(GTX|RTX|GTS?|RX|HD|R[579]|FX|MI|TITAN|Titan|Tesla|Quadro|Radeon|GeForce|Arc|Vega)\b')
AMD_ARCH = re.compile(r'(GCN|RDNA|CDNA|TeraScale|Vega)\s*(\d(?:\.\d)?)?', re.IGNORECASE)

NVIDIA_SERIES_ARCH = [
    (re.compile(r'GeForce (8|9|100|200|300)\b'), 'tesla-g80'),
    (re.compile(r'GeForce (400|500)\b'), 'fermi'),
    (re.compile(r'GeForce (600|700)\b'), 'kepler'),
    (re.compile(r'GeForce 900\b'), 'maxwell'),
    (re.compile(r'GeForce 10 series'), 'pascal'),
    (re.compile(r'Volta'), 'volta'),
    (re.compile(r'GeForce (GTX )?16 series'), 'turing'),
    (re.compile(r'RTX 20 series'), 'turing'),
    (re.compile(r'RTX 30 series'), 'ampere'),
    (re.compile(r'RTX 40 series'), 'ada'),
    (re.compile(r'RTX 50 series'), 'blackwell'),
    (re.compile(r'Quadro K'), 'kepler'),
    (re.compile(r'Quadro M'), 'maxwell'),
    (re.compile(r'Quadro P'), 'pascal'),
    (re.compile(r'Quadro GV'), 'volta'),
    (re.compile(r'Quadro RTX'), 'turing'),
    (re.compile(r'RTX Ax000'), 'ampere'),
    (re.compile(r'RTX Ada'), 'ada'),
    (re.compile(r'RTX PRO Blackwell'), 'blackwell'),
]

TPU_EXTRA_ALIASES = {'v6e': [{'pattern': 'Trillium', 'gate': 'tpu', 'case': 'sensitive'}],
                     'v7': [{'pattern': 'Ironwood', 'gate': 'tpu', 'case': 'sensitive'}]}


def fetch(name: str, url: str, use_cache: bool = True) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f'{name}.html'
    if use_cache and cache.exists():
        return cache.read_text()
    html = requests.get(url, headers=UA, timeout=60).text
    cache.write_text(html)
    return html


def parse_release(cell: str) -> str | None:
    cell = FOOTNOTE.sub(' ', str(cell))
    m = DATE_FULL.search(cell) or DATE_MONTH.search(cell)
    if m:
        return f'{m.group(2)}-{MONTH_NUM[m.group(1)[:3].lower()]:02d}'
    m = DATE_QUARTER.search(cell)
    if m:
        return f'{m.group(2)}-{int(m.group(1)) * 3 - 2:02d}'
    m = DATE_YEAR.search(cell)
    if m:
        return m.group(0)
    return None


def release_key(release: str) -> tuple[int, int]:
    parts = release.split('-')
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 13


def merge_release(a: str, b: str) -> str:
    ka, kb = release_key(a), release_key(b)
    if ka[0] != kb[0]:
        return a if ka[0] < kb[0] else b
    # same year: prefer the month-precise one; both precise -> earlier month
    if ka[1] == 13:
        return b
    if kb[1] == 13:
        return a
    return a if ka[1] <= kb[1] else b


def clean_model_name(raw: str) -> str:
    s = FOOTNOTE.sub(' ', str(raw))
    s = QUOTED.sub(' ', s)
    s = PAREN.sub(' ', s)
    s = DC_SUFFIX.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip(' –-')
    # 'Radeon' is a meaningful prefix except in 'Radeon Instinct <code>' names,
    # which must merge with the dedicated Instinct page's bare codes.
    s = re.sub(r'^Radeon Instinct\b', 'Instinct', s, flags=re.IGNORECASE)
    tokens = s.split(' ')
    while tokens and (tokens[0].lower() in BRAND_TOKENS or re.match(r'^\d+×$', tokens[0])):
        tokens.pop(0)
    while tokens and VARIANT_TOKEN.match(tokens[-1]):
        tokens.pop()
    return ' '.join(tokens).strip()


def slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def escape_name(name: str) -> str:
    # regex-escape everything except spaces (spaces are separator syntax)
    return ' '.join(re.escape(tok) for tok in name.split(' '))


def build_aliases(name: str, manufacturer: str, release: str) -> list:
    aliases: list = []
    seen: set[str] = set()

    def add(pattern: str, gate: str | None = None):
        if pattern in seen:
            return
        seen.add(pattern)
        aliases.append({'pattern': pattern, 'gate': gate} if gate else pattern)

    if SHORT_CODE.match(name):
        # bare data-center style code: A100, K80, MI250X ...
        if manufacturer == 'nvidia':
            add(f'NVIDIA {escape_name(name)}')
            if release < '2021':  # Tesla branding retired after Ampere
                add(f'Tesla {escape_name(name)}')
        elif manufacturer == 'amd':
            add(f'Instinct {escape_name(name)}')
            add(f'AMD {escape_name(name)}')
        add(escape_name(name), gate='gpu')
        return aliases

    if len(name) >= 6 and (re.search(r'\d', name) or 'titan' in name.lower()):
        add(escape_name(name))

    tokens = name.split(' ')
    if tokens[0] in ('GeForce', 'Radeon') and len(tokens) > 1:
        stripped = ' '.join(tokens[1:])
        if DISTINCTIVE_PREFIX.match(stripped) and re.search(r'\d', stripped):
            add(escape_name(stripped))
        elif SHORT_CODE.match(stripped):
            add(escape_name(stripped), gate='gpu')

    # gated bare code for the trailing token of workstation/prefixed names:
    # 'Quadro GV100' -> GV100, 'RTX A6000' -> A6000
    if len(tokens) > 1 and DISTINCTIVE_PREFIX.match(name) and SHORT_CODE.match(tokens[-1]):
        add(re.escape(tokens[-1]), gate='gpu')

    if not aliases:
        add(escape_name(name), gate='gpu')
    return aliases


SPEC_FIELDS = ('fp32_gflops', 'fp64_gflops', 'fp16_tensor_gflops', 'vram_gb', 'tdp_w')
SPEC_NULL = re.compile(r'^(nan|unknown|\?|—|–|-|n/a|tba|tbd)$', re.IGNORECASE)


def sanitize_table_html(table) -> str:
    """Wikipedia occasionally emits malformed colspan/rowspan (e.g. `2' `),
    which makes pandas.read_html raise and would silently drop a whole table
    (this hid the entire RTX 40 series). Coerce them to clean integers."""
    html = str(table)
    def fix(m):
        digits = re.sub(r'\D', '', m.group(2)) or '1'
        return f'{m.group(1)}="{digits}"'
    return re.sub(r'\b(colspan|rowspan)\s*=\s*"([^"]*)"', fix, html)


def classify_spec_cols(cols: list[str]) -> dict[str, tuple[int, float]]:
    """Map spec field -> (column index, unit factor). Dense-only tensor rule.

    Handles NVIDIA ('Processing Power ... Single/Double/Tensor') and AMD
    ('Processing power ... Vector/Matrix', 'Vector TFLOPS') namings."""
    spec: dict[str, tuple[int, float]] = {}
    for i, c in enumerate(cols):
        lc = c.lower()
        is_perf = 'processing power' in lc or 'tflops' in lc or 'gflops' in lc
        if is_perf:
            unit = 1000.0 if 'tflops' in lc else 1.0
            if 'sparse' in lc or 'ray tracing' in lc or 'speedup' in lc:
                continue
            # Tensor/matrix MUST be tested first: the data-center table names
            # its tensor column "Half precision Tensor Core FP32 Accumulate",
            # which would otherwise be captured as FP32 (it is not — that is
            # the accumulate type, and the throughput is ~8x true FP32).
            if 'tensor' in lc or 'matrix' in lc:
                dense_ok = ('dense' in lc or 'tensor core' in lc
                            or re.search(r'fp16|bf16|half', lc))
                if dense_ok and 'fp16_tensor_gflops' not in spec:
                    spec['fp16_tensor_gflops'] = (i, unit)
                continue
            # AMD lists FP32/FP64 as "vector"; NVIDIA as single/double
            if re.search(r'\b(single|fp32)\b', lc) and 'fp32_gflops' not in spec:
                spec['fp32_gflops'] = (i, unit)
            elif re.search(r'\b(double|fp64)\b', lc) and 'fp64_gflops' not in spec:
                spec['fp64_gflops'] = (i, unit)
            elif ('vector' in lc and 'fp32_gflops' not in spec
                  and not re.search(r'fp16|bf16|int', lc)):
                spec['fp32_gflops'] = (i, unit)   # bare "Vector TFLOPS" -> FP32
        elif 'memory' in lc and 'size' in lc and 'vram_gb' not in spec:
            unit = 1 / 1024 if '(mb' in lc or '(mib' in lc else 1.0
            spec['vram_gb'] = (i, unit)
        elif 'tdp' in lc and 'tdp_w' not in spec:
            spec['tdp_w'] = (i, 1.0)
    return spec


def parse_spec_value(cell, factor: float) -> float | None:
    s = FOOTNOTE.sub(' ', str(cell)).replace(',', '').replace('\xa0', ' ').strip()
    if not s or SPEC_NULL.match(s):
        return None
    nums = re.findall(r'\d+(?:\.\d+)?', s)
    if not nums:
        return None
    return round(max(float(x) for x in nums) * factor, 3)


def flatten_columns(df: pd.DataFrame) -> list[str]:
    if isinstance(df.columns, pd.MultiIndex):
        return [' '.join(str(p) for p in tup if 'Unnamed' not in str(p)).strip()
                for tup in df.columns]
    return [str(c) for c in df.columns]


def find_col(cols: list[str], *needles: str) -> int | None:
    for needle in needles:
        for i, c in enumerate(cols):
            if needle in c.lower():
                return i
    return None


def iter_tables(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    for table in soup.find_all('table', class_='wikitable'):
        h = table.find_previous(['h3', 'h4'])
        h2 = table.find_previous('h2')
        heading = h.get_text(' ', strip=True) if h else ''
        section = h2.get_text(' ', strip=True) if h2 else ''
        yield table, heading, section


def parse_arch(raw: str, manufacturer: str) -> str | None:
    raw = FOOTNOTE.sub('', raw)
    raw = re.split(r'[&(\n]', raw)[0].strip()
    if not raw or raw.lower() == 'nan':
        return None
    if manufacturer == 'amd':
        m = AMD_ARCH.search(raw)
        if not m:
            return None
        return slugify(f'{m.group(1)} {m.group(2) or ""}'.strip())
    return slugify(raw)


def parse_gpu_page(html: str, manufacturer: str, url: str, entries: dict, stats: dict) -> None:
    for table, heading, section in iter_tables(html):
        if SKIP_HEADING.search(heading) or SKIP_HEADING.search(section):
            stats['skipped_tables'] += 1
            continue
        try:
            df = pd.read_html(io.StringIO(sanitize_table_html(table)))[0]
        except Exception as e:
            # loud, not silent: a dropped table = missing models (see RTX 40)
            print(f'[build_registry] UNPARSEABLE table under {heading!r}: {e}',
                  file=sys.stderr)
            stats['unparseable_tables'] += 1
            continue
        cols = flatten_columns(df)
        model_i = find_col(cols, 'model', 'accelerator')
        launch_i = find_col(cols, 'launch', 'release')
        if model_i is None or launch_i is None:
            stats['skipped_tables'] += 1
            continue
        arch_i = find_col(cols, 'microarchitecture', 'architecture')

        is_datacenter = ('data center' in section.lower()
                         or 'instinct' in url.lower())
        is_workstation = bool(re.search(r'Quadro|RTX [AP]|RTX Ada|RTX PRO', heading))
        segment = ('datacenter' if is_datacenter
                   else 'workstation' if is_workstation else 'consumer')
        subtype = ('datacenter-gpu' if is_datacenter
                   else 'workstation-gpu' if is_workstation else 'consumer-gpu')

        spec_cols = classify_spec_cols(cols)
        spec_src = f'{url}#{heading}' if heading else url

        arch_from_heading = None
        if manufacturer == 'nvidia':
            for pat, arch in NVIDIA_SERIES_ARCH:
                if pat.search(heading):
                    arch_from_heading = arch
                    break

        for _, row in df.iterrows():
            stats['rows'] += 1
            raw_name = str(row.iloc[model_i])
            if SKIP_MODEL.search(raw_name):
                stats['skipped_models'] += 1
                continue
            name = clean_model_name(raw_name)
            if (not name or name.lower().startswith('model') or len(name) < 3
                    or not re.search(r'[A-Za-z]', name) or '×' in name):
                continue
            release = parse_release(row.iloc[launch_i])
            if release is None:
                stats['no_date'] += 1
                continue
            if release_key(release)[0] < 2000:
                stats['pre2000'] += 1
                continue
            arch = arch_from_heading
            if arch is None and arch_i is not None:
                arch = parse_arch(str(row.iloc[arch_i]), manufacturer)

            specs = {f: parse_spec_value(row.iloc[i], factor)
                     for f, (i, factor) in spec_cols.items()}
            specs = {f: v for f, v in specs.items() if v is not None}

            brand = 'NVIDIA' if manufacturer == 'nvidia' else 'AMD'
            display = f'{brand} {name}'
            mid = f'{manufacturer}-{slugify(name)}'
            prev = entries.get(mid)
            if prev is None:
                entries[mid] = {
                    'id': mid, 'display': display, 'manufacturer': manufacturer,
                    'family': slugify(heading) or None, 'architecture': arch,
                    'subtype': subtype, 'segment': segment, 'release': release,
                    'release_source': url,
                    **{f: specs.get(f) for f in SPEC_FIELDS},
                    'spec_source': spec_src if specs else None,
                    'aliases': build_aliases(name, manufacturer, release),
                }
            else:
                prev['release'] = merge_release(prev['release'], release)
                if prev['architecture'] is None:
                    prev['architecture'] = arch
                # variant rows (V100 16/32GB, PCIe/SXM) collapse to top config
                for f, v in specs.items():
                    prev[f] = v if prev.get(f) is None else max(prev[f], v)
                if specs and not prev.get('spec_source'):
                    prev['spec_source'] = spec_src
                have = {yaml.safe_dump(a) for a in prev['aliases']}
                for a in build_aliases(name, manufacturer, release):
                    if yaml.safe_dump(a) not in have:
                        prev['aliases'].append(a)


def parse_tpu_page(html: str, url: str, entries: dict) -> None:
    """TPU table is transposed (rows = attributes, cols = generations)."""
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', class_='wikitable')
    rows = table.find_all('tr')
    versions = [c.get_text(' ', strip=True) for c in rows[0].find_all(['th', 'td'])][1:]

    def row_values(needle: str) -> list[str]:
        for r in rows:
            cells = r.find_all(['th', 'td'])
            if cells and needle in cells[0].get_text(' ', strip=True).lower():
                return [c.get_text(' ', strip=True) for c in cells][1:]
        return []

    dates = row_values('introduced')
    mem = row_values('memory')                     # 'Memory' row: "16 GiB HBM"
    tdp = row_values('thermal design power')
    perf = row_values('computational performance')  # trillion ops/s (bf16 for v4+)

    def at(vals: list[str], i: int) -> str | None:
        return vals[i] if i < len(vals) else None

    for i, ver_raw in enumerate(versions):
        m = re.match(r'(v\d+\w*)', FOOTNOTE.sub('', ver_raw).strip())
        release = parse_release(at(dates, i) or '')
        if not m or release is None:
            continue
        ver = m.group(1)
        aliases: list = [f'TPU {ver}', f'TPU{ver}', f'Cloud TPU {ver}']
        aliases += TPU_EXTRA_ALIASES.get(ver, [])
        # TPUs report integer/bf16 ops, not FP32/FP64 — populate only the
        # tensor axis and memory/TDP; fp32/fp64 stay null (correctly excluded).
        tops = parse_spec_value(at(perf, i) or '', 1000.0)  # tera-ops -> giga-ops
        entries[f'google-tpu-{ver}'] = {
            'id': f'google-tpu-{ver}', 'display': f'Google TPU {ver}',
            'manufacturer': 'google', 'family': 'tpu', 'architecture': f'tpu-{ver}',
            'subtype': 'tpu', 'segment': 'cloud', 'release': release,
            'release_source': url,
            'fp32_gflops': None, 'fp64_gflops': None,
            'fp16_tensor_gflops': tops,
            'vram_gb': parse_spec_value(at(mem, i) or '', 1.0),
            'tdp_w': parse_spec_value(at(tdp, i) or '', 1.0),
            'spec_source': url,
            'aliases': aliases,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    stats = {'rows': 0, 'no_date': 0, 'pre2000': 0, 'skipped_tables': 0,
             'skipped_models': 0, 'unparseable_tables': 0}
    for name, (manufacturer, url) in SOURCES.items():
        html = fetch(name, url, use_cache=not args.no_cache)
        if name == 'tpu':
            parse_tpu_page(html, url, entries)
        else:
            parse_gpu_page(html, manufacturer, url, entries, stats)

    models = sorted(entries.values(), key=lambda m: (m['release'], m['id']))
    out = {'sources': {k: v[1] for k, v in SOURCES.items()},
           'retrieved': datetime.date.today().isoformat(),
           'models': models}
    OUT_PATH.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=100))
    print(f'{len(models)} models -> {OUT_PATH} {stats}', file=sys.stderr)
    cov = {f: sum(m.get(f) is not None for m in models) for f in SPEC_FIELDS}
    print('spec coverage: ' + ' | '.join(f'{f}={n} ({100*n/len(models):.0f}%)'
                                         for f, n in cov.items()), file=sys.stderr)


if __name__ == '__main__':
    main()
