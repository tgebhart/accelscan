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
from collections import Counter, defaultdict
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
    'apple': ('apple', 'https://en.wikipedia.org/wiki/Apple_silicon'),
    'xeon_phi': ('intel', 'https://en.wikipedia.org/wiki/Xeon_Phi'),
    'intel_arc': ('intel', 'https://en.wikipedia.org/wiki/Intel_Arc'),
}

SKIP_HEADING = re.compile(
    r'Mobility|\bGo\b|\dM\b|M series|MX series|IGP|NVS|Console|GRID|Chipset'
    r'|All-in-Wonder|Rage|Mach series|Wonder series|Quadro Plex|laptop',
    re.IGNORECASE)
SKIP_MODEL = re.compile(r'\b(Mobile|Laptop|Max-?Q|Embedded|Deskside)\b', re.IGNORECASE)

MONTHS = ('Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?'
          '|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?')
DATE_FULL = re.compile(rf'({MONTHS})\.?\s+\d{{1,2}},?\s+(\d{{4}})')
DATE_MONTH = re.compile(rf'({MONTHS})\.?,?\s+(\d{{4}})')
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
    r'|\d{3,4}SP|Eyefinity|Edition|Founders|Workstation)$', re.IGNORECASE)
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

# Two cards were marketed as TITAN X: the 2015 Maxwell one (listed as GeForce GTX
# TITAN X) and the 2016 Pascal one. Papers disambiguate with the architecture, and
# the Pascal entry's generated alias already has that shape ('TITAN X Pascal'),
# so the Maxwell one needs the mirror. Bare 'TITAN X' stays unresolved on purpose:
# it is genuinely ambiguous between the two.
EXTRA_MODEL_ALIASES = {'nvidia-geforce-gtx-titan-x': ['TITAN X Maxwell']}


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
        # TITAN names carry no digit, so the digit test alone left 'GTX TITAN X'
        # and 'GTX TITAN Black' reachable only by their full GeForce name
        if DISTINCTIVE_PREFIX.match(stripped) and (re.search(r'\d', stripped)
                                                   or 'titan' in stripped.lower()):
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


XEON_PHI_NAME = re.compile(r'^Xeon\s+Phi\s+([0-9][0-9A-Za-z]{2,5})$')
# trailing 'M' is a mobile-workstation part (A30M, A60M): out of scope
ARC_CODE = re.compile(r'^([AB]\d{2,3}[A-LN-Za-z]?)$')
ARC_HEADINGS = ('desktop', 'workstation')          # per the user: not Mobile
# Family-level entries: most papers write "Intel Xeon Phi" or "Intel Arc" with no
# model number, so a per-model-only registry would drop them. Specs stay null --
# the family's top configuration is not what an unnamed mention reports.
FAMILY_ENTRIES = {
    'intel-xeon-phi': {
        'display': 'Intel Xeon Phi', 'manufacturer': 'intel', 'family': 'xeon-phi',
        'architecture': 'mic', 'subtype': 'manycore', 'segment': 'datacenter',
        'aliases': ['Xeon Phi', 'Knights Corner', 'Knights Landing', 'Knights Mill',
                    {'pattern': 'KNL', 'gate': 'gpu'}],
    },
    'intel-arc': {
        'display': 'Intel Arc', 'manufacturer': 'intel', 'family': 'arc',
        'architecture': 'xe', 'subtype': 'consumer-gpu', 'segment': 'consumer',
        # bare 'Arc' is arc length, electric arc, arc minutes: vendor-qualified only
        'aliases': ['Intel Arc'],
    },
}


def bare_code_alias(code: str, entries: dict) -> list:
    """Gated bare-code alias, unless another vendor already claims that code.

    Intel's Arc Pro A40 and NVIDIA's A40 are different GPUs with the same short
    code, and the NVIDIA part carries ~2,600 papers. Two entries claiming one
    span makes `canonicalize_one`'s tie-break arbitrary, so the newer, rarer
    claimant gets no bare alias and stays reachable as 'Arc Pro A40'.
    """
    for e in entries.values():
        for a in e.get('aliases', []):
            if (a if isinstance(a, str) else a['pattern']) == re.escape(code):
                return []
    return [{'pattern': re.escape(code), 'gate': 'gpu'}]


def _family_entry(mid: str, release: str, url: str) -> dict:
    return {'id': mid, **FAMILY_ENTRIES[mid], 'release': release,
            'release_source': url, 'kind': 'model',
            **{f: None for f in SPEC_FIELDS}, 'spec_source': None}


def parse_xeon_phi_page(html: str, url: str, entries: dict, stats: dict) -> None:
    """Xeon Phi coprocessors and many-core processors from the Models tables.

    The Knights Corner table reports `Peak DP compute (GFLOPS)` directly, which is
    the only FLOPS figure Intel published for these parts: they have no FP32 or
    tensor number, so those axes stay null rather than being back-computed.
    On-package memory (GDDR5 or MCDRAM) is the device memory.
    """
    for table, heading, section in iter_tables(html):
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
        except ValueError:
            stats['unparseable_tables'] += 1
            continue
        cols = flatten_columns(df)
        i_name = 0
        i_dp = find_col(cols, 'peak dp compute')
        i_mem = find_col(cols, 'mcdram memory quantity', 'gddr5 ecc memory quantity')
        i_tdp = find_col(cols, 'tdp')
        i_rel = find_col(cols, 'released', 'release date')
        if i_dp is None or i_rel is None:
            stats['skipped_tables'] += 1
            continue
        for row in df.itertuples(index=False):
            cells = list(row)
            name = ' '.join(FOOTNOTE.sub(' ', str(cells[i_name])).split())
            m = XEON_PHI_NAME.match(name)
            if not m:
                stats['skipped_models'] += 1
                continue
            release = parse_release(str(cells[i_rel]))
            if release is None:
                stats['no_date'] += 1
                continue
            stats['rows'] += 1
            code = m.group(1)
            mid = f'intel-xeon-phi-{code.lower()}'
            specs = {
                'fp64_gflops': parse_spec_value(cells[i_dp], 1.0),
                'vram_gb': parse_spec_value(cells[i_mem], 1.0) if i_mem is not None else None,
                'tdp_w': parse_spec_value(cells[i_tdp], 1.0) if i_tdp is not None else None,
            }
            prev = entries.get(mid)
            if prev is None:
                entries[mid] = {
                    'id': mid, 'display': f'Intel Xeon Phi {code}',
                    'manufacturer': 'intel', 'family': 'xeon-phi',
                    'architecture': 'mic', 'subtype': 'manycore',
                    'segment': 'datacenter', 'release': release,
                    'release_source': url,
                    'fp32_gflops': None, 'fp16_tensor_gflops': None, **specs,
                    'spec_source': f'{url}#{heading or section}',
                    # '7120P' is distinctive but numeric; gate it like other bare codes
                    'aliases': [f'Xeon Phi {code}',
                                *bare_code_alias(code, entries)],
                }
            else:
                prev['release'] = min(prev['release'], release)
                for f, v in specs.items():
                    if v is not None:
                        prev[f] = v if prev[f] is None else max(prev[f], v)
        fam = [e for e in entries.values() if e.get('family') == 'xeon-phi']
        if fam and 'intel-xeon-phi' not in entries:
            entries['intel-xeon-phi'] = _family_entry(
                'intel-xeon-phi', min(e['release'] for e in fam), url)


def parse_intel_arc_page(html: str, url: str, entries: dict, stats: dict) -> None:
    """Intel Arc desktop and workstation cards.

    Only the XMX matrix-engine column may feed the tensor axis; the plain
    'Half precision' column is vector FP16 and would inflate it, the same
    conflation the data-center-table bug produced for NVIDIA.
    """
    for table, heading, section in iter_tables(html):
        if not any(h in f'{heading} {section}'.lower() for h in ARC_HEADINGS):
            stats['skipped_tables'] += 1
            continue
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
        except ValueError:
            stats['unparseable_tables'] += 1
            continue
        cols = flatten_columns(df)
        i_brand, i_code = 0, 1
        i_rel = find_col(cols, 'launch')
        i_fp32 = find_col(cols, 'single precision')
        i_fp64 = find_col(cols, 'double precision')
        i_xmx = find_col(cols, 'xmx half precision')
        i_mem = find_col(cols, 'memory size')
        i_tdp = find_col(cols, 'tdp')
        if i_rel is None or i_fp32 is None:
            stats['skipped_tables'] += 1
            continue
        for row in df.itertuples(index=False):
            cells = list(row)
            code = ' '.join(FOOTNOTE.sub(' ', str(cells[i_code])).split())
            code = clean_model_name(code)            # 'A770 16GB' -> 'A770'
            if not ARC_CODE.match(code):
                stats['skipped_models'] += 1
                continue
            release = parse_release(str(cells[i_rel]))
            if release is None:
                stats['no_date'] += 1
                continue
            stats['rows'] += 1
            pro = 'pro' in str(cells[i_brand]).lower()
            line = 'Arc Pro' if pro else 'Arc'
            mid = f'intel-arc-{"pro-" if pro else ""}{code.lower()}'
            specs = {
                'fp32_gflops': parse_spec_value(cells[i_fp32], 1000.0),
                'fp64_gflops': parse_spec_value(cells[i_fp64], 1000.0) if i_fp64 is not None else None,
                'fp16_tensor_gflops': parse_spec_value(cells[i_xmx], 1000.0) if i_xmx is not None else None,
                'vram_gb': parse_spec_value(cells[i_mem], 1.0) if i_mem is not None else None,
                'tdp_w': parse_spec_value(cells[i_tdp], 1.0) if i_tdp is not None else None,
            }
            prev = entries.get(mid)
            if prev is None:
                entries[mid] = {
                    'id': mid, 'display': f'Intel {line} {code}',
                    'manufacturer': 'intel', 'family': 'arc',
                    'architecture': 'xe', 'subtype': 'workstation-gpu' if pro
                    else 'consumer-gpu',
                    'segment': 'workstation' if pro else 'consumer',
                    'release': release, 'release_source': url, **specs,
                    'spec_source': f'{url}#{heading or section}',
                    # 'Intel Arc A770' must be listed explicitly: the family
                    # alias 'Intel Arc' is the longer span against a bare
                    # 'Arc A770', and longest-span resolution would hand the
                    # mention to the family entry.
                    'aliases': [f'Intel {line} {code}', f'{line} {code}',
                                *bare_code_alias(code, entries)],
                }
            else:
                prev['release'] = min(prev['release'], release)
                for f, v in specs.items():
                    if v is not None:
                        prev[f] = v if prev[f] is None else max(prev[f], v)
        fam = [e for e in entries.values() if e.get('family') == 'arc']
        if fam and 'intel-arc' not in entries:
            entries['intel-arc'] = _family_entry(
                'intel-arc', min(e['release'] for e in fam), url)


APPLE_HEADING = 'comparisonofm-seriesprocessors'
APPLE_NAME = re.compile(r'^M(\d+)(?:\s+(Pro|Max|Ultra))?$')


def parse_apple_page(html: str, url: str, entries: dict, stats: dict) -> None:
    """Apple M-series from the 'Comparison of M-series processors' table.

    Structurally the hardest page in SOURCES: three levels of column headers and
    rowspans up to 36 rows, so the cell grid is only correct after pandas expands
    the spans -- hand-walking `<tr>`s here silently mis-aligns columns. One chip
    occupies several rows (its GPU-core bins: M4 Max ships 32- and 40-core), which
    collapse per the same rule used for the NVIDIA/AMD variant rows: top
    configuration for specs, earliest date for release.

    Three axes are deliberately left null rather than filled with a
    plausible-looking number:

    - **fp64**: Metal exposes no double precision, so Apple GPUs have no FP64
      figure to report. Null is the correct value and excludes them from that axis.
    - **fp16 tensor**: the table's AI-accelerator column is the Neural Engine in
      integer TOPS -- a separate engine, a different unit. Writing it into
      `fp16_tensor_gflops` would corrupt the tensor-capability series with a
      quantity that is not GPU dense FP16 throughput.
    - **tdp**: Apple publishes none, and this table has no TDP column.

    `vram_gb` is *unified* memory (shared with the CPU, and the maximum orderable
    configuration), which is the GPU-addressable pool and so the right analogue
    for the reported-memory estimand -- but it is not dedicated VRAM, hence the
    per-entry note.
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = None
    for h in soup.find_all(['h2', 'h3', 'h4']):
        # the heading renders as "Comparison of M -series processors"
        if APPLE_HEADING in h.get_text('', strip=True).lower().replace(' ', ''):
            table = h.find_next('table', class_='wikitable')
            break
    if table is None:
        raise SystemExit(f'{url}: no table under a heading matching '
                         f'{APPLE_HEADING!r} -- the page changed, fix the parser '
                         f'rather than silently emitting no Apple entries')

    df = pd.read_html(io.StringIO(str(table)))[0]
    cols = flatten_columns(df)
    idx = {k: find_col(cols, n) for k, n in
           (('name', 'name'), ('fp32', 'fp32 flops'), ('vram', 'available capacity'),
            ('release', 'first release'), ('cores', 'gpu cores'))}
    missing = [k for k, v in idx.items() if v is None and k != 'cores']
    if missing:
        raise SystemExit(f'{url}: M-series table lacks column(s) {missing}; '
                         f'saw {cols}')

    per_chip: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        cells = list(row)
        name = ' '.join(FOOTNOTE.sub(' ', str(cells[idx['name']])).split())
        m = APPLE_NAME.match(name)
        if not m:
            stats['skipped_models'] += 1
            continue
        release = parse_release(str(cells[idx['release']]))
        if release is None:
            stats['no_date'] += 1
            continue
        stats['rows'] += 1
        # tera- -> giga-FLOPS; the memory cell lists every orderable size
        fp32 = parse_spec_value(cells[idx['fp32']], 1000.0)
        vram = parse_spec_value(cells[idx['vram']], 1.0)
        cur = per_chip.setdefault(name, {'gen': m.group(1), 'suffix': m.group(2),
                                        'release': release, 'fp32': None,
                                        'vram': None})
        cur['release'] = min(cur['release'], release)   # 'YYYY-MM' sorts as a date
        for k, v in (('fp32', fp32), ('vram', vram)):
            if v is not None:
                cur[k] = v if cur[k] is None else max(cur[k], v)

    for name, c in per_chip.items():
        suffix = c['suffix']
        mid = f'apple-m{c["gen"]}' + (f'-{suffix.lower()}' if suffix else '')
        entries[mid] = {
            'id': mid, 'display': f'Apple {name}', 'manufacturer': 'apple',
            'family': 'm-series', 'architecture': f'apple-m{c["gen"]}',
            'subtype': 'mobile-gpu', 'segment': 'consumer',
            'release': c['release'], 'release_source': url,
            'fp32_gflops': c['fp32'], 'fp64_gflops': None,
            'fp16_tensor_gflops': None, 'vram_gb': c['vram'], 'tdp_w': None,
            'spec_source': f'{url}#Comparison_of_M-series_processors',
            'notes': 'vram_gb is max unified memory (shared with the CPU), not '
                     'dedicated VRAM; fp64 undefined (no double precision in '
                     'Metal); Neural Engine TOPS deliberately not mapped to the '
                     'fp16 tensor axis',
            # Gated on the 'apple' vocabulary: bare M1/M2/M4 are also the money
            # supply, BMW models and manifold notation. 'M1 Pro' beats 'M1' by
            # the longest-span rule in registry.match_paragraph.
            'aliases': [f'Apple {name}', {'pattern': name, 'gate': 'apple'}],
        }


# --- vendor/brand triggers, derived from the parsed tables -------------------
# These catch "an NVIDIA graphics card" / "a FirePro card" -- a vendor named with
# no model number. Hand-listing them produced a lopsided vocabulary: NVIDIA had
# GeForce, Quadro and Tesla while AMD had only Radeon, even though these tables
# carry FirePro (70 models), FireGL (29) and Instinct (12). Deriving the tokens
# from the data makes the vocabulary a function of each vendor's product lines
# rather than of what we happened to remember.
BRAND_MIN_MODELS = 5      # a product line, not a one-off
BRAND_MIN_LEN = 4         # excludes series prefixes (GTX, RTX, RX, HD)
# Leading tokens that are not usable hardware triggers: a different product
# category, a company name, or an ordinary English word. Vendor-neutral by
# construction -- each entry is a fact about the word, not about the vendor.
BRAND_DENY = {
    # company names come from VENDOR_NAMES below, gated; letting the derivation
    # add them ungated would make a bare 'Intel' (i.e. every Core i7 paper) a GPU
    # brand mention
    'nvidia': 'company name, see VENDOR_NAMES',
    'amd': 'company name, see VENDOR_NAMES',
    'ati': 'company name, see VENDOR_NAMES',
    'intel': 'company name, and overwhelmingly CPUs',
    'google': 'company name, and overwhelmingly not hardware',
    'apple': 'the company and the fruit; the apple gate vocabulary covers it',
    'cloud': 'ordinary word, from "Cloud TPU v4"',
    'xeon': 'overwhelmingly CPUs; "Xeon Phi" is its own family entry',
    'knights': 'ordinary word; covered by the Xeon Phi family entry',
}
BRAND_GATE = {'tesla': 'gpu', 'titan': 'gpu'}     # a car and a moon
# The company name itself, gated uniformly. The rule is symmetric; the outcome
# is not, because 'NVIDIA' and 'Radeon' are themselves gpu context terms while
# 'AMD' and 'ATI' are not -- those vendors also sell CPUs, which is a documented
# property of the gate vocabulary rather than a per-vendor exception here.
VENDOR_NAMES = {'nvidia': ['NVIDIA'], 'amd': ['AMD', 'ATI']}


def build_vendor_entries(entries: dict) -> None:
    brands: dict[str, Counter] = defaultdict(Counter)
    for e in entries.values():
        if e.get('kind', 'model') != 'model':
            continue
        for a in e.get('aliases', []):
            pattern = a if isinstance(a, str) else a['pattern']
            token = pattern.split(' ')[0].replace('\\', '')
            if token.isalpha() and len(token) >= BRAND_MIN_LEN:
                brands[e['manufacturer']][token] = brands[e['manufacturer']][token] + 1

    for mfr, counts in brands.items():
        # fold case variants ('TITAN'/'Titan') onto the most common spelling
        folded: dict[str, tuple[int, str]] = {}
        for token, n in counts.items():
            key = token.lower()
            prev = folded.get(key)
            folded[key] = (n + (prev[0] if prev else 0),
                           token if not prev or n > prev[0] else prev[1])
        lines = [(k, spelling) for k, (n, spelling) in sorted(folded.items())
                 if n >= BRAND_MIN_MODELS and k not in BRAND_DENY]
        if not lines:
            continue
        aliases: list = []
        for key, spelling in lines:
            gate = BRAND_GATE.get(key)
            aliases.append({'pattern': spelling, 'gate': gate, 'case': 'sensitive'}
                           if gate else spelling)
        for name in VENDOR_NAMES.get(mfr, []):
            # case-insensitive: papers write 'Nvidia' as often as 'NVIDIA', and
            # the auto rule would make an all-caps pattern case-sensitive
            aliases.append({'pattern': name, 'gate': 'gpu', 'case': 'insensitive'})
        entries[f'vendor-{mfr}'] = {
            'id': f'vendor-{mfr}', 'display': f'{mfr.upper()} (brand mention)',
            'manufacturer': mfr, 'family': None, 'architecture': None,
            'subtype': 'generic-gpu', 'segment': None, 'kind': 'generic',
            'release': None, 'release_source': None,
            **{f: None for f in SPEC_FIELDS}, 'spec_source': None,
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
        elif name == 'apple':
            parse_apple_page(html, url, entries, stats)
        elif name == 'xeon_phi':
            parse_xeon_phi_page(html, url, entries, stats)
        elif name == 'intel_arc':
            parse_intel_arc_page(html, url, entries, stats)
        else:
            parse_gpu_page(html, manufacturer, url, entries, stats)

    for mid, extra in EXTRA_MODEL_ALIASES.items():
        if mid not in entries:
            raise SystemExit(f'EXTRA_MODEL_ALIASES targets {mid}, which the scrape '
                             f'no longer produces')
        entries[mid]['aliases'].extend(extra)
    build_vendor_entries(entries)
    models = sorted(entries.values(), key=lambda m: (m['release'] or '', m['id']))
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
