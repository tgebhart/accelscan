"""Load and compile the hardware alias registry.

Sources: registry/hardware.yaml (hand-maintained: context gates, generics,
architectures, overrides) merged with registry/generated/*.yaml (built from
Wikipedia/vendor tables by scripts/build_registry.py). A hand entry with the
same id replaces a generated one.

Matching is three-layered:
  1. Aho-Corasick automaton over lowercase literal cores — cheap paragraph
     prefilter; a hit must sit on word boundaries in the haystack.
  2. Confirming regex per alias, with word-boundary wrapping, optional plural
     's', and literal spaces compiled as [\\s-]* separators.
  3. Context gating: tier-B aliases need a context_terms regex nearby (or a
     paper-level tier-A model hit, applied by the caller).

Overlap resolution keeps the longest span, so 'RTX 3090 Ti' beats 'RTX 3090'
without hand-authored lookaheads.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import ahocorasick
import yaml

REGISTRY_DIR = Path('registry')
MIN_CORE_LEN = 2


@dataclass(frozen=True)
class Alias:
    model_id: str
    pattern: str
    regex: re.Pattern
    gate: str | None
    core: str


@dataclass
class Model:
    id: str
    display: str
    manufacturer: str
    kind: str = 'model'
    family: str | None = None
    architecture: str | None = None
    subtype: str | None = None
    segment: str | None = None
    release: str | None = None
    release_source: str | None = None
    notes: str | None = None
    # per-device specs, scraped from the source tables (never hardcoded);
    # None = unknown -> excluded per-axis by the capacity estimands
    fp32_gflops: float | None = None
    fp64_gflops: float | None = None
    fp16_tensor_gflops: float | None = None
    vram_gb: float | None = None
    tdp_w: float | None = None
    spec_source: str | None = None
    aliases: list = field(default_factory=list)


@dataclass(frozen=True)
class Match:
    model_id: str
    kind: str
    surface: str
    start: int
    end: int
    gate_required: bool
    gate_ok: bool  # gate satisfied locally (always True for tier-A)


def _compile_pattern(pattern: str, case: str) -> re.Pattern:
    body = pattern.replace(' ', r'[\s-]*')
    wrapped = rf'(?<!\w)(?:{body})s?(?!\w)'
    if case == 'auto':
        letters = [c for c in re.sub(r'\\.', '', pattern) if c.isalpha()]
        case = 'sensitive' if letters and all(c.isupper() for c in letters) else 'insensitive'
    flags = 0 if case == 'sensitive' else re.IGNORECASE
    return re.compile(wrapped, flags)


def _literal_core(pattern: str) -> str:
    m = re.search(r'[A-Za-z0-9]{%d,}' % MIN_CORE_LEN, re.sub(r'\\.', ' ', pattern))
    if not m:
        raise ValueError(f'no literal core of length >= {MIN_CORE_LEN} in pattern {pattern!r}')
    return m.group(0).lower()


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == '_'


class CompiledRegistry:
    def __init__(self, version: str, models: dict[str, Model],
                 gates: dict[str, re.Pattern], aliases: list[Alias]):
        self.version = version
        self.models = models
        self.gates = gates
        self.aliases = aliases
        self.automaton = ahocorasick.Automaton()
        core_map: dict[str, list[int]] = {}
        for i, a in enumerate(aliases):
            core_map.setdefault(a.core, []).append(i)
        for core, idxs in core_map.items():
            self.automaton.add_word(core, (len(core), idxs))
        self.automaton.make_automaton()

    def match_paragraph(self, text: str, context_before: str = '',
                        context_after: str = '') -> list[Match]:
        lowered = text.lower()
        candidate_ids: set[int] = set()
        for end_idx, (core_len, idxs) in self.automaton.iter(lowered):
            start_idx = end_idx - core_len + 1
            if start_idx > 0 and _is_word_char(lowered[start_idx - 1]):
                continue
            if end_idx + 1 < len(lowered) and _is_word_char(lowered[end_idx + 1]):
                # allow plural / model suffixes to be resolved by the regex
                if not lowered[end_idx + 1].isalnum():
                    continue
            candidate_ids.update(idxs)
        if not candidate_ids:
            return []

        raw: list[tuple[Alias, re.Match]] = []
        for i in candidate_ids:
            alias = self.aliases[i]
            for m in alias.regex.finditer(text):
                raw.append((alias, m))
        if not raw:
            return []

        # Longest span wins; drop matches fully contained in a kept span.
        raw.sort(key=lambda am: (-(am[1].end() - am[1].start()), am[1].start()))
        kept: list[tuple[Alias, re.Match]] = []
        for alias, m in raw:
            if any(k.start() <= m.start() and m.end() <= k.end() and
                   (k.start(), k.end()) != (m.start(), m.end())
                   for _, k in kept):
                continue
            if any((k.start(), k.end()) == (m.start(), m.end()) and
                   ka.model_id == alias.model_id for ka, k in kept):
                continue
            kept.append((alias, m))

        window = f'{context_before[-250:]} {text} {context_after[:250]}'
        out = []
        for alias, m in kept:
            gate_ok = True
            if alias.gate is not None:
                gate_ok = bool(self.gates[alias.gate].search(window))
            out.append(Match(
                model_id=alias.model_id,
                kind=self.models[alias.model_id].kind,
                surface=m.group(0),
                start=m.start(),
                end=m.end(),
                gate_required=alias.gate is not None,
                gate_ok=gate_ok,
            ))
        out.sort(key=lambda x: x.start)
        return out


def _normalize_alias(entry) -> dict:
    if isinstance(entry, str):
        return {'pattern': entry, 'gate': None, 'case': 'auto'}
    return {'pattern': entry['pattern'], 'gate': entry.get('gate'),
            'case': entry.get('case', 'auto')}


def load_registry(registry_dir: Path | str = REGISTRY_DIR) -> CompiledRegistry:
    registry_dir = Path(registry_dir)
    hand = yaml.safe_load((registry_dir / 'hardware.yaml').read_text())
    version = str(hand['version'])

    entries: dict[str, dict] = {}
    for gen_path in sorted((registry_dir / 'generated').glob('*.yaml')):
        gen = yaml.safe_load(gen_path.read_text())
        for e in gen.get('models', []):
            if e['id'] in entries:
                raise ValueError(f'duplicate generated id {e["id"]} in {gen_path.name}')
            entries[e['id']] = e
    for e in hand.get('models', []):
        entries[e['id']] = e  # hand entries override generated

    gates = {
        name: re.compile('|'.join(rf'(?<!\w)(?:{re.escape(t)})(?!\w)' for t in terms),
                         re.IGNORECASE)
        for name, terms in hand.get('context_terms', {}).items()
    }

    models: dict[str, Model] = {}
    aliases: list[Alias] = []
    for e in entries.values():
        model = Model(
            id=e['id'], display=e['display'], manufacturer=e['manufacturer'],
            kind=e.get('kind', 'model'), family=e.get('family'),
            architecture=e.get('architecture'), subtype=e.get('subtype'),
            segment=e.get('segment'), release=str(e['release']) if e.get('release') else None,
            release_source=e.get('release_source'), notes=e.get('notes'),
            fp32_gflops=e.get('fp32_gflops'), fp64_gflops=e.get('fp64_gflops'),
            fp16_tensor_gflops=e.get('fp16_tensor_gflops'),
            vram_gb=e.get('vram_gb'), tdp_w=e.get('tdp_w'),
            spec_source=e.get('spec_source'),
        )
        models[model.id] = model
        for raw in e.get('aliases', []):
            a = _normalize_alias(raw)
            if a['gate'] is not None and a['gate'] not in gates:
                raise ValueError(f'{model.id}: unknown gate {a["gate"]!r}')
            aliases.append(Alias(
                model_id=model.id,
                pattern=a['pattern'],
                regex=_compile_pattern(a['pattern'], a['case']),
                gate=a['gate'],
                core=_literal_core(a['pattern']),
            ))

    seen = set()
    for a in aliases:
        key = (a.model_id, a.pattern, a.gate)
        if key in seen:
            raise ValueError(f'duplicate alias {key}')
        seen.add(key)

    return CompiledRegistry(version=version, models=models, gates=gates, aliases=aliases)
