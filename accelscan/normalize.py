"""Stage 3: canonicalize LLM mentions against the registry.

The LLM emits a free-text `model_normalized` (e.g. "NVIDIA Tesla V100",
"NVIDIA Titan X"). We resolve each distinct string to a registry canonical id
by running the registry matcher over it and keeping the longest match,
preferring model-kind entries over architecture/generic. This collapses
surface variants ("Titan X" / "TITAN X") and gives every mention a stable
`canonical_model` / `canonical_display` / `canonical_kind`.

`manufacturer` is taken straight from the LLM output (a clean enum); the
registry mapping is only used for model/architecture identity.
"""

import re
from functools import lru_cache

import polars as pl

from accelscan.registry import CompiledRegistry, load_registry

# rank when a string matches several registry entries
_KIND_RANK = {'model': 0, 'architecture': 1, 'generic': 2}
_PARENS = re.compile(r'[()\[\]]')

# Strings the LLM emits when it decides the registry hit was NOT hardware, or
# that name the non-accelerator meaning outright. These are *successful*
# rejections, not canonicalization failures — drop them before auditing.
REJECT_PATTERNS = re.compile(
    r'not[- ]an[- ]accelerator|^n/?a$|^none$|^unknown$'
    r'|thermoplastic|polyurethane'          # TPU in materials-science papers
    r'|gamma[- ]ray|telescope|space telescope'  # "Fermi" the observatory
    r'|antibod|cell line|money supply',     # K80 antibody, M2 money supply
    re.IGNORECASE)


def is_reject(text: str | None) -> bool:
    return bool(text) and bool(REJECT_PATTERNS.search(text))


def canonicalize_one(reg: CompiledRegistry, text: str | None) -> dict:
    """Resolve one model string to (id, display, kind). Longest span wins;
    ties broken toward model > architecture > generic.

    Gates are deliberately KEPT here: a bare 'Fermi' in a hardware context is
    the architecture, but 'Fermi Gamma-ray Space Telescope' must not become a
    GPU — so context gating still applies to the model string itself."""
    if not text or is_reject(text):
        return {'canonical_model': None, 'canonical_display': None, 'canonical_kind': None}
    matches = reg.match_paragraph(text)
    # The LLM parenthesises the disambiguator the vendor wrote inline, and a
    # parenthesis is not one of the separators a multi-word alias tolerates: so
    # "NVIDIA Titan X (Pascal)" matched only 'Pascal' and 883 used-mentions across
    # the two corpora were attributed to the *architecture* instead of to the
    # Titan X that was actually used. Retry on the depunctuated form and let the
    # longest-span rule choose between the two candidate sets.
    depunct = _PARENS.sub(' ', text)
    if depunct != text:
        matches = matches + reg.match_paragraph(depunct)
    matches = [m for m in matches if not m.gate_required or m.gate_ok]
    if not matches:
        return {'canonical_model': None, 'canonical_display': None, 'canonical_kind': None}
    # A specific model outranks an architecture or a brand outright, not merely on
    # a tie. Architecture names are long English words and model codes are short,
    # so longest-span alone sent "NVIDIA Grace Hopper GH200" to the Hopper
    # architecture rather than to the GH200: 'Hopper' is six characters, 'GH200'
    # five. No string that names both is better described by the architecture.
    models = [m for m in matches if m.kind == 'model']
    best = max(models or matches,
               key=lambda m: (m.end - m.start, -_KIND_RANK.get(m.kind, 9)))
    model = reg.models[best.model_id]
    return {'canonical_model': model.id, 'canonical_display': model.display,
            'canonical_kind': model.kind}


def canonicalize_column(df: pl.DataFrame, reg: CompiledRegistry,
                        source_col: str = 'model_normalized') -> pl.DataFrame:
    """Add canonical_model/display/kind by resolving each distinct source
    string once, then joining back (avoids per-row matcher calls)."""
    distinct = df.select(source_col).unique().to_series().to_list()
    rows = [{source_col: s, **canonicalize_one(reg, s)} for s in distinct]
    mapping = pl.DataFrame(rows, schema={
        source_col: pl.Utf8, 'canonical_model': pl.Utf8,
        'canonical_display': pl.Utf8, 'canonical_kind': pl.Utf8})
    return df.join(mapping, on=source_col, how='left')


@lru_cache(maxsize=1)
def _registry() -> CompiledRegistry:
    return load_registry()


# Non-GPU accelerator families deliberately left out of the registry (the
# paper is GPU/TPU-centric and FPGA "FLOPS" is bitstream-dependent, so a
# nameplate figure would be meaningless). Classified here so the unresolved
# bucket is *documented* rather than an unexplained measurement gap.
OUT_OF_SCOPE = {
    # Deliberately descoped in registry v0.3.0, so their strings must land in a
    # named bucket rather than in `unknown`, where a scope decision would be
    # indistinguishable from a registry gap: embedded modules, and the two
    # wafer-scale/IPU vendors no source we parse publishes per-device specs for.
    # Architecture-only reporting ("we used NVIDIA Ampere GPUs"). Not a scope
    # decision and not a gap: the paper named a generation, not a device, so there
    # is no model to resolve to and no per-device spec to attach. Bucketed so the
    # share of architecture-grain reporting stays measurable.
    'architecture-only': re.compile(
        r'\b(Fermi|Kepler|Maxwell|Pascal|Volta|Turing|Ampere|Ada Lovelace|Hopper'
        r'|Blackwell|GCN|RDNA\d?|CDNA\d?|Graphics Core Next)\b'),
    'embedded-module': re.compile(r'jetson|tegra|\borin\b|xavier', re.IGNORECASE),
    # A rented instance type names a bundle, not a device: 'p3.16xlarge' is 8
    # V100s and 'g4dn.xlarge' is one T4, and resolving one would require a
    # vendor-published instance->device-and-count table that none of our sources
    # carries. Measured on SH-NER, where these are 5 of 174 annotated
    # hardware spans, so the omission is bounded rather than unknown.
    'cloud-instance': re.compile(
        r'\b(([a-z]\d[a-z]{0,3})\.(nano|micro|small|medium|\d*x?large)'
        r'|n1-|n2-|a2-|a3-|standard_n[cdv]|nc\d+ads?|nd\d+'
        r'|colab|sagemaker|paperspace|runpod|lambda labs)', re.IGNORECASE),
    'wafer-scale-ipu': re.compile(r'cerebras|graphcore|\bIPU\b|\bWSE-?\d?\b|\bCS-[123]\b'),
    'fpga-xilinx': re.compile(r'xilinx|virtex|artix|kintex|spartan|zynq|alveo|zcu|vc707|ultrascale',
                              re.IGNORECASE),
    'fpga-intel': re.compile(r'\b(arria|stratix|cyclone|agilex)\b', re.IGNORECASE),
    'fpga-generic': re.compile(r'\bfpga\b', re.IGNORECASE),
    'cpu': re.compile(r'xeon|core i[3579]|ryzen|threadripper|epyc|\bcpu\b|arm cortex|cortex-[amr]',
                      re.IGNORECASE),
    # A named machine is a facility, not a device: its node composition is
    # published elsewhere and changes over the machine's life, so resolving one to
    # a model would be an inference, not an extraction.
    'named-system': re.compile(
        r'\bsupercomput\w+\b|\b(Frontier|Summit|LUMI'
        r'|Fugaku|JUWELS|Leonardo|Perlmutter|Polaris|Aurora|Sierra|Piz Daint'
        r'|TSUBAME|Stampede|Bridges|Jean Zay|MareNostrum)\b', re.IGNORECASE),
    'sbc-embedded': re.compile(r'raspberry pi|arduino|beaglebone|myriad|coral|edge tpu',
                               re.IGNORECASE),
    'dsp-asic-other': re.compile(r'\bdsp\b|\basic\b|kirin|snapdragon|exynos|neural engine',
                                 re.IGNORECASE),
}


def classify_unresolved(text: str | None) -> str:
    """Bucket a string that failed to canonicalize, for the audit table."""
    if not text:
        return 'empty'
    if is_reject(text):
        return 'llm-rejected-non-hardware'
    for label, pat in OUT_OF_SCOPE.items():
        if pat.search(text):
            return label
    return 'unknown'


def unresolved_audit(df: pl.DataFrame, source_col: str = 'model_normalized',
                     top_n: int = 40) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (bucket summary, top unresolved strings) for a canonicalized
    mention frame. Publish alongside capacity results so the share of
    extracted-but-unregistered hardware is explicit."""
    named = df.filter(pl.col(source_col).is_not_null())
    unres = named.filter(pl.col('canonical_model').is_null())
    buckets = {s: classify_unresolved(s) for s in unres[source_col].unique().to_list()}
    unres = unres.with_columns(
        bucket=pl.col(source_col).replace_strict(buckets, default='unknown'))
    summary = (unres.group_by('bucket').agg(mentions=pl.len())
               .with_columns(share_of_named=pl.col('mentions') / named.height)
               .sort('mentions', descending=True))
    top = (unres.group_by(source_col, 'bucket').agg(mentions=pl.len())
           .sort('mentions', descending=True).head(top_n))
    return summary, top
