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

from functools import lru_cache

import polars as pl

from accelscan.registry import CompiledRegistry, load_registry

# rank when a string matches several registry entries
_KIND_RANK = {'model': 0, 'architecture': 1, 'generic': 2}


def canonicalize_one(reg: CompiledRegistry, text: str | None) -> dict:
    """Resolve one model string to (id, display, kind). Longest span wins;
    ties broken toward model > architecture > generic."""
    if not text:
        return {'canonical_model': None, 'canonical_display': None, 'canonical_kind': None}
    matches = reg.match_paragraph(text)
    matches = [m for m in matches if not m.gate_required or m.gate_ok]
    if not matches:
        return {'canonical_model': None, 'canonical_display': None, 'canonical_kind': None}
    best = max(matches, key=lambda m: (m.end - m.start, -_KIND_RANK.get(m.kind, 9)))
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
