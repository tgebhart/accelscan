# Methodology review: accelscan extraction pipeline

Review of the extraction methodology (paper draft + `docs/methods.tex` + registry/matching code) ahead of publication. Goal: not maximal precision, but a **robust, repeatable pipeline whose choices are defensible and simply describable**. Recommendations are mostly small (docs, one test, two alias-generation rules, one audit notebook cell), not a refactor.

## Overall verdict

The architecture is sound and unusually defensible for this genre: a *generated* registry with per-entry `spec_source`, a three-layer matcher (Aho–Corasick prefilter → confirming regex → context gate) shared byte-for-byte across corpora and pinned by tests, LLM adjudication as the precision stage, and version-namespaced outputs. The simple one-sentence description already exists: *"a registry of device names generated from public specification tables; exact alias matching with context gating for ambiguous short codes; LLM adjudication of every candidate passage; canonicalization of extracted names back to the registry."* Nothing in the design reads as contrived — the hand-maintained surface is now just 3 gate vocabularies + 7 generic terms, and every hand judgment (`BRAND_DENY`, `BRAND_GATE`, `MIN_BARE_CODE`) is enumerated with a stated reason and pinned by a test fixture with measured pilot counts.

The gaps are (a) reproducibility of the registry *generators*, (b) no end-to-end accuracy evaluation, (c) places where the code and the methods text disagree, and (d) two cross-source inconsistencies in alias generation that a careful reviewer could call arbitrary.

## Recommendations, prioritized

### P0 — required before submission

1. **Archive the registry source snapshots.** `registry/cache/` is gitignored, so neither `build_registry.py` nor `build_epoch_registry.py` can be re-run from the repo; only the emitted YAML is archival. Fix: commit the cached HTML + `ml_hardware.csv` (or a tarball / Zenodo deposit), and record **Wikipedia `oldid` permalinks** next to each URL in `SOURCES` so `release_source`/`spec_source` point at immutable revisions. This is the single biggest "repeatable" gap: Wikipedia tables mutate.

2. **Add an external validation of the extraction.** The methods section currently reports structural validity checks (pre-release rate, offsets, drift) but no accuracy measurement of the extraction itself. Evaluate the candidate-generation stage against **SH-NER (Anjum et al. 2025)** — already cited in Related Work — by running the registry matcher over its 1,128 annotated papers and reporting what fraction of its hardware-device annotations are recovered (and classifying the misses: out-of-scope FPGA/CPU vs. genuine registry gaps). This converts the "lower bound" caveat into a measured number.

3. **Fix the gate-window description mismatch.** `methods.tex` says a tier-B context term must occur "within ±250 characters of the match"; `registry.py:148` actually searches the *entire paragraph* plus 250 chars of each neighbor, position-independent (~3,000 chars for a max paragraph). Fix the **prose**, not the code (per-match ±250 windows would force a re-scan): "within the matched paragraph or the adjacent 250 characters of its neighbours." Same subsection: quantify the **paper-level rescue** — report what share of tier-B used-mentions survive only via rescue rather than local context (computable from stored `gate_required`/`gate_ok` columns, no re-scan). If small, it's a footnote; if large, report the local-gate-only series as robustness.

### P1 — makes the "universal, not contrived" claim true

4. **Unify bare-code alias policy across the two generators.** Today the Wikipedia builder gates any `SHORT_CODE` bare code (so gated bare `K80`, `A100` exist) while the Epoch builder refuses bare codes under 4 chars (`MIN_BARE_CODE = 4`: no bare `T4`, `L4`, `H20`) and gates 4–5-char ones. Both rules are individually reasoned, but they are *two* rules, and they produce a recall asymmetry between NVIDIA datacenter parts sourced from Wikipedia vs. Epoch (T4 is reachable only as "Tesla T4"/"NVIDIA T4"). Adopt **one rule stated once** — recommend the Epoch rule (bare codes: <4 chars never, 4+ chars gated) applied in both builders, with the deliberately-ambiguous exceptions (`TITAN X`) as the enumerated exception list. State the rule in methods as a single sentence. Note this changes matching → registry minor-version bump → affects only future scans; bound the effect the same way the RTX-40 gap was bounded.

5. **Fix `case: auto` on vendor-qualified Epoch aliases.** `NVIDIA L4`, `NVIDIA A800`, `NVIDIA HGX H20` compile case-**sensitive** (all letters uppercase), so "Nvidia L4" never matches — the exact reason `vendor-nvidia` needed a manual `case: insensitive`. Since several of these are the export-control SKUs the paper reports on, this is a directional recall bias on a headline series. Fix: in alias generation, force `case: insensitive` when the pattern contains a vendor name token (or make the vendor prefix lowercase-tolerant in `_compile_pattern`).

6. **Make the registry version cover the generated files.** `version` is a scalar in `hardware.yaml`; a re-scrape mutates matching without touching it. Cheapest fix, no schema change: a test that hashes `registry/generated/*.yaml` + `hardware.yaml` against a committed digest, failing on any content change without a version bump.

### P2 — documentation and completeness

7. **Sync methods.tex/README with v0.3.1 reality.** Stale statements found: methods.tex:159 says the hand file holds "the NVIDIA architecture words" (removed in `ce490ca`; `hardware.yaml` now holds only gates + generics); tier-B count "236 of 1,862" is the v0.2.0 scan registry, current is 278/2,099 — fine if labeled as scan-time numbers, currently ambiguous; README §Registry still describes `hardware.yaml` as holding "generic/vendor/architecture entries, long-tail accelerators, and overrides" and omits `generated/epoch.yaml`.

8. **Actually produce the unresolved audit.** `normalize.unresolved_audit()` is documented as a publication artefact ("publish alongside capacity results") and methods.tex quotes its numbers, but no notebook calls it. Add a cell to the trends/capacity notebooks that emits the bucket table per corpus, so the printed 94.9%/97.0% resolution rates and the out-of-scope shares regenerate with the tables.

9. **State two matcher semantics explicitly in methods** (they are fine, just undocumented): literal spaces compile to `[\s-]*` which is zero-or-more, so `TeslaV100`/`RTX3090` match by design; equal-length spans from different models are both retained (the LLM disambiguates). Also note the 20-vs-5 passage caps truncate in document order — worth one sentence since it interacts with per-paper mention counts.

### LLM stage (deprioritized; brief)

- The design is already strong (guided JSON, versioned prompt, evidence_quote for hallucination audit, status accounting).
- The `matched_surfaces` hint passes registry attention bias into the LLM; the worked negative examples mitigate it. Worth one methods sentence acknowledging the hint and pointing at the `no_mention` rates (5.5%/9.1%) as evidence the model does reject hints.

## Files touched if recommendations are implemented

- `docs/methods.tex` — P0-3 prose fix, P2-7 sync, P2-9 semantics
- `accelscan/scripts/build_registry.py` + `build_epoch_registry.py` — P1-4 unified bare-code rule, P1-5 case rule, oldid permalinks in `SOURCES`
- `registry/` — commit cache snapshots (P0-1)
- `tests/` — registry content-hash test (P1-6), fixtures for the new alias rules
- a SH-NER evaluation script in `accelscan/scripts/` — P0-2
- notebooks — unresolved-audit cell (P2-8), rescue-share robustness cell (P0-3)

## Verification

- `pytest tests/` green after alias-rule changes; new fixture cases for "Nvidia L4", bare `T4` (negative), gated `A800`.
- Re-run `build_registry.py`/`build_epoch_registry.py` from committed caches on a clean checkout → byte-identical `generated/*.yaml`.
- Rescue-share and SH-NER numbers land in methods.tex with the notebooks that produce them.
