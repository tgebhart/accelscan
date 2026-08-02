# accelscan

GPU/accelerator mention extraction from scientific full text, for a metascience
study of hardware diffusion in science. **Two corpora**, extracted by identical
code and reported separately: **S2ORC** (2005–2025, GROBID-over-PDF body text) and
**arXiv** (1991–2025, author LaTeX source).

Pipeline: **registry/regex candidate filter → local LLM structured extraction
→ normalization → analysis panels.** All bulk outputs are sharded parquet
under `s3://scirec-embeddings/accelscan/`, namespaced by registry/prompt/model
version. Restartability everywhere is per-shard `.done` markers: rerun the
same command/array and completed shards are skipped.

## Setup

```bash
conda create -n accelscan python=3.12 -y
conda activate accelscan
pip install -e '.[dev]'          # + '.[gpu]' (vllm) on GPU nodes for stage 2
cp .env.example ~/.config/accelscan.env   # fill in MSI S3 keys
```

## Registry

- `registry/hardware.yaml` — hand-maintained: context-term gate vocabularies,
  generic/vendor/architecture entries, long-tail accelerators, and overrides
  (same id replaces a generated entry).
- `registry/generated/wikipedia.yaml` — built from the Wikipedia GPU lists:
  `python -m accelscan.scripts.build_registry` (cached HTML in
  `registry/cache/`; `--no-cache` to refetch).
- Every false positive/negative found in audits becomes a fixture in
  `tests/fixtures/registry_cases.yaml`. Bump `version` in hardware.yaml on any
  change; outputs are namespaced by it.

## Pipeline

```bash
# dev smoke test on the local sample shard (no SLURM, no S3)
python -m accelscan.scan --local-file <shard-path> --out-dir output/smoke --limit 20000

# stage 1: pilot scan (8 random shards) then full scan  [msibigmem]
sbatch slurm/pilot_scan.txt
sbatch slurm/stage1_scan.txt

# stage 1.5: repack candidates into ~25k-passage GPU shards  [login node, minutes]
python -m accelscan.repack

# stage 2: LLM extraction  [preempt-gpu]
#   (a) SLURM array, one shard per task — set --array=0-(n_shards-1) from repack's manifest:
sbatch slurm/stage2_infer.txt
#   (b) single GPU, many shards in one process (model loaded once, .done-skip per shard):
python -m accelscan.infer --shards 0-35 --model Qwen/Qwen3-14B

# stage 3.5: reported compute capacity per paper (CPU, minutes)
python -m accelscan.compute                         # primary (floor-1 counts)
python -m accelscan.compute --count-policy explicit # robustness variant

# stage 1.6 (optional): numerical-precision flags from stored passages
python -m accelscan.precision

# topical clustering (see "Topics" below)
sbatch slurm/embed_specter2.txt        # SPECTER2 abstract embeddings [GPU]
sbatch slurm/cluster.txt               # BERTopic UMAP+HDBSCAN [msibigmem]
```

## Capacity estimands (stage 3.5)

`accelscan/compute.py` joins per-device specs to used-in-this-work mentions:
`reported_flops = Σ device_count × per-device peak`, likewise VRAM and TDP.

- **Specs are scraped, never hardcoded.** `scripts/build_registry.py` parses the
  Processing-power / Memory / TDP columns out of the same cached source tables
  that supply release dates, into `fp32_gflops`, `fp64_gflops`,
  `fp16_tensor_gflops`, `vram_gb`, `tdp_w` + `spec_source`. Validated against
  published values in `tests/test_specs.py` (V100, A100, H100, K80, GTX 1080/Ti,
  RTX 3090/4090, MI250X, 8800 GTX).
- **Tensor throughput is never mixed into FP32.** The data-center table names its
  tensor column "Half precision Tensor Core FP32 Accumulate"; a test asserts the
  separation (it caused an ~8× inflation before being caught).
- **Interpretation: reported nameplate peak, not utilized FLOPs.** Every figure
  carries this caveat.
- Device counts are winsorized (per-year p99.9, hard cap 65,536) — the raw
  extraction contains a 13M outlier. Verified max after winsorizing is 18,688
  (Titan's real GPU count).
- Missing specs are **null, never zero-filled**, with `spec_missing_*` flags and a
  published spec-coverage-by-year table (fp32 coverage: 55% in 2010 → ~83% after 2019).

## Corpora

`accelscan.paths.Corpus` carries the output namespace and the per-paper key; every
stage takes `--corpus {s2orc,arxiv}`, defaulting to `s2orc`, so all pre-existing
commands are unchanged.

| | s2orc | arxiv |
|---|---|---|
| source | `s2orc_v2/*.gz` on MSI (gzip JSONL) | `s3://arxiv/src/*.tar` (requester-pays, streamed) |
| text | GROBID body paragraphs | LaTeX source, de-TeXed by `accelscan/latex.py` |
| key column | `corpusid` (Int64) | `paper_id` (`arxiv:2301.01234`); `corpusid` is null |
| output prefix | `accelscan/…` (unchanged) | `accelscan/arxiv/…` |
| `year` | Semantic Scholar metadata | free from the arXiv id (`YYMM`) |
| analysis window | 2005–2025 | 2005–2026 (arXiv has no ingestion lag; 2026 partial) |
| `field` | `s2fieldsofstudy` | arXiv's own taxonomy (`registry/arxiv_categories.yaml`) |
| citations | `citations` table (`i_3`/`i_5`, CD index) | **none** — no S2 identifiers by design |

**S2ORC keys never move.** `tests/test_paths.py` asserts every S2ORC key
byte-for-byte against the pre-refactor literals and asserts no arXiv key is
reachable from an S2ORC prefix listing — which is what keeps `.done` diffing and
mentions-version discovery corpus-safe.

The matcher, gating, passage caps and the whole LLM stage are shared: both corpora
enter through `scan.scan_paragraphs`, and
`tests/test_scan.py::test_shared_core_is_corpus_agnostic` fails if that is ever
forked.

## arXiv pipeline

```bash
# 0. metadata snapshot (HF parquet mirror, 3.11M rows, ~2.9GB, no auth)
#    -> field/primary_category/title/abstract  [msismall]
sbatch slurm/arxiv_metadata.txt

# 1a. PILOT from MSI: 20 tars over four eras (~10GB, ~$1 egress) + conversion audit
sbatch slurm/arxiv_pilot_scan.txt

# 1b. FULL history on EC2 in us-east-1 -- reads 2.9TB for $0 and writes to MSI S3
#     (the same run from MSI would be ~$260 of AWS egress; arxiv_scan refuses
#     to start outside us-east-1 without --yes-i-know)
python -m accelscan.arxiv_scan --dry-run           # bytes + bill, downloads nothing
# EC2 user-data (Ubuntu), one instance per YYMM slice; clones this repo, builds a
# venv, and re-checks the region before spending anything:
YYMM_RANGE=9108-0512 bash scripts/ec2_stage1_bootstrap.sh

# 2. repack + LLM + analytics, all on MSI
sbatch slurm/arxiv_stage2_infer.txt                # set --array from repack's manifest
sbatch slurm/arxiv_analytics.txt
```

`arxiv_scan` writes a third product per tar, `arxiv/ingest/parts/{shard}.parquet`,
holding per-paper skip reasons (`pdf_only`, `postscript`, `no_tex`, `timeout`, …) and
LaTeX conversion stats (`encoding`, `body_found`, `includes_missing`, `had_bibliography`,
`n_ref_paras_filtered`). Plot these **by year before trusting any trend**: a
converter that degrades across eras manufactures exactly the upward trend this
project measures.

### LaTeX handling: the matcher reads the raw source

`accelscan/latex.py` does **no TeX parsing**. Hardware names are ASCII literals that
appear verbatim in source (`NVIDIA V100`, `GPU`, `CUDA`), so the matcher reads the
`.tex` directly. Two parsing designs preceded this — a hand-rolled de-TeXer that hung
two full-history runs on an unbalanced `\cite{`, then pylatexenc — and measuring raw
search against the fixture suite showed the benefit of parsing was confined to
*exclusions*, not text quality: 30 contract violations, all of them captions, listings,
tables, comments or math markup contributing mentions, plus one recall loss where an
unresolved `\input` hid the methods file.

**That trade was taken deliberately** (2026-07-31): the bias is upward and lands mostly
in the any-mention series, `usage_context` already labels a caption or reference as
not-used-in-this-work, and no parser can hang a worker. 2.15 ms/paper (1.8 core-hours
for all 3.01M).

Five gotchas are kept, each one regex, each a large rather than slight effect:

| | |
|---|---|
| all text members concatenated | simpler *and* higher recall than resolving `\input` — the methods section is often its own `.tex`, and an unresolved include loses it entirely (measured: 0 candidates vs 1) |
| `$8$` → `8` | numeric inline math unwrapped; symbolic becomes a space so words cannot glue. Device counts are a headline estimand |
| bibliography excluded | `.bbl` never read, truncate at `\begin{thebibliography}` past the halfway guard, reference-shape paragraph filter. Cited *titles* routinely say "GPU-accelerated …" |
| preamble and post-`\end{document}` dropped | per file, so an included fragment ordered before the root is not discarded |
| blank-line paragraphs, long ones re-split | TeX's own rule; `PASSAGE_CHAR_CAP` is a character budget |

**Must be stated in the paper.** arXiv any-mention prevalence is biased upward relative
to S2ORC, which gets these exclusions free (GROBID drops floats; `bibliography` is a
separate object). Cross-corpus *levels* were already non-comparable; this widens the gap
in a known direction, and `reported use` / model-specific series are far better protected
because the LLM sees the passage. `tests/fixtures/latex_cases.yaml` pins the accepted
contamination as explicit `contains` assertions, so moving one back to `absent` is a
visible policy change. A robustness column is cheap if a reviewer asks: rerun the matcher
over the stored passages with exclusions applied — no LLM, no re-download.

`arxiv_scan` caps each paper at `--paper-timeout` seconds (`skip_reason='timeout'`).
With no parser left there is nothing to hang, but the guard costs nothing.

## Versioning note

Outputs are namespaced by the registry version at the stage that produced them.
Mentions were extracted under registry `0.1.0`; adding models/specs bumped the
registry to `0.2.0` **without re-running the LLM** (extraction is
registry-independent — verified: RTX 4090 was already extracted 13k times before
the registry knew the card). Downstream stages therefore call
`accelscan.s3.mentions_glob()`, which discovers the extraction-time version
rather than assuming the current one.

## Stage 4a: analytic tables (`accelscan/denominator.py`)

```bash
python -m accelscan.denominator        # denominator + paper_flags + citations
```

Writes to `s3://…/accelscan/analytic/{registry}/{prompt}/{model_tag}/`:

| table | rows | contents |
|---|---|---|
| `denominator` | 12,408,584 | one row per S2ORC full-text paper: year, field, `is_candidate`. **The population** — required so prevalence shares have a correct base. |
| `paper_flags` | 352,822 | per-paper accelerator numerators: any mention / reported use / model-specific, manufacturers, subtypes, model vintage, max device count. |
| `citations` | 347,829 | `citations_5y` (`i_5`) and `disruption_5y` (`cd_5`, CD index) from the lab's precomputed outcomes table, plus `window_complete` (≤2020) and `has_outcomes` (83.4% match). |

Very large inputs (papers, citations) are read part-by-part (`_by_part`) so memory
stays bounded regardless of table size.

## Notebooks (one per paper section)

| notebook | section | contents |
|---|---|---|
| `trends.ipynb` | §4 | corpus composition, prevalence, **reporting specificity (2.8%→75.8%)**, mention-context mix, field adoption lag, measurement funnel |
| `capacity.ipynb` | §5 | FLOPS/VRAM growth vs the vendor frontier, spec coverage, scale-out, precision capability-vs-choice, **inequality: Lorenz + bootstrapped Gini + Theil between/within-field** |
| `productivity.ipynb` | §6 | paper support (count + per year since release, support over device life), **productive half-life with CIs + censoring + per-model diagnostics**, adoption lag by generation, 3-yr citation support, rank-stability |
| `gpu_topics.ipynb` | §7 | GPU overlay on SPECTER2/BERTopic topics (frontier-lag by topic, segment mix) |
| `manufacturer.ipynb` | §8 | vendor shares, HHI/entropy, **CUDA lock-in proxy**, multi-vendor papers, **export-control parts (A800/H800/H20)** |
| `gpu_usage.ipynb` | — | original exploratory manufacturer/model view |

Every notebook starts with a chdir-to-repo-root guard (they use repo-relative
paths like `registry/hardware.yaml`) and reads the `Python (accelscan)` kernel.

### Figure export

Each notebook's setup cell calls `accelscan.plotting.setup_figures('<notebook>')`,
which patches `plt.show` once so no plot cell carries figure bookkeeping. It:

- writes every figure to `output/analysis/<notebook>/<nn>_<slug>.pdf`, where the
  slug is derived from the figure's suptitle (or its axes titles) — figures are
  named after what they plot, and `nn` preserves notebook order;
- forces **integer year ticks**, so no axis ever reads `2007.5`. Year axes are
  detected by view range (span inside 1980–2040 on a linear scale), not by label,
  so "hardware age (years)" and log axes are left alone;
- puts the **`%` on the tick labels** of any axis whose label opens with
  `Percentage` / `Proportion` / `Cumulative share`. Axis labels therefore spell the
  word out and must never open with a bare `%` glyph — the wording is also what
  selects the scale (`Percentage…` = 0–100, `Proportion…`/`Cumulative share…` =
  0–1), since a percentage axis topping out at 0.3% is otherwise
  indistinguishable from a proportion.

Re-render everything:

```bash
for nb in trends capacity productivity gpu_topics manufacturer gpu_usage; do jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=1800 notebooks/$nb.ipynb; done
```

Remaining stub: `accelscan/panels.py` (tidy year×field×model export panels).

## Tests

```bash
python -m pytest -q
```

Matcher behavior cases (including hard negatives like the K80 antibody, BMW
M2, P100 ECG latency, graph-convolutional "GCN") are in
`tests/fixtures/registry_cases.yaml`.
