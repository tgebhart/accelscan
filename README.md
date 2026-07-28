# accelscan

GPU/accelerator mention extraction from S2ORC full text, for a metascience
study of hardware diffusion in science (2005–2025).

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
```

Stage 3 (normalize to paper×model) and stage 4 (denominator + panels) live in
`accelscan/normalize.py` / `accelscan/denominator.py` / `accelscan/panels.py`
(to be added after the pilot freeze).

## Tests

```bash
python -m pytest -q
```

Matcher behavior cases (including hard negatives like the K80 antibody, BMW
M2, P100 ECG latency, graph-convolutional "GCN") are in
`tests/fixtures/registry_cases.yaml`.
