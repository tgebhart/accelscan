"""Constants and env-var-backed secrets. No keys are hardcoded here.

Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and optionally
ACCELSCAN_S3_ENDPOINT) in the environment; see .env.example.
"""

import os

BUCKET = 'scirec-embeddings'
S2ORC_PREFIX = 's2orc_v2/'
OUT_PREFIX = 'accelscan'

# --- Semantic Scholar bulk release -----------------------------------------
# One release id drives the metadata prefixes, so a re-download cannot leave the
# analytic stages reading last release's tables. 2026-07-28 is the latest release
# the bulk API offers as of 2026-08-06; the previous tables (2026-02-10) are still
# on S3, so `S2_RELEASE_ID=2026-02-10` in the environment pins back to them --
# which is what to do until slurm/s2_download.txt has run for the new release,
# because these prefixes are read, not created, by the analytic stages.
S2_RELEASE_ID = os.environ.get('S2_RELEASE_ID', '2026-07-28')
S2_DATA_DIR = os.environ.get('S2_DATA_DIR', 'data/semanticscholar')
S2_API_BASE = 'https://api.semanticscholar.org/datasets/v1/release/'
S2_PROCESSED_PREFIX = f'data/semanticscholar/{S2_RELEASE_ID}/processed'
PAPERS_PREFIX = f'{S2_PROCESSED_PREFIX}/papers/parts/'
ABSTRACTS_PREFIX = f'{S2_PROCESSED_PREFIX}/abstracts/parts/'
AUTHORS_PREFIX = f'{S2_PROCESSED_PREFIX}/authors/parts/'
# Citation/disruption outcomes, produced by accelscan.semanticscholar.aggregate.
# Defaults to the lab table the analytic stage has always read, so nothing moves
# until a fresh one is built.
OUTCOMES_PATH = os.environ.get(
    'ACCELSCAN_OUTCOMES_PATH',
    '/projects/standard/funkr/gebhart/projects/rfunklab/novelty_corp/data'
    '/semanticscholar/processed/citation_outcomes/semanticscholar_outcomes.parquet.gz')


# Paragraphs matched per paper caps (stage 1). Papers whose only hits are
# generic terms (GPU/CUDA/...) carry less marginal information per passage.
MAX_PASSAGES_MODEL_SPECIFIC = 20
MAX_PASSAGES_GENERIC_ONLY = 5
PASSAGE_CHAR_CAP = 2500
GATE_WINDOW_CHARS = 250
# LaTeX authors write far longer paragraphs than GROBID emits, so an arXiv
# paragraph is re-split at sentence boundaries above this length. Must be <=
# PASSAGE_CHAR_CAP: at 3000 against a 2500 cap it *guaranteed* that a match in the
# tail of a split paragraph was truncated out of its own passage (0.56% of the
# 2026-07-31 run). scan._assemble_passage now also windows on the match, so this is
# belt and braces, but the ordering is asserted in tests/test_scan.py.
SPLIT_LONG_PARA_CHARS = PASSAGE_CHAR_CAP


def s2_api_key() -> str | None:
    """Semantic Scholar bulk-API key. Optional: the API serves dataset listings
    unauthenticated but rate-limits them hard, so a key is strongly advised for a
    full download. Never hardcode it -- put S2_API_KEY in .env."""
    return os.environ.get('S2_API_KEY')


def s3_config() -> dict:
    return {
        'aws_access_key_id': os.environ['AWS_ACCESS_KEY_ID'],
        'aws_secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY'],
        'endpoint_url': os.environ.get('ACCELSCAN_S3_ENDPOINT', 'https://s3.msi.umn.edu'),
    }
