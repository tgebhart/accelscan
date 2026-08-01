"""Constants and env-var-backed secrets. No keys are hardcoded here.

Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and optionally
ACCELSCAN_S3_ENDPOINT) in the environment; see .env.example.
"""

import os

BUCKET = 'scirec-embeddings'
S2ORC_PREFIX = 's2orc_v2/'
PAPERS_PREFIX = 'data/semanticscholar/2026-02-10/processed/papers/parts/'
ABSTRACTS_PREFIX = 'data/semanticscholar/2026-02-10/processed/abstracts/parts/'
OUT_PREFIX = 'accelscan'


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


def s3_config() -> dict:
    return {
        'aws_access_key_id': os.environ['AWS_ACCESS_KEY_ID'],
        'aws_secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY'],
        'endpoint_url': os.environ.get('ACCELSCAN_S3_ENDPOINT', 'https://s3.msi.umn.edu'),
    }
