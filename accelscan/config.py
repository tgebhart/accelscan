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


def s3_config() -> dict:
    return {
        'aws_access_key_id': os.environ['AWS_ACCESS_KEY_ID'],
        'aws_secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY'],
        'endpoint_url': os.environ.get('ACCELSCAN_S3_ENDPOINT', 'https://s3.msi.umn.edu'),
    }
