#!/bin/bash -e
# Rebuild + redeploy the public arXiv-results website (site/) to S3+CloudFront.
#
# Run this after each ~monthly bulk-release re-ingestion (re-extraction, or a
# new registry/analytic version landing on MSI S3). See docs/site_deploy.md
# for the one-time bucket/CloudFront setup this assumes already exists.
#
# Two AWS identities are involved and must not be conflated: MSI's S3
# (AWS_ACCESS_KEY_ID/SECRET from .env) is where step 1 READS the source data
# from; the site-hosting bucket + CloudFront live in a SEPARATE, full-fledged
# AWS account, credentialed by SITE_AWS_ACCESS_KEY_ID/SECRET (same pattern as
# ARXIV_AWS_* in arxiv_bulk.py, for the same reason -- a second real-AWS
# account distinct from MSI's env vars). Steps 3-4 override AWS_ACCESS_KEY_ID/
# SECRET/DEFAULT_REGION from the SITE_AWS_* vars right before running any
# `aws` CLI command, so nothing downstream of that point can silently inherit
# MSI's credentials instead.
#
# Required in the environment:
#   SITE_BUCKET                 the S3 bucket serving the site (one-time setup)
#   SITE_CLOUDFRONT_DIST_ID     CloudFront distribution id, for the invalidation
#   SITE_AWS_ACCESS_KEY_ID      site-hosting account credentials (put in .env)
#   SITE_AWS_SECRET_ACCESS_KEY
# Optional: SITE_AWS_REGION (default us-east-1 -- CloudFront's API is global,
#   but the CLI still needs a signing region). Optional
#   (accelscan.scripts.export_site_data flags): CORPUS, PROMPT_VERSION,
#   MODEL_TAG, COUNT_POLICY -- defaults match the notebooks' knobs cells.
#
#   set -a; source .env; set +a   # MSI creds (step 1) + SITE_AWS_* (steps 3-4)
#   export SITE_BUCKET=... SITE_CLOUDFRONT_DIST_ID=...
#   ./scripts/deploy_site.sh
set -o pipefail
: "${SITE_BUCKET:?set SITE_BUCKET, the S3 bucket serving the site}"
: "${SITE_CLOUDFRONT_DIST_ID:?set SITE_CLOUDFRONT_DIST_ID}"
: "${SITE_AWS_ACCESS_KEY_ID:?set SITE_AWS_ACCESS_KEY_ID (site-hosting account, in .env)}"
: "${SITE_AWS_SECRET_ACCESS_KEY:?set SITE_AWS_SECRET_ACCESS_KEY (site-hosting account, in .env)}"

CORPUS="${CORPUS:-arxiv}"
PROMPT_VERSION="${PROMPT_VERSION:-p1}"
MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
COUNT_POLICY="${COUNT_POLICY:-floor1}"

echo "== 1/4  exporting chart JSON + drill-down bundle from MSI S3 =="
python -m accelscan.scripts.export_site_data --out site_data \
  --corpus "$CORPUS" --prompt-version "$PROMPT_VERSION" \
  --model-tag "$MODEL_TAG" --count-policy "$COUNT_POLICY"

echo "== 2/4  building the static site (npm) =="
npm --prefix site run build

# From here on, every `aws` call uses the site-hosting account -- MSI's
# AWS_ACCESS_KEY_ID/SECRET (still exported from .env, needed by step 1 above)
# are overridden, not merely shadowed by a second pair sitting alongside them.
export AWS_ACCESS_KEY_ID="$SITE_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$SITE_AWS_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="${SITE_AWS_REGION:-us-east-1}"

echo "== 3/4  syncing to s3://$SITE_BUCKET =="
aws s3 sync site/dist/ "s3://$SITE_BUCKET/" --delete
aws s3 sync site_data/ "s3://$SITE_BUCKET/data/" --delete

echo "== 4/4  invalidating CloudFront cache =="
aws cloudfront create-invalidation --distribution-id "$SITE_CLOUDFRONT_DIST_ID" --paths '/*'

echo "done."
