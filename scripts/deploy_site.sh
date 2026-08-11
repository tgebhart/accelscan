#!/bin/bash -e
# Rebuild + redeploy the public arXiv-results website (site/) to S3+CloudFront.
#
# Run this after each ~monthly bulk-release re-ingestion (re-extraction, or a
# new registry/analytic version landing on MSI S3). See docs/site_deploy.md
# for the one-time bucket/CloudFront setup this assumes already exists.
#
# Required in the environment:
#   SITE_BUCKET               the S3 bucket serving the site (one-time setup)
#   SITE_CLOUDFRONT_DIST_ID   CloudFront distribution id, for the invalidation
# Optional (accelscan.scripts.export_site_data flags): CORPUS, PROMPT_VERSION,
#   MODEL_TAG, COUNT_POLICY -- defaults match the notebooks' knobs cells.
#
#   set -a; source .env; set +a   # MSI S3 read credentials
#   export SITE_BUCKET=... SITE_CLOUDFRONT_DIST_ID=...
#   ./scripts/deploy_site.sh
set -o pipefail
: "${SITE_BUCKET:?set SITE_BUCKET, the S3 bucket serving the site}"
: "${SITE_CLOUDFRONT_DIST_ID:?set SITE_CLOUDFRONT_DIST_ID}"

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

echo "== 3/4  syncing to s3://$SITE_BUCKET =="
aws s3 sync site/dist/ "s3://$SITE_BUCKET/" --delete
aws s3 sync site_data/ "s3://$SITE_BUCKET/data/" --delete

echo "== 4/4  invalidating CloudFront cache =="
aws cloudfront create-invalidation --distribution-id "$SITE_CLOUDFRONT_DIST_ID" --paths '/*'

echo "done."
