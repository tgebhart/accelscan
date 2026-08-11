# Deploying the public arXiv-results website

The site (`site/`) is a static bundle: 7 headline charts (Plotly.js) plus a
paper drill-down browser that queries published parquet directly from the
visitor's browser (DuckDB-WASM). There is no backend, no database service,
and no server to operate — hosting is a public-read S3 bucket behind
CloudFront, and "redeploying" is re-running one script.

## One-time setup

Everything below is plain `aws` CLI — no Terraform/CDK, to keep this a small,
auditable list of commands rather than another thing to maintain.

```bash
BUCKET=accelscan-site        # pick a globally-unique name
REGION=us-east-1

# 1. bucket, block-public-access disabled ONLY on this bucket (the site is
#    meant to be public; nothing else in the AWS account should be)
aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false

# 2. bucket policy: public READ of objects only (no bucket listing, no writes)
cat > /tmp/site-bucket-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadOnly",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::$BUCKET/*"
  }]
}
EOF
aws s3api put-bucket-policy --bucket "$BUCKET" --policy file:///tmp/site-bucket-policy.json

# 3. static website hosting (index/error documents)
aws s3api put-bucket-website --bucket "$BUCKET" --website-configuration \
  '{"IndexDocument": {"Suffix": "index.html"}, "ErrorDocument": {"Key": "index.html"}}'

# 4. CloudFront distribution in front, for HTTPS (S3 website endpoints are
#    HTTP-only) and caching. S3 and CloudFront both support HTTP Range
#    requests by default, which is what DuckDB-WASM's partial parquet reads
#    depend on -- nothing extra to configure for that.
aws cloudfront create-distribution --origin-domain-name "$BUCKET.s3-website-$REGION.amazonaws.com" \
  --default-root-object index.html
```

Note the distribution id and domain name (`aws cloudfront list-distributions`
if you lose them) — they're the `SITE_CLOUDFRONT_DIST_ID` env var and the
public URL respectively.

## Recurring deploy (~monthly, after a bulk-release re-ingestion)

```bash
cd accelscan
set -a; source .env; set +a               # MSI S3 read credentials
export SITE_BUCKET=accelscan-site
export SITE_CLOUDFRONT_DIST_ID=E1234EXAMPLE
./scripts/deploy_site.sh
```

This runs `accelscan.scripts.export_site_data` (reads the arXiv analytic/
capacity/mentions tables from MSI S3, writes `site_data/` locally), builds
`site/` with `npm run build`, syncs both to the bucket, and invalidates the
CloudFront cache. Nothing runs on a schedule — it's a single command to run
by hand whenever there's new data.

`site_data/` and `site/dist/` are build artifacts, not checked in; deploying
from a clean checkout regenerates both from scratch every time.

## Local development / verification before deploying

```bash
python -m accelscan.scripts.export_site_data --out site_data
ln -s ../../site_data site/public/data      # Vite serves this at /data/...
npm --prefix site install
npm --prefix site run dev                   # http://localhost:5173
```

Check: all 7 charts render on the home page, the paper browser's filters and
pagination work, and every "view on arXiv" link resolves. `npm --prefix site
run build && npx --prefix site vite preview` serves the production build the
same way, for a closer check before syncing to S3.

## Cost

Storage: `site/dist/` (a few hundred KB) + `site_data/` (~40MB at the
2026-08 data volume) ≈ pennies/month in S3. CloudFront's free tier covers
1TB/month of data transfer and 10M requests — this site's traffic is expected
to sit well inside that, so ongoing cost should be near $0 unless traffic
grows substantially.
