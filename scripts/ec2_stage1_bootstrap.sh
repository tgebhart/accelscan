#!/bin/bash -e
# EC2 user-data: run arXiv stage 1 in us-east-1, write results to MSI S3.
#
# WHY EC2: reading s3://arxiv (requester-pays) from EC2 in us-east-1 costs $0 in
# transfer; reading the same 2.9 TB from MSI is AWS internet egress (~$260). Only
# stage 1 moves -- it is CPU-only and needs neither vLLM nor torch. Everything
# after it (repack -> infer -> analytics) runs on MSI against the candidates this
# writes, so the handoff is just an S3 prefix.
#
# Launch (one instance per YYMM slice; c7i.8xlarge spot is ~$0.30-0.50/h):
#   aws ec2 run-instances --image-id <al2023-ami> --instance-type c7i.8xlarge \
#     --instance-market-options MarketType=spot \
#     --user-data file://scripts/ec2_stage1_bootstrap.sh \
#     --iam-instance-profile Name=<profile-that-can-read-requester-pays>
#
# Required in the environment (SSM parameters, or baked into user-data):
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   MSI keys, for writing outputs
#   ARXIV_AWS_ACCESS_KEY_ID / ..._SECRET        AWS keys, if no instance role
#   GITHUB_TOKEN                                read access; the repo is private
#   YYMM_RANGE                                  e.g. 9108-0512 (this slice)
# Optional: ACCELSCAN_REPO (default below), ACCELSCAN_REF (default main)
set -o pipefail
: "${YYMM_RANGE:?set YYMM_RANGE, e.g. 9108-0512}"
ACCELSCAN_REPO="${ACCELSCAN_REPO:-github.com/tgebhart/accelscan.git}"
ACCELSCAN_REF="${ACCELSCAN_REF:-main}"

REGION=$(curl -s -X PUT http://169.254.169.254/latest/api/token \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  | xargs -I{} curl -s -H "X-aws-ec2-metadata-token: {}" \
    http://169.254.169.254/latest/meta-data/placement/region)
if [ "$REGION" != "us-east-1" ]; then
  echo "REFUSING: instance is in '$REGION', not us-east-1 -- that would bill egress." >&2
  exit 1
fi
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1

dnf install -y -q python3.12 python3.12-pip git

# private repo: token over HTTPS on an ephemeral instance, else an ssh deploy key
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git clone --depth 1 --branch "$ACCELSCAN_REF" \
    "https://x-access-token:${GITHUB_TOKEN}@${ACCELSCAN_REPO}" /opt/accelscan
elif [ -f /root/.ssh/id_ed25519 ]; then
  GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new' \
    git clone --depth 1 --branch "$ACCELSCAN_REF" \
      "git@${ACCELSCAN_REPO/\//:}" /opt/accelscan
else
  echo 'REFUSING: set GITHUB_TOKEN or install a deploy key at /root/.ssh/id_ed25519' >&2
  exit 1
fi
cd /opt/accelscan
git log --oneline -1                       # record the exact code being run

# stage-1 deps only: no torch, no vllm, no bertopic
python3.12 -m pip install -q --upgrade pip
python3.12 -m pip install -q polars boto3 orjson tenacity pyahocorasick pyyaml

# registry/ is repo-relative, so run from the repo root (see accelscan/registry.py)
python3.12 -m accelscan.arxiv_scan --dry-run --yymm "$YYMM_RANGE" | tail -25
python3.12 -m accelscan.arxiv_scan --yymm "$YYMM_RANGE" --max-workers "$(nproc)" \
  2>&1 | tee "/var/log/arxiv_scan_${YYMM_RANGE}.log"

echo "stage 1 complete for $YYMM_RANGE; outputs are on MSI S3 under accelscan/arxiv/"
# shutdown -h now   # uncomment for fire-and-forget spot instances
