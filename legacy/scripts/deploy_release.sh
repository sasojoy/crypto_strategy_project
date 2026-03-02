#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ssh_target>  # e.g. ubuntu@trader-1"
  exit 1
fi

HOST="$1"
BASE="/opt/crypto_strategy_project"
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
SHA="$(git rev-parse --short=8 HEAD)"
REL_DIR="${BASE}/releases/${STAMP}_${SHA}"

ssh "$HOST" "mkdir -p '$REL_DIR'"
rsync -a --delete --exclude '.venv' ./ "$HOST:$REL_DIR/"
ssh "$HOST" "cd '$REL_DIR' && ${BASE}/.venv/bin/pip install -r requirements.txt"
ssh "$HOST" "find '$REL_DIR' -name '__pycache__' -type d -exec rm -rf {} +; find '$REL_DIR' -name '*.pyc' -delete"
ssh "$HOST" "ln -sfn '$REL_DIR' ${BASE}/current"
ssh "$HOST" "sudo systemctl daemon-reload && sudo systemctl restart trader-once.timer"
echo "Deployed to $HOST  build=${SHA}  at ${REL_DIR}"
